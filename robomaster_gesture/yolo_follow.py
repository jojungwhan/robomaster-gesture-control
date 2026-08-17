"""YOLO object tracking with fail-closed RoboMaster target following."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Optional, Sequence, Tuple

from .control_lease import ControlLeaseError, ControllerLease
from .control_status import DEFAULT_CONTROL_STATUS_PATH, ControlStatusPublisher
from .expanded_scene import ExpandedSceneRecognizer
from .models import GestureDecision, VelocityCommand, translation_directions
from .robot_adapter import (
    CommandPump,
    CommandPumpConfig,
    DjiRobotAdapter,
    DryRunRobot,
    RobotError,
    S1AppKeyboardAdapter,
)
from .scene_speech import (
    PiperSceneSpeaker,
    SceneNarrationPolicy,
    describe_scene,
    is_look_query,
    merge_scene_detections,
)
from .voice_control import (
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_PYTHON,
    WhisperSpeechRecognizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARENT = PROJECT_ROOT.parent
DEFAULT_PIPER_PYTHON = WORKSPACE_PARENT / ".venv-piper" / "Scripts" / "python.exe"
DEFAULT_PIPER_MODEL = (
    WORKSPACE_PARENT / "piper-voices" / "en_US-kristin-medium.onnx"
)
DEFAULT_PIPER_WORKER = PROJECT_ROOT / "scripts" / "piper_scene_worker.py"
DEFAULT_EXPANDED_SCENE_MODEL = PROJECT_ROOT / "yoloe-26n-seg-pf.pt"
DEFAULT_EXPANDED_SCENE_WORKER = PROJECT_ROOT / "scripts" / "expanded_scene_worker.py"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]
    track_id: Optional[int] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class FollowConfig:
    target_label: str = "bottle"
    target_confidence: float = 0.35
    minimum_lock_frames: int = 3
    center_deadzone_ratio: float = 0.12
    stop_height_ratio: float = 0.38
    resume_height_ratio: float = 0.31
    forward_speed_m_s: float = 0.15
    strafe_speed_m_s: float = 0.12
    person_stop_confidence: float = 0.30
    spatial_match_iou: float = 0.20

    def validate(self) -> None:
        if not self.target_label.strip():
            raise ValueError("target label cannot be empty")
        if not 0.0 <= self.target_confidence <= 1.0:
            raise ValueError("target confidence must be between 0 and 1")
        if self.minimum_lock_frames < 1:
            raise ValueError("minimum lock frames must be positive")
        if not 0.01 <= self.center_deadzone_ratio <= 0.45:
            raise ValueError("center deadzone ratio must be between 0.01 and 0.45")
        if not 0.05 <= self.resume_height_ratio < self.stop_height_ratio <= 0.90:
            raise ValueError("target height ratios must have resume < stop")
        if not 0.01 <= self.forward_speed_m_s <= 0.50:
            raise ValueError("forward speed must be between 0.01 and 0.50 m/s")
        if not 0.01 <= self.strafe_speed_m_s <= 0.50:
            raise ValueError("strafe speed must be between 0.01 and 0.50 m/s")
        if not 0.0 <= self.person_stop_confidence <= 1.0:
            raise ValueError("person stop confidence must be between 0 and 1")


@dataclass(frozen=True)
class FollowDecision:
    state: str
    reason: str
    command: VelocityCommand
    target: Optional[Detection] = None
    lock_frames: int = 0
    horizontal_error: float = 0.0
    height_ratio: float = 0.0


@dataclass(frozen=True)
class CameraFreshness:
    fresh: bool
    reason: str
    changed_fraction: float = 0.0
    confirmation_frames: int = 0


class CameraFreshnessGuard:
    """Require changing camera pixels before allowing follower decisions.

    RoboMaster app capture also succeeds on menus and on a frozen render.  This
    guard samples the unobstructed center of the frame and fails closed until
    several meaningful changes have arrived.  Once armed, a prolonged freeze
    disarms it and requires the changing-frame sequence again.
    """

    def __init__(
        self,
        minimum_changed_frames: int = 3,
        minimum_changed_fraction: float = 0.002,
        pixel_delta: float = 2.0,
        freeze_after_s: float = 0.75,
        sample_step: int = 8,
    ):
        if minimum_changed_frames < 1:
            raise ValueError("minimum changed frames must be positive")
        if not 0.0 < minimum_changed_fraction <= 1.0:
            raise ValueError("minimum changed fraction must be in (0, 1]")
        if pixel_delta <= 0.0:
            raise ValueError("pixel delta must be positive")
        if freeze_after_s <= 0.0:
            raise ValueError("freeze timeout must be positive")
        if sample_step < 1:
            raise ValueError("sample step must be positive")
        self.minimum_changed_frames = minimum_changed_frames
        self.minimum_changed_fraction = minimum_changed_fraction
        self.pixel_delta = pixel_delta
        self.freeze_after_s = freeze_after_s
        self.sample_step = sample_step
        self._previous = None
        self._last_change_at = None  # type: Optional[float]
        self._confirmation_frames = 0
        self._fresh = False

    def update(self, frame, now_s: Optional[float] = None) -> CameraFreshness:
        import numpy as np

        now_s = time.monotonic() if now_s is None else float(now_s)
        if frame is None or getattr(frame, "ndim", 0) not in (2, 3):
            return self.reset("invalid camera frame - stopped")
        height, width = frame.shape[:2]
        if height < 5 or width < 5:
            return self.reset("invalid camera frame - stopped")

        # Avoid the app's outer chrome and most live-view HUD elements.
        sample = frame[
            height // 5 : (4 * height) // 5 : self.sample_step,
            width // 5 : (4 * width) // 5 : self.sample_step,
        ]
        if sample.ndim == 3:
            sample = sample[..., :3].astype(np.float32).mean(axis=2)
        else:
            sample = sample.astype(np.float32)

        if self._previous is None or self._previous.shape != sample.shape:
            self._previous = sample.copy()
            self._last_change_at = now_s
            self._confirmation_frames = 0
            self._fresh = False
            return self._waiting(0.0)

        difference = np.abs(sample - self._previous)
        self._previous = sample.copy()
        changed_fraction = float(np.count_nonzero(difference >= self.pixel_delta)) / float(
            difference.size
        )
        if changed_fraction >= self.minimum_changed_fraction:
            self._last_change_at = now_s
            if not self._fresh:
                self._confirmation_frames += 1
                if self._confirmation_frames >= self.minimum_changed_frames:
                    self._fresh = True
            return CameraFreshness(
                self._fresh,
                "camera stream changing"
                if self._fresh
                else self._waiting_reason(),
                changed_fraction,
                self._confirmation_frames,
            )

        if (
            self._fresh
            and self._last_change_at is not None
            and now_s - self._last_change_at >= self.freeze_after_s
        ):
            self._fresh = False
            self._confirmation_frames = 0
            return CameraFreshness(
                False,
                "camera frame frozen - stopped",
                changed_fraction,
                0,
            )
        if (
            not self._fresh
            and self._last_change_at is not None
            and now_s - self._last_change_at >= self.freeze_after_s
        ):
            self._confirmation_frames = 0
            return CameraFreshness(
                False,
                "camera frame frozen - stopped",
                changed_fraction,
                0,
            )
        if self._fresh:
            return CameraFreshness(
                True,
                "camera stream changing",
                changed_fraction,
                self._confirmation_frames,
            )
        return self._waiting(changed_fraction)

    def reset(self, reason: str = "camera freshness reset") -> CameraFreshness:
        self._previous = None
        self._last_change_at = None
        self._confirmation_frames = 0
        self._fresh = False
        return CameraFreshness(False, reason)

    def _waiting_reason(self) -> str:
        return "verifying camera freshness ({}/{})".format(
            self._confirmation_frames,
            self.minimum_changed_frames,
        )

    def _waiting(self, changed_fraction: float) -> CameraFreshness:
        return CameraFreshness(
            False,
            self._waiting_reason(),
            changed_fraction,
            self._confirmation_frames,
        )


def _intersection_over_union(first: Detection, second: Detection) -> float:
    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = first.area + second.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


class TargetFollower:
    """Lock one target and translate its fresh position into safe movement."""

    def __init__(self, config: FollowConfig = None):
        self.config = config or FollowConfig()
        self.config.validate()
        self._target_label = self.config.target_label.casefold()
        self._locked_track_id = None  # type: Optional[int]
        self._last_target = None  # type: Optional[Detection]
        self._lock_frames = 0
        self._near_target = False

    def reset(self, reason: str = "reset") -> FollowDecision:
        self._locked_track_id = None
        self._last_target = None
        self._lock_frames = 0
        self._near_target = False
        return FollowDecision("STOPPED", reason, VelocityCommand.stopped())

    def update(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
    ) -> FollowDecision:
        if frame_width <= 0 or frame_height <= 0:
            return self.reset("invalid frame dimensions")

        people = [
            item
            for item in detections
            if item.label.casefold() == "person"
            and item.confidence >= self.config.person_stop_confidence
        ]
        if people and self._target_label != "person":
            return self.reset("person detected - protective stop")

        candidates = [
            item
            for item in detections
            if item.label.casefold() == self._target_label
            and item.confidence >= self.config.target_confidence
        ]
        if not candidates:
            return self.reset("target lost - stopped")

        target = self._select_target(candidates)
        if target is None:
            return self.reset("locked target lost - stopped")

        if self._same_target(target):
            self._lock_frames += 1
        else:
            self._locked_track_id = target.track_id
            self._lock_frames = 1
            self._near_target = False
        self._last_target = target

        if self._lock_frames < self.config.minimum_lock_frames:
            return FollowDecision(
                "ACQUIRING",
                "locking {} ({}/{})".format(
                    target.label,
                    self._lock_frames,
                    self.config.minimum_lock_frames,
                ),
                VelocityCommand.stopped(),
                target=target,
                lock_frames=self._lock_frames,
            )

        center_x, _ = target.center
        horizontal_error = (center_x - frame_width / 2.0) / (frame_width / 2.0)
        height_ratio = max(0.0, target.box[3] - target.box[1]) / frame_height

        if self._near_target:
            if height_ratio > self.config.resume_height_ratio:
                return FollowDecision(
                    "HOLDING",
                    "target within stop range",
                    VelocityCommand.stopped(),
                    target,
                    self._lock_frames,
                    horizontal_error,
                    height_ratio,
                )
            self._near_target = False
        if height_ratio >= self.config.stop_height_ratio:
            self._near_target = True
            return FollowDecision(
                "HOLDING",
                "target reached stop size",
                VelocityCommand.stopped(),
                target,
                self._lock_frames,
                horizontal_error,
                height_ratio,
            )

        if abs(horizontal_error) > self.config.center_deadzone_ratio:
            command = VelocityCommand(
                right_m_s=math.copysign(
                    self.config.strafe_speed_m_s, horizontal_error
                )
            )
            direction = "right" if horizontal_error > 0.0 else "left"
            return FollowDecision(
                "FOLLOWING",
                "align {} toward {}".format(direction, target.label),
                command,
                target,
                self._lock_frames,
                horizontal_error,
                height_ratio,
            )

        return FollowDecision(
            "FOLLOWING",
            "approach {}".format(target.label),
            VelocityCommand(forward_m_s=self.config.forward_speed_m_s),
            target,
            self._lock_frames,
            horizontal_error,
            height_ratio,
        )

    def _select_target(
        self, candidates: Sequence[Detection]
    ) -> Optional[Detection]:
        if self._locked_track_id is not None:
            for item in candidates:
                if item.track_id == self._locked_track_id:
                    return item
            return None

        if self._last_target is not None:
            spatial = max(
                candidates,
                key=lambda item: _intersection_over_union(self._last_target, item),
            )
            if (
                _intersection_over_union(self._last_target, spatial)
                >= self.config.spatial_match_iou
            ):
                return spatial

        return max(candidates, key=lambda item: (item.area, item.confidence))

    def _same_target(self, target: Detection) -> bool:
        if self._last_target is None:
            return False
        if self._locked_track_id is not None or target.track_id is not None:
            return (
                self._locked_track_id is not None
                and target.track_id == self._locked_track_id
            )
        return (
            _intersection_over_union(self._last_target, target)
            >= self.config.spatial_match_iou
        )


class UltralyticsTracker:
    def __init__(
        self,
        model_name: str,
        confidence: float,
        image_size: int,
        device: str,
        tracker: str,
        target_label: str,
        persist_tracks: bool = True,
        describe_all: bool = False,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run .\\setup_yolo.ps1 first."
            ) from exc
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.tracker = tracker
        self.persist_tracks = persist_tracks
        raw_names = self.model.names
        self.names = (
            dict(raw_names)
            if isinstance(raw_names, dict)
            else dict(enumerate(raw_names))
        )
        if describe_all:
            self.class_ids = None
        else:
            wanted = {target_label.casefold(), "person"}
            self.class_ids = [
                class_id
                for class_id, label in self.names.items()
                if str(label).casefold() in wanted
            ]
        if not any(
            str(label).casefold() == target_label.casefold()
            for label in self.names.values()
        ):
            raise RuntimeError(
                "YOLO model has no class named {!r}. Available examples: {}".format(
                    target_label,
                    ", ".join(str(item) for item in list(self.names.values())[:20]),
                )
            )

    def track(self, frame) -> Tuple[Detection, ...]:
        arguments = dict(
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if self.class_ids is not None:
            arguments["classes"] = self.class_ids
        if self.persist_tracks:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                **arguments
            )
        else:
            results = self.model.predict(frame, **arguments)
        detections = []
        if not results:
            return ()
        boxes = results[0].boxes
        if boxes is None:
            return ()
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(xyxy)
        )
        for coordinates, confidence, class_id, track_id in zip(
            xyxy, confidences, classes, track_ids
        ):
            detections.append(
                Detection(
                    label=str(self.names[int(class_id)]),
                    confidence=float(confidence),
                    box=tuple(float(value) for value in coordinates),
                    track_id=None if track_id is None else int(track_id),
                )
            )
        return tuple(detections)


class OpenCvFrameSource:
    def __init__(self, source, loop_file: bool = False):
        self.source = source
        self.loop_file = loop_file
        self.capture = None

    def open(self) -> None:
        import cv2

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError("Could not open video source: {}".format(self.source))

    def read(self):
        if self.capture is None:
            raise RuntimeError("Video source is not open")
        ok, frame = self.capture.read()
        if ok:
            return frame
        if self.loop_file and not isinstance(self.source, int):
            import cv2

            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if ok:
                return frame
        return None

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class StaticImageSource:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.frame = None

    def open(self) -> None:
        import cv2

        self.frame = cv2.imread(str(self.path))
        if self.frame is None:
            raise RuntimeError("Could not read image: {}".format(self.path))

    def read(self):
        return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.frame = None


class DjiSdkFrameSource:
    def __init__(self, adapter: DjiRobotAdapter, resolution: str = "360p"):
        self.adapter = adapter
        self.resolution = resolution
        self._camera = None

    def open(self) -> None:
        self._camera = self.adapter.camera
        result = self._camera.start_video_stream(
            display=False, resolution=self.resolution
        )
        if result is False:
            self._camera = None
            raise RuntimeError("RoboMaster SDK camera stream did not start")

    def read(self):
        if self._camera is None:
            raise RuntimeError("RoboMaster SDK camera is not open")
        return self._camera.read_cv2_image(timeout=0.5, strategy="newest")

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop_video_stream()
            finally:
                self._camera = None


class RoboMasterAppFrameSource:
    """Capture the DJI app client without activating or focusing its window."""

    def __init__(self, window_title: str = "RoboMaster"):
        self.window_title = window_title
        self._window = 0
        self._ctypes = None
        self._wintypes = None
        self._user32 = None
        self._gdi32 = None

    def open(self) -> None:
        if os.name != "nt":
            raise RuntimeError("RoboMaster app capture is only available on Windows")
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        self._user32.FindWindowW.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = (wintypes.HWND,)
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.GetWindowRect.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.RECT),
        )
        self._user32.GetWindowRect.restype = wintypes.BOOL
        self._user32.GetClientRect.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.RECT),
        )
        self._user32.GetClientRect.restype = wintypes.BOOL
        self._user32.ClientToScreen.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.POINT),
        )
        self._user32.ClientToScreen.restype = wintypes.BOOL
        self._user32.GetWindowDC.argtypes = (wintypes.HWND,)
        self._user32.GetWindowDC.restype = wintypes.HDC
        self._user32.ReleaseDC.argtypes = (wintypes.HWND, wintypes.HDC)
        self._user32.ReleaseDC.restype = ctypes.c_int
        self._user32.PrintWindow.argtypes = (
            wintypes.HWND,
            wintypes.HDC,
            wintypes.UINT,
        )
        self._user32.PrintWindow.restype = wintypes.BOOL
        self._gdi32.CreateCompatibleDC.argtypes = (wintypes.HDC,)
        self._gdi32.CreateCompatibleDC.restype = wintypes.HDC
        self._gdi32.CreateCompatibleBitmap.argtypes = (
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
        )
        self._gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        self._gdi32.SelectObject.argtypes = (wintypes.HDC, wintypes.HGDIOBJ)
        self._gdi32.SelectObject.restype = wintypes.HGDIOBJ
        self._gdi32.DeleteObject.argtypes = (wintypes.HGDIOBJ,)
        self._gdi32.DeleteObject.restype = wintypes.BOOL
        self._gdi32.DeleteDC.argtypes = (wintypes.HDC,)
        self._gdi32.DeleteDC.restype = wintypes.BOOL
        self._gdi32.GetDIBits.argtypes = (
            wintypes.HDC,
            wintypes.HBITMAP,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.UINT,
        )
        self._gdi32.GetDIBits.restype = ctypes.c_int
        self._window = int(self._user32.FindWindowW(None, self.window_title) or 0)
        if not self._window:
            raise RuntimeError(
                "RoboMaster app window not found; open its live drive view first"
            )

    def read(self):
        import ctypes
        import numpy as np

        if not self._window or not self._user32.IsWindow(self._window):
            raise RuntimeError("RoboMaster app window closed")
        wintypes = self._wintypes
        user32 = self._user32
        gdi32 = self._gdi32

        window_rect = wintypes.RECT()
        client_rect = wintypes.RECT()
        client_origin = wintypes.POINT(0, 0)
        if not user32.GetWindowRect(self._window, ctypes.byref(window_rect)):
            raise RuntimeError("Could not read RoboMaster window dimensions")
        if not user32.GetClientRect(self._window, ctypes.byref(client_rect)):
            raise RuntimeError("Could not read RoboMaster client dimensions")
        if not user32.ClientToScreen(self._window, ctypes.byref(client_origin)):
            raise RuntimeError("Could not locate RoboMaster client area")

        width = window_rect.right - window_rect.left
        height = window_rect.bottom - window_rect.top
        client_width = client_rect.right - client_rect.left
        client_height = client_rect.bottom - client_rect.top
        if min(width, height, client_width, client_height) <= 0:
            raise RuntimeError("RoboMaster app window has invalid dimensions")

        window_dc = user32.GetWindowDC(self._window)
        if not window_dc:
            raise RuntimeError("Could not access the RoboMaster window image")
        memory_dc = gdi32.CreateCompatibleDC(window_dc)
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        if not memory_dc or not bitmap:
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if memory_dc:
                gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(self._window, window_dc)
            raise RuntimeError("Could not allocate a RoboMaster capture frame")
        old_bitmap = gdi32.SelectObject(memory_dc, bitmap)
        if not old_bitmap:
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(self._window, window_dc)
            raise RuntimeError("Could not prepare the RoboMaster capture frame")
        try:
            if not user32.PrintWindow(self._window, memory_dc, 2):
                raise RuntimeError("Windows could not capture the RoboMaster app")

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = (
                    ("biSize", wintypes.DWORD),
                    ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD),
                )

            class BITMAPINFO(ctypes.Structure):
                _fields_ = (
                    ("bmiHeader", BITMAPINFOHEADER),
                    ("bmiColors", wintypes.DWORD * 3),
                )

            info = BITMAPINFO()
            info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = 0
            buffer = (ctypes.c_ubyte * (width * height * 4))()
            rows = gdi32.GetDIBits(
                memory_dc,
                bitmap,
                0,
                height,
                ctypes.byref(buffer),
                ctypes.byref(info),
                0,
            )
            if rows != height:
                raise RuntimeError("Windows returned an incomplete RoboMaster frame")
            bgra = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
            offset_x = client_origin.x - window_rect.left
            offset_y = client_origin.y - window_rect.top
            return bgra[
                offset_y : offset_y + client_height,
                offset_x : offset_x + client_width,
                :3,
            ].copy()
        finally:
            gdi32.SelectObject(memory_dc, old_bitmap)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(self._window, window_dc)

    def close(self) -> None:
        self._window = 0


class NonActivatingPreview:
    """Always-on-top preview that cannot steal S1 keyboard focus."""

    def __init__(self, width: int = 720, height: int = 480):
        import tkinter as tk

        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("YOLO Follow Preview")
        self.root.overrideredirect(True)
        self.root.configure(bg="#071018")
        self.label = tk.Label(self.root, bg="#071018", bd=0)
        self.label.pack(fill="both", expand=True)
        self.root.geometry("{}x{}+18+72".format(width, height))
        self.root.update_idletasks()
        self._photo = None
        self.closed = False
        self._show_without_activation()

    def _show_without_activation(self) -> None:
        if os.name != "nt":
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            return
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetParent.argtypes = (wintypes.HWND,)
        user32.GetParent.restype = wintypes.HWND
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.GetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowLongPtrW.argtypes = (
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_ssize_t,
        )
        user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        child = wintypes.HWND(self.root.winfo_id())
        wrapper = user32.GetParent(child) or child
        foreground = user32.GetForegroundWindow()
        exstyle = int(user32.GetWindowLongPtrW(wrapper, -20))
        user32.SetWindowLongPtrW(
            wrapper,
            -20,
            exstyle | 0x00000020 | 0x00000080 | 0x08000000,
        )
        self.root.deiconify()
        self.root.update_idletasks()
        user32.SetWindowPos(
            wrapper,
            wintypes.HWND(-1),
            18,
            72,
            self.width,
            self.height,
            0x0010 | 0x0040 | 0x0020,
        )
        if foreground and user32.GetForegroundWindow() != foreground:
            user32.SetForegroundWindow(foreground)

    def show(self, frame) -> None:
        if self.closed:
            return
        import cv2
        from PIL import Image, ImageTk

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self._photo)
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.closed = True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass


def annotate_frame(
    frame,
    detections,
    decision: FollowDecision,
    fps: float,
    narration_only=(),
):
    import cv2

    annotated = frame.copy()
    target = decision.target
    for item in narration_only:
        x1, y1, x2, y2 = (int(round(value)) for value in item.box)
        color = (220, 80, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            "SCENE {} {:.2f}".format(item.label, item.confidence),
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    for item in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in item.box)
        selected = target is item or (
            target is not None
            and target.track_id is not None
            and item.track_id == target.track_id
        )
        if item.label.casefold() == "person":
            color = (50, 50, 255)
        elif selected:
            color = (70, 240, 120)
        else:
            color = (255, 190, 70)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if selected else 2)
        identity = " #{}".format(item.track_id) if item.track_id is not None else ""
        cv2.putText(
            annotated,
            "{}{} {:.2f}".format(item.label, identity, item.confidence),
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    height, width = annotated.shape[:2]
    cv2.line(annotated, (width // 2, 0), (width // 2, height), (80, 140, 160), 1)
    directions = translation_directions(decision.command)
    movement = " + ".join(directions) if directions else "STOP"
    cv2.rectangle(annotated, (0, 0), (width, 62), (7, 16, 24), -1)
    cv2.putText(
        annotated,
        "{} | {} | {:.1f} FPS".format(decision.state, movement, fps),
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (80, 245, 170),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        decision.reason,
        (14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (210, 235, 245),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _write_frame_atomically(path: Path, frame) -> None:
    """Publish the latest annotated frame; a reader never sees a partial file.

    Used to embed the preview inside the RoboMaster Control Center, which runs a
    different Python and reads this JPEG instead of showing the pop-up window.
    """
    import cv2

    # The captured frame can be very large (full app window on a high-DPI
    # display); the embedded panel only needs a modest size, so downscale for the
    # feed. YOLO already ran on its own (letterboxed) input, so this is display
    # only.
    height, width = frame.shape[:2]
    max_width = 960
    if width > max_width:
        scale = max_width / float(width)
        frame = cv2.resize(
            frame,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    # Encode by format string, not by the temp file's extension, then write the
    # bytes and rename so a reader never observes a half-written JPEG.
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    temporary = Path(str(path) + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(buffer.tobytes())
        os.replace(str(temporary), str(path))
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a named YOLO object and safety-gate RoboMaster following. "
            "Dry-run is the default."
        )
    )
    parser.add_argument(
        "--source",
        choices=("robomaster-app", "sdk", "webcam", "file"),
        default="robomaster-app",
    )
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--window-title", default="RoboMaster")
    parser.add_argument("--target", default="bottle")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-lock-frames", type=int, default=3)
    parser.add_argument("--center-deadzone", type=float, default=0.12)
    parser.add_argument("--stop-height", type=float, default=0.38)
    parser.add_argument("--resume-height", type=float, default=0.31)
    parser.add_argument("--forward-speed", type=float, default=0.15)
    parser.add_argument("--strafe-speed", type=float, default=0.12)
    parser.add_argument("--person-stop-confidence", type=float, default=0.30)
    parser.add_argument(
        "--max-inference-seconds",
        type=float,
        default=1.50,
        help="stop rather than move when one inference exceeds this age",
    )
    parser.add_argument("--frame-interval", type=float, default=0.08)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--camera-check",
        action="store_true",
        help=(
            "verify a changing robot-mounted camera stream and exit without "
            "loading YOLO or enabling motion"
        ),
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--frame-out",
        type=Path,
        default=None,
        help="write each annotated frame to this JPEG so the Control Center can "
        "embed the view instead of showing the pop-up window",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="describe recognized objects aloud with local Piper neural TTS",
    )
    parser.add_argument(
        "--voice-describe",
        action="store_true",
        help=(
            "listen for 'tell me what you see' and speak the current scene on "
            "demand (uses the microphone and Piper TTS)"
        ),
    )
    parser.add_argument("--whisper-python", type=Path, default=DEFAULT_WHISPER_PYTHON)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-model-name", default="base.en")
    parser.add_argument(
        "--describe-request-file",
        type=Path,
        default=None,
        help="describe the current scene aloud whenever this file is touched, "
        "so a single external recognizer can drive both movement and scene "
        "queries without a second microphone listener",
    )
    parser.add_argument("--speech-confidence", type=float, default=0.45)
    parser.add_argument("--speech-stable-frames", type=int, default=3)
    parser.add_argument("--speech-repeat-seconds", type=float, default=12.0)
    parser.add_argument("--speech-clear-seconds", type=float, default=3.0)
    parser.add_argument("--speech-max-groups", type=int, default=4)
    parser.add_argument(
        "--basic-scene-only",
        action="store_true",
        help="disable the narration-only expanded furniture recognizer",
    )
    parser.add_argument("--expanded-scene-confidence", type=float, default=0.35)
    parser.add_argument("--expanded-scene-image-size", type=int, default=320)
    parser.add_argument("--expanded-scene-interval", type=float, default=1.5)
    parser.add_argument(
        "--expanded-scene-model", type=Path, default=DEFAULT_EXPANDED_SCENE_MODEL
    )
    parser.add_argument(
        "--expanded-scene-worker", type=Path, default=DEFAULT_EXPANDED_SCENE_WORKER
    )
    parser.add_argument("--piper-python", type=Path, default=DEFAULT_PIPER_PYTHON)
    parser.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--piper-worker", type=Path, default=DEFAULT_PIPER_WORKER)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--transport", choices=("sdk", "s1-app"), default="s1-app")
    parser.add_argument("--connection", choices=("ap", "sta", "rndis"), default="sta")
    parser.add_argument("--protocol", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--robot-ip")
    parser.add_argument("--local-ip")
    parser.add_argument("--serial-number")
    parser.add_argument("--status-file", type=Path, default=DEFAULT_CONTROL_STATUS_PATH)
    return parser


def _validate_args(args) -> FollowConfig:
    if args.camera_check and args.live:
        raise ValueError("--camera-check cannot be combined with --live")
    if args.camera_check and args.source not in ("robomaster-app", "sdk"):
        raise ValueError(
            "--camera-check requires --source robomaster-app or --source sdk"
        )
    if args.camera_check and args.speak:
        raise ValueError("--camera-check cannot be combined with --speak")
    if args.source == "file" and args.input_file is None:
        raise ValueError("--source file requires --input-file")
    if args.input_file is not None and not args.input_file.is_file():
        raise ValueError("input file not found: {}".format(args.input_file))
    if args.source == "sdk" and args.transport != "sdk":
        raise ValueError("--source sdk requires --transport sdk")
    if args.live and args.source not in ("robomaster-app", "sdk"):
        raise ValueError(
            "live following requires the robot-mounted camera: "
            "--source robomaster-app or --source sdk"
        )
    if args.live and args.source == "robomaster-app" and args.transport != "s1-app":
        raise ValueError("--source robomaster-app requires --transport s1-app in live mode")
    if args.live and args.target.casefold() == "person":
        raise ValueError("live following of people is prohibited")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.image_size < 160:
        raise ValueError("--image-size must be at least 160")
    if not 0.02 <= args.frame_interval <= 1.0:
        raise ValueError("--frame-interval must be between 0.02 and 1 second")
    if not 0.10 <= args.max_inference_seconds <= 5.0:
        raise ValueError("--max-inference-seconds must be between 0.10 and 5 seconds")
    if args.duration < 0.0:
        raise ValueError("--duration cannot be negative")
    if not 0.0 <= args.speech_confidence <= 1.0:
        raise ValueError("--speech-confidence must be between 0 and 1")
    if args.speech_stable_frames < 1:
        raise ValueError("--speech-stable-frames must be positive")
    if args.speech_repeat_seconds < 1.0:
        raise ValueError("--speech-repeat-seconds must be at least one second")
    if args.speech_clear_seconds < 0.0:
        raise ValueError("--speech-clear-seconds cannot be negative")
    if not 1 <= args.speech_max_groups <= 8:
        raise ValueError("--speech-max-groups must be between 1 and 8")
    if not 0.0 <= args.expanded_scene_confidence <= 1.0:
        raise ValueError("--expanded-scene-confidence must be between 0 and 1")
    if args.expanded_scene_image_size < 160:
        raise ValueError("--expanded-scene-image-size must be at least 160")
    if not 0.5 <= args.expanded_scene_interval <= 60.0:
        raise ValueError("--expanded-scene-interval must be between 0.5 and 60 seconds")
    config = FollowConfig(
        target_label=args.target,
        target_confidence=args.confidence,
        minimum_lock_frames=args.minimum_lock_frames,
        center_deadzone_ratio=args.center_deadzone,
        stop_height_ratio=args.stop_height,
        resume_height_ratio=args.resume_height,
        forward_speed_m_s=args.forward_speed,
        strafe_speed_m_s=args.strafe_speed,
        person_stop_confidence=args.person_stop_confidence,
    )
    config.validate()
    return config


def _make_motion_adapter(args):
    if not args.live:
        return DryRunRobot()
    if args.transport == "s1-app":
        return S1AppKeyboardAdapter()
    return DjiRobotAdapter(
        conn_type=args.connection,
        proto_type=args.protocol,
        robot_ip=args.robot_ip,
        local_ip=args.local_ip,
        serial_number=args.serial_number,
    )


def _make_frame_source(args, motion_adapter):
    camera_adapter = None
    if args.source == "robomaster-app":
        return RoboMasterAppFrameSource(args.window_title), camera_adapter
    if args.source == "webcam":
        return OpenCvFrameSource(args.webcam_index), camera_adapter
    if args.source == "file":
        image_suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
        if args.input_file.suffix.casefold() in image_suffixes:
            return StaticImageSource(args.input_file), camera_adapter
        return OpenCvFrameSource(str(args.input_file), loop_file=False), camera_adapter

    if args.live:
        camera_adapter = motion_adapter
    else:
        camera_adapter = DjiRobotAdapter(
            conn_type=args.connection,
            proto_type=args.protocol,
            robot_ip=args.robot_ip,
            local_ip=args.local_ip,
            serial_number=args.serial_number,
        )
        camera_adapter.connect()
    return DjiSdkFrameSource(camera_adapter), camera_adapter


def _run_camera_check(args) -> int:
    frame_source = None
    camera_adapter = None
    freshness_guard = CameraFreshnessGuard()
    timeout_s = args.duration if args.duration > 0.0 else 8.0
    deadline_s = time.monotonic() + timeout_s
    last_result = CameraFreshness(False, "waiting for first camera frame")

    try:
        if args.source == "robomaster-app":
            frame_source = RoboMasterAppFrameSource(args.window_title)
        else:
            camera_adapter = DjiRobotAdapter(
                conn_type=args.connection,
                proto_type=args.protocol,
                robot_ip=args.robot_ip,
                local_ip=args.local_ip,
                serial_number=args.serial_number,
            )
            camera_adapter.connect()
            frame_source = DjiSdkFrameSource(camera_adapter)
        frame_source.open()
        print(
            "CAMERA CHECK: verifying a changing {} stream for up to {:.1f}s. "
            "YOLO, narration, and robot motion are disabled.".format(
                args.source,
                timeout_s,
            ),
            flush=True,
        )
        while time.monotonic() < deadline_s:
            frame = frame_source.read()
            if frame is None:
                raise RuntimeError("Camera preflight failed: frame unavailable")
            last_result = freshness_guard.update(frame)
            if last_result.fresh:
                print(
                    "Camera preflight passed: changing robot-camera stream "
                    "verified over {} consecutive frames. No model or movement "
                    "controller was started.".format(
                        last_result.confirmation_frames,
                    ),
                    flush=True,
                )
                return 0
            remaining = args.frame_interval
            if remaining > 0.0:
                time.sleep(remaining)
        raise RuntimeError(
            "Camera preflight failed after {:.1f}s: {}".format(
                timeout_s,
                last_result.reason,
            )
        )
    finally:
        if frame_source is not None:
            frame_source.close()
        if camera_adapter is not None:
            camera_adapter.close()


def run(args) -> int:
    config = _validate_args(args)
    if args.camera_check:
        return _run_camera_check(args)
    follower = TargetFollower(config)
    inference_confidence = min(args.confidence, args.person_stop_confidence)
    describe_scene_enabled = (
        args.speak
        or args.voice_describe
        or args.describe_request_file is not None
    )
    if describe_scene_enabled:
        inference_confidence = min(inference_confidence, args.speech_confidence)
    tracker = UltralyticsTracker(
        model_name=args.model,
        confidence=inference_confidence,
        image_size=args.image_size,
        device=args.device,
        tracker=args.tracker,
        target_label=args.target,
        persist_tracks=not (
            args.source == "file"
            and args.input_file is not None
            and args.input_file.suffix.casefold()
            in {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
        ),
        describe_all=describe_scene_enabled,
    )
    motion_adapter = _make_motion_adapter(args)
    frame_source = None
    camera_adapter = None
    controller_lease = None  # type: Optional[ControllerLease]
    pump = None
    preview = None
    speaker = None
    narration_policy = None
    expanded_scene = None
    recognizer = None
    last_scene_detections = ()  # type: Sequence[Detection]
    last_scene_frame_width = 0
    describe_request_mtime = None
    if args.describe_request_file is not None:
        try:
            describe_request_mtime = args.describe_request_file.stat().st_mtime
        except OSError:
            describe_request_mtime = None
    camera_freshness = (
        CameraFreshnessGuard()
        if args.source in ("robomaster-app", "sdk")
        else None
    )
    publisher = ControlStatusPublisher(
        path=args.status_file,
        live=args.live,
        transport=args.transport,
    )
    current = follower.reset("controller starting")
    last_signature = None
    last_frame_at = time.monotonic()
    last_loop_at = 0.0
    fps = 0.0
    start_s = None

    try:
        if args.live:
            controller_lease = ControllerLease()
            controller_lease.acquire()
        motion_adapter.connect()
        frame_source, camera_adapter = _make_frame_source(args, motion_adapter)
        frame_source.open()
        if describe_scene_enabled:
            speaker = PiperSceneSpeaker(
                python_executable=args.piper_python,
                worker=args.piper_worker,
                model=args.piper_model,
            )
            speaker.start()
            print(
                "English scene speech enabled with Piper voice {}.".format(
                    args.piper_model.stem
                ),
                flush=True,
            )
        if args.voice_describe:
            recognizer = WhisperSpeechRecognizer(
                python_executable=args.whisper_python,
                model=args.whisper_model,
                model_name=args.whisper_model_name,
            )
            recognizer.start()
            print(
                "Voice scene queries enabled: say 'tell me what you see'.",
                flush=True,
            )
        if args.speak:
            narration_policy = SceneNarrationPolicy(
                stable_frames=args.speech_stable_frames,
                repeat_seconds=args.speech_repeat_seconds,
                clear_seconds=args.speech_clear_seconds,
            )
            print(
                "Continuous scene narration enabled.",
                flush=True,
            )
            if not args.basic_scene_only:
                try:
                    expanded_scene = ExpandedSceneRecognizer(
                        python_executable=Path(sys.executable),
                        worker=args.expanded_scene_worker,
                        model=args.expanded_scene_model,
                        confidence=args.expanded_scene_confidence,
                        image_size=args.expanded_scene_image_size,
                        interval_s=args.expanded_scene_interval,
                    )
                    expanded_scene.start(timeout_s=90.0)
                    print(
                        "Expanded furniture recognition enabled with {}.".format(
                            args.expanded_scene_model.stem
                        ),
                        flush=True,
                    )
                except RuntimeError as exc:
                    if expanded_scene is not None:
                        expanded_scene.close()
                    expanded_scene = None
                    print(
                        "WARNING: expanded scene recognition disabled: {}".format(exc),
                        file=sys.stderr,
                        flush=True,
                    )
        if not args.no_preview:
            preview = NonActivatingPreview()
        pump = CommandPump(
            motion_adapter,
            CommandPumpConfig(
                rate_hz=15.0,
                stale_after_s=args.max_inference_seconds,
                moving_keepalive_s=0.12,
                robot_timeout_s=0.35,
            ),
        )
        pump.start()
        publisher.publish(
            GestureDecision("YOLO", current.reason, current.command), force=True
        )
        print(
            "{} YOLO follow; source={}, target={!r}, model={}.".format(
                "LIVE ROBOT" if args.live else "DRY RUN",
                args.source,
                args.target,
                args.model,
            ),
            flush=True,
        )
        print(
            "Motion requires {} consecutive target frames. Target/person/frame "
            "loss stops immediately.".format(config.minimum_lock_frames),
            flush=True,
        )
        if camera_freshness is not None:
            print(
                "Camera freshness interlock requires 3 changing frames and "
                "stops a frozen stream.",
                flush=True,
            )
        start_s = time.monotonic()

        while True:
            loop_started = time.monotonic()
            if (
                args.duration > 0.0
                and start_s is not None
                and loop_started - start_s >= args.duration
            ):
                print("Duration reached; stopping.", flush=True)
                break
            frame = frame_source.read()
            if frame is None:
                current = follower.reset("video ended or frame unavailable")
                pump.halt()
                publisher.publish(
                    GestureDecision("YOLO", current.reason, current.command),
                    force=True,
                )
                print("YOLO -> STOP: {}".format(current.reason), flush=True)
                break
            last_frame_at = time.monotonic()
            freshness = (
                camera_freshness.update(frame, now_s=last_frame_at)
                if camera_freshness is not None
                else CameraFreshness(True, "camera freshness not required")
            )
            detections = tracker.track(frame) if freshness.fresh else ()
            inference_done = time.monotonic()
            narration_only = ()
            if freshness.fresh:
                last_scene_detections = tuple(
                    item
                    for item in detections
                    if item.confidence >= args.speech_confidence
                )
                last_scene_frame_width = int(frame.shape[1])
            elapsed = inference_done - last_loop_at if last_loop_at else 0.0
            if elapsed > 0.0:
                fps = 1.0 / elapsed
            last_loop_at = inference_done

            if not freshness.fresh:
                current = follower.reset(freshness.reason)
            elif inference_done - last_frame_at > args.max_inference_seconds:
                current = follower.reset("inference stale - stopped")
            else:
                current = follower.update(
                    detections,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )
            pump.submit(current.command)
            publisher.publish(
                GestureDecision("YOLO", current.reason, current.command)
            )

            if (
                freshness.fresh
                and speaker is not None
                and narration_policy is not None
            ):
                expanded_detections = ()
                narration_ready = True
                if expanded_scene is not None:
                    expanded_scene.submit(frame, now_s=inference_done)
                    expanded_detections = tuple(
                        Detection(
                            label=str(item["label"]),
                            confidence=float(item["confidence"]),
                            box=tuple(float(value) for value in item["box"]),
                        )
                        for item in expanded_scene.detections(
                            max_age_s=max(2.0, args.expanded_scene_interval * 1.75)
                        )
                        if float(item["confidence"])
                        >= args.expanded_scene_confidence
                    )
                    narration_ready = expanded_scene.has_confirmed_scan
                    if expanded_scene.error is not None:
                        print(
                            "WARNING: expanded scene recognition disabled: {}".format(
                                expanded_scene.error
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        expanded_scene.close()
                        expanded_scene = None
                        narration_ready = True
                narration_detections = merge_scene_detections(
                    tuple(
                        item
                        for item in detections
                        if item.confidence >= args.speech_confidence
                    ),
                    expanded_detections,
                )
                narration_only = tuple(
                    item
                    for item in narration_detections
                    if not any(item is primary for primary in detections)
                )
                if narration_ready:
                    scene = describe_scene(
                        narration_detections,
                        frame_width=int(frame.shape[1]),
                        confidence=0.0,
                        maximum_groups=args.speech_max_groups,
                    )
                    spoken_text = narration_policy.update(
                        scene, now_s=inference_done
                    )
                    if spoken_text:
                        speaker.speak(spoken_text)
                        print("ROBOT SAYS: {}".format(spoken_text), flush=True)
                if speaker.error is not None:
                    print(
                        "WARNING: scene speech disabled: {}".format(speaker.error),
                        file=sys.stderr,
                        flush=True,
                    )
                    speaker.close()
                    speaker = None
                    narration_policy = None
                    if expanded_scene is not None:
                        expanded_scene.close()
                        expanded_scene = None

            # On-demand: answer "tell me what you see" by describing the latest
            # detections aloud, independent of the continuous narration policy.
            if recognizer is not None and speaker is not None:
                event = recognizer.get(timeout_s=0.0)
                while event is not None:
                    if event.event == "recognized" and is_look_query(event.text):
                        scene = describe_scene(
                            last_scene_detections,
                            frame_width=max(1, last_scene_frame_width),
                            confidence=0.0,
                            maximum_groups=args.speech_max_groups,
                        )
                        spoken = (
                            scene.text
                            if scene is not None
                            else "I do not see any recognized objects right now."
                        )
                        speaker.speak(spoken)
                        print("ROBOT SAYS: {}".format(spoken), flush=True)
                    elif event.event == "error":
                        print(
                            "WARNING: voice query recognizer error: {}".format(
                                event.message
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    event = recognizer.get(timeout_s=0.0)

            # External trigger: a single shared recognizer (the voice controller)
            # touches this file when it hears "what do you see", so scene queries
            # need no second microphone listener here.
            if args.describe_request_file is not None and speaker is not None:
                try:
                    request_mtime = args.describe_request_file.stat().st_mtime
                except OSError:
                    request_mtime = None
                if request_mtime is not None and request_mtime != describe_request_mtime:
                    describe_request_mtime = request_mtime
                    scene = describe_scene(
                        last_scene_detections,
                        frame_width=max(1, last_scene_frame_width),
                        confidence=0.0,
                        maximum_groups=args.speech_max_groups,
                    )
                    spoken = (
                        scene.text
                        if scene is not None
                        else "I do not see any recognized objects right now."
                    )
                    speaker.speak(spoken)
                    print("ROBOT SAYS: {}".format(spoken), flush=True)

            signature = (current.state, current.reason, translation_directions(current.command))
            if signature != last_signature:
                movement = " + ".join(signature[2]) if signature[2] else "STOP"
                print(
                    "YOLO -> {:<14} {:<15} {}".format(
                        current.state, movement, current.reason
                    ),
                    flush=True,
                )
                last_signature = signature

            if preview is not None or args.frame_out is not None:
                annotated = annotate_frame(
                    frame,
                    detections,
                    current,
                    fps,
                    narration_only=narration_only,
                )
                if preview is not None:
                    preview.show(annotated)
                    if preview.closed:
                        preview = None
                if args.frame_out is not None:
                    _write_frame_atomically(args.frame_out, annotated)
            if pump.error is not None:
                raise RobotError("Command sender stopped: {}".format(pump.error))

            remaining = args.frame_interval - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopping immediately.", flush=True)
    finally:
        if pump is not None:
            pump.halt()
            pump.close()
        if frame_source is not None:
            frame_source.close()
        if preview is not None:
            preview.close()
        if recognizer is not None:
            recognizer.close()
        if speaker is not None:
            speaker.close()
        if expanded_scene is not None:
            expanded_scene.close()
        if camera_adapter is not None and camera_adapter is not motion_adapter:
            camera_adapter.close()
        motion_adapter.close()
        if controller_lease is not None:
            controller_lease.close()
        publisher.publish(
            GestureDecision(
                "YOLO", "YOLO controller stopped", VelocityCommand.stopped()
            ),
            force=True,
        )
    return 0


def _make_process_dpi_aware() -> None:
    """Report true pixel sizes so the RoboMaster app capture is not clipped.

    Without this, on a scaled (high-DPI) display GetWindowRect/GetClientRect
    return logical pixels while PrintWindow renders physical pixels, so only the
    top-left fraction of the window is captured (about a quarter at 200%).
    """
    if os.name != "nt":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main(argv: Sequence[str] = None) -> int:
    _make_process_dpi_aware()
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, ControlLeaseError, RuntimeError, RobotError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
