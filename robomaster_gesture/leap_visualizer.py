"""Interactive, non-activating RoboMaster Control Center for Ultraleap.

The hand topology and desktop x/z projection are based on Ultraleap's
``leapc-python-bindings/examples/visualiser.py``.  This implementation was
rewritten around Tk and the installed LeapC CFFI module so it needs no OpenCV.
Its controls accept mouse clicks. The window opens without taking focus; live
controllers then return focus to the stock RoboMaster app before movement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence, Tuple

from .control_center import (
    DEFAULT_MODELS_DIRECTORY,
    DESCRIBE_REQUEST_PATH,
    LEAP_MODE,
    MIC_TEST_MODE,
    PROJECT_ROOT,
    VOICE_MODE,
    VOICE_MODES,
    ControlCenterProcessManager,
)
from .control_status import DEFAULT_CONTROL_STATUS_PATH, ControlStatusReader
from .models import translation_directions
from .robomaster_app import RoboMasterAppLauncher
from .ultraleap_app import UltraleapControlPanelLauncher
from .whisper_models import default_choice, discover_whisper_models


MIC_HELP_STEPS = (
    (
        "UNBLOCK SAMSUNG",
        "Press Fn+F10 once. Keep that setting only if the toast says "
        '"Block Recording off". You can also use Samsung Settings > '
        "Security and privacy > Block Recording / Block camera and mic > Off.",
    ),
    (
        "ALLOW DESKTOP APPS",
        "In Windows Settings, open Privacy & security > Microphone and turn on "
        '"Let desktop apps access your microphone". PowerShell may not appear '
        "by name in the app list; that is expected for desktop apps.",
    ),
    (
        "VERIFY WITHOUT MOVING",
        "Choose TEST MIC and watch the Control Center meter rise while you "
        'speak, then say "robot forward". The 15-second dry run cannot move '
        "the robot. If it stays at zero, open Sound > Input > Microphone "
        "Array > Start test and watch the Windows input meter.",
    ),
)

MIC_PRIVACY_SETTINGS_URI = "ms-settings:privacy-microphone"
SOUND_INPUT_SETTINGS_URI = "ms-settings:sound"
SAMSUNG_SETTINGS_URI = "samsungsettings15:"


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class MicrophoneMeterState:
    label: str
    level: int
    tone: str


@dataclass(frozen=True)
class TranscriptionDisplayState:
    label: str
    tone: str


def microphone_meter_state(
    mode: Optional[str],
    phase: str,
    audio_level: Optional[int],
    audio_updated_at_s: float,
    now_s: float,
    stale_after_s: float = 1.5,
) -> MicrophoneMeterState:
    """Describe the microphone meter without depending on Tk."""
    if mode not in VOICE_MODES:
        return MicrophoneMeterState("MIC OFF", 0, "muted")
    if phase in ("SWITCHING", "STARTING"):
        return MicrophoneMeterState("MIC STARTING", 0, "muted")
    if phase == "STOPPING":
        return MicrophoneMeterState("MIC STOPPING", 0, "muted")
    if phase == "ERROR":
        return MicrophoneMeterState("MIC ERROR", 0, "error")
    if (
        phase != "RUNNING"
        or audio_level is None
        or audio_updated_at_s <= 0.0
        or now_s - audio_updated_at_s > stale_after_s
    ):
        return MicrophoneMeterState("MIC WAITING", 0, "muted")

    level = max(0, min(100, int(audio_level)))
    if level == 0:
        return MicrophoneMeterState("NO SIGNAL  0%", 0, "warn")
    if level >= 85:
        return MicrophoneMeterState("LOUD  {}%".format(level), level, "warn")
    return MicrophoneMeterState("INPUT  {}%".format(level), level, "good")


def transcription_display_state(
    mode: Optional[str],
    phase: str,
    transcript_text: str,
    transcript_confidence: float,
) -> TranscriptionDisplayState:
    """Describe the persistent one-line speech transcription readout."""
    clean_text = " ".join(str(transcript_text).split())
    if len(clean_text) > 64:
        clean_text = clean_text[:61].rstrip() + "..."
    if clean_text:
        confidence = max(0.0, min(1.0, float(transcript_confidence)))
        prefix = "TRANSCRIPT" if mode in VOICE_MODES else "LAST HEARD"
        return TranscriptionDisplayState(
            '{}  "{}"  {}% confidence'.format(
                prefix,
                clean_text,
                int(round(confidence * 100.0)),
            ),
            "good",
        )
    if mode in VOICE_MODES:
        if phase == "ERROR":
            return TranscriptionDisplayState(
                "TRANSCRIPT  Recognition error",
                "error",
            )
        if phase == "STOPPING":
            return TranscriptionDisplayState(
                "TRANSCRIPT  Stopping listener...",
                "muted",
            )
        if phase in ("SWITCHING", "STARTING"):
            return TranscriptionDisplayState(
                "TRANSCRIPT  Loading model, please wait...",
                "muted",
            )
        return TranscriptionDisplayState(
            "TRANSCRIPT  Listening for a command...",
            "muted",
        )
    return TranscriptionDisplayState(
        "TRANSCRIPT  No speech recognized yet",
        "muted",
    )


def build_diagnostics_text(
    tracking: str,
    detail: str,
    command: str,
    transcript: str,
    status: str,
) -> str:
    """Assemble a single copyable block of the current on-screen readouts."""
    return "\n".join(
        (
            "RoboMaster Control Center diagnostics",
            "Tracking:   {}".format(tracking),
            "Hand:       {}".format(detail),
            "Command:    {}".format(command),
            "Transcript: {}".format(transcript),
            "Status:",
            status,
        )
    )


@dataclass(frozen=True)
class BoneSnapshot:
    start: Point3
    end: Point3


@dataclass(frozen=True)
class HandSnapshot:
    hand_id: int
    handedness: str
    confidence: float
    pinch_strength: float
    grab_strength: float
    palm: Point3
    elbow: Point3
    wrist: Point3
    digits: Tuple[Tuple[BoneSnapshot, ...], ...]


@dataclass(frozen=True)
class TrackingSnapshot:
    frame_id: int
    framerate: float
    hands: Tuple[HandSnapshot, ...]
    status: str
    service_connected: bool
    device_present: bool
    updated_at_s: float


def project_xz(
    point: Point3,
    center_x: float,
    center_y: float,
    pixels_per_mm: float,
) -> Tuple[int, int]:
    """Project Leap desktop coordinates into a stable top-down view."""
    return (
        int(round(center_x + point.x * pixels_per_mm)),
        int(round(center_y + point.z * pixels_per_mm)),
    )


class LeapCReader(threading.Thread):
    """Copy LeapC events into immutable snapshots for the Tk UI thread."""

    def __init__(self, sdk_root: Path):
        super().__init__(name="LeapVisualizerReader", daemon=True)
        self.sdk_root = Path(sdk_root)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = TrackingSnapshot(
            frame_id=0,
            framerate=0.0,
            hands=(),
            status="CONNECTING",
            service_connected=False,
            device_present=False,
            updated_at_s=time.monotonic(),
        )

    def stop(self) -> None:
        self._stop_event.set()

    def latest(self) -> TrackingSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(self, snapshot: TrackingSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    @staticmethod
    def _point(native) -> Point3:
        return Point3(float(native.x), float(native.y), float(native.z))

    def _copy_tracking(self, event, left_hand_value: int) -> TrackingSnapshot:
        hands = []
        for hand_index in range(int(event.nHands)):
            hand = event.pHands[hand_index]
            digits = []
            for digit_index in range(5):
                digit = hand.digits[digit_index]
                bones = []
                for bone_index in range(4):
                    bone = digit.bones[bone_index]
                    bones.append(
                        BoneSnapshot(
                            start=self._point(bone.prev_joint),
                            end=self._point(bone.next_joint),
                        )
                    )
                digits.append(tuple(bones))

            hands.append(
                HandSnapshot(
                    hand_id=int(hand.id),
                    handedness=(
                        "LEFT" if int(hand.type) == left_hand_value else "RIGHT"
                    ),
                    confidence=float(hand.confidence),
                    pinch_strength=float(hand.pinch_strength),
                    grab_strength=float(hand.grab_strength),
                    palm=self._point(hand.palm.position),
                    elbow=self._point(hand.arm.prev_joint),
                    wrist=self._point(hand.arm.next_joint),
                    digits=tuple(digits),
                )
            )

        return TrackingSnapshot(
            frame_id=int(event.info.frame_id),
            framerate=float(event.framerate),
            hands=tuple(hands),
            status="TRACKING",
            service_connected=True,
            device_present=True,
            updated_at_s=time.monotonic(),
        )

    def _status_snapshot(
        self,
        status: str,
        service_connected: bool,
        device_present: bool,
    ) -> TrackingSnapshot:
        current = self.latest()
        return TrackingSnapshot(
            frame_id=current.frame_id,
            framerate=current.framerate,
            hands=(),
            status=status,
            service_connected=service_connected,
            device_present=device_present,
            updated_at_s=time.monotonic(),
        )

    def run(self) -> None:
        sdk_text = str(self.sdk_root)
        if sdk_text not in sys.path:
            sys.path.insert(0, sdk_text)

        try:
            from leapc_cffi import ffi, libleapc as leap
        except Exception as exc:
            self._publish(
                self._status_snapshot(
                    "SDK ERROR: {}".format(exc), False, False
                )
            )
            return

        connection_pointer = ffi.new("LEAP_CONNECTION *")
        result = leap.LeapCreateConnection(ffi.NULL, connection_pointer)
        if result != leap.eLeapRS_Success:
            self._publish(
                self._status_snapshot(
                    "CREATE ERROR 0x{:08X}".format(int(result) & 0xFFFFFFFF),
                    False,
                    False,
                )
            )
            return

        connection = connection_pointer[0]
        try:
            result = leap.LeapOpenConnection(connection)
            if result != leap.eLeapRS_Success:
                self._publish(
                    self._status_snapshot(
                        "OPEN ERROR 0x{:08X}".format(int(result) & 0xFFFFFFFF),
                        False,
                        False,
                    )
                )
                return

            message = ffi.new("LEAP_CONNECTION_MESSAGE *")
            service_connected = False
            device_present = False
            while not self._stop_event.is_set():
                result = leap.LeapPollConnection(connection, 100, message)
                if result == leap.eLeapRS_Timeout:
                    continue
                if result != leap.eLeapRS_Success:
                    self._publish(
                        self._status_snapshot(
                            "POLL ERROR 0x{:08X}".format(
                                int(result) & 0xFFFFFFFF
                            ),
                            service_connected,
                            device_present,
                        )
                    )
                    time.sleep(0.1)
                    continue

                if message.type == leap.eLeapEventType_Connection:
                    service_connected = True
                    self._publish(
                        self._status_snapshot(
                            "WAITING FOR DEVICE", True, device_present
                        )
                    )
                elif message.type == leap.eLeapEventType_ConnectionLost:
                    service_connected = False
                    device_present = False
                    self._publish(
                        self._status_snapshot("SERVICE LOST", False, False)
                    )
                elif message.type == leap.eLeapEventType_Device:
                    device_present = True
                    self._publish(
                        self._status_snapshot("DEVICE READY", True, True)
                    )
                elif message.type in (
                    leap.eLeapEventType_DeviceLost,
                    leap.eLeapEventType_DeviceFailure,
                ):
                    device_present = False
                    self._publish(
                        self._status_snapshot(
                            "DEVICE LOST", service_connected, False
                        )
                    )
                elif message.type == leap.eLeapEventType_Tracking:
                    service_connected = True
                    device_present = True
                    self._publish(
                        self._copy_tracking(
                            message.tracking_event,
                            int(leap.eLeapHandType_Left),
                        )
                    )
        except Exception as exc:
            self._publish(
                self._status_snapshot(
                    "READER ERROR: {}".format(exc),
                    False,
                    False,
                )
            )
        finally:
            leap.LeapCloseConnection(connection)
            leap.LeapDestroyConnection(connection)


class ControlCenterWindow:
    BACKGROUND = "#071018"
    PANEL = "#0C1A26"
    GRID = "#163247"
    TEXT = "#D9F7FF"
    MUTED = "#7EA5B8"
    GOOD = "#43F0A6"
    WARN = "#FFBE55"
    ERROR = "#FF6B73"
    RIGHT = "#42D8FF"
    LEFT = "#FF5CCB"
    CONTROL_PANEL_HEIGHT = 268

    def __init__(
        self,
        reader: LeapCReader,
        width: int,
        height: int,
        x: int,
        y: int,
        opacity: float,
        duration_s: float,
        control_reader: ControlStatusReader,
        controller_manager: ControlCenterProcessManager,
        shutdown_file: Optional[Path] = None,
        app_launcher: Optional[RoboMasterAppLauncher] = None,
        ultraleap_launcher: Optional[UltraleapControlPanelLauncher] = None,
    ):
        import tkinter as tk

        self.tk = tk
        self.reader = reader
        self.width = width
        self.total_height = height
        self.height = height - self.CONTROL_PANEL_HEIGHT
        self.duration_s = duration_s
        self.control_reader = control_reader
        self.controller_manager = controller_manager
        self.app_launcher = app_launcher or RoboMasterAppLauncher()
        self.ultraleap_launcher = (
            ultraleap_launcher or UltraleapControlPanelLauncher()
        )
        self.shutdown_file = Path(shutdown_file) if shutdown_file is not None else None
        self.started_at_s = time.monotonic()
        self._closing = False
        self._drag_origin = None
        self._external_controller_active = False
        self._mic_help_window = None
        self._mic_meter_render_key = None
        self._app_launch_in_progress = False
        self._app_status = None  # type: Optional[Tuple[str, str]]
        self._app_lock = threading.Lock()
        self._pending_app_result = None
        self._pending_ultraleap_result = None
        self._vision_process = None
        self._vision_frame_path = PROJECT_ROOT / "logs" / "vision_frame.jpg"
        self._vision_frame_mtime = 0.0
        self._vision_photo = None
        # Whisper model choices installed under <workspace>/models.
        self.whisper_models = discover_whisper_models(DEFAULT_MODELS_DIRECTORY)
        self.selected_model = (
            default_choice(self.whisper_models) if self.whisper_models else None
        )
        if self.selected_model is not None:
            self.controller_manager.set_whisper_model(self.selected_model)
        self._diag_tracking = ""
        self._diag_detail = ""
        self._diag_control = ""
        self._diag_status = "OFF  |  Robot control is off."
        self._diag_transcript = "TRANSCRIPT  No speech recognized yet"

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("RoboMaster Control Center")
        self.root.overrideredirect(True)
        self.root.configure(bg=self.BACKGROUND)
        self.root.geometry("{}x{}+{}+{}".format(width, height, x, y))
        self.root.resizable(False, False)
        self.root.attributes("-alpha", max(0.35, min(1.0, opacity)))
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=self.height,
            bg=self.BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="x")
        self.canvas.bind("<ButtonPress-1>", self._begin_drag)
        self.canvas.bind("<B1-Motion>", self._drag_window)
        self._build_control_panel()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()
        self._show_without_initial_activation(x, y)
        self.root.after(0, self._render)
        # Hand tracking needs the Ultraleap software, so open its Control Panel
        # right away (like we auto-open DJI's app for the robot).
        self.root.after(300, self._autostart_ultraleap)
        # Object detection is on by default; start it shortly after the window
        # is up so the panel is visible while the models load.
        self.root.after(1200, self._autostart_vision)

    def _build_control_panel(self) -> None:
        tk = self.tk
        panel = tk.Frame(
            self.root,
            width=self.width,
            height=self.CONTROL_PANEL_HEIGHT,
            bg=self.PANEL,
            highlightbackground="#28516A",
            highlightthickness=1,
        )
        panel.pack(fill="both", expand=True)
        panel.pack_propagate(False)

        top = tk.Frame(panel, bg=self.PANEL)
        top.pack(fill="x", padx=12, pady=(7, 1))
        tk.Label(
            top,
            text="ROBOT CONTROL",
            fg=self.TEXT,
            bg=self.PANEL,
            font=("Segoe UI Semibold", 10),
        ).pack(side="left")
        tk.Label(
            top,
            text="  LEAP + VOICE",
            fg=self.MUTED,
            bg=self.PANEL,
            font=("Segoe UI", 8),
        ).pack(side="left")
        tk.Button(
            top,
            text="CLOSE",
            command=self.close,
            fg=self.MUTED,
            bg="#112B3B",
            activeforeground=self.TEXT,
            activebackground="#1C4256",
            relief="flat",
            bd=0,
            padx=10,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        ).pack(side="right")
        self.copy_button = tk.Button(
            top,
            text="COPY",
            command=self._copy_diagnostics,
            fg=self.TEXT,
            bg="#163247",
            activeforeground=self.TEXT,
            activebackground="#22516B",
            relief="flat",
            bd=0,
            padx=10,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )
        self.copy_button.pack(side="right", padx=(0, 6))

        arm_row = tk.Frame(panel, bg=self.PANEL)
        arm_row.pack(fill="x", padx=10)
        self.safety_var = tk.BooleanVar(value=False)
        self.safety_check = tk.Checkbutton(
            arm_row,
            text="SAFE & READY  -  floor clear and RoboMaster connected",
            variable=self.safety_var,
            command=self._safety_changed,
            fg=self.WARN,
            bg=self.PANEL,
            activeforeground=self.WARN,
            activebackground=self.PANEL,
            selectcolor="#112B3B",
            anchor="w",
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        self.safety_check.pack(side="left")
        # Dry run lets Leap and Voice be tested without a robot: commands are
        # computed and shown but never sent to the S1.
        self.dry_run_var = tk.BooleanVar(value=False)
        self.dry_run_check = tk.Checkbutton(
            arm_row,
            text="DRY RUN (no S1)",
            variable=self.dry_run_var,
            command=self._dry_run_changed,
            fg=self.RIGHT,
            bg=self.PANEL,
            activeforeground=self.RIGHT,
            activebackground=self.PANEL,
            selectcolor="#112B3B",
            anchor="e",
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        self.dry_run_check.pack(side="right")

        button_row = tk.Frame(panel, bg=self.PANEL)
        button_row.pack(fill="x", padx=10, pady=(4, 5))
        self.leap_button = self._make_mode_button(
            button_row,
            "HAND CONTROL",
            lambda: self._start_controller(LEAP_MODE),
            self.RIGHT,
        )
        self.leap_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.voice_button = self._make_mode_button(
            button_row,
            "VOICE CONTROL",
            lambda: self._start_controller(VOICE_MODE),
            self.LEFT,
        )
        self.voice_button.pack(side="left", fill="x", expand=True, padx=4)
        self.stop_button = self._make_mode_button(
            button_row,
            "STOP CONTROL",
            self._stop_controller,
            self.ERROR,
        )
        self.stop_button.pack(side="left", fill="x", expand=True, padx=4)
        self.close_button = self._make_mode_button(
            button_row,
            "CLOSE APP",
            self.close,
            "#DCE8ED",
        )
        self.close_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(4, 0),
        )

        voice_box = tk.Frame(
            panel,
            bg="#0A1620",
            highlightthickness=1,
            highlightbackground=self.GRID,
        )
        voice_box.pack(fill="x", padx=10, pady=(0, 5))
        voice_left = tk.Frame(voice_box, bg="#0A1620")
        voice_left.pack(side="left", fill="both", expand=True, padx=8, pady=6)
        voice_heading = tk.Frame(voice_left, bg="#0A1620")
        voice_heading.pack(fill="x")
        tk.Label(
            voice_heading,
            text="VOICE  /  LOCAL WHISPER",
            fg=self.GOOD,
            bg="#0A1620",
            anchor="w",
            font=("Segoe UI Semibold", 8),
        ).pack(side="left")
        self._build_model_picker(voice_heading)
        self.mic_level_var = tk.StringVar(value="MIC OFF")
        self.mic_level_label = tk.Label(
            voice_heading,
            textvariable=self.mic_level_var,
            fg=self.MUTED,
            bg="#0A1620",
            anchor="e",
            font=("Consolas", 8),
        )
        self.mic_level_label.pack(side="right")
        self.mic_meter = tk.Canvas(
            voice_left,
            height=12,
            bg="#08131C",
            highlightbackground=self.GRID,
            highlightthickness=1,
            bd=0,
        )
        self.mic_meter.pack(fill="x", pady=(3, 3))
        self.transcript_text = self._make_readonly_text(
            voice_left,
            height=1,
            bg="#0D202D",
            fg=self.MUTED,
            font=("Consolas", 8),
            wrap="none",
        )
        self.transcript_text.pack(fill="x", pady=(0, 3))
        self._set_readonly_text(
            self.transcript_text,
            "TRANSCRIPT  No speech recognized yet",
            self.MUTED,
        )
        tk.Label(
            voice_left,
            text=(
                "forward / back / left / right / stop  |  "
                "'what do you see'  |  'do you see a person?'"
            ),
            fg=self.MUTED,
            bg="#0A1620",
            anchor="w",
            font=("Segoe UI", 8),
        ).pack(fill="x")
        voice_actions = tk.Frame(voice_box, bg="#0A1620")
        voice_actions.pack(side="right", padx=6, pady=6)
        self.mic_test_button = tk.Button(
            voice_actions,
            text="TEST MIC  /  15s SAFE",
            command=self._start_mic_test,
            fg=self.TEXT,
            bg="#163247",
            activeforeground=self.TEXT,
            activebackground="#22516B",
            disabledforeground="#527184",
            relief="flat",
            bd=0,
            width=18,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )
        self.mic_test_button.pack(fill="x", pady=(0, 3))
        self.mic_help_button = tk.Button(
            voice_actions,
            text="MIC BLOCKED?",
            command=self._show_mic_help,
            fg=self.WARN,
            bg="#112B3B",
            activeforeground=self.WARN,
            activebackground="#1C4256",
            relief="flat",
            bd=0,
            width=18,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )
        self.mic_help_button.pack(fill="x")
        # Object detection is on by default: the webcam feed appears in the panel
        # above and "what do you see" describes the scene aloud.
        self.vision_var = tk.BooleanVar(value=True)
        self.vision_check = tk.Checkbutton(
            voice_actions,
            text="OBJECT DETECTION",
            variable=self.vision_var,
            command=self._vision_changed,
            fg=self.GOOD,
            bg="#0A1620",
            activeforeground=self.GOOD,
            activebackground="#0A1620",
            selectcolor="#112B3B",
            anchor="w",
            font=("Segoe UI Semibold", 8),
            cursor="hand2",
        )
        self.vision_check.pack(fill="x", pady=(3, 0))

        # Always-visible readout of the command going to the S1. It lives in the
        # panel (not the canvas, which the object-detection feed can cover) so the
        # operator can always see what is being sent and whether it is live.
        cmd_bar = tk.Frame(
            panel,
            bg="#08131C",
            highlightthickness=1,
            highlightbackground=self.GRID,
        )
        cmd_bar.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(
            cmd_bar,
            text="COMMAND → S1",
            fg=self.MUTED,
            bg="#08131C",
            font=("Segoe UI Semibold", 8),
        ).pack(side="left", padx=(8, 6), pady=3)
        self.command_badge_label = tk.Label(
            cmd_bar,
            text="CONTROL OFF",
            fg=self.MUTED,
            bg="#08131C",
            font=("Segoe UI Semibold", 9),
        )
        self.command_badge_label.pack(side="right", padx=(6, 8))
        self.command_motion_label = tk.Label(
            cmd_bar,
            text="—",
            fg=self.MUTED,
            bg="#08131C",
            anchor="w",
            font=("Consolas", 12, "bold"),
        )
        self.command_motion_label.pack(side="left", fill="x", expand=True)

        self.controller_status_text = self._make_readonly_text(
            panel,
            height=2,
            bg="#071018",
            fg=self.MUTED,
            font=("Consolas", 8),
            wrap="word",
        )
        self.controller_status_text.pack(
            fill="both", expand=True, padx=10, pady=(0, 7)
        )
        self._set_controller_status("OFF  |  Robot control is off.", self.MUTED)

    def _make_mode_button(self, parent, text: str, command, color: str):
        return self.tk.Button(
            parent,
            text=text,
            command=command,
            fg="#061017",
            bg=color,
            activeforeground="#061017",
            activebackground=color,
            disabledforeground="#527184",
            relief="flat",
            bd=0,
            pady=7,
            cursor="hand2",
            font=("Segoe UI Semibold", 9),
        )

    def _build_model_picker(self, parent) -> None:
        tk = self.tk
        if not self.whisper_models:
            tk.Label(
                parent,
                text="  /  MODEL n/a",
                fg=self.MUTED,
                bg="#0A1620",
                font=("Segoe UI", 8),
            ).pack(side="left")
            self.model_menubutton = None
            return
        self.model_display_var = tk.StringVar(
            value="MODEL {}  ▾".format(self.selected_model.name)
        )
        self.model_menubutton = tk.Menubutton(
            parent,
            textvariable=self.model_display_var,
            fg=self.RIGHT,
            bg="#0A1620",
            activeforeground=self.RIGHT,
            activebackground="#0A1620",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )
        menu = tk.Menu(self.model_menubutton, tearoff=0)
        for choice in self.whisper_models:
            menu.add_command(
                label=choice.menu_text,
                command=lambda c=choice: self._select_whisper_model(c),
            )
        self.model_menubutton.configure(menu=menu)
        self.model_menubutton.pack(side="left", padx=(6, 0))

    def _select_whisper_model(self, choice) -> None:
        self.selected_model = choice
        self.controller_manager.set_whisper_model(choice)
        self.model_display_var.set("MODEL {}  ▾".format(choice.name))
        # A different model needs a fresh recognizer, so stop any running
        # listener; the next VOICE / QUICK / TEST MIC press loads the model.
        if self.controller_manager.latest().mode in VOICE_MODES:
            self.controller_manager.stop_async()

    def _vision_is_running(self) -> bool:
        proc = self._vision_process
        return proc is not None and proc.poll() is None

    def _autostart_vision(self) -> None:
        if self._closing:
            return
        if self.vision_var.get() and not self._vision_is_running():
            self._start_vision()

    def _vision_changed(self) -> None:
        """React to the OBJECT DETECTION checkbox toggling."""
        if self.vision_var.get():
            self._start_vision()
        else:
            self._stop_vision()

    def _start_vision(self) -> None:
        """Start S1-camera object detection with on-demand voice narration."""
        if self._vision_is_running():
            return
        python = self.controller_manager.python_executable
        if not Path(python).is_file():
            self.vision_var.set(False)
            self._set_controller_status(
                "Object detection unavailable: controller Python not found.",
                self.ERROR,
            )
            return
        # Detection reads the S1 camera from DJI's RoboMaster app by default. For
        # a temporary laptop-camera test when the S1 is unavailable (e.g. a dead
        # battery), set ROBOMASTER_VISION_SOURCE=webcam; a fresh launch without it
        # goes back to the S1 camera.
        vision_source = (
            os.environ.get("ROBOMASTER_VISION_SOURCE", "").strip()
            or "robomaster-app"
        )
        use_webcam = vision_source == "webcam"
        # The S1 app source needs DJI's live-drive camera view open; the laptop
        # webcam does not.
        if not use_webcam and not self.app_launcher.is_app_present():
            self.vision_var.set(False)
            self._set_controller_status(
                "OBJECT DETECTION uses the S1 camera via the RoboMaster app.\n"
                "Open the app's live drive (camera) view, then tick OBJECT "
                "DETECTION.",
                self.WARN,
            )
            return
        command = [
            str(python),
            "-m",
            "robomaster_gesture.yolo_follow",
            "--source",
            vision_source,
            "--target",
            "person",
            # Stream frames into this window instead of the pop-up preview.
            "--frame-out",
            str(self._vision_frame_path),
            "--no-preview",
            # Throttle detection to ~3 fps so it leaves CPU for voice recognition.
            "--frame-interval",
            "0.3",
            # Downscale the WHOLE frame to 256 px for inference (letterboxed, so
            # the full field of view is kept) - fast on CPU and enough for
            # room-scale objects (people, furniture); raise it for small/distant
            # ones.
            "--image-size",
            "256",
            # No microphone listener here: VOICE CONTROL touches this file on
            # "what do you see" so a single recognizer serves both.
            "--describe-request-file",
            str(DESCRIBE_REQUEST_PATH),
        ]
        try:
            log_directory = PROJECT_ROOT / "logs"
            log_directory.mkdir(parents=True, exist_ok=True)
            # Drop any stale frame so the panel only shows this session's feed.
            try:
                self._vision_frame_path.unlink()
            except OSError:
                pass
            self._vision_frame_mtime = 0.0
            self._vision_photo = None
            self._vision_log = open(
                log_directory / "vision.log", "w", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            self._vision_process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._vision_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # surfaced to the operator
            self._vision_process = None
            self.vision_var.set(False)
            self._set_controller_status(
                "Could not start object detection: {}".format(exc), self.ERROR
            )
            return

    def _stop_vision(self) -> None:
        proc = self._vision_process
        self._vision_process = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        log = getattr(self, "_vision_log", None)
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
            self._vision_log = None
        try:
            self.vision_var.set(False)
        except self.tk.TclError:
            pass

    def _current_vision_photo(self):
        """Decode the latest embedded detection frame, or None when unavailable."""
        if not self._vision_is_running():
            return None
        try:
            mtime = self._vision_frame_path.stat().st_mtime
        except OSError:
            return self._vision_photo  # no frame yet; keep whatever we had
        if mtime == self._vision_frame_mtime and self._vision_photo is not None:
            return self._vision_photo
        try:
            from PIL import Image, ImageTk

            with Image.open(self._vision_frame_path) as image:
                frame = image.convert("RGB")
                frame.thumbnail(
                    (self.width - 14, self.height - 46),
                    Image.Resampling.LANCZOS,
                )
                photo = ImageTk.PhotoImage(frame)
        except Exception:
            # A transient half-written read: keep the previous frame.
            return self._vision_photo
        self._vision_frame_mtime = mtime
        self._vision_photo = photo
        return photo

    def _make_readonly_text(self, parent, *, height, bg, fg, font, wrap):
        """Build a status readout that renders as a label but can be selected.

        The widget is a real ``Text`` so a user can drag-select and copy an
        error or transcript, but every editing keystroke and paste is blocked so
        it stays read-only.
        """
        tk = self.tk
        text = tk.Text(
            parent,
            height=height,
            # A Text defaults to 80 characters wide; pin the requested width to 1
            # and let ``fill`` size it so it cannot push neighbouring controls
            # (the TEST MIC / MIC BLOCKED? buttons) off the panel.
            width=1,
            bg=bg,
            fg=fg,
            font=font,
            wrap=wrap,
            bd=0,
            highlightthickness=0,
            relief="flat",
            padx=8,
            pady=4,
            insertwidth=0,
            selectbackground="#24506B",
            selectforeground=self.TEXT,
            inactiveselectbackground="#24506B",
            cursor="xterm",
        )
        text.bind("<Key>", self._readonly_text_key)
        text.bind("<<Paste>>", lambda event: "break")
        text.bind("<Button-2>", lambda event: "break")
        text.bind(
            "<Button-3>",
            lambda event, widget=text: self._show_text_menu(event, widget),
        )
        return text

    @staticmethod
    def _readonly_text_key(event):
        """Allow copy/select-all and cursor motion; block any edit keystroke."""
        control_pressed = bool(event.state & 0x0004)
        if control_pressed and event.keysym.lower() in ("c", "a", "insert"):
            return None
        navigation = {
            "Left",
            "Right",
            "Up",
            "Down",
            "Home",
            "End",
            "Prior",
            "Next",
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
        }
        if event.keysym in navigation:
            return None
        return "break"

    def _show_text_menu(self, event, widget) -> None:
        menu = self.tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Copy",
            command=lambda: self._copy_text_selection(widget),
        )
        menu.add_command(
            label="Copy all",
            command=lambda: self._copy_to_clipboard(widget.get("1.0", "end-1c")),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _copy_text_selection(self, widget) -> None:
        try:
            text = widget.get("sel.first", "sel.last")
        except self.tk.TclError:
            text = widget.get("1.0", "end-1c")
        self._copy_to_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
        except self.tk.TclError:
            pass

    def _set_readonly_text(self, widget, value: str, color: str) -> None:
        widget.configure(fg=color)
        # Only rewrite on a real change so a user can select a static error line
        # without the 30 Hz refresh dropping their selection under them.
        if getattr(widget, "_shown_value", None) == value:
            return
        widget._shown_value = value
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def _set_controller_status(self, text: str, color: str) -> None:
        self._diag_status = text
        self._set_readonly_text(self.controller_status_text, text, color)

    def _diagnostics_text(self) -> str:
        return build_diagnostics_text(
            self._diag_tracking,
            self._diag_detail,
            self._diag_control,
            self._diag_transcript,
            self._diag_status,
        )

    def _copy_diagnostics(self) -> None:
        self._copy_to_clipboard(self._diagnostics_text())
        try:
            self.copy_button.configure(text="COPIED")
            self.root.after(1200, lambda: self.copy_button.configure(text="COPY"))
        except self.tk.TclError:
            pass

    def _begin_drag(self, event) -> None:
        self._drag_origin = (event.x_root, event.y_root, self.root.winfo_x(), self.root.winfo_y())

    def _drag_window(self, event) -> None:
        if self._drag_origin is None:
            return
        start_x, start_y, window_x, window_y = self._drag_origin
        self.root.geometry(
            "+{}+{}".format(window_x + event.x_root - start_x, window_y + event.y_root - start_y)
        )

    def _safety_changed(self) -> None:
        if not self.safety_var.get() and self.controller_manager.latest().live:
            self.controller_manager.stop_async()

    def _dry_run_changed(self) -> None:
        # Changing the S1/no-S1 target mid-session is ambiguous, so stop the
        # current controller; the next drive press starts in the new target.
        if self.controller_manager.latest().mode in (LEAP_MODE, VOICE_MODE):
            self.controller_manager.stop_async()

    def _start_controller(self, mode: str) -> None:
        if self._external_controller_active:
            self._set_controller_status(
                "EXTERNAL CONTROL  |  Stop its console with Ctrl+C before switching here.",
                self.WARN,
            )
            return
        live = not self.dry_run_var.get()
        if live and not self.safety_var.get():
            self._set_controller_status(
                "ARM REQUIRED  |  Confirm the floor is clear, or tick DRY RUN to "
                "test without the S1.",
                self.WARN,
            )
            return
        # The stock-S1 transport drives DJI's desktop app, so open it first when
        # a live mode is requested and its window is not already present. Dry run
        # never contacts the S1, so it needs neither the app nor the safety arm.
        if (
            live
            and mode in (LEAP_MODE, VOICE_MODE)
            and not self.app_launcher.is_app_present()
        ):
            self._ensure_robomaster_app()
            return
        self._app_status = None
        self.controller_manager.switch_async(mode, live=live)

    def _ensure_robomaster_app(self) -> None:
        if self._app_launch_in_progress:
            return
        self._app_launch_in_progress = True
        self._app_status = (
            "STARTING ROBOMASTER APP  |  Opening DJI's RoboMaster app (first run "
            "may copy ~1.1 GB). Connect the S1 and open its live drive view, then "
            "press HAND or VOICE again.",
            self.WARN,
        )

        def worker() -> None:
            result = self.app_launcher.launch()
            # Publish to shared state; the Tk render loop consumes it. Tkinter
            # is not thread-safe, so the worker never touches widgets directly.
            with self._app_lock:
                self._pending_app_result = result

        threading.Thread(
            target=worker,
            name="RoboMasterAppLaunch",
            daemon=True,
        ).start()

    def _consume_pending_app_result(self) -> None:
        with self._app_lock:
            result = self._pending_app_result
            self._pending_app_result = None
        if result is not None:
            self._on_app_launch_finished(result)

    def _autostart_ultraleap(self) -> None:
        """Open Ultraleap's Control Panel so the sensor is tracking on startup."""
        def worker() -> None:
            result = self.ultraleap_launcher.launch()
            with self._app_lock:
                self._pending_ultraleap_result = result

        threading.Thread(
            target=worker, name="UltraleapLaunch", daemon=True
        ).start()

    def _consume_pending_ultraleap_result(self) -> None:
        with self._app_lock:
            result = self._pending_ultraleap_result
            self._pending_ultraleap_result = None
        # Success (opened or already open) needs no status line - the Control
        # Panel window is its own confirmation. Only a missing install or a
        # launch failure is worth surfacing, and only while idle.
        if (
            result is not None
            and not result.started
            and not result.already_running
        ):
            self._app_status = ("LEAP  |  {}".format(result.message), self.WARN)

    def _on_app_launch_finished(self, result) -> None:
        self._app_launch_in_progress = False
        if result.already_running:
            self._app_status = (
                "ROBOMASTER APP READY  |  {} Then press HAND or VOICE.".format(
                    result.message
                ),
                self.GOOD,
            )
        elif result.started:
            self._app_status = (
                "STARTING ROBOMASTER APP  |  {}".format(result.message),
                self.GOOD,
            )
        else:
            self._app_status = (
                "ROBOMASTER APP  |  {}".format(result.message),
                self.ERROR,
            )

    def _start_mic_test(self) -> None:
        if self._external_controller_active:
            self._set_controller_status(
                "EXTERNAL CONTROL  |  Stop its console before starting a microphone test.",
                self.WARN,
            )
            return
        self._app_status = None
        self.controller_manager.switch_async(MIC_TEST_MODE)

    def _show_mic_help(self) -> None:
        if self._mic_help_window is not None:
            try:
                if self._mic_help_window.winfo_exists():
                    self._mic_help_window.lift()
                    return
            except self.tk.TclError:
                pass

        # Settings and help should never compete with a live controller or an
        # open microphone. Stop and disarm first, even if this is clicked while
        # a mode is transitioning.
        self.safety_var.set(False)
        self.controller_manager.stop_async()

        tk = self.tk
        help_window = tk.Toplevel(self.root)
        self._mic_help_window = help_window
        help_window.withdraw()
        help_window.overrideredirect(True)
        help_window.configure(bg="#28516A")
        help_window.attributes("-topmost", True)

        width = 550
        height = 440
        x = max(0, self.root.winfo_x() + (self.width - width) // 2)
        y = max(0, self.root.winfo_y() + 70)
        help_window.geometry("{}x{}+{}+{}".format(width, height, x, y))

        shell = tk.Frame(help_window, bg=self.PANEL)
        shell.pack(fill="both", expand=True, padx=1, pady=1)

        header = tk.Frame(shell, bg=self.PANEL)
        header.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(
            header,
            text="MICROPHONE BLOCKED?",
            fg=self.TEXT,
            bg=self.PANEL,
            font=("Segoe UI Semibold", 13),
        ).pack(side="left")
        tk.Button(
            header,
            text="CLOSE",
            command=self._close_mic_help,
            fg=self.MUTED,
            bg="#112B3B",
            activeforeground=self.TEXT,
            activebackground="#1C4256",
            relief="flat",
            bd=0,
            padx=10,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        ).pack(side="right")

        tk.Label(
            shell,
            text="Voice uses this laptop's microphone. Follow these steps in order.",
            fg=self.MUTED,
            bg=self.PANEL,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x", padx=18, pady=(0, 10))

        for number, (title, detail) in enumerate(MIC_HELP_STEPS, start=1):
            self._make_mic_help_step(shell, number, title, detail)

        self.mic_help_status_var = tk.StringVar(
            value="Opening Settings also leaves robot control safely disarmed."
        )
        tk.Label(
            shell,
            textvariable=self.mic_help_status_var,
            fg=self.GOOD,
            bg=self.PANEL,
            anchor="w",
            font=("Segoe UI", 8),
        ).pack(fill="x", padx=18, pady=(7, 5))

        actions = tk.Frame(shell, bg=self.PANEL)
        actions.pack(fill="x", padx=18, pady=(0, 14))
        self._make_settings_button(
            actions,
            "SAMSUNG SETTINGS",
            SAMSUNG_SETTINGS_URI,
            "Samsung Settings",
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._make_settings_button(
            actions,
            "MIC PRIVACY",
            MIC_PRIVACY_SETTINGS_URI,
            "Windows microphone privacy",
        ).pack(side="left", fill="x", expand=True, padx=4)
        self._make_settings_button(
            actions,
            "SOUND INPUT",
            SOUND_INPUT_SETTINGS_URI,
            "Windows sound input",
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        drag_origin = {"value": None}

        def begin_drag(event) -> None:
            drag_origin["value"] = (
                event.x_root,
                event.y_root,
                help_window.winfo_x(),
                help_window.winfo_y(),
            )

        def drag(event) -> None:
            if drag_origin["value"] is None:
                return
            start_x, start_y, window_x, window_y = drag_origin["value"]
            help_window.geometry(
                "+{}+{}".format(
                    window_x + event.x_root - start_x,
                    window_y + event.y_root - start_y,
                )
            )

        header.bind("<ButtonPress-1>", begin_drag)
        header.bind("<B1-Motion>", drag)
        help_window.protocol("WM_DELETE_WINDOW", self._close_mic_help)
        help_window.deiconify()
        help_window.lift()

    def _make_mic_help_step(self, parent, number: int, title: str, detail: str) -> None:
        tk = self.tk
        row = tk.Frame(parent, bg="#0A1620")
        row.pack(fill="x", padx=18, pady=3)
        tk.Label(
            row,
            text=str(number),
            fg="#061017",
            bg=self.GOOD,
            width=2,
            font=("Segoe UI Semibold", 10),
        ).pack(side="left", padx=(9, 10), pady=9)
        copy = tk.Frame(row, bg="#0A1620")
        copy.pack(side="left", fill="both", expand=True, pady=7, padx=(0, 9))
        tk.Label(
            copy,
            text=title,
            fg=self.TEXT,
            bg="#0A1620",
            anchor="w",
            font=("Segoe UI Semibold", 8),
        ).pack(fill="x")
        tk.Label(
            copy,
            text=detail,
            fg=self.MUTED,
            bg="#0A1620",
            anchor="w",
            justify="left",
            wraplength=465,
            font=("Segoe UI", 8),
        ).pack(fill="x")

    def _make_settings_button(self, parent, text: str, uri: str, label: str):
        return self.tk.Button(
            parent,
            text=text,
            command=lambda: self._open_settings(uri, label),
            fg=self.TEXT,
            bg="#163247",
            activeforeground=self.TEXT,
            activebackground="#22516B",
            relief="flat",
            bd=0,
            pady=7,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )

    def _open_settings(self, uri: str, label: str) -> None:
        self.safety_var.set(False)
        self.controller_manager.stop_async()
        try:
            if os.name != "nt":
                raise OSError("Settings shortcuts require Windows")
            os.startfile(uri)
        except OSError as exc:
            self.mic_help_status_var.set(
                "Could not open {}: {}".format(label, exc)
            )
        else:
            self.mic_help_status_var.set("Opened {}.".format(label))

    def _close_mic_help(self) -> None:
        help_window = self._mic_help_window
        self._mic_help_window = None
        if help_window is None:
            return
        try:
            help_window.destroy()
        except self.tk.TclError:
            pass

    def _stop_controller(self) -> None:
        self.safety_var.set(False)
        self.controller_manager.stop_async()

    def _show_without_initial_activation(self, x: int, y: int) -> None:
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

        gwl_exstyle = -20
        ws_ex_transparent = 0x00000020
        ws_ex_toolwindow = 0x00000080
        ws_ex_noactivate = 0x08000000
        old_style = int(user32.GetWindowLongPtrW(wrapper, gwl_exstyle))
        user32.SetWindowLongPtrW(
            wrapper,
            gwl_exstyle,
            # Clear the old overlay's click-through and permanent NOACTIVATE
            # flags. Mapping below still uses SWP_NOACTIVATE, while a later
            # deliberate button click can activate Tk and dispatch normally.
            (old_style & ~(ws_ex_transparent | ws_ex_noactivate)) | ws_ex_toolwindow,
        )

        # Map topmost without transferring the user's current keyboard focus.
        self.root.deiconify()
        self.root.update_idletasks()
        hwnd_topmost = wintypes.HWND(-1)
        swp_noactivate = 0x0010
        swp_showwindow = 0x0040
        swp_framechanged = 0x0020
        user32.SetWindowPos(
            wrapper,
            hwnd_topmost,
            x,
            y,
            self.width,
            self.total_height,
            swp_noactivate | swp_showwindow | swp_framechanged,
        )
        if foreground and user32.GetForegroundWindow() != foreground:
            user32.SetForegroundWindow(foreground)

    def _inside_view(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return 8 <= x < self.width - 8 and 45 <= y < self.height - 78

    def _draw_bar(
        self,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        color: str,
    ) -> None:
        value = max(0.0, min(1.0, value))
        self.canvas.create_text(
            x,
            y,
            text=label,
            fill=self.MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        left = x + 48
        self.canvas.create_rectangle(
            left,
            y - 5,
            left + width,
            y + 5,
            fill="#112B3B",
            outline="#244A5E",
        )
        self.canvas.create_rectangle(
            left,
            y - 5,
            left + int(width * value),
            y + 5,
            fill=color,
            outline="",
        )

    def _render_hand(
        self,
        hand: HandSnapshot,
        center_x: float,
        center_y: float,
        scale: float,
    ) -> None:
        color = self.LEFT if hand.handedness == "LEFT" else self.RIGHT

        def project(point: Point3) -> Tuple[int, int]:
            return project_xz(point, center_x, center_y, scale)

        wrist = project(hand.wrist)
        elbow = project(hand.elbow)
        palm = project(hand.palm)
        if self._inside_view(elbow) and self._inside_view(wrist):
            self.canvas.create_line(
                elbow[0], elbow[1], wrist[0], wrist[1], fill=color, width=4
            )

        finger_bases = []
        for digit_index, digit in enumerate(hand.digits):
            if not digit:
                continue
            base = project(digit[0].start)
            finger_bases.append(base)
            if self._inside_view(wrist) and self._inside_view(base):
                self.canvas.create_line(
                    wrist[0], wrist[1], base[0], base[1], fill=color, width=2
                )
            for bone_index, bone in enumerate(digit):
                start = project(bone.start)
                end = project(bone.end)
                if self._inside_view(start) and self._inside_view(end):
                    self.canvas.create_line(
                        start[0],
                        start[1],
                        end[0],
                        end[1],
                        fill=color,
                        width=3 if bone_index < 2 else 2,
                    )
                    self.canvas.create_oval(
                        start[0] - 2,
                        start[1] - 2,
                        start[0] + 2,
                        start[1] + 2,
                        fill=self.TEXT,
                        outline="",
                    )
                    radius = 4 if bone_index == 3 else 2
                    self.canvas.create_oval(
                        end[0] - radius,
                        end[1] - radius,
                        end[0] + radius,
                        end[1] + radius,
                        fill=color,
                        outline="",
                    )

        for first, second in zip(finger_bases, finger_bases[1:]):
            if self._inside_view(first) and self._inside_view(second):
                self.canvas.create_line(
                    first[0], first[1], second[0], second[1], fill=color, width=2
                )

        if self._inside_view(palm):
            self.canvas.create_oval(
                palm[0] - 7,
                palm[1] - 7,
                palm[0] + 7,
                palm[1] + 7,
                fill=color,
                outline=self.TEXT,
                width=2,
            )
            self.canvas.create_text(
                palm[0] + 12,
                palm[1] - 12,
                text=hand.handedness,
                fill=color,
                font=("Segoe UI Semibold", 9),
                anchor="w",
            )

    def _render(self) -> None:
        if self._closing:
            return
        if self.shutdown_file is not None and self.shutdown_file.is_file():
            self.close()
            return
        if self.duration_s > 0.0 and time.monotonic() - self.started_at_s >= self.duration_s:
            self.close()
            return

        snapshot = self.reader.latest()
        stale = time.monotonic() - snapshot.updated_at_s > 0.5
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            1,
            1,
            self.width - 2,
            self.height - 2,
            fill=self.BACKGROUND,
            outline="#28516A",
            width=2,
        )

        # When object detection is running, embed its camera feed here in place
        # of the hand skeleton, so it is part of this window rather than a pop-up.
        vision_photo = self._current_vision_photo()
        if vision_photo is not None:
            self.canvas.create_image(
                self.width // 2,
                34 + (self.height - 34) // 2,
                image=vision_photo,
                anchor="center",
            )
            self.canvas.create_text(
                16, 17, text="OBJECT DETECTION", fill=self.GOOD,
                font=("Segoe UI Semibold", 12), anchor="w",
            )
            self.canvas.create_text(
                self.width - 14, 17, text="say 'tell me what you see'",
                fill=self.MUTED, font=("Consolas", 9), anchor="e",
            )
            self._update_control_widgets(None, False)
            self.root.after(33, self._render)
            return
        if self._vision_is_running():
            # Running but no frame yet (models still loading).
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text="Loading object detection...", fill=self.MUTED,
                font=("Segoe UI Semibold", 12),
            )
            self.canvas.create_text(
                16, 17, text="OBJECT DETECTION", fill=self.GOOD,
                font=("Segoe UI Semibold", 12), anchor="w",
            )
            self._update_control_widgets(None, False)
            self.root.after(33, self._render)
            return

        self.canvas.create_text(
            16,
            17,
            text="ROBOMASTER CONTROL CENTER",
            fill=self.TEXT,
            font=("Segoe UI Semibold", 12),
            anchor="w",
        )
        status_color = (
            self.GOOD
            if snapshot.service_connected and snapshot.device_present and not stale
            else self.WARN
        )
        status_text = "STALE" if stale else snapshot.status
        self.canvas.create_oval(
            self.width - 142,
            12,
            self.width - 132,
            22,
            fill=status_color,
            outline="",
        )
        self.canvas.create_text(
            self.width - 14,
            17,
            text="{}  {:>2.0f} FPS".format(status_text, snapshot.framerate),
            fill=status_color,
            font=("Consolas", 9),
            anchor="e",
        )
        self._diag_tracking = "{}  {:.0f} FPS".format(status_text, snapshot.framerate)

        view_top = 42
        view_bottom = self.height - 76
        center_x = self.width / 2.0
        center_y = (view_top + view_bottom) / 2.0
        scale = min((self.width - 36) / 400.0, (view_bottom - view_top) / 300.0)
        self.canvas.create_rectangle(
            10,
            view_top,
            self.width - 10,
            view_bottom,
            fill=self.PANEL,
            outline=self.GRID,
        )
        self.canvas.create_line(
            center_x, view_top + 6, center_x, view_bottom - 6, fill=self.GRID
        )
        self.canvas.create_line(
            16, center_y, self.width - 16, center_y, fill=self.GRID
        )
        self.canvas.create_text(
            center_x,
            view_top + 10,
            text="HAND AWAY = ROBOT FORWARD",
            fill=self.MUTED,
            font=("Segoe UI", 8),
            anchor="n",
        )
        self.canvas.create_text(
            18,
            center_y,
            text="<  ROBOT LEFT",
            fill=self.MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.canvas.create_text(
            self.width - 18,
            center_y,
            text="ROBOT RIGHT  >",
            fill=self.MUTED,
            font=("Segoe UI", 8),
            anchor="e",
        )
        self.canvas.create_text(
            center_x,
            view_bottom - 10,
            text="HAND TOWARD YOU = ROBOT BACK",
            fill=self.MUTED,
            font=("Segoe UI", 8),
            anchor="s",
        )

        if snapshot.hands and not stale:
            for hand in snapshot.hands:
                self._render_hand(hand, center_x, center_y, scale)
        else:
            message = (
                "SHOW HAND ABOVE SENSOR"
                if snapshot.service_connected and snapshot.device_present and not stale
                else status_text
            )
            self.canvas.create_text(
                center_x,
                center_y,
                text=message,
                fill=self.MUTED,
                font=("Segoe UI Semibold", 12),
            )

        control = self.control_reader.latest()
        control_fresh = (
            control is not None
            and 0.0 <= time.time() - control.updated_at_epoch_s <= 0.75
        )
        if control_fresh:
            motion, badge, control_color, _badge_color = self._command_readout(
                control, True
            )
            control_text = "{}  [{}]  |  {}".format(motion, control.state, badge)
        else:
            control_text = "OPEN HAND  >  LIGHT PINCH (NO GRIP)  |  CONTROL OFF"
            control_color = self.MUTED
        self._diag_control = control_text
        self.canvas.create_text(
            self.width / 2,
            self.height - 68,
            text=control_text,
            fill=control_color,
            font=("Segoe UI Semibold", 9),
            anchor="center",
        )

        if snapshot.hands and not stale:
            primary = snapshot.hands[0]
            self._draw_bar(
                16,
                self.height - 50,
                112,
                "PINCH",
                primary.pinch_strength,
                self.RIGHT,
            )
            self._draw_bar(
                214,
                self.height - 50,
                112,
                "FIST",
                primary.grab_strength,
                self.LEFT,
            )
            detail = "{}  palm x={:+.0f} y={:+.0f} z={:+.0f} mm  id={}".format(
                primary.handedness,
                primary.palm.x,
                primary.palm.y,
                primary.palm.z,
                primary.hand_id,
            )
        else:
            detail = "{} hand(s)  frame {}".format(
                len(snapshot.hands), snapshot.frame_id
            )
        self._diag_detail = detail
        self.canvas.create_text(
            16,
            self.height - 20,
            text=detail,
            fill=self.MUTED,
            font=("Consolas", 9),
            anchor="w",
        )
        self._update_control_widgets(control, control_fresh)
        self.root.after(33, self._render)

    def _update_microphone_meter(self, managed) -> None:
        state = microphone_meter_state(
            managed.mode,
            managed.phase,
            managed.audio_level,
            managed.audio_updated_at_s,
            time.monotonic(),
        )
        color = {
            "muted": self.MUTED,
            "good": self.GOOD,
            "warn": self.WARN,
            "error": self.ERROR,
        }[state.tone]
        self.mic_level_var.set(state.label)
        self.mic_level_label.configure(fg=color)

        width = self.mic_meter.winfo_width()
        height = self.mic_meter.winfo_height()
        if width < 40 or height < 4:
            return
        render_key = (state.level, state.tone, width, height)
        if render_key == self._mic_meter_render_key:
            return
        self._mic_meter_render_key = render_key

        segment_count = 18
        active_count = (state.level * segment_count + 99) // 100
        gap = 3
        padding = 2
        segment_width = (
            width - 2 * padding - gap * (segment_count - 1)
        ) / float(segment_count)
        self.mic_meter.delete("all")
        for index in range(segment_count):
            x1 = padding + index * (segment_width + gap)
            x2 = x1 + segment_width
            fill = color if index < active_count else self.GRID
            self.mic_meter.create_rectangle(
                x1,
                2,
                x2,
                height - 2,
                fill=fill,
                outline="",
            )

    def _update_transcription(self, managed) -> None:
        state = transcription_display_state(
            managed.mode,
            managed.phase,
            managed.transcript_text,
            managed.transcript_confidence,
        )
        color = {
            "muted": self.MUTED,
            "good": self.GOOD,
            "warn": self.WARN,
            "error": self.ERROR,
        }[state.tone]
        self._diag_transcript = state.label
        self._set_readonly_text(self.transcript_text, state.label, color)

    _DIRECTION_ARROWS = {
        "FORWARD": "↑",
        "BACK": "↓",
        "LEFT": "←",
        "RIGHT": "→",
        "CW": "↻",
        "CCW": "↺",
    }

    def _command_readout(self, control, control_fresh: bool):
        """Return (motion, badge, motion_color, badge_color) for the S1 command.

        Shared by the always-visible panel indicator and the on-canvas line so
        the two never disagree about what is being sent.
        """
        if not control_fresh or control is None:
            return ("—  no command", "CONTROL OFF", self.MUTED, self.MUTED)
        parts = list(translation_directions(control.command))
        yaw = control.command.clockwise_deg_s
        if yaw >= 5.0:
            parts.append("CW")
        elif yaw <= -5.0:
            parts.append("CCW")
        moving = bool(parts)
        speed = max(
            abs(control.command.forward_m_s), abs(control.command.right_m_s)
        )
        if moving:
            labeled = "  ".join(
                "{} {}".format(self._DIRECTION_ARROWS.get(part, ""), part).strip()
                for part in parts
            )
            motion = (
                "{}   {:.2f} m/s".format(labeled, speed)
                if speed >= 0.015
                else labeled
            )
        else:
            motion = "■ STOP"
        if control.live:
            # Amber while the robot is actually moving, calm green when stopped.
            color = self.WARN if moving else self.GOOD
            return (motion, "● SENT TO S1", color, color)
        motion_color = self.RIGHT if moving else self.MUTED
        return (motion, "○ DRY RUN — NOT SENT", motion_color, self.RIGHT)

    def _update_command_indicator(self, managed, control, control_fresh: bool) -> None:
        active = (
            control_fresh
            and control is not None
            and managed.mode in (LEAP_MODE, VOICE_MODE)
            and managed.process_id is not None
            and control.process_id == managed.process_id
        )
        motion, badge, motion_color, badge_color = self._command_readout(
            control, bool(active)
        )
        self.command_motion_label.config(text=motion, fg=motion_color)
        self.command_badge_label.config(text=badge, fg=badge_color)

    def _update_control_widgets(self, control, control_fresh: bool) -> None:
        self._consume_pending_app_result()
        self._consume_pending_ultraleap_result()
        self._update_command_indicator(
            self.controller_manager.latest(), control, control_fresh
        )
        if self._vision_process is not None and self._vision_process.poll() is not None:
            # The object-detection window closed or exited on its own.
            self._stop_vision()
        managed = self.controller_manager.latest()
        self._update_microphone_meter(managed)
        self._update_transcription(managed)
        busy = managed.phase in ("SWITCHING", "STARTING", "STOPPING")
        self._external_controller_active = bool(
            control_fresh
            and control.live
            and control.process_id != managed.process_id
            and "controller stopped" not in control.reason.casefold()
        )

        # Dry run cannot move the robot, so it does not require the safety arm.
        dry_run = self.dry_run_var.get()
        can_start = (
            (dry_run or self.safety_var.get())
            and not busy
            and not self._external_controller_active
        )
        leap_state = (
            "disabled"
            if not can_start or managed.mode == LEAP_MODE
            else "normal"
        )
        voice_state = (
            "disabled"
            if not can_start or managed.mode == VOICE_MODE
            else "normal"
        )
        mic_state = (
            "disabled"
            if busy
            or managed.mode == MIC_TEST_MODE
            or self._external_controller_active
            else "normal"
        )
        idle_suffix = "  (DRY)" if dry_run else ""
        self.leap_button.configure(
            state=leap_state,
            text=(
                ("HAND DRY" if not managed.live else "HAND ACTIVE")
                if managed.mode == LEAP_MODE
                else "HAND CONTROL" + idle_suffix
            ),
            bg=(
                self.RIGHT
                if managed.mode == LEAP_MODE or leap_state == "normal"
                else "#112B3B"
            ),
            disabledforeground="#061017" if managed.mode == LEAP_MODE else "#527184",
        )
        self.voice_button.configure(
            state=voice_state,
            text=(
                ("VOICE DRY" if not managed.live else "VOICE ACTIVE")
                if managed.mode == VOICE_MODE
                else "VOICE CONTROL" + idle_suffix
            ),
            bg=(
                self.LEFT
                if managed.mode == VOICE_MODE or voice_state == "normal"
                else "#112B3B"
            ),
            disabledforeground="#061017" if managed.mode == VOICE_MODE else "#527184",
        )
        self.mic_test_button.configure(
            state=mic_state,
            text=(
                "MIC TEST ACTIVE"
                if managed.mode == MIC_TEST_MODE
                else "TEST MIC  /  15s SAFE"
            ),
        )

        if self._external_controller_active:
            status_text = (
                "EXTERNAL LIVE CONTROL  PID {}\n"
                "Stop its console with Ctrl+C before selecting a Control Center mode."
            ).format(control.process_id)
            color = self.WARN
            self._app_status = None
        elif (
            self._app_status is not None
            and managed.mode is None
            and managed.phase == "OFF"
        ):
            # A pending or finished RoboMaster app launch owns the status line
            # until a controller actually starts.
            status_text, color = self._app_status
        elif managed.phase == "OFF" and not self.safety_var.get():
            if self.dry_run_var.get():
                status_text = (
                    "DRY RUN  |  HAND or VOICE will run without the S1; commands "
                    "are shown but not sent.\n"
                    "Untick DRY RUN and check the safety box to drive a real S1."
                )
            else:
                status_text = (
                    "DISARMED  |  Leap visualization is active; robot control is off.\n"
                    "Check the safety box, or tick DRY RUN to test without the S1."
                )
            color = self.MUTED
        else:
            mode_name = {
                LEAP_MODE: "LEAP",
                VOICE_MODE: "VOICE",
                MIC_TEST_MODE: "MIC TEST / DRY RUN",
                None: "CONTROL",
            }[managed.mode]
            if managed.mode in (LEAP_MODE, VOICE_MODE) and not managed.live:
                mode_name += " / DRY RUN"
            status_text = "{} {}  |  {}".format(
                mode_name, managed.phase, managed.message
            )
            if (
                control_fresh
                and managed.process_id is not None
                and control.process_id == managed.process_id
            ):
                status_text += "\n{}  |  {}".format(control.state, control.reason)
            color = (
                self.ERROR
                if managed.phase == "ERROR"
                else self.WARN
                if managed.live and managed.phase in ("STARTING", "RUNNING")
                else self.GOOD
                if managed.phase in ("RUNNING", "OFF")
                else self.MUTED
            )
            self._app_status = None
        if self._vision_is_running():
            # Vision owns the status while it runs. The camera feed appears in
            # the panel above once YOLO, the voice model, and Piper finish
            # loading (~15-55 s on CPU).
            status_text = (
                "OBJECT DETECTION  |  the S1 camera feed appears above once "
                "models load (~15-55 s on CPU).\n"
                "Start VOICE CONTROL and say 'what do you see' or 'do you see a "
                "chair?' to hear it. Untick OBJECT DETECTION to turn it off."
            )
            color = self.GOOD
        self._set_controller_status(status_text, color)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.safety_var.set(False)
        self._close_mic_help()
        self._stop_vision()
        self.controller_manager.close()
        if self.shutdown_file is not None:
            try:
                self.shutdown_file.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        self.reader.stop()
        self.reader.join(timeout=1.0)
        # Let Tk finish any queued theme/display callbacks before destruction.
        self.root.after_idle(self.root.destroy)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if not self._closing:
                self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show Leap hands and safely select RoboMaster control input"
    )
    parser.add_argument(
        "--sdk-root",
        type=Path,
        default=Path(
            os.environ.get(
                "LEAPSDK_INSTALL_LOCATION",
                r"C:\Program Files\Ultraleap\LeapSDK",
            )
        ),
    )
    parser.add_argument("--width", type=int, default=620)
    parser.add_argument("--height", type=int, default=680)
    parser.add_argument("--x", type=int, default=-1)
    parser.add_argument("--y", type=int, default=72)
    parser.add_argument("--opacity", type=float, default=0.90)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--control-status",
        type=Path,
        default=DEFAULT_CONTROL_STATUS_PATH,
    )
    parser.add_argument(
        "--shutdown-file",
        type=Path,
        help="close and safely stop managed control when this request file appears",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.sdk_root.is_dir():
        print("Ultraleap SDK not found: {}".format(args.sdk_root), file=sys.stderr)
        return 1
    if args.width < 560 or args.height < 620:
        print("Control Center is too small.", file=sys.stderr)
        return 1

    # Import Tk only after argument validation so pure projection tests remain
    # usable in environments without a graphical session.
    import tkinter as tk

    probe = tk.Tk()
    probe.withdraw()
    screen_width = probe.winfo_screenwidth()
    # Drain Tk's initial theme notification before destroying this short-lived
    # probe; otherwise Tcl can report a late <<ThemeChanged>> during shutdown.
    probe.update_idletasks()
    probe.update()
    probe.destroy()
    x = args.x if args.x >= 0 else max(0, screen_width - args.width - 18)

    reader = LeapCReader(args.sdk_root)
    reader.start()
    center = ControlCenterWindow(
        reader=reader,
        width=args.width,
        height=args.height,
        x=x,
        y=max(0, args.y),
        opacity=args.opacity,
        duration_s=max(0.0, args.duration),
        control_reader=ControlStatusReader(args.control_status),
        controller_manager=ControlCenterProcessManager(
            control_status_path=args.control_status
        ),
        shutdown_file=args.shutdown_file,
    )
    center.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
