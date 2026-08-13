"""Non-activating Windows overlay for Ultraleap hand tracking.

The hand topology and desktop x/z projection are based on Ultraleap's
``leapc-python-bindings/examples/visualiser.py``.  This implementation was
rewritten around Tk and the installed LeapC CFFI module so it needs no OpenCV
and, importantly for RoboMaster control, never takes keyboard focus.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional, Sequence, Tuple

from .control_status import DEFAULT_CONTROL_STATUS_PATH, ControlStatusReader
from .models import translation_directions


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


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


class OverlayWindow:
    BACKGROUND = "#071018"
    PANEL = "#0C1A26"
    GRID = "#163247"
    TEXT = "#D9F7FF"
    MUTED = "#7EA5B8"
    GOOD = "#43F0A6"
    WARN = "#FFBE55"
    RIGHT = "#42D8FF"
    LEFT = "#FF5CCB"

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
    ):
        import tkinter as tk

        self.tk = tk
        self.reader = reader
        self.width = width
        self.height = height
        self.duration_s = duration_s
        self.control_reader = control_reader
        self.started_at_s = time.monotonic()
        self._closing = False

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Leap Hand Overlay")
        self.root.overrideredirect(True)
        self.root.configure(bg=self.BACKGROUND)
        self.root.geometry("{}x{}+{}+{}".format(width, height, x, y))
        self.root.attributes("-alpha", max(0.35, min(1.0, opacity)))
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            bg=self.BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._show_without_activation(x, y)
        self.root.after(0, self._render)

    def _show_without_activation(self, x: int, y: int) -> None:
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
            old_style
            | ws_ex_transparent
            | ws_ex_toolwindow
            | ws_ex_noactivate,
        )

        # Let Tk map the already-NOACTIVATE window, then reinforce topmost and
        # click-through state without transferring keyboard focus.
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
            self.height,
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
        self.canvas.create_text(
            16,
            17,
            text="LEAP HAND TRACKING",
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
            directions = translation_directions(control.command)
            mode = "ROBOT" if control.live else "DRY RUN"
            movement = " + ".join(directions) if directions else "STOP"
            control_text = "{}  {}  [{}]".format(mode, movement, control.state)
            control_color = self.WARN if control.live and directions else self.GOOD
        else:
            control_text = "ROBOT CONTROL  OFF"
            control_color = self.MUTED
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
                "GRAB",
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
        self.canvas.create_text(
            16,
            self.height - 20,
            text=detail,
            fill=self.MUTED,
            font=("Consolas", 9),
            anchor="w",
        )
        self.root.after(33, self._render)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
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
    parser = argparse.ArgumentParser(description="Show Leap hands without taking focus")
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
    parser.add_argument("--width", type=int, default=460)
    parser.add_argument("--height", type=int, default=390)
    parser.add_argument("--x", type=int, default=-1)
    parser.add_argument("--y", type=int, default=72)
    parser.add_argument("--opacity", type=float, default=0.90)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--control-status",
        type=Path,
        default=DEFAULT_CONTROL_STATUS_PATH,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.sdk_root.is_dir():
        print("Ultraleap SDK not found: {}".format(args.sdk_root), file=sys.stderr)
        return 1
    if args.width < 320 or args.height < 260:
        print("Overlay is too small.", file=sys.stderr)
        return 1

    # Import Tk only after argument validation so pure projection tests remain
    # usable in environments without a graphical session.
    import tkinter as tk

    probe = tk.Tk()
    probe.withdraw()
    screen_width = probe.winfo_screenwidth()
    probe.destroy()
    x = args.x if args.x >= 0 else max(0, screen_width - args.width - 18)

    reader = LeapCReader(args.sdk_root)
    reader.start()
    overlay = OverlayWindow(
        reader=reader,
        width=args.width,
        height=args.height,
        x=x,
        y=max(0, args.y),
        opacity=args.opacity,
        duration_s=max(0.0, args.duration),
        control_reader=ControlStatusReader(args.control_status),
    )
    overlay.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
