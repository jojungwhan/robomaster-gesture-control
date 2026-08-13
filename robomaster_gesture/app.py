from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time
from typing import Optional, Sequence

from .control_status import DEFAULT_CONTROL_STATUS_PATH, ControlStatusPublisher
from .gesture import GestureConfig, GestureController
from .leap_source import LeapSource, LeapSourceError
from .models import GestureDecision, VelocityCommand, translation_directions
from .robot_adapter import (
    CommandPump,
    CommandPumpConfig,
    DjiRobotAdapter,
    DryRunRobot,
    RobotAdapter,
    RobotError,
    S1AppKeyboardAdapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_DLL = PROJECT_ROOT / "build" / "leap_hand_bridge.dll"


class ConsoleReporter:
    def __init__(self, interval_s: float = 0.20):
        self.interval_s = interval_s
        self._last_at = 0.0
        self._last_signature = None
        self._last_state = None

    @staticmethod
    def _play_cue(pattern) -> None:
        if sys.platform != "win32":
            return

        def play() -> None:
            try:
                import winsound

                for frequency, duration_ms in pattern:
                    winsound.Beep(frequency, duration_ms)
                    time.sleep(0.04)
            except Exception:
                pass

        threading.Thread(target=play, name="GestureAudioCue", daemon=True).start()

    def report(
        self,
        decision: GestureDecision,
        service_connected: bool,
        device_present: bool,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        signature = (decision.state, decision.reason)
        if decision.state != self._last_state:
            if decision.state == "READY":
                self._play_cue(((880, 100),))
            elif decision.state == "DRIVING":
                self._play_cue(((1200, 90), (1500, 110)))
            elif self._last_state == "DRIVING":
                self._play_cue(((420, 180),))
            self._last_state = decision.state
        if not force and signature == self._last_signature:
            if decision.state != "DRIVING" or now - self._last_at < self.interval_s:
                return

        command = decision.command
        directions = translation_directions(command)
        motion_text = "+".join(directions) if directions else "STOP"
        hand_text = "none"
        if decision.hand is not None:
            hand_text = (
                "{} pinch={:.2f} grab={:.2f} palm=({:+.0f},{:+.0f})mm".format(
                    decision.hand.handedness,
                    decision.hand.pinch_strength,
                    decision.hand.grab_strength,
                    decision.hand.palm_x_mm,
                    decision.hand.palm_z_mm,
                )
            )
        print(
            "[{:<7}] x={:+.2f}m/s y={:+.2f}m/s z={:+.0f}deg/s | "
            "{} | motion={} | hand={} | leap={}/{}".format(
                decision.state,
                command.forward_m_s,
                command.right_m_s,
                command.clockwise_deg_s,
                decision.reason,
                motion_text,
                hand_text,
                "service" if service_connected else "no-service",
                "device" if device_present else "no-device",
            ),
            flush=True,
        )
        self._last_at = now
        self._last_signature = signature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Control a DJI RoboMaster chassis with an Ultraleap hand tracker. "
            "Dry-run is the default; --live is required to send robot commands."
        )
    )
    parser.add_argument("--live", action="store_true", help="send commands to a robot")
    parser.add_argument(
        "--transport",
        choices=("sdk", "s1-app"),
        default="sdk",
        help="DJI EP SDK or stock-S1 Windows app keyboard transport (default: sdk)",
    )
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="connect, send a stop command, and disconnect without opening LeapC",
    )
    parser.add_argument(
        "--connection",
        choices=("ap", "sta", "rndis"),
        default="sta",
        help="RoboMaster connection mode (default: sta)",
    )
    parser.add_argument(
        "--protocol",
        choices=("tcp", "udp"),
        default="tcp",
        help="RoboMaster command transport (default: tcp)",
    )
    parser.add_argument("--robot-ip", help="robot IP for STA mode; otherwise discover")
    parser.add_argument("--local-ip", help="local adapter IP when auto-selection is wrong")
    parser.add_argument("--serial-number", help="14-character RoboMaster serial number")
    parser.add_argument(
        "--hand",
        choices=("right", "left", "any"),
        default="right",
        help="hand allowed to control the robot (default: right)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=0.35,
        help="maximum forward/strafe speed in m/s (default: 0.35)",
    )
    parser.add_argument(
        "--max-yaw",
        type=float,
        default=35.0,
        help="maximum rotation speed in degrees/s (default: 35)",
    )
    parser.add_argument(
        "--invert-strafe", action="store_true", help="reverse left/right mapping"
    )
    parser.add_argument(
        "--invert-yaw", action="store_true", help="reverse wrist-yaw mapping"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="stop automatically after this many seconds (useful for testing)",
    )
    parser.add_argument(
        "--bridge-dll",
        type=Path,
        default=DEFAULT_BRIDGE_DLL,
        help="path to leap_hand_bridge.dll",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_CONTROL_STATUS_PATH,
        help="local command-status file used by the hand overlay",
    )
    return parser


def make_robot(args: argparse.Namespace) -> RobotAdapter:
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


def run(args: argparse.Namespace) -> int:
    if args.connect_only and not args.live:
        raise ValueError("--connect-only requires --live")
    if not 0.05 <= args.max_speed <= 1.0:
        raise ValueError("--max-speed must be between 0.05 and 1.0 m/s")
    if not 5.0 <= args.max_yaw <= 120.0:
        raise ValueError("--max-yaw must be between 5 and 120 degrees/s")
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be positive")

    robot = make_robot(args)
    source = None  # type: Optional[LeapSource]
    pump = None  # type: Optional[CommandPump]
    status_publisher = None  # type: Optional[ControlStatusPublisher]

    if args.live:
        print(
            "LIVE MODE: physical motion is enabled, but the chassis remains stopped "
            "until the open-hand then pinch sequence completes.",
            flush=True,
        )
        if args.transport == "s1-app":
            print(
                "S1 APP MODE: DJI's documented W/A/S/D controls provide "
                "forward/back/strafe only; wrist-yaw rotation is disabled. "
                "Keep the RoboMaster live drive view in the foreground.",
                flush=True,
            )
    else:
        print(
            "DRY RUN: gesture-derived commands are displayed but no robot is contacted.",
            flush=True,
        )

    try:
        if args.connect_only:
            robot.connect()
            robot.stop()
            connected_ip = getattr(robot, "connected_ip", None)
            if args.transport == "s1-app":
                print(
                    "RoboMaster Windows-app transport verified; W/A/S/D released. "
                    "This verifies the app window and input permissions, not the "
                    "app-to-robot radio link."
                )
            else:
                print(
                    "RoboMaster connection verified{}; zero speed sent.".format(
                        " at {}".format(connected_ip) if connected_ip else ""
                    )
                )
            return 0

        source = LeapSource(args.bridge_dll)
        source.open()
        robot.connect()
        connected_ip = getattr(robot, "connected_ip", None)
        if connected_ip:
            print("Connected to RoboMaster at {}.".format(connected_ip), flush=True)

        pump = CommandPump(
            robot,
            CommandPumpConfig(
                rate_hz=15.0,
                stale_after_s=0.20,
                moving_keepalive_s=0.15,
                robot_timeout_s=0.35,
            ),
        )
        pump.start()

        controller = GestureController(
            GestureConfig(
                preferred_hand=args.hand,
                translation_deadzone_mm=(
                    10.0 if args.transport == "s1-app" else 25.0
                ),
                translation_full_scale_mm=(
                    65.0 if args.transport == "s1-app" else 120.0
                ),
                max_translation_m_s=args.max_speed,
                max_yaw_deg_s=args.max_yaw,
                response_curve=(1.0 if args.transport == "s1-app" else 1.35),
                smoothing_time_s=(0.08 if args.transport == "s1-app" else 0.12),
                strafe_sign=-1.0 if args.invert_strafe else 1.0,
                yaw_sign=(
                    0.0
                    if args.transport == "s1-app"
                    else (-1.0 if args.invert_yaw else 1.0)
                ),
            )
        )
        status_publisher = ControlStatusPublisher(
            path=args.status_file,
            live=args.live,
            transport=args.transport,
        )
        status_publisher.publish(
            GestureDecision(
                state=controller.state.value,
                reason="controller started - stopped",
                command=VelocityCommand.stopped(),
            ),
            force=True,
        )
        reporter = ConsoleReporter()
        start_s = time.monotonic()
        last_tracking_frame_s = start_s
        stale_reported = False
        current_decision = GestureDecision(
            state=controller.state.value,
            reason="controller started - stopped",
            command=VelocityCommand.stopped(),
        )

        print(
            "Controls: hold one open {} hand until READY, pinch for 0.35s, "
            "then move it. Release, make a fist, or remove the hand to stop. "
            "Press Ctrl+C to exit.".format(args.hand),
            flush=True,
        )
        if args.transport == "s1-app":
            print(
                "Audio cues: one beep=READY, two rising beeps=DRIVING, low "
                "beep=STOP. After the rising beeps, move the whole pinched "
                "hand at least 2 cm from its anchor.",
                flush=True,
            )

        while True:
            now_s = time.monotonic()
            if args.duration is not None and now_s - start_s >= args.duration:
                print("Duration reached; stopping.", flush=True)
                break

            frame = source.poll(timeout_ms=50)
            if frame is None:
                if now_s - last_tracking_frame_s > 0.20 and not stale_reported:
                    current_decision = controller.on_tracking_timeout(now_s)
                    pump.submit(VelocityCommand.stopped())
                    status_publisher.publish(current_decision, force=True)
                    reporter.report(
                        current_decision,
                        source.service_connected,
                        source.device_present,
                        force=True,
                    )
                    stale_reported = True
                else:
                    status_publisher.publish(current_decision)
                continue

            last_tracking_frame_s = frame.arrival_time_s
            stale_reported = False
            current_decision = controller.update(frame)
            pump.submit(current_decision.command)
            status_publisher.publish(current_decision)
            reporter.report(
                current_decision, source.service_connected, source.device_present
            )

            if pump.error is not None:
                raise RobotError("Command sender stopped: {}".format(pump.error))

    except KeyboardInterrupt:
        print("\nCtrl+C received; stopping immediately.", flush=True)
    finally:
        if status_publisher is not None:
            try:
                status_publisher.publish(
                    GestureDecision(
                        state="WAITING",
                        reason="controller stopped",
                        command=VelocityCommand.stopped(),
                    ),
                    force=True,
                )
            except OSError:
                pass
        if pump is not None:
            pump.halt()
            pump.close()
        try:
            robot.close()
        finally:
            if source is not None:
                source.close()

    return 0


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, LeapSourceError, RobotError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print("UNEXPECTED ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
