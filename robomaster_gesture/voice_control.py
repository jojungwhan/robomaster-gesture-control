"""Offline Windows voice control for a RoboMaster chassis.

Movement phrases require a wake word by default.  Every accepted phrase creates
one bounded motion pulse; silence can therefore never leave the robot moving.
STOP phrases are accepted with or without the wake word.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence

from .control_lease import ControlLeaseError, ControllerLease
from .control_status import DEFAULT_CONTROL_STATUS_PATH, ControlStatusPublisher
from .models import GestureDecision, VelocityCommand, translation_directions
from .robot_adapter import (
    CommandPump,
    CommandPumpConfig,
    DjiRobotAdapter,
    DryRunRobot,
    RobotError,
    S1AppKeyboardAdapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEECH_HELPER = PROJECT_ROOT / "scripts" / "windows_speech_recognizer.ps1"


@dataclass(frozen=True)
class VoiceIntent:
    name: str
    command: VelocityCommand
    stop: bool = False


@dataclass(frozen=True)
class SpeechEvent:
    event: str
    text: str = ""
    confidence: float = 0.0
    message: str = ""


def _normalized_words(text: str):
    return tuple(re.findall(r"[a-z]+", text.casefold()))


def parse_voice_command(
    text: str,
    speed_m_s: float = 0.20,
    yaw_deg_s: float = 25.0,
    wake_word: str = "robot",
    require_wake_word: bool = True,
) -> Optional[VoiceIntent]:
    """Turn a constrained speech phrase into one unambiguous chassis command."""
    words = list(_normalized_words(text))
    if not words:
        return None

    stop_words = {"stop", "halt", "freeze", "cancel"}
    if stop_words.intersection(words) or (
        "emergency" in words and "stop" in words
    ):
        return VoiceIntent("STOP", VelocityCommand.stopped(), stop=True)

    normalized_wake = _normalized_words(wake_word)
    has_wake_word = bool(normalized_wake) and all(
        item in words for item in normalized_wake
    )
    if require_wake_word and not has_wake_word:
        return None
    for item in normalized_wake:
        if item in words:
            words.remove(item)

    word_set = set(words)
    left = "left" in word_set
    right = "right" in word_set
    forward = bool({"forward", "ahead"}.intersection(word_set))
    back = bool({"back", "backward", "backwards", "reverse"}.intersection(word_set))

    if left and right or forward and back:
        return None

    is_turn = bool({"turn", "rotate", "spin"}.intersection(word_set))
    if is_turn:
        if left == right or forward or back:
            return None
        clockwise = yaw_deg_s if right else -yaw_deg_s
        return VoiceIntent(
            "TURN RIGHT" if right else "TURN LEFT",
            VelocityCommand(clockwise_deg_s=clockwise),
        )

    forward_speed = speed_m_s if forward else (-speed_m_s if back else 0.0)
    right_speed = speed_m_s if right else (-speed_m_s if left else 0.0)
    command = VelocityCommand(
        forward_m_s=forward_speed,
        right_m_s=right_speed,
    )
    if command.is_stopped():
        return None

    names = translation_directions(command)
    return VoiceIntent(" + ".join(names), command)


class WindowsSpeechRecognizer:
    """JSON-line bridge to Windows' installed offline speech recognizer."""

    def __init__(
        self,
        helper: Path,
        culture: str,
        wake_word: str,
        audio_file: Optional[Path] = None,
    ):
        self.helper = Path(helper)
        self.culture = culture
        self.wake_word = wake_word
        self.audio_file = Path(audio_file) if audio_file else None
        self._process = None
        self._events = queue.Queue()
        self._stderr = []
        self._threads = []

    def start(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows System.Speech is only available on Windows.")
        if not self.helper.is_file():
            raise RuntimeError("Speech helper not found: {}".format(self.helper))

        arguments = [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.helper),
            "-Culture",
            self.culture,
            "-WakeWord",
            self.wake_word,
        ]
        if self.audio_file is not None:
            arguments.extend(("-AudioFile", str(self.audio_file)))

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8-sig",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        self._threads = [
            threading.Thread(
                target=self._read_stdout,
                name="WindowsSpeechStdout",
                daemon=True,
            ),
            threading.Thread(
                target=self._read_stderr,
                name="WindowsSpeechStderr",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                self._events.put(
                    SpeechEvent(
                        event=str(data.get("event", "")),
                        text=str(data.get("text", "")),
                        confidence=float(data.get("confidence", 0.0)),
                        message=str(data.get("message", "")),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                self._events.put(
                    SpeechEvent(event="error", message="Invalid speech event: " + line)
                )
        self._events.put(SpeechEvent(event="process-ended"))

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())

    def get(self, timeout_s: float = 0.05) -> Optional[SpeechEvent]:
        try:
            return self._events.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None

    @property
    def returncode(self) -> Optional[int]:
        return None if self._process is None else self._process.poll()

    @property
    def stderr_text(self) -> str:
        return "\n".join(self._stderr).strip()

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
        self._process = None


def _make_robot(args):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Control RoboMaster with bounded offline voice commands from the "
            "default PC microphone or a PCM WAV file. Dry-run is the default."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--audio-file",
        type=Path,
        help="recognize commands from a PCM WAV file instead of the microphone",
    )
    source.add_argument(
        "--list-recognizers",
        action="store_true",
        help="list installed Windows speech recognizers and exit",
    )
    parser.add_argument("--culture", default="en-US")
    parser.add_argument("--wake-word", default="robot")
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="accept movement phrases without 'robot' (less safe)",
    )
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--command-duration", type=float, default=0.60)
    parser.add_argument("--speed", type=float, default=0.20)
    parser.add_argument("--yaw-speed", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help=(
            "verify the live robot transport, send STOP, and exit without "
            "starting speech recognition"
        ),
    )
    parser.add_argument("--transport", choices=("sdk", "s1-app"), default="s1-app")
    parser.add_argument("--connection", choices=("ap", "sta", "rndis"), default="sta")
    parser.add_argument("--protocol", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--robot-ip")
    parser.add_argument("--local-ip")
    parser.add_argument("--serial-number")
    parser.add_argument("--status-file", type=Path, default=DEFAULT_CONTROL_STATUS_PATH)
    parser.add_argument("--speech-helper", type=Path, default=DEFAULT_SPEECH_HELPER)
    return parser


def _validate_args(args) -> None:
    if args.connect_only and not args.live:
        raise ValueError("--connect-only requires --live")
    if args.connect_only and args.audio_file is not None:
        raise ValueError("--connect-only cannot be combined with --audio-file")
    if args.connect_only and args.list_recognizers:
        raise ValueError("--connect-only cannot be combined with --list-recognizers")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")
    if not 0.10 <= args.command_duration <= 3.0:
        raise ValueError("--command-duration must be between 0.10 and 3 seconds")
    if not 0.05 <= args.speed <= 0.50:
        raise ValueError("--speed must be between 0.05 and 0.50 m/s")
    if not 5.0 <= args.yaw_speed <= 90.0:
        raise ValueError("--yaw-speed must be between 5 and 90 degrees/s")
    if args.duration < 0.0:
        raise ValueError("--duration cannot be negative")
    if not args.no_wake_word and not _normalized_words(args.wake_word):
        raise ValueError("--wake-word must contain at least one letter")
    if args.audio_file is not None:
        if not args.audio_file.is_file():
            raise ValueError("audio file not found: {}".format(args.audio_file))
        if args.audio_file.suffix.casefold() != ".wav":
            raise ValueError("Windows offline recognition requires a PCM WAV file")


def _list_recognizers() -> int:
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "[System.Speech.Recognition.SpeechRecognitionEngine]::"
        "InstalledRecognizers() | ForEach-Object { "
        "Write-Output ($_.Culture.Name + \"`t\" + $_.Description) }"
    )
    completed = subprocess.run(
        ("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command),
        check=False,
        text=True,
    )
    return int(completed.returncode)


def _run_connect_only(args, robot) -> int:
    controller_lease = ControllerLease()
    try:
        controller_lease.acquire()
        robot.connect()
        robot.stop()
        connected_ip = getattr(robot, "connected_ip", None)
        if args.transport == "s1-app":
            print(
                "Voice transport preflight passed: RoboMaster app input access "
                "verified and W/A/S/D released. Speech recognition was not started. "
                "This does not verify the app-to-robot radio link.",
                flush=True,
            )
        else:
            print(
                "Voice transport preflight passed{}; STOP sent and speech "
                "recognition was not started.".format(
                    " at {}".format(connected_ip) if connected_ip else ""
                ),
                flush=True,
            )
        return 0
    finally:
        robot.close()
        controller_lease.close()


def run(args) -> int:
    _validate_args(args)
    if args.list_recognizers:
        return _list_recognizers()

    robot = _make_robot(args)
    if args.connect_only:
        return _run_connect_only(args, robot)
    pump = None
    controller_lease = None  # type: Optional[ControllerLease]
    recognizer = WindowsSpeechRecognizer(
        helper=args.speech_helper,
        culture=args.culture,
        wake_word="" if args.no_wake_word else args.wake_word,
        audio_file=args.audio_file,
    )
    publisher = ControlStatusPublisher(
        path=args.status_file,
        live=args.live,
        transport=args.transport,
    )
    current = VelocityCommand.stopped()
    motion_deadline_s = 0.0
    input_completed = False
    start_s = None

    try:
        if args.live:
            controller_lease = ControllerLease()
            controller_lease.acquire()
        robot.connect()
        pump = CommandPump(
            robot,
            CommandPumpConfig(
                rate_hz=20.0,
                stale_after_s=0.20,
                moving_keepalive_s=0.10,
                robot_timeout_s=0.30,
            ),
        )
        pump.start()
        # Do not accept or queue speech until the robot connection and the
        # independent stale-command watchdog are both active.  This prevents a
        # phrase spoken during a slow live connection from moving the robot
        # later, after the operator's context may have changed.
        recognizer.start()
        start_s = time.monotonic()
        mode = "LIVE ROBOT" if args.live else "DRY RUN"
        source_name = str(args.audio_file) if args.audio_file else "default microphone"
        print("{} voice control; source={}.".format(mode, source_name), flush=True)
        print(
            "Say 'robot forward/back/left/right' or 'stop'. Each movement lasts "
            "{:.2f}s unless STOP is heard first.".format(args.command_duration),
            flush=True,
        )

        publisher.publish(
            GestureDecision("VOICE", "voice controller started", current), force=True
        )

        while True:
            now_s = time.monotonic()
            if (
                args.duration > 0.0
                and start_s is not None
                and now_s - start_s >= args.duration
            ):
                print("Duration reached; stopping.", flush=True)
                break

            event = recognizer.get(timeout_s=0.05)
            if event is not None:
                if event.event == "ready":
                    print("Speech recognizer ready: {}".format(event.message), flush=True)
                elif event.event == "recognized":
                    print(
                        "Heard {!r} confidence={:.2f}".format(
                            event.text, event.confidence
                        ),
                        flush=True,
                    )
                    intent = parse_voice_command(
                        event.text,
                        speed_m_s=args.speed,
                        yaw_deg_s=args.yaw_speed,
                        wake_word=args.wake_word,
                        require_wake_word=not args.no_wake_word,
                    )
                    if (
                        event.confidence < args.min_confidence
                        and not (intent is not None and intent.stop)
                    ):
                        print("Ignored: below confidence threshold.", flush=True)
                    else:
                        if intent is None:
                            print("Ignored: no unambiguous movement command.", flush=True)
                        elif intent.stop:
                            current = VelocityCommand.stopped()
                            motion_deadline_s = 0.0
                            pump.halt()
                            publisher.publish(
                                GestureDecision("VOICE", "voice STOP", current),
                                force=True,
                            )
                            print("VOICE -> STOP", flush=True)
                        elif (
                            args.transport == "s1-app"
                            and abs(intent.command.clockwise_deg_s) > 0.0
                            and intent.command.forward_m_s == 0.0
                            and intent.command.right_m_s == 0.0
                        ):
                            current = VelocityCommand.stopped()
                            motion_deadline_s = 0.0
                            pump.halt()
                            publisher.publish(
                                GestureDecision(
                                    "VOICE",
                                    "turn unavailable in S1 app mode",
                                    current,
                                ),
                                force=True,
                            )
                            print(
                                "VOICE -> STOP (stock S1 app cannot rotate with W/A/S/D)",
                                flush=True,
                            )
                        else:
                            current = intent.command
                            motion_deadline_s = now_s + args.command_duration
                            pump.submit(current)
                            publisher.publish(
                                GestureDecision(
                                    "VOICE",
                                    "voice command: " + intent.name,
                                    current,
                                ),
                                force=True,
                            )
                            print("VOICE -> {}".format(intent.name), flush=True)
                elif event.event == "error":
                    raise RuntimeError(event.message or "speech recognition error")
                elif event.event in ("completed", "process-ended"):
                    input_completed = True

            now_s = time.monotonic()
            if not current.is_stopped():
                if now_s >= motion_deadline_s:
                    current = VelocityCommand.stopped()
                    motion_deadline_s = 0.0
                    pump.halt()
                    publisher.publish(
                        GestureDecision("VOICE", "voice pulse expired", current),
                        force=True,
                    )
                    print("VOICE -> STOP (pulse expired)", flush=True)
                else:
                    pump.submit(current)

            if pump.error is not None:
                raise RobotError("Command sender stopped: {}".format(pump.error))
            if input_completed and current.is_stopped():
                if recognizer.returncode not in (None, 0):
                    raise RuntimeError(
                        recognizer.stderr_text or "speech recognizer exited with an error"
                    )
                break
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopping immediately.", flush=True)
    finally:
        if pump is not None:
            pump.halt()
            pump.close()
        try:
            robot.close()
        finally:
            recognizer.close()
        if controller_lease is not None:
            controller_lease.close()
        publisher.publish(
            GestureDecision(
                "VOICE", "voice controller stopped", VelocityCommand.stopped()
            ),
            force=True,
        )
    return 0


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, ControlLeaseError, RuntimeError, RobotError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
