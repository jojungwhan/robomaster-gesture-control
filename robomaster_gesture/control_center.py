"""Safe subprocess lifecycle for the interactive RoboMaster Control Center."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional, Sequence
from uuid import uuid4

from .control_status import DEFAULT_CONTROL_STATUS_PATH
from .whisper_models import WhisperModelChoice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_PYTHON = (
    PROJECT_ROOT.parent / ".venv-robomaster" / "Scripts" / "python.exe"
)
DEFAULT_BRIDGE_DLL = PROJECT_ROOT / "build" / "leap_hand_bridge.dll"
DEFAULT_STOP_DIRECTORY = PROJECT_ROOT / "logs" / "control_center"
DEFAULT_MODELS_DIRECTORY = PROJECT_ROOT.parent / "models"
# Touched by the voice controller on "what do you see"; watched by the running
# object-detection process, so one recognizer serves movement and scene queries.
DESCRIBE_REQUEST_PATH = PROJECT_ROOT / "logs" / "describe_request.flag"

LEAP_MODE = "leap"
VOICE_MODE = "voice"
MIC_TEST_MODE = "mic-test"
CONTROLLER_MODES = frozenset((LEAP_MODE, VOICE_MODE, MIC_TEST_MODE))
# Modes that listen to the microphone (drive the mic meter and transcript).
VOICE_MODES = frozenset((VOICE_MODE, MIC_TEST_MODE))
MIC_LEVEL_PREFIX = "MIC LEVEL "
SPEECH_TRANSCRIPT_PREFIX = "MIC TRANSCRIPT "
MAX_TRANSCRIPT_LENGTH = 160

# Voice control needs no wake word: say just the direction ("forward"). Movement
# is gated by the confidence threshold below and the strict command-word filter.
# Trailing silence before transcribing (4 blocks x 0.10 s = 0.40 s) balances low
# latency against capturing the whole word, which matters for accuracy.
VOICE_END_SILENCE_BLOCKS = "4"

# A correct command must still arrive within this window or it is ignored as
# stale. CPU Whisper transcription of a short command can take a few seconds, so
# this is wider than the recognizer default; it bounds how old a command the
# robot will still act on.
VOICE_MAX_COMMAND_AGE = "6"

# Forward/back travel less per pulse than a strafe on the stock S1, so give them
# a longer key-press than the 0.35 s used for left/right.
VOICE_COMMAND_DURATION = "0.35"
VOICE_FORWARD_COMMAND_DURATION = "0.6"

# Minimum Whisper confidence (exp of the average token log-probability) for a
# movement command. Short single words spoken near the robot score lower than in
# a quiet room, so this is kept modest; the strict command-word filter is the
# real guard against noise, and "stop" is always accepted regardless.
LIVE_MIN_CONFIDENCE = "0.30"


def stop_requested(path: Optional[Path]) -> bool:
    """Return whether a controller received its private graceful-stop request."""
    return path is not None and Path(path).is_file()


def parse_microphone_level(line: str) -> Optional[int]:
    """Parse a private Control Center microphone-level status line."""
    if not line.startswith(MIC_LEVEL_PREFIX):
        return None
    try:
        level = int(line[len(MIC_LEVEL_PREFIX) :].strip())
    except ValueError:
        return None
    return level if 0 <= level <= 100 else None


@dataclass(frozen=True)
class SpeechTranscript:
    text: str
    confidence: float


def format_speech_transcript(text: str, confidence: float) -> str:
    """Format one recognized phrase as a private Control Center status line."""
    clean_text = " ".join(str(text).split())[:MAX_TRANSCRIPT_LENGTH]
    try:
        clean_confidence = float(confidence)
    except (TypeError, ValueError):
        clean_confidence = 0.0
    if not math.isfinite(clean_confidence):
        clean_confidence = 0.0
    clean_confidence = max(0.0, min(1.0, clean_confidence))
    payload = json.dumps(
        {"text": clean_text, "confidence": round(clean_confidence, 6)},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return SPEECH_TRANSCRIPT_PREFIX + payload


def parse_speech_transcript(line: str) -> Optional[SpeechTranscript]:
    """Parse a structured recognized phrase emitted for the Control Center."""
    if not line.startswith(SPEECH_TRANSCRIPT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(SPEECH_TRANSCRIPT_PREFIX) :])
        text = " ".join(str(payload["text"]).split())[:MAX_TRANSCRIPT_LENGTH]
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not text or not math.isfinite(confidence):
        return None
    return SpeechTranscript(text, max(0.0, min(1.0, confidence)))


@dataclass(frozen=True)
class ControllerProcessSnapshot:
    mode: Optional[str] = None
    phase: str = "OFF"
    message: str = "Robot control is off."
    process_id: Optional[int] = None
    live: bool = False
    audio_level: Optional[int] = None
    audio_updated_at_s: float = 0.0
    transcript_text: str = ""
    transcript_confidence: float = 0.0
    transcript_updated_at_s: float = 0.0


class ControlCenterProcessManager:
    """Switch one Control Center-owned motion process at a time.

    The machine-local ``ControllerLease`` in each live child remains the final
    exclusivity guard.  This manager additionally performs orderly switching
    and provides a file-based stop request so controller ``finally`` blocks can
    release the robot and all synthesized movement keys.
    """

    def __init__(
        self,
        python_executable: Path = DEFAULT_CONTROLLER_PYTHON,
        bridge_dll: Path = DEFAULT_BRIDGE_DLL,
        control_status_path: Path = DEFAULT_CONTROL_STATUS_PATH,
        stop_directory: Path = DEFAULT_STOP_DIRECTORY,
        popen_factory=subprocess.Popen,
    ):
        self.python_executable = Path(python_executable)
        self.bridge_dll = Path(bridge_dll)
        self.control_status_path = Path(control_status_path)
        self.stop_directory = Path(stop_directory)
        self._popen_factory = popen_factory
        self._lock = threading.RLock()
        self._snapshot = ControllerProcessSnapshot()
        self._process = None
        self._stop_path = None  # type: Optional[Path]
        self._expected_stop_process = None
        self._transition_thread = None  # type: Optional[threading.Thread]
        self._cancel_start = threading.Event()
        self._closed = False
        # None means "let the child use its default model"; a choice overrides
        # it for the next Voice or microphone-test start.
        self._whisper_model = None  # type: Optional[WhisperModelChoice]

    def set_whisper_model(self, choice: Optional[WhisperModelChoice]) -> None:
        with self._lock:
            self._whisper_model = choice

    @property
    def whisper_model(self) -> Optional[WhisperModelChoice]:
        with self._lock:
            return self._whisper_model

    def latest(self) -> ControllerProcessSnapshot:
        with self._lock:
            return self._snapshot

    def build_command(
        self,
        mode: str,
        stop_path: Path,
        live: bool = True,
    ) -> Sequence[str]:
        """Build the shell-free child command for a selected input mode.

        ``live`` drives whether Leap or Voice attach to the S1. When it is
        false the child uses the dry-run robot: it computes and reports commands
        but never connects to the RoboMaster app, so voice and Leap can be
        tested without a robot. The microphone test is always a dry run.
        """
        if mode not in CONTROLLER_MODES:
            raise ValueError("unsupported Control Center mode: {}".format(mode))

        common = (
            str(self.python_executable),
            "-m",
            (
                "robomaster_gesture.voice_control"
                if mode in VOICE_MODES
                else "robomaster_gesture"
            ),
        )
        live_flag = ("--live",) if live else ()
        if mode == LEAP_MODE:
            return common + live_flag + (
                "--transport",
                "s1-app",
                "--hand",
                "right",
                "--bridge-dll",
                str(self.bridge_dll),
                "--status-file",
                str(self.control_status_path),
                "--stop-file",
                str(stop_path),
            )
        if mode == VOICE_MODE:
            return common + live_flag + (
                "--speech-engine",
                "whisper",
                "--transport",
                "s1-app",
                "--command-duration",
                VOICE_COMMAND_DURATION,
                "--forward-command-duration",
                VOICE_FORWARD_COMMAND_DURATION,
                "--min-confidence",
                LIVE_MIN_CONFIDENCE,
                # No wake word: say just "forward"/"back"/"left"/"right"/"stop".
                "--no-wake-word",
                "--whisper-end-silence-blocks",
                VOICE_END_SILENCE_BLOCKS,
                "--max-command-age",
                VOICE_MAX_COMMAND_AGE,
                # "what do you see" asks the object-detection process to describe.
                "--describe-request-file",
                str(DESCRIBE_REQUEST_PATH),
                "--status-file",
                str(self.control_status_path),
                "--stop-file",
                str(stop_path),
                "--emit-audio-level",
                "--emit-transcript",
            ) + self._selected_whisper_model_args()
        return common + (
            "--speech-engine",
            "whisper",
            "--transport",
            "s1-app",
            "--duration",
            "15",
            "--command-duration",
            VOICE_COMMAND_DURATION,
            "--forward-command-duration",
            VOICE_FORWARD_COMMAND_DURATION,
            "--min-confidence",
            LIVE_MIN_CONFIDENCE,
            "--no-wake-word",
            "--whisper-end-silence-blocks",
            VOICE_END_SILENCE_BLOCKS,
            "--max-command-age",
            VOICE_MAX_COMMAND_AGE,
            "--describe-request-file",
            str(DESCRIBE_REQUEST_PATH),
            "--status-file",
            str(self.control_status_path),
            "--stop-file",
            str(stop_path),
            "--emit-audio-level",
            "--emit-transcript",
        ) + self._selected_whisper_model_args()

    def _selected_whisper_model_args(self) -> Sequence[str]:
        """Override the child's default model only when one has been picked."""
        choice = self.whisper_model
        if choice is None:
            return ()
        return (
            "--whisper-model",
            str(choice.path),
            "--whisper-model-name",
            choice.name,
        )

    def switch_async(self, mode: str, live: bool = True) -> bool:
        """Gracefully stop the current child, then start ``mode`` in a worker."""
        if mode not in CONTROLLER_MODES:
            raise ValueError("unsupported Control Center mode: {}".format(mode))
        if mode == MIC_TEST_MODE:
            live = False
        with self._lock:
            if self._closed:
                return False
            if self._transition_thread is not None and self._transition_thread.is_alive():
                self._snapshot = replace(
                    self._snapshot,
                    message="Please wait for the current mode change to finish.",
                )
                return False
            self._cancel_start.clear()
            if (
                self._process is not None
                and self._process.poll() is None
                and self._snapshot.mode == mode
                and self._snapshot.live == live
            ):
                self._snapshot = replace(
                    self._snapshot,
                    message="{} is already active.".format(self._mode_label(mode)),
                )
                return False
            self._snapshot = replace(
                self._snapshot,
                phase="SWITCHING",
                message="Switching to {}...".format(self._mode_label(mode)),
            )
            worker = threading.Thread(
                target=self._switch_worker,
                args=(mode, live),
                name="ControlCenterModeSwitch",
                daemon=True,
            )
            self._transition_thread = worker
            worker.start()
            return True

    def stop_async(self) -> bool:
        """Request an immediate orderly stop without blocking the Tk thread."""
        with self._lock:
            if self._closed:
                return False
            if self._transition_thread is not None and self._transition_thread.is_alive():
                self._cancel_start.set()
                self._snapshot = replace(
                    self._snapshot,
                    phase="STOPPING",
                    message="Cancelling the mode change and stopping robot control...",
                )
                stop_path = self._stop_path
                if stop_path is not None:
                    try:
                        stop_path.parent.mkdir(parents=True, exist_ok=True)
                        stop_path.write_text("stop\n", encoding="ascii")
                    except OSError:
                        pass
                return True
            self._snapshot = replace(
                self._snapshot,
                phase="STOPPING",
                message="Stopping robot control and releasing W/A/S/D...",
            )
            worker = threading.Thread(
                target=self._switch_worker,
                args=(None, False),
                name="ControlCenterStop",
                daemon=True,
            )
            self._transition_thread = worker
            worker.start()
            return True

    def _switch_worker(self, mode: Optional[str], live: bool = True) -> None:
        try:
            self._stop_current(release_keys=True)
            with self._lock:
                closed = self._closed
            if mode is not None and not closed and not self._cancel_start.is_set():
                self._start_current(mode, live)
            if self._cancel_start.is_set():
                self._stop_current(release_keys=True)
            if (mode is None or self._cancel_start.is_set()) and not closed:
                self._set_snapshot(self._idle_snapshot_with_transcript())
        except Exception as exc:
            self._set_snapshot(
                ControllerProcessSnapshot(
                    phase="ERROR",
                    message="Could not change control mode: {}".format(exc),
                )
            )

    def _start_current(self, mode: str, live: bool = True) -> None:
        if mode == MIC_TEST_MODE:
            live = False
        if not self.python_executable.is_file():
            raise RuntimeError(
                "controller Python was not found: {}".format(self.python_executable)
            )
        if mode == LEAP_MODE and not self.bridge_dll.is_file():
            raise RuntimeError(
                "Leap bridge was not found; run build.ps1 first: {}".format(
                    self.bridge_dll
                )
            )

        self.stop_directory.mkdir(parents=True, exist_ok=True)
        stop_path = self.stop_directory / ("stop-{}.request".format(uuid4().hex))
        command = self.build_command(mode, stop_path, live)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        # Read the child pipe as UTF-8 (below), so the child must also write
        # UTF-8. Without this, a redirected child on a non-UTF-8 Windows locale
        # (for example CP949 on a Korean install) encodes non-ASCII output such
        # as the microphone device name in the ANSI code page, and it arrives as
        # mojibake. This also propagates to the whisper helper the child spawns.
        environment["PYTHONIOENCODING"] = "utf-8"
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = self._popen_factory(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        with self._lock:
            if self._closed:
                try:
                    stop_path.write_text("stop\n", encoding="ascii")
                except OSError:
                    pass
            self._process = process
            self._stop_path = stop_path
            self._expected_stop_process = None
            self._snapshot = ControllerProcessSnapshot(
                mode=mode,
                phase="STARTING",
                message="Starting {}...".format(self._mode_label(mode)),
                process_id=int(process.pid),
                live=live and mode != MIC_TEST_MODE,
            )
        monitor = threading.Thread(
            target=self._monitor_output,
            args=(process, mode, stop_path),
            name="ControlCenterOutput",
            daemon=True,
        )
        monitor.start()

    def _monitor_output(self, process, mode: str, stop_path: Path) -> None:
        last_line = ""
        stream = process.stdout
        if stream is not None:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                microphone_level = parse_microphone_level(line)
                transcript = parse_speech_transcript(line)
                with self._lock:
                    if self._process is not process:
                        continue
                    if microphone_level is not None:
                        self._snapshot = replace(
                            self._snapshot,
                            audio_level=microphone_level,
                            audio_updated_at_s=time.monotonic(),
                        )
                        continue
                    if transcript is not None:
                        self._snapshot = replace(
                            self._snapshot,
                            transcript_text=transcript.text,
                            transcript_confidence=transcript.confidence,
                            transcript_updated_at_s=time.monotonic(),
                        )
                        continue
                    last_line = line
                    phase = self._snapshot.phase
                    # Voice only becomes RUNNING once the recognizer is actually
                    # ready to listen; while the model is still loading it stays
                    # STARTING so the panel does not claim it is listening.
                    if (
                        line.startswith("Controls:")
                        or line.startswith("Speech recognizer ready:")
                    ):
                        phase = "RUNNING"
                    self._snapshot = replace(
                        self._snapshot,
                        phase=phase,
                        message=line,
                    )

        return_code = int(process.wait())
        with self._lock:
            if self._process is not process:
                self._remove_stop_file(stop_path)
                return
            expected = self._expected_stop_process is process
            self._process = None
            self._stop_path = None
            self._expected_stop_process = None
            retained_transcript = {
                "transcript_text": self._snapshot.transcript_text,
                "transcript_confidence": self._snapshot.transcript_confidence,
                "transcript_updated_at_s": self._snapshot.transcript_updated_at_s,
            }
            if expected:
                snapshot = ControllerProcessSnapshot(**retained_transcript)
            elif return_code == 0:
                snapshot = ControllerProcessSnapshot(
                    message=(
                        "Laptop microphone test finished."
                        if mode == MIC_TEST_MODE
                        else "{} ended.".format(self._mode_label(mode))
                    ),
                    **retained_transcript,
                )
            else:
                snapshot = ControllerProcessSnapshot(
                    phase="ERROR",
                    message=last_line
                    or "{} exited with code {}.".format(
                        self._mode_label(mode), return_code
                    ),
                    **retained_transcript,
                )
            self._snapshot = snapshot
        self._remove_stop_file(stop_path)

    def _stop_current(self, release_keys: bool) -> None:
        with self._lock:
            process = self._process
            stop_path = self._stop_path
            if process is not None and process.poll() is None:
                self._expected_stop_process = process

        if process is None or process.poll() is not None:
            if release_keys:
                self._release_s1_keys()
            return

        if stop_path is not None:
            try:
                stop_path.parent.mkdir(parents=True, exist_ok=True)
                stop_path.write_text("stop\n", encoding="ascii")
            except OSError:
                pass
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            # This is a last-resort path (for example, a child stuck during a
            # robot connection). Release keys both before and after termination.
            self._release_s1_keys()
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        finally:
            if release_keys:
                self._release_s1_keys()
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._stop_path = None
                    self._expected_stop_process = None
            if stop_path is not None:
                self._remove_stop_file(stop_path)

    @staticmethod
    def _release_s1_keys() -> None:
        """Best-effort emergency key-up independent of the child process."""
        if os.name != "nt":
            return
        try:
            from .robot_adapter import _Win32RoboMasterKeyboard

            backend = _Win32RoboMasterKeyboard("RoboMaster")
            window = backend.find_window()
            for key in ("w", "a", "s", "d"):
                # ``release`` always submits a global key-up; its window message
                # is simply ignored when the app has already closed (HWND 0).
                backend.release(window, key)
        except Exception:
            pass

    @staticmethod
    def _remove_stop_file(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            LEAP_MODE: "Leap hand control",
            VOICE_MODE: "voice control",
            MIC_TEST_MODE: "laptop microphone test",
        }[mode]

    def _idle_snapshot_with_transcript(self) -> ControllerProcessSnapshot:
        with self._lock:
            current = self._snapshot
            return ControllerProcessSnapshot(
                transcript_text=current.transcript_text,
                transcript_confidence=current.transcript_confidence,
                transcript_updated_at_s=current.transcript_updated_at_s,
            )

    def _set_snapshot(self, snapshot: ControllerProcessSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def close(self) -> None:
        """Synchronously disarm a managed child before the window exits."""
        with self._lock:
            self._closed = True
            self._cancel_start.set()
            transition = self._transition_thread
        self._stop_current(release_keys=True)
        if transition is not None and transition is not threading.current_thread():
            transition.join(timeout=3.5)
        # Catch a child that was created concurrently just before ``_closed``.
        self._stop_current(release_keys=True)
