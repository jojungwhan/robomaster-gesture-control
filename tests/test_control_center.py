import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from robomaster_gesture.control_center import (
    LEAP_MODE,
    MIC_TEST_MODE,
    VOICE_MODE,
    ControlCenterProcessManager,
    format_speech_transcript,
    parse_microphone_level,
    parse_speech_transcript,
    stop_requested,
)
from robomaster_gesture.leap_visualizer import MIC_HELP_STEPS


class ControlCenterCommandTests(unittest.TestCase):
    def setUp(self):
        self.manager = ControlCenterProcessManager(
            python_executable=Path(r"C:\controller\python.exe"),
            bridge_dll=Path(r"C:\project\leap_hand_bridge.dll"),
            control_status_path=Path(r"C:\project\status.json"),
        )
        self.stop_path = Path(r"C:\project\stop.request")

    def test_leap_mode_is_live_and_uses_the_guarded_app_controller(self):
        command = tuple(self.manager.build_command(LEAP_MODE, self.stop_path))

        self.assertEqual("robomaster_gesture", command[2])
        self.assertIn("--live", command)
        self.assertEqual("s1-app", command[command.index("--transport") + 1])
        self.assertEqual(str(self.stop_path), command[command.index("--stop-file") + 1])
        self.assertEqual(
            str(self.manager.bridge_dll), command[command.index("--bridge-dll") + 1]
        )
        self.assertNotIn("--emit-audio-level", command)
        self.assertNotIn("--emit-transcript", command)

    def test_voice_mode_is_live_no_wake_word_with_a_fast_endpoint(self):
        command = tuple(self.manager.build_command(VOICE_MODE, self.stop_path))

        self.assertEqual("robomaster_gesture.voice_control", command[2])
        self.assertIn("--live", command)
        self.assertEqual(
            "0.35", command[command.index("--command-duration") + 1]
        )
        # Forward/back get a longer pulse than a strafe so they travel more.
        self.assertEqual(
            "0.6", command[command.index("--forward-command-duration") + 1]
        )
        self.assertEqual(
            "0.30", command[command.index("--min-confidence") + 1]
        )
        # No wake word (say just the direction) with a 0.40 s endpoint that
        # captures the whole word for accuracy while staying responsive.
        self.assertIn("--no-wake-word", command)
        self.assertNotIn("--wake-word", command)
        self.assertEqual(
            "4", command[command.index("--whisper-end-silence-blocks") + 1]
        )
        # The one recognizer also answers "what do you see" by touching this file.
        self.assertIn("--describe-request-file", command)
        self.assertIn("--emit-audio-level", command)
        self.assertIn("--emit-transcript", command)
        self.assertEqual(
            "whisper", command[command.index("--speech-engine") + 1]
        )

    def test_dry_run_drops_live_for_leap_and_voice(self):
        leap = tuple(self.manager.build_command(LEAP_MODE, self.stop_path, live=False))
        voice = tuple(
            self.manager.build_command(VOICE_MODE, self.stop_path, live=False)
        )

        # Dry run must never attach to the S1; the child then uses the dry-run
        # robot and never opens the RoboMaster app.
        self.assertNotIn("--live", leap)
        self.assertNotIn("--live", voice)
        # Everything else is unchanged so the same command pipeline is exercised.
        self.assertEqual("s1-app", leap[leap.index("--transport") + 1])
        self.assertIn("--no-wake-word", voice)

    def test_microphone_test_is_timed_and_never_live(self):
        command = tuple(self.manager.build_command(MIC_TEST_MODE, self.stop_path))
        # The mic test is always dry, even if a live build is requested.
        live_request = tuple(
            self.manager.build_command(MIC_TEST_MODE, self.stop_path, live=True)
        )

        self.assertNotIn("--live", command)
        self.assertNotIn("--live", live_request)
        self.assertEqual("15", command[command.index("--duration") + 1])
        self.assertIn("--emit-audio-level", command)
        self.assertIn("--emit-transcript", command)
        self.assertEqual(
            "whisper", command[command.index("--speech-engine") + 1]
        )
        # The dry test uses the same acceptance threshold and no-wake-word
        # behavior as live voice so it reflects what would drive the robot.
        self.assertEqual(
            "0.30", command[command.index("--min-confidence") + 1]
        )
        self.assertIn("--no-wake-word", command)

    def test_selected_whisper_model_overrides_the_child_default(self):
        from robomaster_gesture.whisper_models import WhisperModelChoice

        # No selection: the child keeps its own default model.
        voice = tuple(self.manager.build_command(VOICE_MODE, self.stop_path))
        self.assertNotIn("--whisper-model", voice)

        choice = WhisperModelChoice(
            name="large-v3-turbo",
            label="Most accurate",
            path=Path(r"C:\models\whisper-large-v3-turbo-ct2"),
            speed_rank=5,
            is_default=False,
        )
        self.manager.set_whisper_model(choice)
        voice = tuple(self.manager.build_command(VOICE_MODE, self.stop_path))
        mic = tuple(self.manager.build_command(MIC_TEST_MODE, self.stop_path))

        self.assertEqual(
            str(choice.path), voice[voice.index("--whisper-model") + 1]
        )
        self.assertEqual(
            "large-v3-turbo", voice[voice.index("--whisper-model-name") + 1]
        )
        # The dry microphone test uses the same selected model.
        self.assertEqual(
            str(choice.path), mic[mic.index("--whisper-model") + 1]
        )
        # Leap control does not take a Whisper model.
        leap = tuple(self.manager.build_command(LEAP_MODE, self.stop_path))
        self.assertNotIn("--whisper-model", leap)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.build_command("camera", self.stop_path)


class ChildEnvironmentTests(unittest.TestCase):
    def test_child_stdio_is_forced_to_utf8(self):
        # A redirected child on a non-UTF-8 Windows locale would otherwise encode
        # non-ASCII output (such as a Korean microphone name) in the ANSI code
        # page, which the UTF-8 pipe reader turns into mojibake.
        captured = {}

        class DoneProcess:
            pid = 7

            def __init__(self):
                self.stdout = iter(())

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **kwargs):
            captured.update(kwargs)
            return DoneProcess()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = ControlCenterProcessManager(
                python_executable=Path(sys.executable),
                control_status_path=root / "status.json",
                stop_directory=root / "stops",
                popen_factory=fake_popen,
            )
            self.assertTrue(manager.switch_async(MIC_TEST_MODE))
            deadline = time.monotonic() + 2.0
            while "env" not in captured and time.monotonic() < deadline:
                time.sleep(0.01)
            manager.close()

        self.assertEqual("utf-8", captured["env"]["PYTHONIOENCODING"])


class MicrophoneLevelStatusTests(unittest.TestCase):
    def test_valid_control_center_levels_are_parsed(self):
        self.assertEqual(0, parse_microphone_level("MIC LEVEL 0"))
        self.assertEqual(57, parse_microphone_level("MIC LEVEL 57"))
        self.assertEqual(100, parse_microphone_level("MIC LEVEL 100"))

    def test_unrelated_or_invalid_lines_are_not_levels(self):
        for line in (
            "Heard 'robot forward' confidence=0.98",
            "MIC LEVEL",
            "MIC LEVEL loud",
            "MIC LEVEL -1",
            "MIC LEVEL 101",
        ):
            with self.subTest(line=line):
                self.assertIsNone(parse_microphone_level(line))

    def test_transcript_status_round_trips_text_and_confidence(self):
        line = format_speech_transcript("  robot   forward  ", 0.984321)
        transcript = parse_speech_transcript(line)

        self.assertEqual("robot forward", transcript.text)
        self.assertAlmostEqual(0.984321, transcript.confidence)

    def test_invalid_transcript_status_is_ignored(self):
        for line in (
            "Heard 'robot forward' confidence=0.98",
            "MIC TRANSCRIPT",
            "MIC TRANSCRIPT not-json",
            'MIC TRANSCRIPT {"text":"","confidence":0.9}',
            'MIC TRANSCRIPT {"text":"robot forward"}',
        ):
            with self.subTest(line=line):
                self.assertIsNone(parse_speech_transcript(line))


class MicrophoneHelpCopyTests(unittest.TestCase):
    def test_help_covers_samsung_windows_and_safe_verification(self):
        copy = " ".join("{} {}".format(title, detail) for title, detail in MIC_HELP_STEPS)

        self.assertIn("Fn+F10", copy)
        self.assertIn("Let desktop apps access your microphone", copy)
        self.assertIn("PowerShell may not appear", copy)
        self.assertIn("robot forward", copy)
        self.assertIn("cannot move the robot", copy)
        self.assertIn("Microphone Array > Start test", copy)


class GracefulStopRequestTests(unittest.TestCase):
    def test_missing_or_unspecified_request_does_not_stop(self):
        self.assertFalse(stop_requested(None))
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(stop_requested(Path(temporary) / "missing.request"))

    def test_existing_private_request_stops_controller(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary) / "stop.request"
            request.write_text("stop\n", encoding="ascii")
            self.assertTrue(stop_requested(request))

    def test_manager_stop_lets_child_cleanup_without_termination(self):
        class FakeStream:
            def __init__(self, stopped):
                self.stopped = stopped

            def __iter__(self):
                yield "Speech recognizer ready: fake\n"
                yield format_speech_transcript("robot forward", 0.98) + "\n"
                yield "MIC LEVEL 0\n"
                yield "MIC LEVEL 57\n"
                self.stopped.wait(timeout=3.0)

        class FakeProcess:
            pid = 4242

            def __init__(self):
                self.stopped = threading.Event()
                self.stdout = FakeStream(self.stopped)
                self.terminated = False
                self.killed = False

            def poll(self):
                return 0 if self.stopped.is_set() else None

            def wait(self, timeout=None):
                if not self.stopped.wait(timeout=timeout):
                    raise subprocess.TimeoutExpired(("fake",), timeout)
                return 0

            def terminate(self):
                self.terminated = True
                self.stopped.set()

            def kill(self):
                self.killed = True
                self.stopped.set()

        def wait_for(predicate, timeout_s=2.0):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(0.01)
            return bool(predicate())

        created = []

        def fake_popen(command, **_kwargs):
            process = FakeProcess()
            created.append(process)
            request = Path(command[command.index("--stop-file") + 1])

            def honor_stop_request():
                if wait_for(request.is_file):
                    process.stopped.set()

            threading.Thread(target=honor_stop_request, daemon=True).start()
            return process

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = ControlCenterProcessManager(
                python_executable=Path(sys.executable),
                control_status_path=root / "status.json",
                stop_directory=root / "stops",
                popen_factory=fake_popen,
            )
            self.assertTrue(manager.switch_async(VOICE_MODE))
            self.assertTrue(wait_for(lambda: manager.latest().process_id == 4242))
            self.assertTrue(wait_for(lambda: manager.latest().audio_level == 57))
            self.assertEqual("robot forward", manager.latest().transcript_text)
            self.assertAlmostEqual(0.98, manager.latest().transcript_confidence)
            self.assertEqual(
                "Speech recognizer ready: fake",
                manager.latest().message,
            )
            self.assertTrue(manager.stop_async())
            self.assertTrue(wait_for(lambda: manager.latest().phase == "OFF"))
            self.assertEqual("robot forward", manager.latest().transcript_text)
            manager.close()

        self.assertEqual(1, len(created))
        self.assertFalse(created[0].terminated)
        self.assertFalse(created[0].killed)


if __name__ == "__main__":
    unittest.main()
