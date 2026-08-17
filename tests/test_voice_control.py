from contextlib import ExitStack
import unittest
from unittest import mock

from robomaster_gesture.voice_control import (
    SpeechEvent,
    WhisperSpeechRecognizer,
    _make_recognizer,
    build_parser,
    movement_transcription_is_stale,
    normalize_audio_level,
    normalize_event_age,
    parse_voice_command,
    run,
    _validate_args,
)


class VoiceCommandTests(unittest.TestCase):
    def test_whisper_engine_builds_the_isolated_local_recognizer(self):
        args = build_parser().parse_args(("--speech-engine", "whisper"))

        recognizer = _make_recognizer(args)

        self.assertIsInstance(recognizer, WhisperSpeechRecognizer)
        self.assertEqual("base.en", recognizer.model_name)
        self.assertEqual("whisper-base-en-ct2", recognizer.model.name)

    def test_whisper_forwards_a_custom_end_silence_endpoint(self):
        from pathlib import Path
        from robomaster_gesture.voice_control import WhisperSpeechRecognizer

        args = build_parser().parse_args(
            ("--speech-engine", "whisper", "--whisper-end-silence-blocks", "2")
        )
        recognizer = _make_recognizer(args)
        self.assertEqual(2, recognizer.end_silence_blocks)

        captured = {}
        recognizer._launch = lambda arguments, name, cwd=None: captured.update(
            arguments=list(arguments)
        )
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            Path, "is_dir", return_value=True
        ):
            recognizer.start()

        arguments = captured["arguments"]
        self.assertIn("--end-silence-blocks", arguments)
        self.assertEqual(
            "2", arguments[arguments.index("--end-silence-blocks") + 1]
        )

        # Without the override the recognizer keeps the helper's own default.
        default_recognizer = _make_recognizer(
            build_parser().parse_args(("--speech-engine", "whisper"))
        )
        self.assertIsNone(default_recognizer.end_silence_blocks)

    def test_recognition_age_is_normalized(self):
        for value, expected in (
            (None, 0.0),
            ("not-an-age", 0.0),
            (-1, 0.0),
            ("2.5", 2.5),
            (float("inf"), 0.0),
        ):
            with self.subTest(value=value):
                self.assertEqual(expected, normalize_event_age(value))

    def test_delayed_movement_is_stale_but_stop_is_always_accepted(self):
        movement = parse_voice_command("forward")
        stop = parse_voice_command("stop")

        self.assertTrue(
            movement_transcription_is_stale(
                SpeechEvent("recognized", age_seconds=3.1),
                movement,
                3.0,
            )
        )
        self.assertFalse(
            movement_transcription_is_stale(
                SpeechEvent("recognized", age_seconds=30.0),
                stop,
                3.0,
            )
        )

    def test_audio_levels_are_normalized_for_the_ui(self):
        cases = (
            (None, 0),
            ("not-a-level", 0),
            (-4, 0),
            (12.6, 13),
            (57, 57),
            (130, 100),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(expected, normalize_audio_level(value))

    def test_connect_only_requires_live_and_rejects_audio(self):
        parser = build_parser()
        with self.assertRaises(ValueError):
            _validate_args(parser.parse_args(("--connect-only",)))
        with self.assertRaises(ValueError):
            _validate_args(
                parser.parse_args(
                    (
                        "--live",
                        "--connect-only",
                        "--audio-file",
                        str(__file__),
                    )
                )
            )

    def test_connect_only_never_starts_recognizer_or_command_pump(self):
        events = []

        class FakeRobot:
            connected_ip = None

            def connect(self):
                events.append("connect")

            def stop(self):
                events.append("stop")

            def close(self):
                events.append("close")

        class FakeLease:
            def acquire(self):
                events.append("lease")

            def close(self):
                events.append("lease-close")

        def forbidden(*args, **kwargs):
            raise AssertionError("voice machinery must not start during preflight")

        args = build_parser().parse_args(("--live", "--connect-only"))
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control._make_robot",
                    return_value=FakeRobot(),
                )
            )
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control.ControllerLease",
                    FakeLease,
                )
            )
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control.WindowsSpeechRecognizer",
                    forbidden,
                )
            )
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control.CommandPump",
                    forbidden,
                )
            )
            self.assertEqual(0, run(args))

        self.assertEqual(
            ["lease", "connect", "stop", "close", "lease-close"],
            events,
        )

    def test_recognizer_starts_only_after_robot_and_watchdog(self):
        events = []

        class FakeRobot:
            def connect(self):
                events.append("robot-connect")

            def close(self):
                events.append("robot-close")

        class FakePump:
            def __init__(self, robot, config):
                events.append("pump-create")
                self.error = None

            def start(self):
                events.append("pump-start")

            def halt(self):
                events.append("pump-halt")

            def close(self):
                events.append("pump-close")

        class FakeRecognizer:
            def __init__(self, **kwargs):
                events.append("recognizer-create")
                self.returncode = 0
                self.stderr_text = ""
                self._completed = False

            def start(self):
                events.append("recognizer-start")

            def get(self, timeout_s=0.05):
                if not self._completed:
                    self._completed = True
                    return SpeechEvent("completed")
                return None

            def close(self):
                events.append("recognizer-close")

        publisher = mock.Mock()
        args = build_parser().parse_args(())
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control._make_robot",
                    return_value=FakeRobot(),
                )
            )
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control.CommandPump",
                    FakePump,
                )
            )
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control.WindowsSpeechRecognizer",
                    FakeRecognizer,
                )
            )
            stack.enter_context(
                mock.patch(
                    "robomaster_gesture.voice_control.ControlStatusPublisher",
                    return_value=publisher,
                )
            )
            self.assertEqual(0, run(args))

        self.assertLess(events.index("robot-connect"), events.index("pump-start"))
        self.assertLess(events.index("pump-start"), events.index("recognizer-start"))

    def test_cardinal_and_diagonal_commands(self):
        cases = (
            ("forward", "FORWARD", 1, 0),
            ("move backward", "BACK", -1, 0),
            ("left", "LEFT", 0, -1),
            ("go right", "RIGHT", 0, 1),
            ("forward left", "FORWARD + LEFT", 1, -1),
            ("back right", "BACK + RIGHT", -1, 1),
        )
        for phrase, name, forward_sign, right_sign in cases:
            with self.subTest(phrase=phrase):
                intent = parse_voice_command(phrase, speed_m_s=0.2)
                self.assertIsNotNone(intent)
                self.assertEqual(name, intent.name)
                self.assertEqual(forward_sign, (intent.command.forward_m_s > 0) - (intent.command.forward_m_s < 0))
                self.assertEqual(right_sign, (intent.command.right_m_s > 0) - (intent.command.right_m_s < 0))

    def test_turn_commands_have_clockwise_sign(self):
        left = parse_voice_command("turn left", yaw_deg_s=25)
        right = parse_voice_command("rotate right", yaw_deg_s=25)
        self.assertLess(left.command.clockwise_deg_s, 0)
        self.assertGreater(right.command.clockwise_deg_s, 0)

    def test_stop_never_requires_wake_word(self):
        for phrase in ("stop", "halt", "emergency stop", "robot stop"):
            with self.subTest(phrase=phrase):
                intent = parse_voice_command(phrase)
                self.assertTrue(intent.stop)
                self.assertTrue(intent.command.is_stopped())

    def test_no_wake_word_is_needed_by_default(self):
        # A bare direction drives; the strict command-word filter still rejects
        # prose. A wake word is opt-in via wake_word + require_wake_word.
        self.assertIsNotNone(parse_voice_command("forward"))
        self.assertIsNone(parse_voice_command("the cat is over there"))
        self.assertEqual("", build_parser().parse_args(()).wake_word)

    def test_wake_word_can_be_opted_into(self):
        # When a wake word is required, a bare direction is rejected and the
        # word must be present.
        self.assertIsNone(
            parse_voice_command(
                "forward", wake_word="robot", require_wake_word=True
            )
        )
        self.assertIsNotNone(
            parse_voice_command(
                "robot forward", wake_word="robot", require_wake_word=True
            )
        )

    def test_configured_wake_word_accepts_common_transcriptions(self):
        # Whisper writes a spoken "S1" several ways; each must still drive when
        # that wake word is configured and required.
        for phrase, name in (
            ("s1 forward", "FORWARD"),
            ("S1 forward.", "FORWARD"),
            ("s one forward", "FORWARD"),
            ("s 1 back", "BACK"),
            ("es1 left", "LEFT"),
            ("s1 forward left", "FORWARD + LEFT"),
            ("s1 turn right", "TURN RIGHT"),
        ):
            with self.subTest(phrase=phrase):
                intent = parse_voice_command(
                    phrase, wake_word="s1", require_wake_word=True
                )
                self.assertIsNotNone(intent)
                self.assertEqual(name, intent.name)

    def test_configured_wake_word_rejects_prose_and_missing_wake(self):
        for phrase in (
            "forward",  # no wake word
            "s1 dance",  # not a command
            "let's go forward",  # incidental "s" is not a command phrase
            "there is a chair beside the table",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(
                    parse_voice_command(
                        phrase, wake_word="s1", require_wake_word=True
                    )
                )
        # "stop" never needs the wake word.
        self.assertTrue(
            parse_voice_command(
                "stop", wake_word="s1", require_wake_word=True
            ).stop
        )

    def test_ambiguous_or_unrelated_phrases_are_rejected(self):
        for phrase in (
            "left right",
            "forward back",
            "turn left forward",
            "dance",
            "I can see a chair beside the table",
            "the weather is nice today",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_voice_command(phrase))


if __name__ == "__main__":
    unittest.main()
