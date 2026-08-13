from contextlib import ExitStack
import unittest
from unittest import mock

from robomaster_gesture.voice_control import (
    SpeechEvent,
    build_parser,
    parse_voice_command,
    run,
)


class VoiceCommandTests(unittest.TestCase):
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
            ("robot forward", "FORWARD", 1, 0),
            ("robot move backward", "BACK", -1, 0),
            ("robot left", "LEFT", 0, -1),
            ("robot go right", "RIGHT", 0, 1),
            ("robot forward left", "FORWARD + LEFT", 1, -1),
            ("robot back right", "BACK + RIGHT", -1, 1),
        )
        for phrase, name, forward_sign, right_sign in cases:
            with self.subTest(phrase=phrase):
                intent = parse_voice_command(phrase, speed_m_s=0.2)
                self.assertIsNotNone(intent)
                self.assertEqual(name, intent.name)
                self.assertEqual(forward_sign, (intent.command.forward_m_s > 0) - (intent.command.forward_m_s < 0))
                self.assertEqual(right_sign, (intent.command.right_m_s > 0) - (intent.command.right_m_s < 0))

    def test_turn_commands_have_clockwise_sign(self):
        left = parse_voice_command("robot turn left", yaw_deg_s=25)
        right = parse_voice_command("robot rotate right", yaw_deg_s=25)
        self.assertLess(left.command.clockwise_deg_s, 0)
        self.assertGreater(right.command.clockwise_deg_s, 0)

    def test_stop_never_requires_wake_word(self):
        for phrase in ("stop", "halt", "emergency stop", "robot stop"):
            with self.subTest(phrase=phrase):
                intent = parse_voice_command(phrase)
                self.assertTrue(intent.stop)
                self.assertTrue(intent.command.is_stopped())

    def test_motion_requires_wake_word_by_default(self):
        self.assertIsNone(parse_voice_command("forward"))
        self.assertIsNotNone(
            parse_voice_command("forward", require_wake_word=False)
        )

    def test_ambiguous_or_unrelated_phrases_are_rejected(self):
        for phrase in (
            "robot left right",
            "robot forward back",
            "robot turn left forward",
            "robot dance",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_voice_command(phrase))


if __name__ == "__main__":
    unittest.main()
