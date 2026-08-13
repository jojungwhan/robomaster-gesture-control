import unittest

from robomaster_gesture.voice_control import parse_voice_command


class VoiceCommandTests(unittest.TestCase):
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
