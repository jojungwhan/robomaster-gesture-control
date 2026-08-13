import unittest

from robomaster_gesture.models import VelocityCommand, translation_directions
from robomaster_gesture.robot_adapter import DjiRobotAdapter


class FakeChassis:
    def __init__(self):
        self.calls = []

    def drive_speed(self, **kwargs):
        self.calls.append(kwargs)
        return True


class DirectionMappingTests(unittest.TestCase):
    def test_direction_labels_cover_four_cardinal_commands(self):
        cases = (
            (VelocityCommand(forward_m_s=0.2), ("FORWARD",)),
            (VelocityCommand(forward_m_s=-0.2), ("BACK",)),
            (VelocityCommand(right_m_s=-0.2), ("LEFT",)),
            (VelocityCommand(right_m_s=0.2), ("RIGHT",)),
            (
                VelocityCommand(forward_m_s=0.2, right_m_s=-0.2),
                ("FORWARD", "LEFT"),
            ),
            (VelocityCommand.stopped(), ()),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(expected, translation_directions(command))

    def test_dji_adapter_preserves_forward_and_right_axis_signs(self):
        cases = (
            ("forward", VelocityCommand(forward_m_s=0.2), (0.2, 0.0)),
            ("back", VelocityCommand(forward_m_s=-0.2), (-0.2, 0.0)),
            ("left", VelocityCommand(right_m_s=-0.2), (0.0, -0.2)),
            ("right", VelocityCommand(right_m_s=0.2), (0.0, 0.2)),
        )
        for name, command, expected in cases:
            with self.subTest(direction=name):
                chassis = FakeChassis()
                adapter = DjiRobotAdapter(conn_type="sta", proto_type="tcp")
                adapter._chassis = chassis
                adapter.drive(command, timeout_s=0.35)
                call = chassis.calls[-1]
                self.assertEqual(expected, (call["x"], call["y"]))
                self.assertEqual(0.0, call["z"])


if __name__ == "__main__":
    unittest.main()
