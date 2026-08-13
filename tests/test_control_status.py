import tempfile
from pathlib import Path
import unittest

from robomaster_gesture.control_status import (
    ControlStatusPublisher,
    ControlStatusReader,
)
from robomaster_gesture.models import GestureDecision, VelocityCommand


class ControlStatusTests(unittest.TestCase):
    def test_published_robot_direction_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            publisher = ControlStatusPublisher(
                path=path, live=True, transport="s1-app", interval_s=0.0
            )
            publisher.publish(
                GestureDecision(
                    state="DRIVING",
                    reason="pinch held",
                    command=VelocityCommand(
                        forward_m_s=-0.2,
                        right_m_s=0.2,
                    ),
                )
            )

            snapshot = ControlStatusReader(path).latest()
            self.assertIsNotNone(snapshot)
            self.assertTrue(snapshot.live)
            self.assertEqual("s1-app", snapshot.transport)
            self.assertEqual("DRIVING", snapshot.state)
            self.assertAlmostEqual(-0.2, snapshot.command.forward_m_s)
            self.assertAlmostEqual(0.2, snapshot.command.right_m_s)

    def test_status_write_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as directory:
            occupied = Path(directory) / "not-a-directory"
            occupied.write_text("occupied", encoding="utf-8")
            publisher = ControlStatusPublisher(
                path=occupied / "status.json", interval_s=0.0
            )
            published = publisher.publish(
                GestureDecision(
                    state="DRIVING",
                    reason="pinch held",
                    command=VelocityCommand(forward_m_s=0.2),
                )
            )
            self.assertFalse(published)

    def test_reader_accepts_windows_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text(
                '{"updated_at_epoch_s":1,"process_id":2,"live":false,'
                '"transport":"sdk","state":"WAITING","reason":"stopped",'
                '"command":{"forward_m_s":0,"right_m_s":0,'
                '"clockwise_deg_s":0}}',
                encoding="utf-8-sig",
            )
            snapshot = ControlStatusReader(path).latest()
            self.assertIsNotNone(snapshot)
            self.assertEqual("WAITING", snapshot.state)


if __name__ == "__main__":
    unittest.main()
