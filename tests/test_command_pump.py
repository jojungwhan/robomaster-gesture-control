import threading
import time
import unittest

from robomaster_gesture.models import VelocityCommand
from robomaster_gesture.robot_adapter import (
    CommandPump,
    CommandPumpConfig,
    RobotAdapter,
)


class RecordingRobot(RobotAdapter):
    def __init__(self):
        self.commands = []
        self.lock = threading.Lock()

    def connect(self):
        pass

    def drive(self, command, timeout_s=None):
        with self.lock:
            self.commands.append((time.monotonic(), command))

    def close(self):
        pass

    def snapshot(self):
        with self.lock:
            return list(self.commands)


def wait_until(predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class CommandPumpTests(unittest.TestCase):
    def make_pump(self, robot):
        return CommandPump(
            robot,
            CommandPumpConfig(
                rate_hz=100.0,
                stale_after_s=0.08,
                moving_keepalive_s=0.03,
                robot_timeout_s=0.10,
            ),
        )

    def test_stale_input_automatically_sends_stop(self):
        robot = RecordingRobot()
        pump = self.make_pump(robot)
        pump.start()
        try:
            pump.submit(VelocityCommand(forward_m_s=0.2))
            self.assertTrue(
                wait_until(
                    lambda: any(
                        not command.is_stopped() for _, command in robot.snapshot()
                    )
                )
            )
            moving_index = max(
                index
                for index, (_, command) in enumerate(robot.snapshot())
                if not command.is_stopped()
            )
            self.assertTrue(
                wait_until(
                    lambda: any(
                        index > moving_index and command.is_stopped()
                        for index, (_, command) in enumerate(robot.snapshot())
                    )
                )
            )
        finally:
            pump.close()

    def test_halt_sends_stop_after_motion(self):
        robot = RecordingRobot()
        pump = self.make_pump(robot)
        pump.start()
        try:
            pump.submit(VelocityCommand(right_m_s=0.2))
            self.assertTrue(
                wait_until(
                    lambda: any(
                        not command.is_stopped() for _, command in robot.snapshot()
                    )
                )
            )
            moving_index = max(
                index
                for index, (_, command) in enumerate(robot.snapshot())
                if not command.is_stopped()
            )
            pump.halt()
            self.assertTrue(
                wait_until(
                    lambda: any(
                        index > moving_index and command.is_stopped()
                        for index, (_, command) in enumerate(robot.snapshot())
                    )
                )
            )
        finally:
            pump.close()


if __name__ == "__main__":
    unittest.main()
