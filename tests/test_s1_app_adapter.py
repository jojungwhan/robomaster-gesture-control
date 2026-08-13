import threading
import time
import unittest

from robomaster_gesture.models import VelocityCommand
from robomaster_gesture.robot_adapter import RobotError, S1AppKeyboardAdapter


class FakeKeyboardBackend:
    def __init__(self):
        self.window = 123
        self.input_access = True
        self.valid = True
        self.foreground = True
        self.activation_succeeds = True
        self.activation_calls = 0
        self.events = []
        self.lock = threading.Lock()

    def has_input_access(self, window):
        return self.input_access

    def find_window(self):
        return self.window

    def is_window(self, window):
        return self.valid and window == self.window

    def is_foreground(self, window):
        return self.foreground and self.is_window(window)

    def activate(self, window):
        self.activation_calls += 1
        if self.activation_succeeds:
            self.foreground = True
        return self.activation_succeeds and self.is_foreground(window)

    def press(self, window, key):
        with self.lock:
            self.events.append(("down", key))

    def release(self, window, key):
        with self.lock:
            self.events.append(("up", key))

    def snapshot(self):
        with self.lock:
            return list(self.events)


def wait_until(predicate, timeout_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class S1AppKeyboardAdapterTests(unittest.TestCase):
    def make_adapter(self, backend):
        return S1AppKeyboardAdapter(
            backend=backend,
            activation_threshold_m_s=0.08,
            watchdog_s=0.08,
        )

    def test_maps_translation_to_wasd_and_ignores_yaw(self):
        backend = FakeKeyboardBackend()
        adapter = self.make_adapter(backend)
        adapter.connect()
        try:
            adapter.drive(
                VelocityCommand(
                    forward_m_s=0.2,
                    right_m_s=-0.2,
                    clockwise_deg_s=30.0,
                ),
                timeout_s=0.08,
            )
            events = backend.snapshot()
            self.assertIn(("down", "w"), events)
            self.assertIn(("down", "a"), events)
            self.assertNotIn(("down", "d"), events)
            self.assertNotIn(("down", "s"), events)
        finally:
            adapter.close()

    def test_each_cardinal_command_maps_to_expected_key(self):
        cases = (
            ("forward", VelocityCommand(forward_m_s=0.2), "w"),
            ("back", VelocityCommand(forward_m_s=-0.2), "s"),
            ("left", VelocityCommand(right_m_s=-0.2), "a"),
            ("right", VelocityCommand(right_m_s=0.2), "d"),
        )
        for name, command, expected_key in cases:
            with self.subTest(direction=name):
                adapter = self.make_adapter(FakeKeyboardBackend())
                self.assertEqual({expected_key}, adapter._keys_for(command))

    def test_connect_activates_robomaster_window(self):
        backend = FakeKeyboardBackend()
        backend.foreground = False
        adapter = self.make_adapter(backend)
        adapter.connect()
        try:
            self.assertEqual(1, backend.activation_calls)
            self.assertTrue(backend.foreground)
        finally:
            adapter.close()

    def test_connect_rejects_failed_window_activation(self):
        backend = FakeKeyboardBackend()
        backend.foreground = False
        backend.activation_succeeds = False
        adapter = self.make_adapter(backend)
        with self.assertRaises(RobotError):
            adapter.connect()

    def test_below_threshold_and_yaw_only_stay_stopped(self):
        backend = FakeKeyboardBackend()
        adapter = self.make_adapter(backend)
        adapter.connect()
        try:
            baseline = len(backend.snapshot())
            adapter.drive(
                VelocityCommand(
                    forward_m_s=0.04,
                    right_m_s=-0.04,
                    clockwise_deg_s=30.0,
                )
            )
            self.assertFalse(
                any(kind == "down" for kind, _ in backend.snapshot()[baseline:])
            )
        finally:
            adapter.close()

    def test_watchdog_releases_movement_keys(self):
        backend = FakeKeyboardBackend()
        adapter = self.make_adapter(backend)
        adapter.connect()
        try:
            adapter.drive(VelocityCommand(forward_m_s=0.2), timeout_s=0.08)
            self.assertTrue(
                wait_until(lambda: ("up", "w") in backend.snapshot(), timeout_s=0.3)
            )
        finally:
            adapter.close()

    def test_focus_loss_latches_stop_until_zero_command(self):
        backend = FakeKeyboardBackend()
        adapter = self.make_adapter(backend)
        adapter.connect()
        try:
            backend.foreground = False
            with self.assertRaises(RobotError):
                adapter.drive(VelocityCommand(right_m_s=0.2), timeout_s=0.08)
            self.assertNotIn(("down", "d"), backend.snapshot())

            backend.foreground = True
            with self.assertRaises(RobotError):
                adapter.drive(VelocityCommand(right_m_s=0.2), timeout_s=0.08)
            self.assertNotIn(("down", "d"), backend.snapshot())

            adapter.drive(VelocityCommand.stopped())
            adapter.drive(VelocityCommand(right_m_s=0.2), timeout_s=0.08)
            self.assertIn(("down", "d"), backend.snapshot())
        finally:
            adapter.close()

    def test_privilege_mismatch_is_rejected_before_input(self):
        backend = FakeKeyboardBackend()
        backend.input_access = False
        adapter = self.make_adapter(backend)
        with self.assertRaises(RobotError):
            adapter.connect()
        self.assertEqual([], backend.snapshot())


if __name__ == "__main__":
    unittest.main()
