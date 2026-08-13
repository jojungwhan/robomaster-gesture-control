import unittest

from robomaster_gesture.control_lease import ControlLeaseError, ControllerLease


class ControllerLeaseTests(unittest.TestCase):
    def test_only_one_controller_can_hold_a_named_lease(self):
        first = ControllerLease("robomaster_gesture_test_lease")
        second = ControllerLease("robomaster_gesture_test_lease")
        try:
            first.acquire()
            with self.assertRaises(ControlLeaseError):
                second.acquire()
            first.close()
            second.acquire()
        finally:
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
