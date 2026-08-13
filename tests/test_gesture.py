import math
import unittest

from robomaster_gesture.gesture import GestureConfig, GestureController
from robomaster_gesture.models import FrameSample, HandSample


def make_hand(
    hand_id=7,
    is_left=False,
    x=0.0,
    z=-100.0,
    yaw=0.0,
    pinch=0.05,
    grab=0.05,
    visible=1.0,
):
    radians = math.radians(yaw)
    return HandSample(
        hand_id=hand_id,
        is_left=is_left,
        visible_time_s=visible,
        palm_x_mm=x,
        palm_y_mm=180.0,
        palm_z_mm=z,
        velocity_x_mm_s=0.0,
        velocity_y_mm_s=0.0,
        velocity_z_mm_s=0.0,
        direction_x=math.sin(radians),
        direction_y=0.0,
        direction_z=-math.cos(radians),
        normal_x=0.0,
        normal_y=-1.0,
        normal_z=0.0,
        pinch_strength=pinch,
        grab_strength=grab,
        pinch_distance_mm=40.0,
    )


def make_frame(at, *hands):
    return FrameSample(
        arrival_time_s=at,
        frame_id=int(at * 1000),
        sensor_timestamp_us=int(at * 1_000_000),
        framerate=60.0,
        total_hand_count=len(hands),
        hands=tuple(hands),
    )


def arm(controller, start=0.0, hand_id=7):
    controller.update(make_frame(start, make_hand(hand_id=hand_id)))
    ready = controller.update(make_frame(start + 0.40, make_hand(hand_id=hand_id)))
    assert ready.state == "READY"
    controller.update(
        make_frame(start + 0.50, make_hand(hand_id=hand_id, pinch=0.95))
    )
    controller.update(
        make_frame(start + 0.70, make_hand(hand_id=hand_id, pinch=0.95))
    )
    engaged = controller.update(
        make_frame(start + 0.90, make_hand(hand_id=hand_id, pinch=0.95))
    )
    assert engaged.state == "DRIVING"


class GestureControllerTests(unittest.TestCase):
    def test_requires_open_hand_before_pinch(self):
        controller = GestureController()
        result = controller.update(make_frame(0.0, make_hand(pinch=0.95)))
        self.assertEqual("WAITING", result.state)
        self.assertTrue(result.command.is_stopped())

    def test_open_then_held_pinch_arms(self):
        controller = GestureController()
        arm(controller)
        self.assertEqual("DRIVING", controller.state.value)

    def test_ready_allows_natural_intermediate_pinch_transition(self):
        controller = GestureController()
        controller.update(make_frame(0.0, make_hand()))
        ready = controller.update(make_frame(0.40, make_hand()))
        self.assertEqual("READY", ready.state)

        transitioning = controller.update(make_frame(0.45, make_hand(pinch=0.60)))
        self.assertEqual("READY", transitioning.state)
        self.assertTrue(transitioning.command.is_stopped())

        arming = controller.update(make_frame(0.50, make_hand(pinch=0.95)))
        self.assertEqual("ARMING", arming.state)

    def test_forward_and_strafe_mapping(self):
        controller = GestureController(GestureConfig(smoothing_time_s=0.01))
        arm(controller)
        result = controller.update(
            make_frame(1.05, make_hand(x=120.0, z=-220.0, pinch=0.95))
        )
        self.assertGreater(result.command.forward_m_s, 0.30)
        self.assertGreater(result.command.right_m_s, 0.30)

    def test_cardinal_hand_motion_maps_to_robot_directions(self):
        cases = (
            ("away from user", 0.0, -220.0, "forward_m_s", 1),
            ("toward user", 0.0, 20.0, "forward_m_s", -1),
            ("left", -120.0, -100.0, "right_m_s", -1),
            ("right", 120.0, -100.0, "right_m_s", 1),
        )
        for name, x, z, axis, expected_sign in cases:
            with self.subTest(hand_motion=name):
                controller = GestureController(
                    GestureConfig(smoothing_time_s=0.01)
                )
                arm(controller)
                result = controller.update(
                    make_frame(1.05, make_hand(x=x, z=z, pinch=0.95))
                )
                value = getattr(result.command, axis)
                self.assertGreater(value * expected_sign, 0.30)
                other_axis = (
                    result.command.right_m_s
                    if axis == "forward_m_s"
                    else result.command.forward_m_s
                )
                self.assertAlmostEqual(0.0, other_axis)

    def test_wrist_yaw_maps_to_clockwise_rotation(self):
        controller = GestureController(GestureConfig(smoothing_time_s=0.01))
        arm(controller)
        result = controller.update(
            make_frame(1.05, make_hand(yaw=45.0, pinch=0.95))
        )
        self.assertGreater(result.command.clockwise_deg_s, 30.0)

    def test_deadzone_produces_zero(self):
        controller = GestureController(GestureConfig(smoothing_time_s=0.01))
        arm(controller)
        result = controller.update(
            make_frame(1.05, make_hand(x=20.0, z=-120.0, yaw=8.0, pinch=0.95))
        )
        self.assertTrue(result.command.is_stopped())

    def test_release_stops_immediately(self):
        controller = GestureController(GestureConfig(smoothing_time_s=0.01))
        arm(controller)
        moving = controller.update(
            make_frame(1.05, make_hand(z=-220.0, pinch=0.95))
        )
        self.assertFalse(moving.command.is_stopped())
        stopped = controller.update(make_frame(1.10, make_hand(z=-220.0, pinch=0.20)))
        self.assertEqual("WAITING", stopped.state)
        self.assertTrue(stopped.command.is_stopped())

    def test_fist_stops_immediately(self):
        controller = GestureController(GestureConfig(smoothing_time_s=0.01))
        arm(controller)
        stopped = controller.update(
            make_frame(1.05, make_hand(z=-220.0, pinch=0.95, grab=0.95))
        )
        self.assertIn("fist", stopped.reason)
        self.assertTrue(stopped.command.is_stopped())

    def test_hand_loss_and_multiple_hands_stop(self):
        controller = GestureController()
        arm(controller)
        lost = controller.update(make_frame(1.05))
        self.assertTrue(lost.command.is_stopped())

        second = make_hand(hand_id=8, is_left=True)
        multiple = controller.update(make_frame(1.10, make_hand(), second))
        self.assertIn("multiple", multiple.reason)
        self.assertTrue(multiple.command.is_stopped())

    def test_hand_identity_change_stops(self):
        controller = GestureController()
        arm(controller, hand_id=7)
        result = controller.update(
            make_frame(1.05, make_hand(hand_id=99, pinch=0.95))
        )
        self.assertIn("identity", result.reason)
        self.assertTrue(result.command.is_stopped())

    def test_tracking_timeout_stops(self):
        controller = GestureController()
        arm(controller)
        result = controller.on_tracking_timeout(1.20)
        self.assertIn("timeout", result.reason)
        self.assertTrue(result.command.is_stopped())

    def test_speed_is_bounded(self):
        controller = GestureController(GestureConfig(smoothing_time_s=0.001))
        arm(controller)
        result = controller.update(
            make_frame(1.05, make_hand(x=1000.0, z=-1100.0, yaw=120.0, pinch=0.95))
        )
        self.assertLessEqual(abs(result.command.forward_m_s), 0.35)
        self.assertLessEqual(abs(result.command.right_m_s), 0.35)
        self.assertLessEqual(abs(result.command.clockwise_deg_s), 35.0)


if __name__ == "__main__":
    unittest.main()
