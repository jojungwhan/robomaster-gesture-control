import unittest
from pathlib import Path

from robomaster_gesture.yolo_follow import (
    Detection,
    FollowConfig,
    TargetFollower,
    _validate_args,
    build_parser,
)


def detection(label="bottle", box=(270, 140, 370, 300), confidence=0.9, track_id=4):
    return Detection(label, confidence, box, track_id)


class TargetFollowerTests(unittest.TestCase):
    def make_follower(self, **kwargs):
        return TargetFollower(
            FollowConfig(
                target_label="bottle",
                minimum_lock_frames=kwargs.pop("minimum_lock_frames", 3),
                **kwargs
            )
        )

    def lock(self, follower, item=None):
        item = item or detection()
        result = None
        for _ in range(follower.config.minimum_lock_frames):
            result = follower.update((item,), 640, 480)
        return result

    def test_requires_fresh_consecutive_target_frames(self):
        follower = self.make_follower(minimum_lock_frames=3)
        first = follower.update((detection(),), 640, 480)
        second = follower.update((detection(),), 640, 480)
        third = follower.update((detection(),), 640, 480)
        self.assertEqual("ACQUIRING", first.state)
        self.assertEqual("ACQUIRING", second.state)
        self.assertTrue(first.command.is_stopped())
        self.assertEqual("FOLLOWING", third.state)

    def test_centered_distant_target_moves_forward(self):
        follower = self.make_follower()
        result = self.lock(follower)
        self.assertGreater(result.command.forward_m_s, 0.0)
        self.assertEqual(0.0, result.command.right_m_s)

    def test_target_on_left_or_right_maps_to_strafe(self):
        cases = (
            ((20, 140, 120, 300), -1),
            ((520, 140, 620, 300), 1),
        )
        for box, sign in cases:
            with self.subTest(box=box):
                follower = self.make_follower()
                result = self.lock(follower, detection(box=box))
                self.assertEqual(0.0, result.command.forward_m_s)
                self.assertGreater(result.command.right_m_s * sign, 0.0)

    def test_target_at_stop_size_holds_and_uses_hysteresis(self):
        follower = self.make_follower(minimum_lock_frames=1)
        near = follower.update((detection(box=(200, 50, 440, 250)),), 640, 480)
        still_near = follower.update(
            (detection(box=(205, 70, 435, 240)),), 640, 480
        )
        resumed = follower.update(
            (detection(box=(220, 120, 420, 260)),), 640, 480
        )
        self.assertEqual("HOLDING", near.state)
        self.assertTrue(still_near.command.is_stopped())
        self.assertGreater(resumed.command.forward_m_s, 0.0)

    def test_target_loss_stops_immediately_and_requires_relock(self):
        follower = self.make_follower()
        self.lock(follower)
        lost = follower.update((), 640, 480)
        reacquiring = follower.update((detection(),), 640, 480)
        self.assertTrue(lost.command.is_stopped())
        self.assertEqual("ACQUIRING", reacquiring.state)

    def test_target_below_target_confidence_is_treated_as_lost(self):
        follower = self.make_follower(target_confidence=0.70)
        result = follower.update((detection(confidence=0.69),), 640, 480)
        self.assertTrue(result.command.is_stopped())
        self.assertIn("target lost", result.reason)

    def test_person_detection_protectively_stops_non_person_target(self):
        follower = self.make_follower()
        self.lock(follower)
        result = follower.update(
            (
                detection(),
                detection("person", (10, 10, 200, 460), 0.31, 8),
            ),
            640,
            480,
        )
        self.assertTrue(result.command.is_stopped())
        self.assertIn("person", result.reason)

    def test_locked_track_id_does_not_silently_switch_objects(self):
        follower = self.make_follower()
        self.lock(follower, detection(track_id=4))
        switched = follower.update((detection(track_id=9),), 640, 480)
        self.assertTrue(switched.command.is_stopped())
        self.assertIn("locked target lost", switched.reason)

    def test_live_mode_rejects_unmounted_file_or_webcam_sources(self):
        parser = build_parser()
        for source in ("file", "webcam"):
            arguments = ["--live", "--source", source]
            if source == "file":
                arguments.extend(("--input-file", str(Path(__file__))))
            with self.subTest(source=source), self.assertRaises(ValueError):
                _validate_args(parser.parse_args(arguments))

    def test_live_app_source_requires_s1_app_transport(self):
        args = build_parser().parse_args(
            ("--live", "--source", "robomaster-app", "--transport", "sdk")
        )
        with self.assertRaises(ValueError):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
