import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robomaster_gesture.yolo_follow import (
    CameraFreshnessGuard,
    DEFAULT_EXPANDED_SCENE_MODEL,
    DEFAULT_PIPER_MODEL,
    Detection,
    FollowConfig,
    TargetFollower,
    _validate_args,
    build_parser,
    run,
)


def detection(label="bottle", box=(270, 140, 370, 300), confidence=0.9, track_id=4):
    return Detection(label, confidence, box, track_id)


class TargetFollowerTests(unittest.TestCase):
    def test_camera_check_rejects_live_or_unmounted_sources(self):
        parser = build_parser()
        for arguments in (
            ("--camera-check", "--live"),
            ("--camera-check", "--source", "webcam"),
            ("--camera-check", "--speak"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                _validate_args(parser.parse_args(arguments))

    def test_app_camera_check_never_starts_yolo_or_motion_adapter(self):
        events = []
        frames = []
        for index in range(4):
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            frame[20:60, 30:90] = index * 10
            frames.append(frame)

        class FakeFrameSource:
            def __init__(self, window_title):
                events.append("source-create")
                self.index = 0

            def open(self):
                events.append("source-open")

            def read(self):
                frame = frames[min(self.index, len(frames) - 1)]
                self.index += 1
                return frame

            def close(self):
                events.append("source-close")

        def forbidden(*args, **kwargs):
            raise AssertionError("YOLO or motion adapter started during camera check")

        args = build_parser().parse_args(("--camera-check",))
        with mock.patch(
            "robomaster_gesture.yolo_follow.RoboMasterAppFrameSource",
            FakeFrameSource,
        ), mock.patch(
            "robomaster_gesture.yolo_follow.UltralyticsTracker",
            forbidden,
        ), mock.patch(
            "robomaster_gesture.yolo_follow._make_motion_adapter",
            forbidden,
        ):
            self.assertEqual(0, run(args))

        self.assertEqual(
            ["source-create", "source-open", "source-close"],
            events,
        )

    def test_app_camera_check_fails_closed_on_static_frames(self):
        events = []
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        class StaticFrameSource:
            def __init__(self, window_title):
                pass

            def open(self):
                events.append("open")

            def read(self):
                return frame

            def close(self):
                events.append("close")

        def forbidden(*args, **kwargs):
            raise AssertionError("YOLO or motion adapter started during camera check")

        args = build_parser().parse_args(
            (
                "--camera-check",
                "--duration",
                "0.05",
                "--frame-interval",
                "0.02",
            )
        )
        with mock.patch(
            "robomaster_gesture.yolo_follow.RoboMasterAppFrameSource",
            StaticFrameSource,
        ), mock.patch(
            "robomaster_gesture.yolo_follow.UltralyticsTracker",
            forbidden,
        ), mock.patch(
            "robomaster_gesture.yolo_follow._make_motion_adapter",
            forbidden,
        ):
            with self.assertRaisesRegex(RuntimeError, "camera freshness"):
                run(args)

        self.assertEqual(["open", "close"], events)

    def test_camera_freshness_requires_consecutive_meaningful_changes(self):
        guard = CameraFreshnessGuard(minimum_changed_frames=3)
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        self.assertFalse(guard.update(frame, now_s=0.0).fresh)
        for index in range(1, 4):
            frame = frame.copy()
            frame[20:60, 30:90] = index * 10
            result = guard.update(frame, now_s=index * 0.1)
            self.assertEqual(index == 3, result.fresh)

    def test_camera_freshness_rejects_tiny_ui_change(self):
        guard = CameraFreshnessGuard(minimum_changed_frames=1)
        frame = np.zeros((800, 1200, 3), dtype=np.uint8)
        guard.update(frame, now_s=0.0)
        changed = frame.copy()
        changed[400, 600] = 255

        result = guard.update(changed, now_s=0.1)

        self.assertFalse(result.fresh)
        self.assertIn("verifying", result.reason)

    def test_camera_freshness_disarms_and_rearms_after_freeze(self):
        guard = CameraFreshnessGuard(
            minimum_changed_frames=2,
            freeze_after_s=0.5,
        )
        base = np.zeros((80, 120, 3), dtype=np.uint8)
        first = base.copy()
        first[20:60, 30:90] = 10
        second = base.copy()
        second[20:60, 30:90] = 20
        guard.update(base, now_s=0.0)
        guard.update(first, now_s=0.1)
        self.assertTrue(guard.update(second, now_s=0.2).fresh)
        self.assertTrue(guard.update(second, now_s=0.4).fresh)

        frozen = guard.update(second, now_s=0.8)
        self.assertFalse(frozen.fresh)
        self.assertIn("frozen", frozen.reason)

        third = base.copy()
        third[20:60, 30:90] = 30
        fourth = base.copy()
        fourth[20:60, 30:90] = 40
        self.assertFalse(guard.update(third, now_s=0.9).fresh)
        self.assertTrue(guard.update(fourth, now_s=1.0).fresh)

    def test_camera_freshness_discards_partial_lock_after_freeze(self):
        guard = CameraFreshnessGuard(
            minimum_changed_frames=3,
            freeze_after_s=0.5,
        )
        base = np.zeros((80, 120, 3), dtype=np.uint8)
        changed = base.copy()
        changed[20:60, 30:90] = 10
        guard.update(base, now_s=0.0)
        partial = guard.update(changed, now_s=0.1)
        self.assertEqual(1, partial.confirmation_frames)

        frozen = guard.update(changed, now_s=0.7)
        self.assertFalse(frozen.fresh)
        self.assertEqual(0, frozen.confirmation_frames)

        changed_again = base.copy()
        changed_again[20:60, 30:90] = 20
        reacquiring = guard.update(changed_again, now_s=0.8)
        self.assertFalse(reacquiring.fresh)
        self.assertEqual(1, reacquiring.confirmation_frames)

    def test_default_expanded_scene_model_is_compact_prompt_free_yoloe(self):
        self.assertEqual(
            "yoloe-26n-seg-pf.pt", DEFAULT_EXPANDED_SCENE_MODEL.name
        )

    def test_default_scene_voice_is_female_kristin(self):
        self.assertEqual("en_US-kristin-medium.onnx", DEFAULT_PIPER_MODEL.name)

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
