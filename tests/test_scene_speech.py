import unittest

from robomaster_gesture.scene_speech import (
    SceneNarrationPolicy,
    describe_scene,
)
from robomaster_gesture.yolo_follow import Detection


class SceneDescriptionTests(unittest.TestCase):
    def test_describes_multiple_classes_and_screen_positions(self):
        scene = describe_scene(
            (
                Detection("person", 0.91, (0, 0, 100, 200)),
                Detection("person", 0.87, (20, 0, 120, 200)),
                Detection("bus", 0.90, (250, 0, 390, 300)),
                Detection("dog", 0.86, (520, 0, 620, 180)),
            ),
            frame_width=640,
        )
        self.assertIsNotNone(scene)
        self.assertEqual(
            "I see a bus ahead, 2 people on the left, and a dog on the right.",
            scene.text,
        )

    def test_filters_low_confidence_and_empty_scenes(self):
        scene = describe_scene(
            (Detection("bottle", 0.44, (250, 0, 350, 200)),),
            frame_width=640,
            confidence=0.45,
        )
        self.assertIsNone(scene)

    def test_groups_repeated_objects(self):
        scene = describe_scene(
            (
                Detection("bottle", 0.9, (260, 0, 300, 100)),
                Detection("bottle", 0.8, (320, 0, 360, 100)),
            ),
            frame_width=640,
        )
        self.assertEqual("I see 2 bottles ahead.", scene.text)

    def test_natural_irregular_and_multiword_plurals(self):
        scene = describe_scene(
            (
                Detection("knife", 0.9, (250, 0, 290, 100)),
                Detection("knife", 0.8, (300, 0, 340, 100)),
                Detection("traffic light", 0.9, (520, 0, 560, 100)),
                Detection("traffic light", 0.8, (570, 0, 610, 100)),
                Detection("scissors", 0.9, (20, 0, 80, 100)),
            ),
            frame_width=640,
        )
        self.assertEqual(
            "I see 2 knives ahead, a pair of scissors on the left, "
            "and 2 traffic lights on the right.",
            scene.text,
        )


class SceneNarrationPolicyTests(unittest.TestCase):
    def test_requires_stable_frames_then_suppresses_repetition(self):
        scene = describe_scene(
            (Detection("bottle", 0.9, (280, 0, 360, 100)),), 640
        )
        policy = SceneNarrationPolicy(stable_frames=3, repeat_seconds=10)
        self.assertIsNone(policy.update(scene, 0.0))
        self.assertIsNone(policy.update(scene, 0.1))
        self.assertEqual(scene.text, policy.update(scene, 0.2))
        self.assertIsNone(policy.update(scene, 5.0))
        self.assertEqual(scene.text, policy.update(scene, 10.2))

    def test_changed_scene_is_announced_after_stabilizing(self):
        left = describe_scene((Detection("dog", 0.9, (0, 0, 100, 100)),), 640)
        right = describe_scene((Detection("dog", 0.9, (540, 0, 640, 100)),), 640)
        policy = SceneNarrationPolicy(stable_frames=2, repeat_seconds=30)
        policy.update(left, 0.0)
        self.assertEqual(left.text, policy.update(left, 0.1))
        self.assertIsNone(policy.update(right, 0.2))
        self.assertEqual(right.text, policy.update(right, 0.3))

    def test_clear_scene_is_announced_once(self):
        scene = describe_scene((Detection("chair", 0.9, (250, 0, 390, 200)),), 640)
        policy = SceneNarrationPolicy(
            stable_frames=1, repeat_seconds=30, clear_seconds=2
        )
        policy.update(scene, 0.0)
        self.assertIsNone(policy.update(None, 1.0))
        self.assertEqual(
            "I do not see any recognized objects now.", policy.update(None, 3.0)
        )
        self.assertIsNone(policy.update(None, 4.0))


if __name__ == "__main__":
    unittest.main()
