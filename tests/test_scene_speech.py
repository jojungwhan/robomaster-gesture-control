import unittest

from robomaster_gesture.scene_speech import (
    SceneNarrationPolicy,
    answer_object_query,
    describe_scene,
    is_look_query,
    merge_scene_detections,
    parse_object_query,
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

    def test_normalizes_furniture_names_and_plurals(self):
        scene = describe_scene(
            (
                Detection("office chair", 0.9, (0, 0, 100, 100)),
                Detection("office chair", 0.8, (110, 0, 200, 100)),
                Detection("bookshelf", 0.9, (250, 0, 390, 200)),
                Detection("dining table", 0.9, (520, 0, 620, 200)),
            ),
            frame_width=640,
        )
        self.assertEqual(
            "I see a bookshelf ahead, 2 chairs on the left, and a table on the right.",
            scene.text,
        )

    def test_expanded_desk_replaces_overlapping_primary_table(self):
        primary = (Detection("dining table", 0.8, (100, 100, 500, 400)),)
        expanded = (
            Detection("desk", 0.7, (110, 110, 490, 390)),
            Detection("office chair", 0.8, (10, 100, 90, 300)),
        )
        merged = merge_scene_detections(primary, expanded)
        self.assertEqual(["desk", "office chair"], [item.label for item in merged])

    def test_expanded_alias_does_not_duplicate_primary_chair(self):
        primary = (Detection("chair", 0.8, (100, 100, 300, 400)),)
        expanded = (Detection("office chair", 0.9, (105, 105, 295, 395)),)
        merged = merge_scene_detections(primary, expanded)
        self.assertEqual(primary, merged)


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


class ObjectQueryParsingTests(unittest.TestCase):
    def test_targeted_questions_return_the_canonical_label(self):
        cases = {
            "do you see a person": "person",
            "Do you see any people?": "person",
            "can you see a chair": "chair",
            "is there a dog": "dog",
            "do you see any bottles": "bottle",
            "can you spot a laptop": "laptop",
            "do you see my phone": "cell phone",
            "look for a cup": "cup",
        }
        for phrase, label in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(label, parse_object_query(phrase))

    def test_synonyms_and_furniture_aliases_map_to_reportable_labels(self):
        self.assertEqual("couch", parse_object_query("do you see a sofa"))
        self.assertEqual("television", parse_object_query("do you see a tv"))
        self.assertEqual("table", parse_object_query("do you see a table"))
        self.assertEqual("laptop", parse_object_query("do you see a computer"))

    def test_general_and_movement_phrases_are_not_object_queries(self):
        for phrase in (
            "what do you see",
            "tell me what you see",
            "do you see anything",
            "forward",
            "turn left",
            "stop",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_object_query(phrase))

    def test_general_look_queries_cover_more_phrasings(self):
        for phrase in (
            "do you see anything",
            "can you see",
            "describe the room",
            "what are you looking at",
            "what's around you",
            "describe your surroundings",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_look_query(phrase))


class ObjectQueryAnswerTests(unittest.TestCase):
    def setUp(self):
        self.detections = (
            Detection("person", 0.92, (300, 100, 400, 500)),
            Detection("chair", 0.81, (40, 200, 150, 480)),
            Detection("tv", 0.85, (600, 100, 760, 300)),
        )

    def test_present_object_is_confirmed_with_location(self):
        self.assertEqual(
            "Yes, I see a person ahead.",
            answer_object_query("person", self.detections, frame_width=800),
        )

    def test_alias_label_matches_detection_alias(self):
        # "tv" detections canonicalize to "television".
        self.assertEqual(
            "Yes, I see a television on the right.",
            answer_object_query("television", self.detections, frame_width=800),
        )

    def test_absent_object_is_denied(self):
        self.assertEqual(
            "No, I do not see a dog right now.",
            answer_object_query("dog", self.detections, frame_width=800),
        )
        self.assertEqual(
            "No, I do not see an elephant right now.",
            answer_object_query("elephant", self.detections, frame_width=800),
        )

    def test_multiple_matches_are_counted(self):
        detections = (
            Detection("person", 0.9, (40, 0, 140, 400)),
            Detection("person", 0.9, (300, 0, 400, 400)),
        )
        self.assertEqual(
            "Yes, I see a person ahead and a person on the left.",
            answer_object_query("person", detections, frame_width=600),
        )


if __name__ == "__main__":
    unittest.main()
