import unittest

from robomaster_gesture.expanded_scene import confirmed_expanded_detections


def detection(label, box, confidence=0.8):
    return {"label": label, "box": list(box), "confidence": confidence}


class ExpandedSceneConfirmationTests(unittest.TestCase):
    def test_confirms_same_label_at_overlapping_position(self):
        previous = (detection("desk", (100, 100, 500, 400)),)
        current = (detection("desk", (110, 105, 510, 405)),)
        self.assertEqual(current, confirmed_expanded_detections(previous, current))

    def test_rejects_one_scan_guess_or_changed_label(self):
        current = (detection("desk", (100, 100, 500, 400)),)
        self.assertEqual((), confirmed_expanded_detections((), current))
        previous = (detection("table", (100, 100, 500, 400)),)
        self.assertEqual((), confirmed_expanded_detections(previous, current))

    def test_rejects_same_label_that_moved_elsewhere(self):
        previous = (detection("chair", (0, 0, 100, 200)),)
        current = (detection("chair", (500, 0, 600, 200)),)
        self.assertEqual((), confirmed_expanded_detections(previous, current))

    def test_one_previous_box_confirms_at_most_one_current_box(self):
        previous = (detection("chair", (100, 100, 300, 400)),)
        current = (
            detection("chair", (105, 105, 295, 395), 0.9),
            detection("chair", (110, 110, 290, 390), 0.7),
        )
        self.assertEqual((current[0],), confirmed_expanded_detections(previous, current))


if __name__ == "__main__":
    unittest.main()
