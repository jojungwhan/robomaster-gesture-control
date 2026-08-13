import unittest

from robomaster_gesture.leap_visualizer import Point3, project_xz


class LeapVisualizerProjectionTests(unittest.TestCase):
    def test_origin_maps_to_view_center(self):
        self.assertEqual((200, 150), project_xz(Point3(0, 220, 0), 200, 150, 1.0))

    def test_desktop_x_z_projection_matches_hand_motion(self):
        self.assertEqual(
            (250, 120), project_xz(Point3(50, 200, -30), 200, 150, 1.0)
        )

    def test_projection_scale_is_applied(self):
        self.assertEqual(
            (180, 190), project_xz(Point3(-10, 100, 20), 200, 150, 2.0)
        )


if __name__ == "__main__":
    unittest.main()
