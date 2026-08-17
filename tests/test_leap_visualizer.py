import unittest

from robomaster_gesture.control_center import MIC_TEST_MODE, VOICE_MODE
from robomaster_gesture.leap_visualizer import (
    Point3,
    build_diagnostics_text,
    microphone_meter_state,
    project_xz,
    transcription_display_state,
)


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


class MicrophoneMeterStateTests(unittest.TestCase):
    def test_meter_is_off_outside_voice_modes(self):
        state = microphone_meter_state(None, "OFF", None, 0.0, 50.0)
        self.assertEqual(
            ("MIC OFF", 0, "muted"),
            (state.label, state.level, state.tone),
        )

    def test_meter_covers_starting_waiting_and_stale_states(self):
        starting = microphone_meter_state(
            VOICE_MODE, "STARTING", None, 0.0, 50.0
        )
        waiting = microphone_meter_state(
            MIC_TEST_MODE, "RUNNING", None, 0.0, 50.0
        )
        stale = microphone_meter_state(
            VOICE_MODE, "RUNNING", 42, 45.0, 50.0
        )
        self.assertEqual("MIC STARTING", starting.label)
        self.assertEqual("MIC WAITING", waiting.label)
        self.assertEqual("MIC WAITING", stale.label)

    def test_zero_input_is_distinct_from_a_working_signal(self):
        silent = microphone_meter_state(
            VOICE_MODE, "RUNNING", 0, 49.9, 50.0
        )
        input_detected = microphone_meter_state(
            MIC_TEST_MODE, "RUNNING", 43, 49.9, 50.0
        )
        loud = microphone_meter_state(
            VOICE_MODE, "RUNNING", 91, 49.9, 50.0
        )
        self.assertEqual(
            ("NO SIGNAL  0%", 0, "warn"),
            (silent.label, silent.level, silent.tone),
        )
        self.assertEqual(
            ("INPUT  43%", 43, "good"),
            (input_detected.label, input_detected.level, input_detected.tone),
        )
        self.assertEqual(
            ("LOUD  91%", 91, "warn"),
            (loud.label, loud.level, loud.tone),
        )


class TranscriptionDisplayStateTests(unittest.TestCase):
    def test_active_voice_mode_shows_listening_then_recognized_phrase(self):
        listening = transcription_display_state(
            VOICE_MODE,
            "RUNNING",
            "",
            0.0,
        )
        recognized = transcription_display_state(
            MIC_TEST_MODE,
            "RUNNING",
            "robot forward",
            0.98,
        )

        self.assertEqual(
            "TRANSCRIPT  Listening for a command...",
            listening.label,
        )
        self.assertEqual(
            'TRANSCRIPT  "robot forward"  98% confidence',
            recognized.label,
        )
        self.assertEqual("good", recognized.tone)

    def test_last_phrase_remains_visible_after_listener_stops(self):
        state = transcription_display_state(
            None,
            "OFF",
            "robot back",
            0.91,
        )

        self.assertEqual(
            'LAST HEARD  "robot back"  91% confidence',
            state.label,
        )

    def test_empty_and_error_states_are_explicit(self):
        empty = transcription_display_state(None, "OFF", "", 0.0)
        error = transcription_display_state(VOICE_MODE, "ERROR", "", 0.0)

        self.assertEqual("TRANSCRIPT  No speech recognized yet", empty.label)
        self.assertEqual("TRANSCRIPT  Recognition error", error.label)
        self.assertEqual("error", error.tone)


class DiagnosticsTextTests(unittest.TestCase):
    def test_copyable_block_includes_every_readout(self):
        block = build_diagnostics_text(
            tracking="TRACKING  60 FPS",
            detail="RIGHT  palm x=+10 y=+220 z=-5 mm  id=7",
            command="DRY RUN  FORWARD  [DRIVING]",
            transcript='TRANSCRIPT  "robot forward"  98% confidence',
            status="LEAP RUNNING  |  Controls: pinch to drive",
        )

        self.assertIn("Tracking:   TRACKING  60 FPS", block)
        self.assertIn("Command:    DRY RUN  FORWARD  [DRIVING]", block)
        self.assertIn('Transcript: TRANSCRIPT  "robot forward"  98% confidence', block)
        self.assertIn("LEAP RUNNING  |  Controls: pinch to drive", block)
        # The status can be multi-line, so it is placed last under its heading.
        self.assertTrue(block.strip().endswith("Controls: pinch to drive"))


if __name__ == "__main__":
    unittest.main()
