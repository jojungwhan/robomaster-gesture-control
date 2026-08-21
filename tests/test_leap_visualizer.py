import time
import unittest
from types import SimpleNamespace

from robomaster_gesture.control_center import LEAP_MODE, MIC_TEST_MODE, VOICE_MODE
from robomaster_gesture.control_status import ControlStatusSnapshot
from robomaster_gesture.leap_visualizer import (
    ControlCenterWindow,
    Point3,
    build_diagnostics_text,
    microphone_meter_state,
    project_xz,
    transcription_display_state,
)
from robomaster_gesture.models import VelocityCommand


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


class CommandReadoutTests(unittest.TestCase):
    """The always-visible "COMMAND -> S1" readout, tested without a Tk window."""

    def _readout(self, *, forward=0.0, right=0.0, yaw=0.0, live=True, fresh=True):
        window = ControlCenterWindow.__new__(ControlCenterWindow)
        control = ControlStatusSnapshot(
            updated_at_epoch_s=0.0,
            process_id=1,
            live=live,
            transport="s1-app",
            state="DRIVING" if (forward or right or yaw) else "READY",
            reason="",
            command=VelocityCommand(
                forward_m_s=forward, right_m_s=right, clockwise_deg_s=yaw
            ),
        )
        return window._command_readout(control, fresh)

    def test_no_fresh_control_reads_off(self):
        motion, badge, motion_color, badge_color = self._readout(fresh=False)
        self.assertIn("no command", motion)
        self.assertEqual("CONTROL OFF", badge)
        self.assertEqual(ControlCenterWindow.MUTED, motion_color)
        self.assertEqual(ControlCenterWindow.MUTED, badge_color)

    def test_live_movement_is_marked_sent_with_direction_and_speed(self):
        motion, badge, motion_color, _ = self._readout(forward=0.25, live=True)
        self.assertIn("FORWARD", motion)
        self.assertIn("0.25 m/s", motion)
        self.assertEqual("● SENT TO S1", badge)
        # Amber while the robot is actually moving.
        self.assertEqual(ControlCenterWindow.WARN, motion_color)

    def test_live_stop_is_sent_but_calm_green(self):
        motion, badge, motion_color, _ = self._readout(forward=0.0, live=True)
        self.assertEqual("■ STOP", motion)
        self.assertEqual("● SENT TO S1", badge)
        self.assertEqual(ControlCenterWindow.GOOD, motion_color)

    def test_dry_run_movement_is_marked_not_sent(self):
        motion, badge, _motion_color, badge_color = self._readout(
            forward=0.2, right=-0.2, live=False
        )
        self.assertIn("FORWARD", motion)
        self.assertIn("LEFT", motion)
        self.assertEqual("○ DRY RUN — NOT SENT", badge)
        self.assertEqual(ControlCenterWindow.RIGHT, badge_color)


class ControlOwnershipTests(unittest.TestCase):
    """A venv launcher can relaunch the base interpreter, so the worker that
    publishes may differ from the PID the Control Center tracked; either PID
    must still count as the Control Center's own controller."""

    def _snap(self, pid, ppid):
        return ControlStatusSnapshot(
            updated_at_epoch_s=0.0,
            process_id=pid,
            parent_process_id=ppid,
            live=False,
            transport="s1-app",
            state="READY",
            reason="",
            command=VelocityCommand.stopped(),
        )

    def test_child_worker_matches_via_parent_pid(self):
        managed = SimpleNamespace(process_id=100)  # tracked launcher PID
        # Worker relaunched as pid 200 with parent 100.
        self.assertTrue(
            ControlCenterWindow._control_matches_managed(self._snap(200, 100), managed)
        )

    def test_direct_worker_matches_via_own_pid(self):
        managed = SimpleNamespace(process_id=200)
        self.assertTrue(
            ControlCenterWindow._control_matches_managed(self._snap(200, 100), managed)
        )

    def test_unrelated_controller_does_not_match(self):
        managed = SimpleNamespace(process_id=100)
        self.assertFalse(
            ControlCenterWindow._control_matches_managed(self._snap(300, 400), managed)
        )

    def test_no_managed_controller_never_matches(self):
        managed = SimpleNamespace(process_id=None)
        self.assertFalse(
            ControlCenterWindow._control_matches_managed(self._snap(200, 100), managed)
        )


class AutoHandStartTests(unittest.TestCase):
    """Hand control auto-starts (dry run) on a steady hand, without Tk."""

    def _window(self, *, mode=None, phase="OFF"):
        window = ControlCenterWindow.__new__(ControlCenterWindow)
        window._auto_hand_enabled = True
        window._closing = False
        window._auto_hand_allowed = True
        window._hand_present_since = None
        window._external_controller_active = False
        window._app_launch_in_progress = False
        window._app_status = ("stale", "x")
        self.calls = []
        managed = SimpleNamespace(mode=mode, phase=phase, process_id=None)
        window.controller_manager = SimpleNamespace(
            latest=lambda: managed,
            switch_async=lambda mode, live: self.calls.append((mode, live)),
        )
        return window

    def test_no_hand_rearms_and_does_not_start(self):
        window = self._window()
        window._auto_hand_allowed = False
        window._maybe_autostart_hand(False)
        self.assertTrue(window._auto_hand_allowed)
        self.assertEqual([], self.calls)

    def test_brief_hand_is_debounced(self):
        window = self._window()
        window._maybe_autostart_hand(True)  # first sighting only starts the timer
        self.assertEqual([], self.calls)

    def test_steady_hand_starts_dry_run_leap(self):
        window = self._window()
        window._hand_present_since = time.monotonic() - 1.0
        window._maybe_autostart_hand(True)
        self.assertEqual([(LEAP_MODE, False)], self.calls)
        # Consumed so it does not retrigger every frame.
        self.assertFalse(window._auto_hand_allowed)

    def test_running_controller_is_not_disturbed(self):
        window = self._window(mode=LEAP_MODE, phase="RUNNING")
        window._hand_present_since = time.monotonic() - 1.0
        window._maybe_autostart_hand(True)
        self.assertEqual([], self.calls)

    def test_disabled_auto_never_starts(self):
        window = self._window()
        window._auto_hand_enabled = False
        window._hand_present_since = time.monotonic() - 1.0
        window._maybe_autostart_hand(True)
        self.assertEqual([], self.calls)


if __name__ == "__main__":
    unittest.main()
