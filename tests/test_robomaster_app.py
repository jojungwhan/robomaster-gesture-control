import os
import tempfile
import unittest
from pathlib import Path

from robomaster_gesture.robomaster_app import (
    AppLaunchResult,
    RoboMasterAppLauncher,
)


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _recording_runner(completed):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return completed

    return runner, calls


class RoboMasterAppLauncherPresenceTests(unittest.TestCase):
    def test_present_app_is_not_relaunched(self):
        runner, calls = _recording_runner(_FakeCompleted())
        launcher = RoboMasterAppLauncher(
            find_window=lambda title: True,
            runner=runner,
        )

        result = launcher.launch()

        self.assertIsInstance(result, AppLaunchResult)
        self.assertFalse(result.started)
        self.assertTrue(result.already_running)
        self.assertEqual([], calls)

    def test_detection_error_is_not_treated_as_running(self):
        def raising_find_window(_title):
            raise OSError("FindWindow failed")

        launcher = RoboMasterAppLauncher(find_window=raising_find_window)
        self.assertFalse(launcher.is_app_present())


@unittest.skipUnless(os.name == "nt", "the launcher only runs the app on Windows")
class RoboMasterAppLauncherLaunchTests(unittest.TestCase):
    def _launcher(self, script, completed):
        runner, calls = _recording_runner(completed)
        launcher = RoboMasterAppLauncher(
            launcher_script=script,
            find_window=lambda title: False,
            runner=runner,
            powershell="powershell.exe",
        )
        return launcher, calls

    def test_absent_app_is_launched_with_the_project_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "launch_robomaster_standard.ps1"
            script.write_text("exit 0", encoding="ascii")
            launcher, calls = self._launcher(script, _FakeCompleted())

            result = launcher.launch()

        self.assertTrue(result.started)
        self.assertFalse(result.already_running)
        self.assertEqual(1, len(calls))
        command, kwargs = calls[0]
        self.assertEqual("powershell.exe", command[0])
        self.assertIn("-File", command)
        self.assertEqual(str(script), command[-1])
        self.assertTrue(kwargs.get("capture_output"))

    def test_launcher_failure_surfaces_the_last_error_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "launch_robomaster_standard.ps1"
            script.write_text("exit 1", encoding="ascii")
            launcher, _calls = self._launcher(
                script,
                _FakeCompleted(
                    returncode=1,
                    stderr="line one\nClose the running RoboMaster app first\n",
                ),
            )

            result = launcher.launch()

        self.assertFalse(result.started)
        self.assertFalse(result.already_running)
        self.assertIn("Close the running RoboMaster app first", result.message)

    def test_missing_script_is_reported_without_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "does_not_exist.ps1"
            launcher, calls = self._launcher(script, _FakeCompleted())

            result = launcher.launch()

        self.assertFalse(result.started)
        self.assertIn("launcher not found", result.message)
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
