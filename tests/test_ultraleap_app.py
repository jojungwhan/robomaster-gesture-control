import unittest
from pathlib import Path
from unittest import mock

from robomaster_gesture import ultraleap_app
from robomaster_gesture.ultraleap_app import UltraleapControlPanelLauncher


EXE = Path(r"C:\Program Files\Ultraleap\ControlPanel\UnityApp\Ultraleap Control Panel.exe")


def _launcher(*, running=False, executable=EXE, launch=None, process_raises=False):
    def is_process_running(_names):
        if process_raises:
            raise OSError("tasklist failed")
        return running

    return UltraleapControlPanelLauncher(
        find_executable=lambda _candidates: executable,
        is_process_running=is_process_running,
        launcher=launch or (lambda _exe: None),
    )


class UltraleapLauncherTests(unittest.TestCase):
    def test_launches_when_installed_and_not_running(self):
        opened = []
        launcher = _launcher(running=False, launch=opened.append)
        result = launcher.launch()
        self.assertTrue(result.started)
        self.assertFalse(result.already_running)
        self.assertEqual([EXE], opened)

    def test_already_running_is_not_relaunched(self):
        opened = []
        launcher = _launcher(running=True, launch=opened.append)
        result = launcher.launch()
        self.assertFalse(result.started)
        self.assertTrue(result.already_running)
        self.assertEqual([], opened)

    def test_missing_install_gives_an_actionable_hint(self):
        launcher = _launcher(running=False, executable=None)
        result = launcher.launch()
        self.assertFalse(result.started)
        self.assertFalse(result.already_running)
        self.assertIn("not found", result.message)
        self.assertIn("Ultraleap", result.message)

    def test_launch_failure_is_reported(self):
        def boom(_exe):
            raise OSError("access denied")

        result = _launcher(running=False, launch=boom).launch()
        self.assertFalse(result.started)
        self.assertIn("Could not open", result.message)

    def test_detection_failure_counts_as_not_running(self):
        launcher = _launcher(process_raises=True)
        self.assertFalse(launcher.is_running())

    def test_non_windows_is_declined_without_launching(self):
        opened = []
        launcher = _launcher(running=False, launch=opened.append)
        with mock.patch.object(ultraleap_app.os, "name", "posix"):
            result = launcher.launch()
        self.assertFalse(result.started)
        self.assertIn("Windows", result.message)
        self.assertEqual([], opened)


if __name__ == "__main__":
    unittest.main()
