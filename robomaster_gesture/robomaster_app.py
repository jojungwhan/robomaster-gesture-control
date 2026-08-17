"""Detect and, when needed, launch DJI's stock RoboMaster Windows app.

The stock-S1 transport drives the robot by sending W/A/S/D into DJI's desktop
app, so that app must be open before ``HAND`` or ``VOICE`` control can attach.
This module gives the Control Center a small, testable boundary for checking
whether the app window is present and, if not, running the project's
``launch_robomaster_standard.ps1`` launcher.  All Win32 and subprocess access is
injectable so the decision logic can be unit tested off Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Callable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER_SCRIPT = PROJECT_ROOT / "launch_robomaster_standard.ps1"

# DJI's desktop app titles its main window "RoboMaster"; this is the same title
# the S1 app keyboard transport matches with FindWindowW.
ROBOMASTER_WINDOW_TITLE = "RoboMaster"


def _default_find_window(title: str) -> bool:
    """Return whether a top-level window with ``title`` currently exists."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    return bool(user32.FindWindowW(None, title))


@dataclass(frozen=True)
class AppLaunchResult:
    """Outcome of a launch attempt, phrased for a Control Center status line."""

    started: bool
    already_running: bool
    message: str


class RoboMasterAppLauncher:
    """Check for DJI's RoboMaster app and start it through the project launcher."""

    def __init__(
        self,
        launcher_script: Path = DEFAULT_LAUNCHER_SCRIPT,
        window_title: str = ROBOMASTER_WINDOW_TITLE,
        find_window: Optional[Callable[[str], bool]] = None,
        runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
        powershell: str = "powershell.exe",
    ):
        self.launcher_script = Path(launcher_script)
        self.window_title = window_title
        self._find_window = find_window or _default_find_window
        self._runner = runner or subprocess.run
        self._powershell = powershell

    def is_app_present(self) -> bool:
        """Return whether the RoboMaster app window is already open."""
        try:
            return bool(self._find_window(self.window_title))
        except Exception:
            # A detection failure must never be treated as "already running",
            # otherwise control would attach to a robot app that is not there.
            return False

    def build_command(self) -> Sequence[str]:
        """Build the shell-free PowerShell command that runs the launcher."""
        return (
            self._powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.launcher_script),
        )

    def launch(self) -> AppLaunchResult:
        """Start the RoboMaster app unless it is already open.

        This may block while the launcher makes its one-time writable runtime
        copy, so callers should run it off the UI thread.
        """
        if self.is_app_present():
            return AppLaunchResult(
                started=False,
                already_running=True,
                message=(
                    "The RoboMaster app is already open. Connect the S1 and "
                    "open its live drive view."
                ),
            )
        if os.name != "nt":
            return AppLaunchResult(
                started=False,
                already_running=False,
                message="The RoboMaster app can only be launched on Windows.",
            )
        if not self.launcher_script.is_file():
            return AppLaunchResult(
                started=False,
                already_running=False,
                message="RoboMaster launcher not found: {}".format(
                    self.launcher_script
                ),
            )

        try:
            completed = self._runner(
                self.build_command(),
                cwd=str(self.launcher_script.parent),
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # surfaced verbatim to the operator
            return AppLaunchResult(
                started=False,
                already_running=False,
                message="Could not launch the RoboMaster app: {}".format(exc),
            )

        if int(getattr(completed, "returncode", 1)) != 0:
            detail = (
                self._last_line(getattr(completed, "stderr", ""))
                or self._last_line(getattr(completed, "stdout", ""))
                or "the launcher reported an error"
            )
            return AppLaunchResult(
                started=False,
                already_running=False,
                message="RoboMaster app did not launch: {}".format(detail),
            )

        return AppLaunchResult(
            started=True,
            already_running=False,
            message=(
                "Starting the RoboMaster app. Connect the S1 and open its live "
                "drive view, then press HAND or VOICE again."
            ),
        )

    @staticmethod
    def _last_line(text: Optional[str]) -> str:
        if not text:
            return ""
        for line in reversed(str(text).splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return ""
