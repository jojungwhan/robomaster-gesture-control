"""Detect and, when needed, launch Ultraleap's Tracking Control Panel.

Leap hand control needs the Ultraleap tracking service, and its Control Panel is
the visual confirmation that the sensor sees a hand. This opens the Control Panel
alongside our app, mirroring :mod:`robomaster_app` for DJI's desktop app. All
filesystem and subprocess access is injectable so the logic is unit-testable off
Windows.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Callable, Optional, Sequence

from .robomaster_app import AppLaunchResult


# Ultraleap Gemini/Hyperion installs the Control Panel and tray here; the last
# entry is the older Leap Motion "Orion" path, tried as a fallback.
KNOWN_CONTROL_PANEL_PATHS = (
    Path(
        r"C:\Program Files\Ultraleap\ControlPanel\UnityApp"
        r"\Ultraleap Control Panel.exe"
    ),
    Path(r"C:\Program Files\Ultraleap\ControlPanel\Ultraleap Tray.exe"),
    Path(r"C:\Program Files\Leap Motion\Core Services\LeapControlPanel.exe"),
)

# Image names that mean the Control Panel *window* is already open. The tray
# (Ultraleap Tray.exe) is deliberately excluded: it runs in the background even
# when the window is closed, so counting it would skip opening the window the
# user wants to see.
CONTROL_PANEL_PROCESS_NAMES = (
    "Ultraleap Control Panel.exe",
    "LeapControlPanel.exe",
)


def _default_find_executable(candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def _default_process_running(process_names: Sequence[str]) -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return False
    haystack = (result.stdout or "").lower()
    return any(name.lower() in haystack for name in process_names)


def _default_launch(executable: Path) -> None:
    subprocess.Popen(
        [str(executable)],
        cwd=str(Path(executable).parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class UltraleapControlPanelLauncher:
    """Check for and open Ultraleap's Tracking Control Panel."""

    def __init__(
        self,
        executable_candidates: Sequence[Path] = KNOWN_CONTROL_PANEL_PATHS,
        process_names: Sequence[str] = CONTROL_PANEL_PROCESS_NAMES,
        find_executable: Optional[Callable[[Sequence[Path]], Optional[Path]]] = None,
        is_process_running: Optional[Callable[[Sequence[str]], bool]] = None,
        launcher: Optional[Callable[[Path], None]] = None,
    ):
        self.executable_candidates = tuple(Path(p) for p in executable_candidates)
        self.process_names = tuple(process_names)
        self._find_executable = find_executable or _default_find_executable
        self._is_process_running = is_process_running or _default_process_running
        self._launcher = launcher or _default_launch

    def find_executable(self) -> Optional[Path]:
        return self._find_executable(self.executable_candidates)

    def is_running(self) -> bool:
        """Return whether a Control Panel / tray process is already up."""
        try:
            return bool(self._is_process_running(self.process_names))
        except Exception:
            # A detection failure must not be treated as "already running", or we
            # would skip a launch the user asked for.
            return False

    def launch(self) -> AppLaunchResult:
        """Open the Ultraleap Control Panel unless it is already running."""
        if os.name != "nt":
            return AppLaunchResult(
                started=False,
                already_running=False,
                message="The Ultraleap Control Panel can only be launched on Windows.",
            )
        if self.is_running():
            return AppLaunchResult(
                started=False,
                already_running=True,
                message="The Ultraleap Control Panel is already open.",
            )
        executable = self.find_executable()
        if executable is None:
            return AppLaunchResult(
                started=False,
                already_running=False,
                message=(
                    "Ultraleap software not found. Install Ultraleap Tracking "
                    "(Gemini) to use hand control."
                ),
            )
        try:
            self._launcher(executable)
        except Exception as exc:  # surfaced verbatim to the operator
            return AppLaunchResult(
                started=False,
                already_running=False,
                message="Could not open the Ultraleap Control Panel: {}".format(exc),
            )
        return AppLaunchResult(
            started=True,
            already_running=False,
            message="Opening the Ultraleap Control Panel; confirm it sees your hand.",
        )
