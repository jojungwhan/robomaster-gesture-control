"""Elevated, self-logging entry point for stock RoboMaster S1 app control."""

from __future__ import annotations

import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
LOG_DIRECTORY = PROJECT_ROOT / "logs"
STATE_PATH = LOG_DIRECTORY / "current_s1_live.json"
BRIDGE_PATH = PROJECT_ROOT / "build" / "leap_hand_bridge.dll"


def _write_state(**values: object) -> None:
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(STATE_PATH))


def _release_movement_keys() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    for virtual_key in (0x57, 0x41, 0x53, 0x44):
        user32.keybd_event(virtual_key, 0, 0x0002, 0)


def _focus_robomaster_after(delay_s: float) -> None:
    def focus() -> None:
        time.sleep(delay_s)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        window = user32.FindWindowW(None, "RoboMaster")
        if window:
            user32.ShowWindowAsync(window, 9)
            user32.SetForegroundWindow(window)
            user32.SwitchToThisWindow(window, True)

    threading.Thread(target=focus, name="DelayedRoboMasterFocus", daemon=True).start()


def main() -> int:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIRECTORY / "s1_live_{}.log".format(timestamp)
    started_at = datetime.now().astimezone().isoformat()
    process_id = os.getpid()

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = log
        sys.stderr = log
        exit_code = 1
        try:
            _write_state(
                controller_process_id=process_id,
                status="starting",
                exit_code=None,
                log_path=str(log_path),
                started_at=started_at,
            )
            print("Elevated S1 gesture launcher PID {} starting.".format(process_id))
            print("Project: {}".format(PROJECT_ROOT))
            print("Bridge: {}".format(BRIDGE_PATH))
            _release_movement_keys()
            os.chdir(str(PROJECT_ROOT))

            from robomaster_gesture.app import main as gesture_main

            _write_state(
                controller_process_id=process_id,
                status="running",
                exit_code=None,
                log_path=str(log_path),
                started_at=started_at,
            )
            # Let the caller finish reading startup status, then make the DJI
            # live-drive view the final foreground window.
            _focus_robomaster_after(8.0)
            exit_code = gesture_main(
                [
                    "--live",
                    "--transport",
                    "s1-app",
                    "--connection",
                    "sta",
                    "--protocol",
                    "tcp",
                    "--hand",
                    "right",
                    "--bridge-dll",
                    str(BRIDGE_PATH),
                ]
            )
            return exit_code
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            try:
                _release_movement_keys()
                _write_state(
                    controller_process_id=process_id,
                    status="exited",
                    exit_code=exit_code,
                    log_path=str(log_path),
                    started_at=started_at,
                    stopped_at=datetime.now().astimezone().isoformat(),
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


if __name__ == "__main__":
    raise SystemExit(main())
