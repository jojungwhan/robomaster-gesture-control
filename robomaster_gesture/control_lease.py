"""Cross-process lease preventing concurrent live motion controllers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading


class ControlLeaseError(RuntimeError):
    pass


class ControllerLease:
    """Own an exclusive machine-local lease until ``close`` is called."""

    _process_guard = threading.Lock()
    _process_names = set()

    def __init__(self, name: str = "robomaster_gesture_motion_controller"):
        if not name or any(not (item.isalnum() or item in "_-") for item in name):
            raise ValueError("controller lease name contains unsupported characters")
        self.name = name
        self._handle = None
        self._file = None
        self._owned = False

    def acquire(self) -> None:
        if self._owned:
            return
        with self._process_guard:
            if self.name in self._process_names:
                raise ControlLeaseError(
                    "Another live RoboMaster controller is already running."
                )
            self._process_names.add(self.name)

        try:
            if os.name == "nt":
                self._acquire_windows()
            else:
                self._acquire_file()
            self._owned = True
        except Exception:
            with self._process_guard:
                self._process_names.discard(self.name)
            raise

    def _acquire_windows(self) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateMutexW(None, False, "Local\\" + self.name)
        if not handle:
            raise ControlLeaseError("Windows could not create the controller lease.")
        result = int(kernel32.WaitForSingleObject(handle, 0))
        if result not in (0x00000000, 0x00000080):  # acquired or abandoned
            kernel32.CloseHandle(handle)
            if result == 0x00000102:
                raise ControlLeaseError(
                    "Another live RoboMaster controller is already running."
                )
            raise ControlLeaseError(
                "Windows could not acquire the controller lease (error {}).".format(
                    ctypes.get_last_error()
                )
            )
        self._handle = (kernel32, handle)

    def _acquire_file(self) -> None:
        import fcntl

        path = Path(tempfile.gettempdir()) / (self.name + ".lock")
        lock_file = path.open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise ControlLeaseError(
                "Another live RoboMaster controller is already running."
            ) from exc
        self._file = lock_file

    def close(self) -> None:
        if not self._owned:
            return
        try:
            if self._handle is not None:
                kernel32, handle = self._handle
                kernel32.ReleaseMutex(handle)
                kernel32.CloseHandle(handle)
                self._handle = None
            if self._file is not None:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._file.close()
                self._file = None
        finally:
            self._owned = False
            with self._process_guard:
                self._process_names.discard(self.name)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
