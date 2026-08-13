from __future__ import annotations

import ctypes
import os
from pathlib import Path
import time
from typing import Optional

from .models import FrameSample, HandSample


_MAX_HANDS = 2
_LHB_OK = 0
_LHB_TIMEOUT = 1
_STATUS_SERVICE_CONNECTED = 1
_STATUS_DEVICE_PRESENT = 2


class _NativeHand(ctypes.Structure):
    _fields_ = [
        ("hand_id", ctypes.c_uint32),
        ("hand_type", ctypes.c_int32),
        ("visible_time_us", ctypes.c_uint64),
        ("palm_x", ctypes.c_float),
        ("palm_y", ctypes.c_float),
        ("palm_z", ctypes.c_float),
        ("velocity_x", ctypes.c_float),
        ("velocity_y", ctypes.c_float),
        ("velocity_z", ctypes.c_float),
        ("direction_x", ctypes.c_float),
        ("direction_y", ctypes.c_float),
        ("direction_z", ctypes.c_float),
        ("normal_x", ctypes.c_float),
        ("normal_y", ctypes.c_float),
        ("normal_z", ctypes.c_float),
        ("pinch_strength", ctypes.c_float),
        ("grab_strength", ctypes.c_float),
        ("pinch_distance_mm", ctypes.c_float),
    ]


class _NativeFrame(ctypes.Structure):
    _fields_ = [
        ("frame_id", ctypes.c_uint64),
        ("sensor_timestamp_us", ctypes.c_int64),
        ("framerate", ctypes.c_float),
        ("hand_count", ctypes.c_uint32),
        ("total_hand_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("hands", _NativeHand * _MAX_HANDS),
    ]


class LeapSourceError(RuntimeError):
    pass


class LeapSource:
    def __init__(self, library_path: Path):
        self.library_path = Path(library_path).resolve()
        self._library = None
        self._context = ctypes.c_void_p()
        self._dll_directory = None

    def open(self) -> None:
        if self._library is not None:
            return
        if not self.library_path.is_file():
            raise LeapSourceError(
                "Leap bridge DLL not found: {}. Build the CMake target first.".format(
                    self.library_path
                )
            )

        if hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(
                str(self.library_path.parent)
            )
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_functions()

        native_frame_size = self._library.lhb_frame_sample_size()
        native_hand_size = self._library.lhb_hand_sample_size()
        if native_frame_size != ctypes.sizeof(_NativeFrame):
            raise LeapSourceError(
                "Leap frame ABI mismatch: native={}, Python={}".format(
                    native_frame_size, ctypes.sizeof(_NativeFrame)
                )
            )
        if native_hand_size != ctypes.sizeof(_NativeHand):
            raise LeapSourceError(
                "Leap hand ABI mismatch: native={}, Python={}".format(
                    native_hand_size, ctypes.sizeof(_NativeHand)
                )
            )

        result = self._library.lhb_create(ctypes.byref(self._context))
        if result != _LHB_OK:
            self._library = None
            raise LeapSourceError("Could not open LeapC bridge (error {}).".format(result))

    def _configure_functions(self) -> None:
        self._library.lhb_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._library.lhb_create.restype = ctypes.c_int
        self._library.lhb_poll.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_NativeFrame),
        ]
        self._library.lhb_poll.restype = ctypes.c_int
        self._library.lhb_status.argtypes = [ctypes.c_void_p]
        self._library.lhb_status.restype = ctypes.c_uint32
        self._library.lhb_frame_sample_size.argtypes = []
        self._library.lhb_frame_sample_size.restype = ctypes.c_uint32
        self._library.lhb_hand_sample_size.argtypes = []
        self._library.lhb_hand_sample_size.restype = ctypes.c_uint32
        self._library.lhb_last_leap_result.argtypes = [ctypes.c_void_p]
        self._library.lhb_last_leap_result.restype = ctypes.c_uint32
        self._library.lhb_destroy.argtypes = [ctypes.c_void_p]
        self._library.lhb_destroy.restype = None

    @property
    def service_connected(self) -> bool:
        return bool(self._status() & _STATUS_SERVICE_CONNECTED)

    @property
    def device_present(self) -> bool:
        return bool(self._status() & _STATUS_DEVICE_PRESENT)

    def _status(self) -> int:
        if self._library is None or not self._context:
            return 0
        return int(self._library.lhb_status(self._context))

    def poll(self, timeout_ms: int = 50) -> Optional[FrameSample]:
        if self._library is None or not self._context:
            raise LeapSourceError("Leap source is not open.")
        native = _NativeFrame()
        result = self._library.lhb_poll(
            self._context, max(0, int(timeout_ms)), ctypes.byref(native)
        )
        if result == _LHB_TIMEOUT:
            return None
        if result != _LHB_OK:
            leap_result = self._library.lhb_last_leap_result(self._context)
            raise LeapSourceError(
                "Leap polling failed (bridge {}, LeapC 0x{:08X}).".format(
                    result, leap_result
                )
            )

        hands = []
        for index in range(min(int(native.hand_count), _MAX_HANDS)):
            hand = native.hands[index]
            hands.append(
                HandSample(
                    hand_id=int(hand.hand_id),
                    is_left=int(hand.hand_type) == 0,
                    visible_time_s=float(hand.visible_time_us) / 1_000_000.0,
                    palm_x_mm=float(hand.palm_x),
                    palm_y_mm=float(hand.palm_y),
                    palm_z_mm=float(hand.palm_z),
                    velocity_x_mm_s=float(hand.velocity_x),
                    velocity_y_mm_s=float(hand.velocity_y),
                    velocity_z_mm_s=float(hand.velocity_z),
                    direction_x=float(hand.direction_x),
                    direction_y=float(hand.direction_y),
                    direction_z=float(hand.direction_z),
                    normal_x=float(hand.normal_x),
                    normal_y=float(hand.normal_y),
                    normal_z=float(hand.normal_z),
                    pinch_strength=float(hand.pinch_strength),
                    grab_strength=float(hand.grab_strength),
                    pinch_distance_mm=float(hand.pinch_distance_mm),
                )
            )

        return FrameSample(
            arrival_time_s=time.monotonic(),
            frame_id=int(native.frame_id),
            sensor_timestamp_us=int(native.sensor_timestamp_us),
            framerate=float(native.framerate),
            total_hand_count=int(native.total_hand_count),
            hands=tuple(hands),
        )

    def close(self) -> None:
        if self._library is not None and self._context:
            self._library.lhb_destroy(self._context)
        self._context = ctypes.c_void_p()
        self._library = None
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None

    def __enter__(self) -> "LeapSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
