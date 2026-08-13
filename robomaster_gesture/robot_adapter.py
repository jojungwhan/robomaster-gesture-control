from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import threading
import time
from typing import Optional, Set

from .models import VelocityCommand


class RobotError(RuntimeError):
    pass


def _canonical_dji_connection_options(conn_module, conn_type, proto_type):
    """Return the SDK's own option objects, not merely equal strings.

    DJI RoboMaster SDK 0.1.1.68 compares connection modes with ``is`` in
    ``SdkConnection.request_connection``.  Strings produced by argparse are
    equal to the SDK constants but are not guaranteed to be the same objects,
    which can skip every mode branch and leave ``proxy_addr`` uninitialized.
    """
    connection_options = {
        "ap": conn_module.CONNECTION_WIFI_AP,
        "sta": conn_module.CONNECTION_WIFI_STA,
        "rndis": conn_module.CONNECTION_USB_RNDIS,
    }
    protocol_options = {
        "tcp": conn_module.CONNECTION_PROTO_TCP,
        "udp": conn_module.CONNECTION_PROTO_UDP,
    }
    try:
        canonical_connection = connection_options[str(conn_type).casefold()]
    except KeyError as exc:
        raise RobotError(
            "Unsupported RoboMaster SDK connection type: {!r}".format(conn_type)
        ) from exc
    try:
        canonical_protocol = protocol_options[str(proto_type).casefold()]
    except KeyError as exc:
        raise RobotError(
            "Unsupported RoboMaster SDK protocol type: {!r}".format(proto_type)
        ) from exc
    return canonical_connection, canonical_protocol


def _discover_dji_sta_robot(
    config_module,
    serial_number=None,
    socket_module=socket,
    timeout_s: float = 3.0,
):
    """Discover a STA robot before DJI can enter its broken fallback client.

    RoboMaster SDK 0.1.1.68 catches discovery failures, returns ``None``, and
    then attempts to construct a default client through additional defective
    code.  Listen directly on the same UDP port so binding and discovery are
    one atomic operation, then pass the discovered address into the SDK.
    """
    discovery_port = (
        int(config_module.ROBOT_BROADCAST_PORT)
        if serial_number
        else 45678
    )
    listener = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_DGRAM)
    try:
        listener.bind(("0.0.0.0", discovery_port))
    except OSError as exc:
        listener.close()
        raise RobotError(
            "RoboMaster STA discovery port {} is already in use. The desktop "
            "RoboMaster app commonly owns this port; close it before SDK "
            "discovery, or supply --robot-ip with the known robot address.".format(
                discovery_port
            )
        ) from exc

    deadline_s = time.monotonic() + max(0.01, float(timeout_s))
    try:
        while True:
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0.0:
                break
            listener.settimeout(remaining_s)
            try:
                data, address = listener.recvfrom(1024)
            except (socket.timeout, TimeoutError):
                break
            if serial_number:
                received_serial = data.split(b"\x00", 1)[0].decode(
                    "utf-8",
                    errors="replace",
                )
                if received_serial != serial_number:
                    continue
            if not address or not address[0]:
                continue
            return str(address[0])
    except OSError as exc:
        raise RobotError(
            "RoboMaster STA discovery failed while receiving broadcasts: {}".format(
                exc
            )
        ) from exc
    finally:
        listener.close()

    raise RobotError(
        "No RoboMaster STA broadcast was received. Confirm the robot is "
        "powered on, connected to this LAN, and in SDK mode, or supply "
        "--robot-ip with its known address."
    )


class RobotAdapter:
    def connect(self) -> None:
        raise NotImplementedError

    def drive(self, command: VelocityCommand, timeout_s: float = None) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.drive(VelocityCommand.stopped())

    def close(self) -> None:
        raise NotImplementedError


class DryRunRobot(RobotAdapter):
    def __init__(self):
        self.connected = False
        self.last_command = VelocityCommand.stopped()
        self.command_count = 0

    def connect(self) -> None:
        self.connected = True

    def drive(self, command: VelocityCommand, timeout_s: float = None) -> None:
        if not self.connected:
            raise RobotError("Dry-run robot is not connected.")
        self.last_command = command
        self.command_count += 1

    def close(self) -> None:
        self.last_command = VelocityCommand.stopped()
        self.connected = False


class DjiRobotAdapter(RobotAdapter):
    def __init__(
        self,
        conn_type: str,
        proto_type: str,
        robot_ip: str = None,
        local_ip: str = None,
        serial_number: str = None,
    ):
        self.conn_type = conn_type
        self.proto_type = proto_type
        self.robot_ip = robot_ip
        self.local_ip = local_ip
        self.serial_number = serial_number
        self._robot = None
        self._chassis = None

    @property
    def connected_ip(self) -> Optional[str]:
        if self._robot is None:
            return None
        try:
            return self._robot.ip
        except Exception:
            return None

    @property
    def camera(self):
        """Return the SDK camera after connection for vision-only consumers."""
        if self._robot is None:
            raise RobotError("RoboMaster is not connected.")
        return self._robot.camera

    def connect(self) -> None:
        try:
            from robomaster import config, conn, robot
        except ImportError as exc:
            raise RobotError(
                "DJI RoboMaster SDK is not installed in this Python environment."
            ) from exc

        sdk_conn_type, sdk_proto_type = _canonical_dji_connection_options(
            conn,
            self.conn_type,
            self.proto_type,
        )
        config.LOCAL_IP_STR = self.local_ip
        if sdk_conn_type is conn.CONNECTION_WIFI_STA:
            config.ROBOT_IP_STR = self.robot_ip or _discover_dji_sta_robot(
                config,
                serial_number=self.serial_number,
            )
        else:
            config.ROBOT_IP_STR = self.robot_ip

        self._robot = robot.Robot()
        try:
            initialized = self._robot.initialize(
                conn_type=sdk_conn_type,
                proto_type=sdk_proto_type,
                sn=self.serial_number,
            )
        except Exception as exc:
            self._safe_close_after_failure()
            raise RobotError("RoboMaster connection failed: {}".format(exc)) from exc

        if initialized is not True:
            self._safe_close_after_failure()
            raise RobotError(
                "RoboMaster SDK did not initialize. Check power, connection mode, "
                "firmware, and the selected network."
            )

        self._chassis = self._robot.chassis
        self.stop()

    def drive(self, command: VelocityCommand, timeout_s: float = None) -> None:
        if self._chassis is None:
            raise RobotError("RoboMaster is not connected.")
        try:
            result = self._chassis.drive_speed(
                x=float(command.forward_m_s),
                y=float(command.right_m_s),
                z=float(command.clockwise_deg_s),
                timeout=timeout_s,
            )
        except Exception as exc:
            raise RobotError("RoboMaster drive command failed: {}".format(exc)) from exc
        if result is False:
            raise RobotError("RoboMaster rejected a chassis speed command.")

    def stop(self) -> None:
        if self._chassis is not None:
            self.drive(VelocityCommand.stopped(), timeout_s=None)

    def close(self) -> None:
        if self._robot is None:
            return
        try:
            self.stop()
        finally:
            try:
                self._robot.close()
            finally:
                self._chassis = None
                self._robot = None

    def _safe_close_after_failure(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
        self._chassis = None
        self._robot = None


class _Win32RoboMasterKeyboard:
    """Small Win32 boundary used by the stock-S1 desktop-app adapter."""

    WM_KEYUP = 0x0101
    KEYEVENTF_KEYUP = 0x0002
    VK_BY_NAME = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44}

    def __init__(self, window_title: str):
        if os.name != "nt":
            raise RobotError("The RoboMaster desktop-app transport requires Windows.")

        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        self._window_title = window_title

        ulong_ptr = wintypes.WPARAM

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = (
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            )

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = (
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            )

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = (
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            )

        class INPUT_UNION(ctypes.Union):
            _fields_ = (
                ("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT),
            )

        class INPUT(ctypes.Structure):
            _anonymous_ = ("union",)
            _fields_ = (("type", wintypes.DWORD), ("union", INPUT_UNION))

        self._input_type = INPUT
        self._keybd_input_type = KEYBDINPUT
        expected_input_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        if ctypes.sizeof(INPUT) != expected_input_size:
            raise RobotError(
                "Unexpected Win32 INPUT structure size: {} (expected {}).".format(
                    ctypes.sizeof(INPUT), expected_input_size
                )
            )

        self._user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        self._user32.FindWindowW.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = (wintypes.HWND,)
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        self._user32.ShowWindow.restype = wintypes.BOOL
        self._user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        self._user32.SetForegroundWindow.restype = wintypes.BOOL
        self._user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(INPUT),
            ctypes.c_int,
        )
        self._user32.SendInput.restype = wintypes.UINT
        self._user32.PostMessageW.argtypes = (
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self._user32.PostMessageW.restype = wintypes.BOOL

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._advapi32.OpenProcessToken.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        )
        self._advapi32.OpenProcessToken.restype = wintypes.BOOL
        self._advapi32.GetTokenInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._advapi32.GetTokenInformation.restype = wintypes.BOOL

    def is_elevated(self) -> bool:
        return bool(self._shell32.IsUserAnAdmin())

    def has_input_access(self, window: int) -> bool:
        # Windows UIPI blocks SendInput from a lower-integrity process. The DJI
        # executable normally runs asInvoker, so elevation is only needed when
        # that particular app instance was itself launched as administrator.
        return self.is_elevated() or not self._window_process_is_elevated(window)

    def find_window(self) -> int:
        return int(self._user32.FindWindowW(None, self._window_title) or 0)

    def is_window(self, window: int) -> bool:
        return bool(self._user32.IsWindow(window))

    def is_foreground(self, window: int) -> bool:
        return int(self._user32.GetForegroundWindow() or 0) == int(window)

    def activate(self, window: int, timeout_s: float = 1.0) -> bool:
        """Restore and focus the DJI window before accepting movement input."""
        sw_restore = 9
        self._user32.ShowWindow(window, sw_restore)
        self._user32.SetForegroundWindow(window)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if self.is_foreground(window):
                return True
            time.sleep(0.025)
        return self.is_foreground(window)

    def _window_process_is_elevated(self, window: int) -> bool:
        process_query_limited_information = 0x1000
        token_query = 0x0008
        token_elevation_class = 20

        class TOKEN_ELEVATION(self._ctypes.Structure):
            _fields_ = (("TokenIsElevated", self._wintypes.DWORD),)

        process_id = self._wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(window, self._ctypes.byref(process_id))
        if not process_id.value:
            raise RobotError("Cannot identify the RoboMaster app process.")

        process = self._kernel32.OpenProcess(
            process_query_limited_information, False, process_id.value
        )
        if not process:
            raise RobotError("Cannot inspect the RoboMaster app permissions.")
        token = self._wintypes.HANDLE()
        try:
            if not self._advapi32.OpenProcessToken(process, token_query, self._ctypes.byref(token)):
                raise RobotError("Cannot inspect the RoboMaster app access token.")
            try:
                elevation = TOKEN_ELEVATION()
                returned = self._wintypes.DWORD()
                if not self._advapi32.GetTokenInformation(
                    token,
                    token_elevation_class,
                    self._ctypes.byref(elevation),
                    self._ctypes.sizeof(elevation),
                    self._ctypes.byref(returned),
                ):
                    raise RobotError("Cannot inspect the RoboMaster app elevation state.")
                return bool(elevation.TokenIsElevated)
            finally:
                self._kernel32.CloseHandle(token)
        finally:
            self._kernel32.CloseHandle(process)

    def press(self, window: int, key: str) -> None:
        if not self.is_foreground(window):
            raise RobotError("RoboMaster app is not the foreground window.")
        self._send_input(self.VK_BY_NAME[key], key_up=False)

    def release(self, window: int, key: str) -> None:
        virtual_key = self.VK_BY_NAME[key]
        # Send a normal key-up when possible and also address the original app
        # directly. The latter prevents a held key if focus changed mid-command.
        self._send_input(virtual_key, key_up=True, required=False)
        scan_code = int(self._user32.MapVirtualKeyW(virtual_key, 0))
        key_up_lparam = 1 | (scan_code << 16) | (1 << 30) | (1 << 31)
        self._user32.PostMessageW(window, self.WM_KEYUP, virtual_key, key_up_lparam)

    def _send_input(self, virtual_key: int, key_up: bool, required: bool = True) -> None:
        flags = self.KEYEVENTF_KEYUP if key_up else 0
        keyboard = self._keybd_input_type(
            wVk=virtual_key,
            wScan=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        item = self._input_type(type=1, ki=keyboard)
        sent = int(self._user32.SendInput(1, self._ctypes.byref(item), self._ctypes.sizeof(item)))
        if required and sent != 1:
            error = self._ctypes.get_last_error()
            raise RobotError(
                "Windows rejected keyboard input to RoboMaster (error {}). "
                "Run the gesture controller as administrator.".format(error)
            )


class S1AppKeyboardAdapter(RobotAdapter):
    """Drive a stock S1 through DJI's documented Windows-app W/A/S/D controls.

    The adapter intentionally does not synthesize mouse movement or Shift, so it
    cannot rotate the chassis and never selects the app's accelerated speed.
    """

    ALL_KEYS = frozenset(("w", "a", "s", "d"))

    def __init__(
        self,
        window_title: str = "RoboMaster",
        activation_threshold_m_s: float = 0.015,
        watchdog_s: float = 0.25,
        backend=None,
    ):
        self.window_title = window_title
        self.activation_threshold_m_s = activation_threshold_m_s
        self.watchdog_s = watchdog_s
        self._backend = backend
        self._window = 0
        self._pressed = set()  # type: Set[str]
        self._deadline = 0.0
        self._focus_interlock = False
        self._shutdown = False
        self._condition = threading.Condition()
        self._watchdog_thread = None  # type: Optional[threading.Thread]

    @property
    def connected_ip(self) -> Optional[str]:
        return None

    def connect(self) -> None:
        if self._backend is None:
            self._backend = _Win32RoboMasterKeyboard(self.window_title)
        self._window = self._backend.find_window()
        if not self._window or not self._backend.is_window(self._window):
            raise RobotError(
                "RoboMaster Windows app was not found. Open it and connect the S1 first."
            )
        if not self._backend.has_input_access(self._window):
            raise RobotError(
                "This RoboMaster app instance is running as administrator. Either "
                "run the gesture controller as administrator too, or close the DJI "
                "app and reopen it normally."
            )
        if not self._backend.activate(self._window):
            raise RobotError(
                "Windows would not bring the RoboMaster app to the foreground. "
                "Click its live drive view, then start the gesture controller again."
            )

        self._shutdown = False
        self._focus_interlock = False
        self._release_all(suppress_errors=True)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="RoboMasterAppKeyWatchdog",
            daemon=False,
        )
        self._watchdog_thread.start()

    def drive(self, command: VelocityCommand, timeout_s: float = None) -> None:
        if not self._window:
            raise RobotError("RoboMaster Windows app is not connected.")

        keys = self._keys_for(command)
        with self._condition:
            if keys:
                if not self._backend.is_window(self._window):
                    self._release_all_locked(suppress_errors=True)
                    raise RobotError("RoboMaster app closed; stopped.")
                if not self._backend.is_foreground(self._window):
                    self._focus_interlock = True
                    self._deadline = 0.0
                    self._release_all_locked(suppress_errors=True)
                    self._condition.notify_all()
                    raise RobotError(
                        "RoboMaster app lost keyboard focus; movement stopped. "
                        "Return to its live drive view and restart the controller."
                    )
                if self._focus_interlock:
                    # Focus loss is a latched stop. A zero command (normally a
                    # released pinch) must arrive before movement can resume.
                    self._deadline = 0.0
                    self._release_all_locked(suppress_errors=True)
                    self._condition.notify_all()
                    raise RobotError(
                        "RoboMaster app focus interlock is latched; movement stopped."
                    )
                self._set_keys_locked(keys)
                requested_timeout = timeout_s if timeout_s is not None else self.watchdog_s
                self._deadline = time.monotonic() + min(
                    self.watchdog_s, max(0.05, requested_timeout)
                )
            else:
                self._deadline = 0.0
                self._release_all_locked(suppress_errors=False)
                self._focus_interlock = False
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._deadline = 0.0
            self._release_all_locked(suppress_errors=True)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._shutdown = True
            self._deadline = 0.0
            self._release_all_locked(suppress_errors=True)
            self._condition.notify_all()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None
        self._window = 0

    def _keys_for(self, command: VelocityCommand) -> Set[str]:
        keys = set()
        if command.forward_m_s >= self.activation_threshold_m_s:
            keys.add("w")
        elif command.forward_m_s <= -self.activation_threshold_m_s:
            keys.add("s")
        if command.right_m_s >= self.activation_threshold_m_s:
            keys.add("d")
        elif command.right_m_s <= -self.activation_threshold_m_s:
            keys.add("a")
        return keys

    def _set_keys_locked(self, wanted: Set[str]) -> None:
        for key in sorted(self._pressed - wanted):
            self._backend.release(self._window, key)
            self._pressed.discard(key)
        for key in sorted(wanted - self._pressed):
            self._backend.press(self._window, key)
            self._pressed.add(key)

    def _release_all(self, suppress_errors: bool) -> None:
        with self._condition:
            self._release_all_locked(suppress_errors=suppress_errors)

    def _release_all_locked(self, suppress_errors: bool) -> None:
        first_error = None
        # Release every movement key, not only our bookkeeping set, so startup
        # and recovery clear keys left by an interrupted prior run.
        for key in sorted(self.ALL_KEYS):
            try:
                self._backend.release(self._window, key)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._pressed.clear()
        if first_error is not None and not suppress_errors:
            raise first_error

    def _watchdog_loop(self) -> None:
        while True:
            with self._condition:
                if self._shutdown:
                    return
                now = time.monotonic()
                focus_lost = bool(self._pressed) and (
                    not self._backend.is_window(self._window)
                    or not self._backend.is_foreground(self._window)
                )
                expired = bool(self._pressed) and self._deadline > 0.0 and now >= self._deadline
                if focus_lost or expired:
                    if focus_lost:
                        self._focus_interlock = True
                    self._deadline = 0.0
                    self._release_all_locked(suppress_errors=True)
                wait_s = 0.025
                if self._deadline > 0.0:
                    wait_s = min(wait_s, max(0.001, self._deadline - now))
                self._condition.wait(wait_s)


@dataclass(frozen=True)
class CommandPumpConfig:
    rate_hz: float = 15.0
    stale_after_s: float = 0.20
    moving_keepalive_s: float = 0.15
    robot_timeout_s: float = 0.35


class CommandPump:
    """Rate-limited sender with an independent stale-input watchdog."""

    def __init__(self, adapter: RobotAdapter, config: CommandPumpConfig = None):
        self.adapter = adapter
        self.config = config or CommandPumpConfig()
        self._condition = threading.Condition()
        self._desired = VelocityCommand.stopped()
        self._desired_at = 0.0
        self._desired_revision = 0
        self._last_sent = None  # type: Optional[VelocityCommand]
        self._last_sent_at = 0.0
        self._urgent = True
        self._shutdown = False
        self._error = None  # type: Optional[Exception]
        self._thread = None  # type: Optional[threading.Thread]

    @property
    def error(self) -> Optional[Exception]:
        with self._condition:
            return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="RoboMasterCommandPump", daemon=False
        )
        self._thread.start()

    def submit(self, command: VelocityCommand) -> None:
        now = time.monotonic()
        with self._condition:
            was_moving = not self._desired.is_stopped()
            self._desired = command
            self._desired_at = now
            self._desired_revision += 1
            if command.is_stopped() and was_moving:
                self._urgent = True
            self._condition.notify_all()

    def halt(self) -> None:
        with self._condition:
            self._desired = VelocityCommand.stopped()
            self._desired_at = time.monotonic()
            self._desired_revision += 1
            self._urgent = True
            self._condition.notify_all()

    def close(self) -> None:
        if self._thread is None:
            return
        with self._condition:
            self._desired = VelocityCommand.stopped()
            self._desired_at = time.monotonic()
            self._desired_revision += 1
            self._urgent = True
            self._shutdown = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            try:
                self.adapter.stop()
            except Exception:
                pass
        self._thread = None

    def _run(self) -> None:
        interval = 1.0 / self.config.rate_hz
        next_send = time.monotonic()

        while True:
            with self._condition:
                now = time.monotonic()
                wait_s = max(0.0, next_send - now)
                if not self._urgent and not self._shutdown and wait_s > 0.0:
                    self._condition.wait(wait_s)
                    continue

                now = time.monotonic()
                shutting_down = self._shutdown
                stale = now - self._desired_at > self.config.stale_after_s
                command = (
                    VelocityCommand.stopped()
                    if shutting_down or stale
                    else self._desired
                )
                command_revision = self._desired_revision
                urgent = self._urgent
                self._urgent = False

                changed = self._last_sent is None or not self._commands_close(
                    command, self._last_sent
                )
                keepalive_due = (
                    not command.is_stopped()
                    and now - self._last_sent_at >= self.config.moving_keepalive_s
                )
                should_send = urgent or changed or keepalive_due
                next_send = max(next_send + interval, now + interval)

            if should_send:
                try:
                    timeout_s = (
                        None if command.is_stopped() else self.config.robot_timeout_s
                    )
                    self.adapter.drive(command, timeout_s=timeout_s)
                    with self._condition:
                        self._last_sent = command
                        self._last_sent_at = time.monotonic()
                        superseded_by_stop = (
                            not command.is_stopped()
                            and command_revision != self._desired_revision
                            and self._desired.is_stopped()
                        )
                        if superseded_by_stop:
                            # A halt arrived after this motion command was
                            # dequeued. Preserve the halt as urgent so the very
                            # next send clears motion instead of losing it to
                            # the in-flight command.
                            self._urgent = True
                            self._condition.notify_all()
                except Exception as exc:
                    with self._condition:
                        self._error = exc
                        self._shutdown = True
                    try:
                        self.adapter.stop()
                    except Exception:
                        pass
                    return

            if shutting_down:
                return

    @staticmethod
    def _commands_close(
        first: VelocityCommand, second: VelocityCommand, epsilon: float = 0.005
    ) -> bool:
        return (
            abs(first.forward_m_s - second.forward_m_s) <= epsilon
            and abs(first.right_m_s - second.right_m_s) <= epsilon
            and abs(first.clockwise_deg_s - second.clockwise_deg_s)
            <= epsilon * 100.0
        )
