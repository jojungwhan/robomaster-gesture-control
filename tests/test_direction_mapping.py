import socket
import unittest

from robomaster_gesture.models import VelocityCommand, translation_directions
from robomaster_gesture.robot_adapter import (
    DjiRobotAdapter,
    RobotError,
    _canonical_dji_connection_options,
    _discover_dji_sta_robot,
)


class FakeChassis:
    def __init__(self):
        self.calls = []

    def drive_speed(self, **kwargs):
        self.calls.append(kwargs)
        return True


class DirectionMappingTests(unittest.TestCase):
    def test_dji_sta_discovery_returns_broadcast_address(self):
        events = []

        class FakeSocket:
            def bind(self, address):
                events.append(("bind", address))

            def settimeout(self, timeout_s):
                events.append(("timeout", timeout_s))

            def recvfrom(self, maximum_bytes):
                events.append(("recv", maximum_bytes))
                return b"robot-data", ("192.0.2.17", 45678)

            def close(self):
                events.append(("close",))

        class FakeSocketModule:
            AF_INET = object()
            SOCK_DGRAM = object()

            @staticmethod
            def socket(family, kind):
                events.append(("socket", family, kind))
                return FakeSocket()

        class FakeConfig:
            ROBOT_BROADCAST_PORT = 40927

        result = _discover_dji_sta_robot(
            FakeConfig,
            socket_module=FakeSocketModule,
        )

        self.assertEqual("192.0.2.17", result)
        self.assertIn(("bind", ("0.0.0.0", 45678)), events)
        self.assertIn(("recv", 1024), events)
        self.assertEqual(("close",), events[-1])

    def test_dji_sta_discovery_filters_by_serial_number(self):
        frames = iter(
            (
                (b"wrong-serial\x00payload", ("192.0.2.10", 40927)),
                (b"wanted-serial\x00payload", ("192.0.2.11", 40927)),
            )
        )
        events = []

        class FakeSocket:
            def bind(self, address):
                events.append(("bind", address))

            def settimeout(self, timeout_s):
                pass

            def recvfrom(self, maximum_bytes):
                return next(frames)

            def close(self):
                pass

        class FakeSocketModule:
            AF_INET = object()
            SOCK_DGRAM = object()

            @staticmethod
            def socket(family, kind):
                return FakeSocket()

        class FakeConfig:
            ROBOT_BROADCAST_PORT = 40927

        result = _discover_dji_sta_robot(
            FakeConfig,
            serial_number="wanted-serial",
            socket_module=FakeSocketModule,
        )

        self.assertEqual("192.0.2.11", result)
        self.assertEqual([("bind", ("0.0.0.0", 40927))], events)

    def test_dji_sta_discovery_reports_occupied_port_without_scanning(self):
        class FakeSocket:
            def bind(self, address):
                raise OSError("address in use")

            def close(self):
                pass

        class FakeSocketModule:
            AF_INET = object()
            SOCK_DGRAM = object()

            @staticmethod
            def socket(family, kind):
                return FakeSocket()

        class FakeConfig:
            ROBOT_BROADCAST_PORT = 40927

        with self.assertRaisesRegex(RobotError, "already in use"):
            _discover_dji_sta_robot(
                FakeConfig,
                socket_module=FakeSocketModule,
            )

    def test_dji_sta_discovery_reports_missing_broadcast(self):
        class FakeSocket:
            def bind(self, address):
                pass

            def settimeout(self, timeout_s):
                pass

            def recvfrom(self, maximum_bytes):
                raise socket.timeout()

            def close(self):
                pass

        class FakeSocketModule:
            AF_INET = object()
            SOCK_DGRAM = object()

            @staticmethod
            def socket(family, kind):
                return FakeSocket()

        class FakeConfig:
            ROBOT_BROADCAST_PORT = 40927

        with self.assertRaisesRegex(RobotError, "No RoboMaster STA broadcast"):
            _discover_dji_sta_robot(
                FakeConfig,
                socket_module=FakeSocketModule,
                timeout_s=0.01,
            )

    def test_dji_sdk_options_use_canonical_constant_objects(self):
        class FakeConnectionModule:
            CONNECTION_WIFI_AP = object()
            CONNECTION_WIFI_STA = object()
            CONNECTION_USB_RNDIS = object()
            CONNECTION_PROTO_TCP = object()
            CONNECTION_PROTO_UDP = object()

        cases = (
            ("ap", "tcp", "CONNECTION_WIFI_AP", "CONNECTION_PROTO_TCP"),
            ("sta", "udp", "CONNECTION_WIFI_STA", "CONNECTION_PROTO_UDP"),
            ("rndis", "tcp", "CONNECTION_USB_RNDIS", "CONNECTION_PROTO_TCP"),
        )
        for connection, protocol, connection_name, protocol_name in cases:
            with self.subTest(connection=connection, protocol=protocol):
                # Build fresh strings as argparse does; equality must not be
                # mistaken for the identity DJI 0.1.1.68 incorrectly requires.
                cli_connection = "".join(connection)
                cli_protocol = "".join(protocol)
                actual_connection, actual_protocol = (
                    _canonical_dji_connection_options(
                        FakeConnectionModule,
                        cli_connection,
                        cli_protocol,
                    )
                )
                self.assertIs(
                    getattr(FakeConnectionModule, connection_name),
                    actual_connection,
                )
                self.assertIs(
                    getattr(FakeConnectionModule, protocol_name),
                    actual_protocol,
                )

    def test_dji_sdk_options_reject_unknown_values(self):
        class FakeConnectionModule:
            CONNECTION_WIFI_AP = object()
            CONNECTION_WIFI_STA = object()
            CONNECTION_USB_RNDIS = object()
            CONNECTION_PROTO_TCP = object()
            CONNECTION_PROTO_UDP = object()

        with self.assertRaisesRegex(RobotError, "connection type"):
            _canonical_dji_connection_options(FakeConnectionModule, "bluetooth", "tcp")
        with self.assertRaisesRegex(RobotError, "protocol type"):
            _canonical_dji_connection_options(FakeConnectionModule, "sta", "serial")

    def test_direction_labels_cover_four_cardinal_commands(self):
        cases = (
            (VelocityCommand(forward_m_s=0.2), ("FORWARD",)),
            (VelocityCommand(forward_m_s=-0.2), ("BACK",)),
            (VelocityCommand(right_m_s=-0.2), ("LEFT",)),
            (VelocityCommand(right_m_s=0.2), ("RIGHT",)),
            (
                VelocityCommand(forward_m_s=0.2, right_m_s=-0.2),
                ("FORWARD", "LEFT"),
            ),
            (VelocityCommand.stopped(), ()),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(expected, translation_directions(command))

    def test_dji_adapter_preserves_forward_and_right_axis_signs(self):
        cases = (
            ("forward", VelocityCommand(forward_m_s=0.2), (0.2, 0.0)),
            ("back", VelocityCommand(forward_m_s=-0.2), (-0.2, 0.0)),
            ("left", VelocityCommand(right_m_s=-0.2), (0.0, -0.2)),
            ("right", VelocityCommand(right_m_s=0.2), (0.0, 0.2)),
        )
        for name, command, expected in cases:
            with self.subTest(direction=name):
                chassis = FakeChassis()
                adapter = DjiRobotAdapter(conn_type="sta", proto_type="tcp")
                adapter._chassis = chassis
                adapter.drive(command, timeout_s=0.35)
                call = chassis.calls[-1]
                self.assertEqual(expected, (call["x"], call["y"]))
                self.assertEqual(0.0, call["z"])


if __name__ == "__main__":
    unittest.main()
