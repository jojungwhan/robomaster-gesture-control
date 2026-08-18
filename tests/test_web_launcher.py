import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import web_launcher


class StatePidTests(unittest.TestCase):
    def test_reads_pid_from_bom_encoded_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            # PowerShell's Set-Content -Encoding UTF8 writes a BOM.
            state.write_text(
                "﻿" + json.dumps({"pid": 4321, "started_at": "now"}),
                encoding="utf-8",
            )
            with mock.patch.object(web_launcher, "STATE_PATH", state):
                self.assertEqual(4321, web_launcher._read_state_pid())

    def test_missing_or_malformed_state_is_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "nope.json"
            with mock.patch.object(web_launcher, "STATE_PATH", missing):
                self.assertIsNone(web_launcher._read_state_pid())
            bad = Path(temporary) / "bad.json"
            bad.write_text("not json", encoding="utf-8")
            with mock.patch.object(web_launcher, "STATE_PATH", bad):
                self.assertIsNone(web_launcher._read_state_pid())


class StatusTests(unittest.TestCase):
    def test_running_when_pid_present_and_alive(self):
        with mock.patch.object(web_launcher, "_read_state_pid", return_value=100), \
             mock.patch.object(web_launcher, "_pid_is_control_center", return_value=True):
            self.assertEqual({"running": True, "pid": 100}, web_launcher.control_center_status())

    def test_not_running_when_pid_dead(self):
        with mock.patch.object(web_launcher, "_read_state_pid", return_value=100), \
             mock.patch.object(web_launcher, "_pid_is_control_center", return_value=False):
            self.assertEqual({"running": False, "pid": None}, web_launcher.control_center_status())

    def test_not_running_when_no_state(self):
        with mock.patch.object(web_launcher, "_read_state_pid", return_value=None):
            self.assertEqual({"running": False, "pid": None}, web_launcher.control_center_status())


class LauncherPageAndArgsTests(unittest.TestCase):
    def test_default_binds_to_localhost_only(self):
        args = web_launcher.build_parser().parse_args([])
        self.assertEqual("127.0.0.1", args.host)
        self.assertEqual(8770, args.port)

    def test_page_has_controls_and_endpoints(self):
        for needle in ('id="start"', 'id="stop"', "/status", "/start", "/stop"):
            with self.subTest(needle=needle):
                self.assertIn(needle, web_launcher.PAGE)


if __name__ == "__main__":
    unittest.main()
