#!/usr/bin/env python3
"""Local web launcher for the RoboMaster Control Center.

A browser page cannot start a desktop program on its own, so this tiny HTTP
server does it: it serves a control page and exposes /start, /stop, and /status
endpoints that drive ``run_control_center.ps1``. It binds to localhost only, so
it is never exposed to the network.

    C:\\Program Files\\Python312\\python.exe web_launcher.py
    # then open http://127.0.0.1:8770  (opened automatically by default)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONTROL_CENTER_SCRIPT = PROJECT_ROOT / "run_control_center.ps1"
STATE_PATH = PROJECT_ROOT / "logs" / "leap_visualizer_state.json"


def _read_state_pid():
    """Return the PID the Control Center recorded on startup, or None."""
    try:
        # PowerShell's Set-Content -Encoding UTF8 writes a BOM; utf-8-sig eats it.
        data = json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
        return int(data["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _pid_is_control_center(pid: int) -> bool:
    """True when a live pythonw process owns the recorded PID (Windows)."""
    if sys.platform != "win32":
        # Best-effort on other platforms: existence check only.
        try:
            import os

            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "PID eq {}".format(pid), "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return False
    line = result.stdout.strip().lower()
    return ('"{}"'.format(pid) in line) and ("pythonw" in line)


def control_center_status() -> dict:
    """Report whether the Control Center is currently running."""
    pid = _read_state_pid()
    running = pid is not None and _pid_is_control_center(pid)
    return {"running": running, "pid": pid if running else None}


def _run_control_center(stop: bool) -> subprocess.CompletedProcess:
    arguments = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(CONTROL_CENTER_SCRIPT),
    ]
    if stop:
        arguments.append("-Stop")
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboMaster Control Center</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center;
    justify-content: center; font-family: "Segoe UI", system-ui, sans-serif;
    background: radial-gradient(1200px 600px at 50% -10%, #123047, #061019 60%);
    color: #E7F0F7;
  }
  .card {
    width: min(460px, 92vw); background: #0A1620; border: 1px solid #1E3A4F;
    border-radius: 16px; padding: 28px 26px 24px; box-shadow: 0 18px 50px rgba(0,0,0,.5);
  }
  h1 { font-size: 20px; margin: 0 0 2px; letter-spacing: .3px; }
  .sub { color: #7FA6BF; font-size: 13px; margin: 0 0 20px; }
  .status {
    display: flex; align-items: center; gap: 10px; padding: 12px 14px;
    background: #06111A; border: 1px solid #1E3A4F; border-radius: 10px;
    margin-bottom: 20px; font-size: 14px;
  }
  .dot { width: 12px; height: 12px; border-radius: 50%; background: #557; flex: none;
    box-shadow: 0 0 0 3px rgba(255,255,255,.03); transition: background .2s; }
  .dot.on { background: #35D07F; box-shadow: 0 0 10px #35D07F; }
  .dot.off { background: #E5533D; }
  .dot.wait { background: #E5B33D; }
  .status .pid { margin-left: auto; color: #6C8BA0; font-variant-numeric: tabular-nums; }
  .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  button {
    font-size: 15px; font-weight: 600; padding: 14px 10px; border-radius: 10px;
    border: 1px solid transparent; cursor: pointer; color: #06111A;
    transition: filter .15s, opacity .15s;
  }
  button:disabled { opacity: .45; cursor: default; }
  button:not(:disabled):hover { filter: brightness(1.08); }
  .start { background: #35D07F; }
  .stop { background: #E5533D; color: #fff; }
  .hint { margin: 18px 0 0; font-size: 12px; line-height: 1.5; color: #6C8BA0; }
  .hint b { color: #9FC4DB; font-weight: 600; }
  .msg { margin-top: 14px; font-size: 12.5px; min-height: 16px; color: #7FA6BF; }
</style>
</head>
<body>
  <div class="card">
    <h1>RoboMaster Control Center</h1>
    <p class="sub">Leap-motion hand control &middot; voice control &middot; object detection</p>
    <div class="status">
      <span id="dot" class="dot wait"></span>
      <span id="state">Checking&hellip;</span>
      <span id="pid" class="pid"></span>
    </div>
    <div class="buttons">
      <button id="start" class="start" disabled>Open Control Center</button>
      <button id="stop" class="stop" disabled>Stop</button>
    </div>
    <div id="msg" class="msg"></div>
    <p class="hint">
      Opens the desktop control window. Inside it: <b>HAND CONTROL</b> or
      <b>VOICE CONTROL</b> to drive, and object detection with
      <b>&ldquo;what do you see&rdquo;</b> / <b>&ldquo;do you see a person?&rdquo;</b>.
      Use <b>DRY RUN</b> to test without the S1.
    </p>
  </div>
<script>
  const $ = (id) => document.getElementById(id);
  let busy = false;

  function render(s) {
    const dot = $("dot"), state = $("state"), pid = $("pid");
    if (busy) return;
    if (s.running) {
      dot.className = "dot on"; state.textContent = "Running";
      pid.textContent = s.pid ? "PID " + s.pid : "";
      $("start").disabled = true; $("stop").disabled = false;
    } else {
      dot.className = "dot off"; state.textContent = "Stopped";
      pid.textContent = "";
      $("start").disabled = false; $("stop").disabled = true;
    }
  }

  async function refresh() {
    try { render(await (await fetch("/status")).json()); }
    catch (e) { $("state").textContent = "Launcher unreachable"; $("dot").className = "dot off"; }
  }

  async function act(path, label) {
    busy = true;
    $("start").disabled = true; $("stop").disabled = true;
    $("dot").className = "dot wait"; $("state").textContent = label;
    $("msg").textContent = "";
    try {
      const r = await (await fetch(path, { method: "POST" })).json();
      $("msg").textContent = (r.message || "").trim();
      busy = false; render(r);
    } catch (e) {
      busy = false; $("msg").textContent = "Request failed: " + e;
      refresh();
    }
  }

  $("start").addEventListener("click", () => act("/start", "Opening…"));
  $("stop").addEventListener("click", () => act("/stop", "Stopping…"));
  refresh();
  setInterval(refresh, 2500);
</script>
</body>
</html>
"""


class LauncherHandler(BaseHTTPRequestHandler):
    server_version = "RoboMasterLauncher/1.0"

    def log_message(self, *args):  # silence default per-request logging
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/status":
            self._send_json(control_center_status())
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self) -> None:
        if self.path not in ("/start", "/stop"):
            self._send_json({"error": "not found"}, code=404)
            return
        stop = self.path == "/stop"
        try:
            result = _run_control_center(stop=stop)
            message = (result.stdout or result.stderr or "").strip()
        except OSError as exc:
            self._send_json(
                {**control_center_status(), "message": "Launch error: {}".format(exc)},
                code=500,
            )
            return
        payload = control_center_status()
        payload["message"] = message.splitlines()[-1] if message else ""
        self._send_json(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboMaster Control Center web launcher")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), LauncherHandler)
    url = "http://{}:{}".format(
        "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host, args.port
    )
    print("RoboMaster Control Center launcher running at {}".format(url), flush=True)
    print("Open that address in a browser. Press Ctrl+C to stop the launcher.", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLauncher stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
