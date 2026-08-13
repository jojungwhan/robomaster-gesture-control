"""Local status exchange between the gesture loop and hand overlay."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Optional

from .models import GestureDecision, VelocityCommand


DEFAULT_CONTROL_STATUS_PATH = (
    Path(__file__).resolve().parents[1] / "logs" / "gesture_control_status.json"
)


@dataclass(frozen=True)
class ControlStatusSnapshot:
    updated_at_epoch_s: float
    process_id: int
    live: bool
    transport: str
    state: str
    reason: str
    command: VelocityCommand


class ControlStatusPublisher:
    """Atomically publish the command actually submitted to the robot pump."""

    def __init__(
        self,
        path: Path = DEFAULT_CONTROL_STATUS_PATH,
        live: bool = False,
        transport: str = "sdk",
        interval_s: float = 0.05,
    ):
        self.path = Path(path)
        self.live = bool(live)
        self.transport = str(transport)
        self.interval_s = max(0.0, float(interval_s))
        self._last_publish_s = 0.0
        self._last_signature = None

    def publish(self, decision: GestureDecision, force: bool = False) -> bool:
        now_s = time.monotonic()
        directions = (
            1 if decision.command.forward_m_s >= 0.015 else
            -1 if decision.command.forward_m_s <= -0.015 else 0,
            1 if decision.command.right_m_s >= 0.015 else
            -1 if decision.command.right_m_s <= -0.015 else 0,
        )
        signature = (decision.state, directions)
        if (
            not force
            and signature == self._last_signature
            and now_s - self._last_publish_s < self.interval_s
        ):
            return False

        payload = {
            "schema": 1,
            "updated_at_epoch_s": time.time(),
            "process_id": os.getpid(),
            "live": self.live,
            "transport": self.transport,
            "state": decision.state,
            "reason": decision.reason,
            "command": {
                "forward_m_s": decision.command.forward_m_s,
                "right_m_s": decision.command.right_m_s,
                "clockwise_deg_s": decision.command.clockwise_deg_s,
            },
        }
        temporary = self.path.with_name(
            ".{}.{}.tmp".format(self.path.name, os.getpid())
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            os.replace(str(temporary), str(self.path))
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass
            # Visualization telemetry is best-effort and must never interrupt
            # or delay the fail-safe robot command path.
            return False
        self._last_publish_s = now_s
        self._last_signature = signature
        return True


class ControlStatusReader:
    """Read atomic controller snapshots without blocking the Tk event loop."""

    def __init__(self, path: Path = DEFAULT_CONTROL_STATUS_PATH):
        self.path = Path(path)
        self._last_mtime_ns = None  # type: Optional[int]
        self._snapshot = None  # type: Optional[ControlStatusSnapshot]

    def latest(self) -> Optional[ControlStatusSnapshot]:
        try:
            modified_ns = self.path.stat().st_mtime_ns
            if modified_ns == self._last_mtime_ns:
                return self._snapshot
            # utf-8-sig accepts both normal UTF-8 from Python and the BOM that
            # Windows PowerShell adds when a diagnostic file is written there.
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            command_data = data["command"]
            snapshot = ControlStatusSnapshot(
                updated_at_epoch_s=float(data["updated_at_epoch_s"]),
                process_id=int(data["process_id"]),
                live=bool(data["live"]),
                transport=str(data["transport"]),
                state=str(data["state"]),
                reason=str(data["reason"]),
                command=VelocityCommand(
                    forward_m_s=float(command_data["forward_m_s"]),
                    right_m_s=float(command_data["right_m_s"]),
                    clockwise_deg_s=float(command_data["clockwise_deg_s"]),
                ),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return self._snapshot

        self._last_mtime_ns = modified_ns
        self._snapshot = snapshot
        return snapshot
