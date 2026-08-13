from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class HandSample:
    hand_id: int
    is_left: bool
    visible_time_s: float
    palm_x_mm: float
    palm_y_mm: float
    palm_z_mm: float
    velocity_x_mm_s: float
    velocity_y_mm_s: float
    velocity_z_mm_s: float
    direction_x: float
    direction_y: float
    direction_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    pinch_strength: float
    grab_strength: float
    pinch_distance_mm: float

    @property
    def handedness(self) -> str:
        return "left" if self.is_left else "right"

    @property
    def yaw_degrees(self) -> float:
        """Palm pointing yaw; zero points away from a flat, upward-facing sensor."""
        return math.degrees(math.atan2(self.direction_x, -self.direction_z))


@dataclass(frozen=True)
class FrameSample:
    arrival_time_s: float
    frame_id: int
    sensor_timestamp_us: int
    framerate: float
    total_hand_count: int
    hands: Tuple[HandSample, ...]


@dataclass(frozen=True)
class VelocityCommand:
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    clockwise_deg_s: float = 0.0

    @classmethod
    def stopped(cls) -> "VelocityCommand":
        return cls()

    def is_stopped(self, epsilon: float = 1e-4) -> bool:
        return (
            abs(self.forward_m_s) <= epsilon
            and abs(self.right_m_s) <= epsilon
            and abs(self.clockwise_deg_s) <= epsilon
        )


@dataclass(frozen=True)
class GestureDecision:
    state: str
    reason: str
    command: VelocityCommand
    hand: HandSample = None
