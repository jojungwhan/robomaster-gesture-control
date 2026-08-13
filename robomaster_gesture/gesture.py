from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional, Tuple

from .models import FrameSample, GestureDecision, HandSample, VelocityCommand


class GestureState(str, Enum):
    WAITING = "WAITING"
    READY = "READY"
    ARMING = "ARMING"
    DRIVING = "DRIVING"


@dataclass(frozen=True)
class GestureConfig:
    preferred_hand: str = "right"
    minimum_visible_s: float = 0.30
    open_hold_s: float = 0.35
    arm_hold_s: float = 0.35
    pinch_engage: float = 0.80
    pinch_release: float = 0.55
    open_pinch_max: float = 0.35
    open_grab_max: float = 0.25
    fist_grab_min: float = 0.85
    pinch_grab_max: float = 0.70
    translation_deadzone_mm: float = 25.0
    translation_full_scale_mm: float = 120.0
    yaw_deadzone_deg: float = 10.0
    yaw_full_scale_deg: float = 45.0
    max_translation_m_s: float = 0.35
    max_yaw_deg_s: float = 35.0
    response_curve: float = 1.35
    smoothing_time_s: float = 0.12
    strafe_sign: float = 1.0
    yaw_sign: float = 1.0

    def validate(self) -> None:
        if self.preferred_hand not in ("right", "left", "any"):
            raise ValueError("preferred_hand must be right, left, or any")
        if not 0.0 <= self.pinch_release < self.pinch_engage <= 1.0:
            raise ValueError("pinch thresholds must have release < engage")
        if self.translation_full_scale_mm <= self.translation_deadzone_mm:
            raise ValueError("translation full scale must exceed its deadzone")
        if self.yaw_full_scale_deg <= self.yaw_deadzone_deg:
            raise ValueError("yaw full scale must exceed its deadzone")
        if self.max_translation_m_s <= 0.0 or self.max_yaw_deg_s <= 0.0:
            raise ValueError("speed limits must be positive")


class GestureController:
    """Stateful, fail-safe virtual joystick gesture recognizer."""

    def __init__(self, config: GestureConfig = None):
        self.config = config or GestureConfig()
        self.config.validate()
        self.state = GestureState.WAITING
        self._open_since = None  # type: Optional[float]
        self._arming_since = None  # type: Optional[float]
        self._active_hand_id = None  # type: Optional[int]
        self._anchor_samples = []  # type: List[Tuple[float, float, float]]
        self._anchor_x_mm = 0.0
        self._anchor_z_mm = 0.0
        self._anchor_yaw_deg = 0.0
        self._filtered = VelocityCommand.stopped()
        self._last_update_s = None  # type: Optional[float]

    def reset(self, reason: str = "reset") -> GestureDecision:
        self._clear_motion_state()
        return GestureDecision(
            state=self.state.value,
            reason=reason,
            command=VelocityCommand.stopped(),
        )

    def on_tracking_timeout(self, now_s: float) -> GestureDecision:
        self._last_update_s = now_s
        self._clear_motion_state()
        return GestureDecision(
            state=self.state.value,
            reason="tracking frame timeout - stopped",
            command=VelocityCommand.stopped(),
        )

    def update(self, frame: FrameSample) -> GestureDecision:
        now_s = frame.arrival_time_s

        if frame.total_hand_count == 0 or not frame.hands:
            return self._stop(now_s, "no hand - stopped")
        if frame.total_hand_count != 1 or len(frame.hands) != 1:
            return self._stop(now_s, "multiple hands - stopped")

        hand = self._select_hand(frame.hands)
        if hand is None:
            return self._stop(now_s, "preferred hand not visible - stopped")
        if hand.visible_time_s < self.config.minimum_visible_s:
            return self._stop(now_s, "hand settling - stopped", hand)
        if hand.grab_strength >= self.config.fist_grab_min:
            return self._stop(now_s, "fist emergency stop", hand)

        if self.state == GestureState.DRIVING:
            return self._update_driving(hand, now_s)
        if self.state == GestureState.ARMING:
            return self._update_arming(hand, now_s)
        if self.state == GestureState.READY:
            return self._update_ready(hand, now_s)
        return self._update_waiting(hand, now_s)

    def _select_hand(self, hands: Tuple[HandSample, ...]) -> Optional[HandSample]:
        if self.config.preferred_hand == "any":
            return max(hands, key=lambda item: item.visible_time_s)
        want_left = self.config.preferred_hand == "left"
        for hand in hands:
            if hand.is_left == want_left:
                return hand
        return None

    def _is_open(self, hand: HandSample) -> bool:
        return (
            hand.pinch_strength <= self.config.open_pinch_max
            and hand.grab_strength <= self.config.open_grab_max
        )

    def _is_engaged_pinch(self, hand: HandSample) -> bool:
        return (
            hand.pinch_strength >= self.config.pinch_engage
            and hand.grab_strength <= self.config.pinch_grab_max
        )

    def _update_waiting(self, hand: HandSample, now_s: float) -> GestureDecision:
        if not self._is_open(hand):
            self._open_since = None
            return self._zero("show an open hand to enable", hand, now_s)

        if self._open_since is None:
            self._open_since = now_s
        elapsed = now_s - self._open_since
        if elapsed >= self.config.open_hold_s:
            self.state = GestureState.READY
            return self._zero("ready - pinch and hold to engage", hand, now_s)

        remaining = max(0.0, self.config.open_hold_s - elapsed)
        return self._zero("hold open hand {:.1f}s".format(remaining), hand, now_s)

    def _update_ready(self, hand: HandSample, now_s: float) -> GestureDecision:
        if self._is_engaged_pinch(hand):
            self.state = GestureState.ARMING
            self._arming_since = now_s
            self._active_hand_id = hand.hand_id
            self._anchor_samples = []
            self._append_anchor_sample(hand)
            return self._zero("pinch detected - hold steady to engage", hand, now_s)
        if self._is_open(hand):
            return self._zero("ready - pinch and hold to engage", hand, now_s)

        # A natural pinch passes through strengths between the open and engaged
        # thresholds. READY is a non-moving state, so retaining it through that
        # transition is both safer and much easier to operate than forcing the
        # user to reopen the hand and start over.
        return self._zero("ready - complete pinch and hold", hand, now_s)

    def _update_arming(self, hand: HandSample, now_s: float) -> GestureDecision:
        if hand.hand_id != self._active_hand_id:
            return self._stop(now_s, "hand identity changed - stopped", hand)
        if hand.pinch_strength < self.config.pinch_release:
            return self._stop(now_s, "pinch released before engage", hand)
        if hand.grab_strength > self.config.pinch_grab_max:
            return self._stop(now_s, "pinch became ambiguous - stopped", hand)

        self._append_anchor_sample(hand)
        elapsed = now_s - self._arming_since
        if elapsed >= self.config.arm_hold_s and len(self._anchor_samples) >= 3:
            self._set_anchor_from_samples()
            self.state = GestureState.DRIVING
            self._filtered = VelocityCommand.stopped()
            self._last_update_s = now_s
            return self._zero("engaged - move hand while holding pinch", hand, now_s)

        remaining = max(0.0, self.config.arm_hold_s - elapsed)
        return self._zero("arming {:.1f}s", hand, now_s, remaining)

    def _update_driving(self, hand: HandSample, now_s: float) -> GestureDecision:
        if hand.hand_id != self._active_hand_id:
            return self._stop(now_s, "hand identity changed - stopped", hand)
        if hand.pinch_strength < self.config.pinch_release:
            return self._stop(now_s, "pinch released - stopped", hand)
        if hand.grab_strength > self.config.pinch_grab_max:
            return self._stop(now_s, "pose ambiguous - stopped", hand)

        forward_mm = -(hand.palm_z_mm - self._anchor_z_mm)
        right_mm = (hand.palm_x_mm - self._anchor_x_mm) * self.config.strafe_sign
        yaw_deg = self._wrap_degrees(hand.yaw_degrees - self._anchor_yaw_deg)
        yaw_deg *= self.config.yaw_sign

        raw = VelocityCommand(
            forward_m_s=self._scale_axis(
                forward_mm,
                self.config.translation_deadzone_mm,
                self.config.translation_full_scale_mm,
                self.config.max_translation_m_s,
            ),
            right_m_s=self._scale_axis(
                right_mm,
                self.config.translation_deadzone_mm,
                self.config.translation_full_scale_mm,
                self.config.max_translation_m_s,
            ),
            clockwise_deg_s=self._scale_axis(
                yaw_deg,
                self.config.yaw_deadzone_deg,
                self.config.yaw_full_scale_deg,
                self.config.max_yaw_deg_s,
            ),
        )

        dt = 1.0 / 60.0
        if self._last_update_s is not None:
            dt = max(0.001, min(0.20, now_s - self._last_update_s))
        self._last_update_s = now_s
        alpha = 1.0 - math.exp(-dt / self.config.smoothing_time_s)
        self._filtered = VelocityCommand(
            forward_m_s=self._lerp(self._filtered.forward_m_s, raw.forward_m_s, alpha),
            right_m_s=self._lerp(self._filtered.right_m_s, raw.right_m_s, alpha),
            clockwise_deg_s=self._lerp(
                self._filtered.clockwise_deg_s, raw.clockwise_deg_s, alpha
            ),
        )
        return GestureDecision(
            state=self.state.value,
            reason="pinch held",
            command=self._filtered,
            hand=hand,
        )

    def _append_anchor_sample(self, hand: HandSample) -> None:
        self._anchor_samples.append(
            (hand.palm_x_mm, hand.palm_z_mm, hand.yaw_degrees)
        )

    def _set_anchor_from_samples(self) -> None:
        self._anchor_x_mm = sum(item[0] for item in self._anchor_samples) / len(
            self._anchor_samples
        )
        self._anchor_z_mm = sum(item[1] for item in self._anchor_samples) / len(
            self._anchor_samples
        )
        sine = sum(math.sin(math.radians(item[2])) for item in self._anchor_samples)
        cosine = sum(math.cos(math.radians(item[2])) for item in self._anchor_samples)
        self._anchor_yaw_deg = math.degrees(math.atan2(sine, cosine))
        self._anchor_samples = []

    def _scale_axis(
        self, value: float, deadzone: float, full_scale: float, maximum: float
    ) -> float:
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        normalized = min(1.0, (magnitude - deadzone) / (full_scale - deadzone))
        curved = normalized ** self.config.response_curve
        return math.copysign(curved * maximum, value)

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return (value + 180.0) % 360.0 - 180.0

    @staticmethod
    def _lerp(start: float, end: float, amount: float) -> float:
        return start + (end - start) * amount

    def _stop(
        self, now_s: float, reason: str, hand: HandSample = None
    ) -> GestureDecision:
        was_open = hand is not None and self._is_open(hand)
        self._clear_motion_state()
        if was_open:
            self._open_since = now_s
        self._last_update_s = now_s
        return GestureDecision(
            state=self.state.value,
            reason=reason,
            command=VelocityCommand.stopped(),
            hand=hand,
        )

    def _zero(
        self,
        reason_template: str,
        hand: HandSample,
        now_s: float,
        *format_args: float
    ) -> GestureDecision:
        self._last_update_s = now_s
        reason = (
            reason_template.format(*format_args)
            if format_args
            else reason_template
        )
        return GestureDecision(
            state=self.state.value,
            reason=reason,
            command=VelocityCommand.stopped(),
            hand=hand,
        )

    def _clear_motion_state(self) -> None:
        self.state = GestureState.WAITING
        self._open_since = None
        self._arming_since = None
        self._active_hand_id = None
        self._anchor_samples = []
        self._filtered = VelocityCommand.stopped()
