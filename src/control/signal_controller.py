"""Simulated traffic-signal controller: normal cycle, priority and fail-safe.

The controller never touches real infrastructure. It publishes a simulated
signal state which is rendered by the dashboard and used by the priority engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.control.state_machine import (
    InvalidTransition,
    SignalState,
    SignalStateMachine,
    Transition,
)
from src.traffic.intersection import Intersection
from src.utils.config import AppConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger("signal_controller")

EventCallback = Callable[[str, Dict[str, object]], None]


def _phase(
    name: str,
    group: Optional[str],
    color: str,
    duration: float,
) -> Dict[str, object]:
    return {"name": name, "group": group, "color": color, "duration": float(duration)}


def build_normal_cycle(intersection: Intersection, config: AppConfig) -> Tuple[List[Dict[str, object]], List[str]]:
    green = float(config.get("signals.green_seconds", 20))
    yellow = float(config.get("signals.yellow_seconds", 3))
    all_red = float(config.get("signals.all_red_seconds", 1))
    groups = list(intersection.groups.keys())
    start = str(config.get("signals.start_phase", "north_south_green"))
    start_group = None
    for group in groups:
        if start.startswith(group):
            start_group = group
            break
    if start_group is None and "north_south" in groups:
        start_group = "north_south"
    if start_group is None:
        start_group = groups[0]
    idx = groups.index(start_group)
    ordered = groups[idx:] + groups[:idx]
    phases: List[Dict[str, object]] = []
    for group in ordered:
        phases.append(_phase(f"{group}_green", group, "GREEN", green))
        phases.append(_phase(f"{group}_yellow", group, "YELLOW", yellow))
        phases.append(_phase("all_red", None, "RED", all_red))
    return phases, ordered


class SignalController:
    """Stateful simulated signal with a guarded emergency-priority mode."""

    def __init__(
        self,
        config: AppConfig,
        intersection: Intersection,
        on_event: Optional[EventCallback] = None,
        start_time: float = 0.0,
    ) -> None:
        self.config = config
        self.intersection = intersection
        self.on_event = on_event
        self.normal_phases, self.group_order = build_normal_cycle(intersection, config)
        self.green_seconds = float(config.get("signals.green_seconds", 20))
        self.yellow_seconds = float(config.get("signals.yellow_seconds", 3))
        self.all_red_seconds = float(config.get("signals.all_red_seconds", 1))
        self.prepare_red_seconds = max(
            self.all_red_seconds,
            float(config.get("signals.priority_prepare_min_red_seconds", 1.5)),
        )
        self.cooldown_seconds = float(config.get("priority.cooldown_seconds", 10))
        self.max_priority_seconds = float(config.get("priority.max_priority_seconds", 45))
        self.state_machine = SignalStateMachine(on_transition=self._on_transition)
        self.phase: str = str(self.normal_phases[0]["name"])
        self._phase_index = 0
        self._phase_start = float(start_time)
        self._priority_lane: Optional[str] = None
        self._priority_started_at: Optional[float] = None
        self._priority_green_at: Optional[float] = None
        self._priority_requested_at: Optional[float] = None
        self._priority_prepare_duration = self.prepare_red_seconds
        self._priority_released_at: Optional[float] = None
        self._cooldown_until = -math.inf
        self._states: Dict[str, str] = {}
        self._last_now = float(start_time)
        self._apply_normal_phase(0, start_time, emit=False)
        self.fail_safe_reason: Optional[str] = None
        self.transition_count = 0

    # --------------------------------------------------------------- helpers
    @property
    def state(self) -> SignalState:
        return self.state_machine.state

    @property
    def priority_lane(self) -> Optional[str]:
        return self._priority_lane

    @property
    def lane_states(self) -> Dict[str, str]:
        return dict(self._states)

    def _emit(self, event_type: str, payload: Optional[Dict[str, object]] = None) -> None:
        if self.on_event:
            data = dict(payload or {})
            data.setdefault("signal_state", self.state.value)
            data.setdefault("phase", self.phase)
            data.setdefault("lane_states", dict(self._states))
            self.on_event(event_type, data)

    def _on_transition(self, transition: Transition) -> None:
        self.transition_count += 1
        self._emit(
            "signal_state_transition",
            {
                "from_state": transition.from_state,
                "to_state": transition.to_state,
                "reason": transition.reason,
                "sim_time": transition.timestamp,
            },
        )

    def _all_red(self) -> Dict[str, str]:
        return {lane: "RED" for lane in self.intersection.lane_names}

    def _group_green(self, group: str) -> Dict[str, str]:
        return {
            lane: ("GREEN" if self.intersection.group_of(lane) == group else "RED")
            for lane in self.intersection.lane_names
        }

    def _group_yellow(self, group: str) -> Dict[str, str]:
        return {
            lane: ("YELLOW" if self.intersection.group_of(lane) == group else "RED")
            for lane in self.intersection.lane_names
        }

    def _priority_states(self, lane: str, keep_group_green: bool = False) -> Dict[str, str]:
        out = self._all_red()
        group = self.intersection.group_of(lane)
        if keep_group_green and group:
            for name in self.intersection.groups.get(group, []):
                out[name] = "GREEN"
        else:
            out[lane] = "GREEN"
        return out

    def _set_phase(self, name: str, states: Dict[str, str], now: float, emit: bool = True) -> None:
        self.phase = name
        self._phase_start = float(now)
        self._states = dict(states)
        self._validate()
        if emit:
            self._emit("signal_phase", {"phase": name})

    def _validate(self) -> None:
        self.intersection.validate_signal_state(self._states)
        allowed = {"RED", "YELLOW", "GREEN"}
        missing = set(self.intersection.lane_names) - set(self._states)
        if missing:
            raise AssertionError(f"Signal state missing lanes: {missing}")
        bad = {k: v for k, v in self._states.items() if v not in allowed}
        if bad:
            raise AssertionError(f"Invalid signal colours: {bad}")

    def _apply_normal_phase(self, index: int, now: float, emit: bool = True) -> None:
        self._phase_index = int(index) % len(self.normal_phases)
        spec = self.normal_phases[self._phase_index]
        group = spec["group"]
        color = spec["color"]
        if color == "GREEN" and group:
            states = self._group_green(str(group))
        elif color == "YELLOW" and group:
            states = self._group_yellow(str(group))
        else:
            states = self._all_red()
        self._set_phase(str(spec["name"]), states, now, emit=emit)

    def _next_normal_index(self) -> int:
        return (self._phase_index + 1) % len(self.normal_phases)

    def _phase_elapsed(self, now: float) -> float:
        return max(0.0, float(now) - self._phase_start)

    def _current_duration(self) -> float:
        if self.state == SignalState.PRIORITY_PREPARING:
            return self._priority_prepare_duration
        if self.phase == "priority_clearance_all_red":
            return self.all_red_seconds
        if self.state in {SignalState.PRIORITY_ACTIVE, SignalState.FAIL_SAFE}:
            return math.inf
        if 0 <= self._phase_index < len(self.normal_phases):
            return float(self.normal_phases[self._phase_index]["duration"])
        return math.inf

    # ---------------------------------------------------------------- update
    def update(self, now: float) -> None:
        now = float(now)
        self._last_now = max(self._last_now, now)
        if self.state == SignalState.FAIL_SAFE:
            if any(v != "RED" for v in self._states.values()):
                self._states = self._all_red()
                self.phase = "fail_safe_all_red"
            return

        if self.state == SignalState.PRIORITY_PREPARING:
            if self._phase_elapsed(now) >= self._priority_prepare_duration:
                self._enter_priority_green(now)
            return

        if self.state == SignalState.PRIORITY_ACTIVE:
            if self._priority_green_at is None:
                self._priority_green_at = now
            if now - float(self._priority_started_at or now) >= self.max_priority_seconds:
                LOGGER.warning("Priority exceeded max duration; initiating safe release")
                self.release_priority(now, reason="max_priority_duration")
            return

        if self.state == SignalState.EMERGENCY_CLEARED:
            if self.phase == "priority_clearance_all_red" and self._phase_elapsed(now) >= self.all_red_seconds:
                self._finish_return_to_normal(now)
            return

        if self.state == SignalState.RETURNING_TO_NORMAL:
            self._finish_return_to_normal(now)
            return

        # Normal cycling. Advance phase boundaries one by one so a large time
        # step (tab restore, dropped frames, paused debugger) cannot illegally
        # skip or freeze the sequence.
        guard = 0
        while self.state == SignalState.NORMAL and guard < 100:
            guard += 1
            duration = self._current_duration()
            elapsed = self._phase_elapsed(now)
            if elapsed < duration:
                break
            boundary = self._phase_start + duration
            self._apply_normal_phase(self._next_normal_index(), boundary, emit=False)
        self._validate()

    def _enter_priority_green(self, now: float, immediate: bool = False) -> None:
        if self._priority_lane is None:
            return
        if self.state == SignalState.EMERGENCY_DETECTED:
            self.state_machine.transition(SignalState.PRIORITY_ACTIVE, now, "priority green")
        elif self.state == SignalState.PRIORITY_PREPARING:
            self.state_machine.transition(SignalState.PRIORITY_ACTIVE, now, "priority green after all-red")
        self._priority_green_at = float(now)
        # Keep same-group greens only if this was an immediate activation; a
        # prepared priority uses a single protected approach.
        states = self._priority_states(self._priority_lane, keep_group_green=immediate)
        self._set_phase(f"priority_green_{self._priority_lane}", states, now)
        self._emit(
            "priority_activated",
            {
                "lane": self._priority_lane,
                "track_id": None,
                "activated_at": now,
                "immediate": immediate,
                "waiting_time_s": max(0.0, now - float(self._priority_requested_at or now)),
            },
        )
        LOGGER.info("PRIORITY ACTIVE | lane=%s | waiting=%.2fs", self._priority_lane, now - float(self._priority_requested_at or now))

    def request_priority(self, lane: str, now: float, **context: object) -> bool:
        now = float(now)
        if lane not in self.intersection.lanes:
            LOGGER.error("Priority request for unknown lane %r rejected", lane)
            return False
        if self.state in {
            SignalState.PRIORITY_ACTIVE,
            SignalState.PRIORITY_PREPARING,
            SignalState.EMERGENCY_DETECTED,
            SignalState.EMERGENCY_CLEARED,
            SignalState.RETURNING_TO_NORMAL,
            SignalState.FAIL_SAFE,
        }:
            return False
        if now < self._cooldown_until:
            LOGGER.info("Priority request rejected: cooldown active for %.1fs", self._cooldown_until - now)
            self._emit("priority_rejected", {"lane": lane, "reason": "cooldown", "cooldown_remaining": self._cooldown_until - now})
            return False

        self._priority_lane = lane
        self._priority_requested_at = now
        self.state_machine.transition(SignalState.EMERGENCY_DETECTED, now, f"emergency in {lane}")
        lane_now_green = self._states.get(lane) == "GREEN"
        if lane_now_green and not self.intersection.conflicting_green_pairs(
            [name for name, colour in self._states.items() if colour == "GREEN"]
        ):
            # The approach already has a safe green: no need to force a red
            # transition. Activate priority immediately while keeping non-
            # conflicting same-group movements green.
            self._priority_started_at = now
            self._enter_priority_green(now, immediate=True)
            return True

        self._priority_started_at = now
        self.state_machine.transition(SignalState.PRIORITY_PREPARING, now, "safe all-red before priority green")
        self._priority_prepare_duration = self.prepare_red_seconds
        self._set_phase("priority_prepare_all_red", self._all_red(), now)
        self._emit(
            "priority_preparing",
            {
                "lane": lane,
                "requested_at": now,
                "prepare_seconds": self._priority_prepare_duration,
            },
        )
        LOGGER.info("PRIORITY PREPARING | lane=%s | all-red %.1fs", lane, self._priority_prepare_duration)
        return True

    def release_priority(self, now: float, reason: str = "emergency cleared") -> bool:
        now = float(now)
        if self.state not in {SignalState.PRIORITY_PREPARING, SignalState.PRIORITY_ACTIVE}:
            return False
        was_active = self.state == SignalState.PRIORITY_ACTIVE
        self.state_machine.transition(SignalState.EMERGENCY_CLEARED, now, reason)
        self._priority_released_at = now
        self._cooldown_until = max(self._cooldown_until, now + self.cooldown_seconds)
        self._set_phase("priority_clearance_all_red", self._all_red(), now)
        self._emit(
            "priority_released",
            {
                "lane": self._priority_lane,
                "reason": reason,
                "released_at": now,
                "duration_s": (now - self._priority_started_at) if self._priority_started_at else 0.0,
                "was_active": was_active,
            },
        )
        LOGGER.info("PRIORITY RELEASED | lane=%s | reason=%s", self._priority_lane, reason)
        return True

    def _finish_return_to_normal(self, now: float) -> None:
        if self.state == SignalState.EMERGENCY_CLEARED:
            self.state_machine.transition(SignalState.RETURNING_TO_NORMAL, now, "clearance all-red complete")
            # Resume normal sequencing with the group following the priority
            # group. The next controller update marks NORMAL, making the
            # restoration state observable and auditable.
            target_index = 0
            priority_group = self.intersection.group_of(self._priority_lane) if self._priority_lane else None
            if priority_group and priority_group in self.group_order:
                gi = self.group_order.index(priority_group)
                next_group = self.group_order[(gi + 1) % len(self.group_order)]
                for i, spec in enumerate(self.normal_phases):
                    if spec["name"] == f"{next_group}_green":
                        target_index = i
                        break
            self._apply_normal_phase(target_index, now, emit=False)
            self._emit("returning_to_normal", {"phase": self.phase, "cooldown_until": self._cooldown_until})
            return

        if self.state == SignalState.RETURNING_TO_NORMAL:
            self.state_machine.transition(SignalState.NORMAL, now, "normal sequence restored")
            self._priority_lane = None
            self._priority_green_at = None
            self._priority_requested_at = None
            self._emit("normal_restored", {"phase": self.phase, "cooldown_until": self._cooldown_until})
            LOGGER.info("NORMAL SEQUENCE RESTORED | phase=%s", self.phase)

    def force_fail_safe(self, now: float, reason: str, error: Optional[str] = None) -> None:
        self.fail_safe_reason = reason
        if self.state != SignalState.FAIL_SAFE:
            self.state_machine.transition(SignalState.FAIL_SAFE, now, reason)
        self.phase = "fail_safe_all_red"
        self._states = self._all_red()
        self._phase_start = float(now)
        self._priority_lane = None
        self._validate()
        self._emit("fail_safe", {"reason": reason, "error": error})
        LOGGER.error("FAIL-SAFE all-red: %s %s", reason, error or "")

    def reset(self, now: float = 0.0, start_phase: int = 0) -> None:
        self.state_machine.reset(now, reason="manual reset")
        self._priority_lane = None
        self._priority_started_at = None
        self._priority_green_at = None
        self._priority_requested_at = None
        self._priority_released_at = None
        self._cooldown_until = -math.inf
        self.fail_safe_reason = None
        self._apply_normal_phase(start_phase, now, emit=False)

    def can_activate(self, now: float) -> bool:
        return (
            self.state == SignalState.NORMAL
            and float(now) >= self._cooldown_until
            and self.state != SignalState.FAIL_SAFE
        )

    def signal_for(self, lane: str) -> str:
        return self._states.get(lane, "RED")

    def snapshot(self, now: Optional[float] = None) -> Dict[str, object]:
        now = float(now) if now is not None else self._last_now
        remaining = None
        duration = self._current_duration()
        if math.isfinite(duration):
            remaining = max(0.0, duration - self._phase_elapsed(now))
        return {
            "state": self.state.value,
            "phase": self.phase,
            "lane_states": dict(self._states),
            "priority_lane": self._priority_lane,
            "priority_active": self.state == SignalState.PRIORITY_ACTIVE,
            "priority_preparing": self.state == SignalState.PRIORITY_PREPARING,
            "priority_started_at": self._priority_started_at,
            "priority_green_at": self._priority_green_at,
            "phase_remaining_s": remaining,
            "cooldown_remaining_s": max(0.0, self._cooldown_until - now),
            "transition_count": self.transition_count,
            "fail_safe_reason": self.fail_safe_reason,
        }


class FixedTimeController:
    """Baseline fixed-time signal with no emergency priority.

    Used for an apples-to-apples baseline: given the same EV arrival times, it
    computes when the fixed cycle would serve that approach.
    """

    def __init__(self, config: AppConfig, intersection: Intersection, start_time: float = 0.0) -> None:
        self.config = config
        self.intersection = intersection
        self.phases, self.group_order = build_normal_cycle(intersection, config)
        self.cycle_duration = sum(float(p["duration"]) for p in self.phases)
        self.start_time = float(start_time)

    def _phase_at(self, now: float) -> Tuple[Dict[str, object], float, int]:
        total = (max(0.0, now - self.start_time)) % self.cycle_duration if self.cycle_duration > 0 else 0.0
        acc = 0.0
        for i, spec in enumerate(self.phases):
            dur = float(spec["duration"])
            if total < acc + dur or i == len(self.phases) - 1:
                return spec, total - acc, i
            acc += dur
        return self.phases[0], 0.0, 0

    def lane_states(self, now: float) -> Dict[str, str]:
        spec, _, _ = self._phase_at(now)
        group = spec.get("group")
        color = str(spec.get("color"))
        if group and color == "GREEN":
            return {lane: ("GREEN" if self.intersection.group_of(lane) == group else "RED") for lane in self.intersection.lane_names}
        if group and color == "YELLOW":
            return {lane: ("YELLOW" if self.intersection.group_of(lane) == group else "RED") for lane in self.intersection.lane_names}
        return {lane: "RED" for lane in self.intersection.lane_names}

    def snapshot(self, now: float) -> Dict[str, object]:
        spec, elapsed, index = self._phase_at(now)
        return {
            "state": "BASELINE_FIXED_TIME",
            "phase": str(spec["name"]),
            "lane_states": self.lane_states(now),
            "priority_active": False,
            "phase_elapsed_s": elapsed,
            "phase_remaining_s": max(0.0, float(spec["duration"]) - elapsed),
            "cycle_s": self.cycle_duration,
        }

    def next_green_time(self, lane: str, now: float) -> float:
        """Absolute earliest time >= now at which ``lane`` has a green light."""
        group = self.intersection.group_of(lane)
        if group is None:
            return float("inf")
        target_index = None
        for i, spec in enumerate(self.phases):
            if spec.get("group") == group and spec.get("color") == "GREEN":
                target_index = i
                break
        if target_index is None:
            return float("inf")
        spec, elapsed, current = self._phase_at(now)
        if current == target_index:
            return float(now)
        # Walk forward through phases until the target green starts.
        offset = 0.0
        idx = current
        guard = 0
        while guard < len(self.phases) * 3:
            guard += 1
            dur = float(self.phases[idx]["duration"])
            offset += max(0.0, dur - (elapsed if idx == current else 0.0))
            idx = (idx + 1) % len(self.phases)
            if idx == target_index:
                return float(now + offset)
            elapsed = 0.0
        return float("inf")

    def baseline_wait_seconds(self, lane: str, arrival_time: float) -> float:
        return max(0.0, self.next_green_time(lane, arrival_time) - float(arrival_time))
