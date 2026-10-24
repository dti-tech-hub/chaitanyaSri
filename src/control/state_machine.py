"""Finite-state machine for the simulated traffic-signal controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional


class SignalState(str, Enum):
    NORMAL = "NORMAL"
    EMERGENCY_DETECTED = "EMERGENCY_DETECTED"
    PRIORITY_PREPARING = "PRIORITY_PREPARING"
    PRIORITY_ACTIVE = "PRIORITY_ACTIVE"
    EMERGENCY_CLEARED = "EMERGENCY_CLEARED"
    RETURNING_TO_NORMAL = "RETURNING_TO_NORMAL"
    FAIL_SAFE = "FAIL_SAFE"


# Explicitly allowed transitions. This prevents arbitrary jumps such as
# NORMAL -> PRIORITY_ACTIVE without a safe transition.
ALLOWED_TRANSITIONS: Dict[SignalState, set[SignalState]] = {
    SignalState.NORMAL: {SignalState.EMERGENCY_DETECTED, SignalState.FAIL_SAFE},
    SignalState.EMERGENCY_DETECTED: {
        SignalState.PRIORITY_PREPARING,
        SignalState.PRIORITY_ACTIVE,
        SignalState.FAIL_SAFE,
    },
    SignalState.PRIORITY_PREPARING: {
        SignalState.PRIORITY_ACTIVE,
        SignalState.EMERGENCY_CLEARED,
        SignalState.FAIL_SAFE,
    },
    SignalState.PRIORITY_ACTIVE: {
        SignalState.EMERGENCY_CLEARED,
        SignalState.FAIL_SAFE,
    },
    SignalState.EMERGENCY_CLEARED: {
        SignalState.RETURNING_TO_NORMAL,
        SignalState.FAIL_SAFE,
    },
    SignalState.RETURNING_TO_NORMAL: {SignalState.NORMAL, SignalState.FAIL_SAFE},
    SignalState.FAIL_SAFE: {SignalState.NORMAL},
}


@dataclass
class Transition:
    timestamp: float
    from_state: str
    to_state: str
    reason: str
    wall_time: str


class InvalidTransition(RuntimeError):
    pass


class SignalStateMachine:
    def __init__(
        self,
        on_transition: Optional[Callable[[Transition], None]] = None,
        initial: SignalState = SignalState.NORMAL,
    ) -> None:
        self.state = initial
        self.history: List[Transition] = []
        self.on_transition = on_transition

    def can_transition(self, target: SignalState) -> bool:
        return target in ALLOWED_TRANSITIONS.get(self.state, set())

    def transition(self, target: SignalState, timestamp: float, reason: str = "") -> Transition:
        target = SignalState(target)
        if not self.can_transition(target):
            raise InvalidTransition(
                f"Illegal signal-state transition {self.state.value} -> {target.value} ({reason})"
            )
        record = Transition(
            timestamp=float(timestamp),
            from_state=self.state.value,
            to_state=target.value,
            reason=reason,
            wall_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        self.history.append(record)
        self.state = target
        if self.on_transition:
            self.on_transition(record)
        return record

    def reset(self, timestamp: float = 0.0, reason: str = "reset") -> None:
        if self.state != SignalState.NORMAL:
            record = Transition(
                timestamp=float(timestamp),
                from_state=self.state.value,
                to_state=SignalState.NORMAL.value,
                reason=reason,
                wall_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )
            self.history.append(record)
            if self.on_transition:
                self.on_transition(record)
        self.state = SignalState.NORMAL

    def is_priority(self) -> bool:
        return self.state in {
            SignalState.EMERGENCY_DETECTED,
            SignalState.PRIORITY_PREPARING,
            SignalState.PRIORITY_ACTIVE,
        }
