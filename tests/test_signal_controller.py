"""Safety tests for the signal FSM and simulated controller."""
from __future__ import annotations

import pytest

from src.control.signal_controller import FixedTimeController, SignalController
from src.control.state_machine import InvalidTransition, SignalState, SignalStateMachine


def test_state_machine_rejects_illegal_jump():
    machine = SignalStateMachine()
    with pytest.raises(InvalidTransition):
        machine.transition(SignalState.PRIORITY_ACTIVE, 1.0, "illegal")
    machine.transition(SignalState.EMERGENCY_DETECTED, 1.0, "detected")
    machine.transition(SignalState.PRIORITY_ACTIVE, 2.0, "direct safe activation")
    assert machine.state == SignalState.PRIORITY_ACTIVE


def test_normal_cycle_sequence(controller: SignalController):
    assert controller.phase == "north_south_green"
    controller.update(20.0)
    assert controller.phase == "north_south_yellow"
    assert controller.signal_for("north") == "YELLOW"
    controller.update(23.0)
    assert controller.phase == "all_red"
    assert all(v == "RED" for v in controller.lane_states.values())
    controller.update(24.0)
    assert controller.phase == "east_west_green"
    assert controller.signal_for("east") == "GREEN"
    assert controller.signal_for("north") == "RED"


def test_large_time_step_advances_all_phases_without_conflict(controller: SignalController):
    controller.update(50.0)
    # 50 s = one full 48 s cycle plus 2 s into north-south green.
    assert controller.phase == "north_south_green"
    assert controller.signal_for("north") == "GREEN"
    controller.update(120.0)
    # 120 = 2*48 + 24 -> east-west green.
    assert controller.phase == "east_west_green"
    for _ in range(3):
        controller.intersection.validate_signal_state(controller.lane_states)


def test_priority_preparation_and_protected_green(controller: SignalController):
    # North-south green at t=0; request for east requires all-red preparation.
    assert controller.request_priority("east", 0.5)
    assert controller.state == SignalState.PRIORITY_PREPARING
    assert all(v == "RED" for v in controller.lane_states.values())
    controller.update(2.0)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    assert controller.signal_for("east") == "GREEN"
    assert controller.signal_for("north") == "RED"
    assert controller.phase == "priority_green_east"


def test_priority_release_restores_normal_with_clearance(controller: SignalController):
    assert controller.request_priority("east", 0.5)
    controller.update(2.0)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    assert controller.release_priority(4.0, "test release")
    assert controller.state == SignalState.EMERGENCY_CLEARED
    assert all(v == "RED" for v in controller.lane_states.values())
    assert controller.signal_for("east") == "RED"
    controller.update(5.0)
    assert controller.state == SignalState.RETURNING_TO_NORMAL
    controller.update(5.1)
    assert controller.state == SignalState.NORMAL
    # The next group after the protected east-west approach is north-south.
    assert controller.phase == "north_south_green"


def test_priority_is_rejected_during_cooldown(controller: SignalController):
    assert controller.request_priority("east", 0.5)
    controller.update(2.0)
    controller.release_priority(4.0, "test")
    controller.update(5.0)
    controller.update(6.0)
    assert controller.state == SignalState.NORMAL
    assert not controller.request_priority("west", 6.5)
    assert controller.state == SignalState.NORMAL


def test_fail_safe_forces_all_red_and_blocks_priority(controller: SignalController):
    controller.force_fail_safe(5.0, "unit test")
    assert controller.state == SignalState.FAIL_SAFE
    assert all(v == "RED" for v in controller.lane_states.values())
    assert not controller.request_priority("north", 6.0)
    controller.update(7.0)
    assert controller.state == SignalState.FAIL_SAFE


def test_fixed_time_baseline_next_green(baseline: FixedTimeController):
    assert baseline.baseline_wait_seconds("north", 0.0) == pytest.approx(0.0)
    assert baseline.baseline_wait_seconds("east", 0.0) == pytest.approx(24.0)
    # East green 24-44; at t=44 it is yellow and the next green is at 72.
    assert baseline.baseline_wait_seconds("east", 44.0) == pytest.approx(28.0)
    assert baseline.baseline_wait_seconds("north", 25.0) == pytest.approx(23.0)


def test_normal_restoration_never_creates_conflicting_greens(controller: SignalController):
    for lane in ("north", "south", "east", "west"):
        controller.reset(0.0)
        controller.update(2.0)
        controller.request_priority(lane, 2.5)
        controller.update(4.5)
        assert controller.state == SignalState.PRIORITY_ACTIVE
        controller.intersection.validate_signal_state(controller.lane_states)
        controller.release_priority(6.0, "test")
        controller.update(8.0)
        controller.update(9.0)
        controller.intersection.validate_signal_state(controller.lane_states)
