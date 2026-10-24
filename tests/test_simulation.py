"""Tests for the deterministic baseline/proposed simulation."""
from __future__ import annotations

from src.control.simulation import build_scenario, simulate_comparison, simulate_proposed
from src.traffic.intersection import Intersection


def test_scenario_is_deterministic(config, intersection):
    a = build_scenario(config, intersection, seed=42)
    b = build_scenario(config, intersection, seed=42)
    c = build_scenario(config, intersection, seed=43)
    assert a.to_dict() == b.to_dict()
    assert [v.arrival_time for v in a.vehicles] == [v.arrival_time for v in b.vehicles]
    assert [v.arrival_time for v in a.vehicles] != [v.arrival_time for v in c.vehicles]


def test_comparison_uses_same_arrivals_and_serves_evs(config, intersection):
    result = simulate_comparison(config, intersection, seed=42)
    baseline = result["baseline"]
    proposed = result["proposed"]
    assert baseline["emergency_total"] == proposed["emergency_total"] == 5
    assert baseline["emergency_served"] == 5
    assert proposed["emergency_served"] == 5
    assert proposed["emergency_waiting_time_s"] is not None
    assert baseline["emergency_waiting_time_s"] is not None
    # Emergency priority must not make emergency vehicles wait longer than the
    # fixed-time baseline on this deterministic scenario. Normal traffic may be
    # traded off against that reduction.
    assert proposed["emergency_waiting_time_s"] <= baseline["emergency_waiting_time_s"]
    assert proposed["priority_events"] >= 0


def test_priority_event_fields(config, intersection):
    result = simulate_comparison(config, intersection, seed=7)
    for event in result["priority_events"]:
        assert event["lane"] in intersection.lanes
        assert event["requested_time"] <= event["green_time"]
        assert event["waiting_time_s"] >= 0
        assert event["baseline_wait_s"] is not None
        assert event["priority_duration_s"] >= 0


def test_proposed_controller_never_has_conflicting_greens(config, intersection):
    # simulate_proposed would raise AssertionError from the controller's own
    # validation if a conflicting state were ever applied. Run a compact case.
    scenario = build_scenario(config, intersection, seed=11)
    scenario.duration_s = min(scenario.duration_s, 120.0)
    summary, _, events = simulate_proposed(config, intersection, scenario, dt=0.25)
    assert summary["emergency_served"] >= 0
    for event in events:
        assert event["lane"] in intersection.lanes
