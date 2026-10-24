"""Deterministic four-approach signal simulation for baseline vs priority studies.

This is a *signal-level* traffic simulation: vehicles arrive according to a
seeded Poisson process, wait for a green light, and cross. No real-world
traffic-flow claims are made. It uses the same SignalController and fixed-time
baseline as the computer-vision pipeline.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.control.signal_controller import FixedTimeController, SignalController
from src.control.state_machine import SignalState
from src.traffic.intersection import Intersection
from src.utils.config import AppConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger("simulation")


@dataclass
class SimVehicle:
    vehicle_id: int
    lane: str
    arrival_time: float
    class_name: str
    is_emergency: bool
    crossing_time: float = 3.0
    confirmation_delay: float = 2.0
    confirmed_time: Optional[float] = None
    priority_requested_time: Optional[float] = None
    green_time: Optional[float] = None
    departure_time: Optional[float] = None
    wait_time: Optional[float] = None
    baseline_wait_time: Optional[float] = None
    status: str = "waiting"
    lane_was_green_on_arrival: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "lane": self.lane,
            "arrival_time": round(self.arrival_time, 3),
            "class_name": self.class_name,
            "is_emergency": self.is_emergency,
            "confirmed_time": None if self.confirmed_time is None else round(self.confirmed_time, 3),
            "priority_requested_time": None if self.priority_requested_time is None else round(self.priority_requested_time, 3),
            "green_time": None if self.green_time is None else round(self.green_time, 3),
            "departure_time": None if self.departure_time is None else round(self.departure_time, 3),
            "wait_time_s": None if self.wait_time is None else round(self.wait_time, 3),
            "baseline_wait_time_s": None if self.baseline_wait_time is None else round(self.baseline_wait_time, 3),
            "status": self.status,
        }


@dataclass
class Scenario:
    vehicles: List[SimVehicle]
    duration_s: float
    seed: int
    arrival_rates: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "seed": self.seed,
            "vehicles": len(self.vehicles),
            "emergency_vehicles": sum(1 for v in self.vehicles if v.is_emergency),
            "arrival_rates_per_hour": self.arrival_rates,
        }


def build_scenario(config: AppConfig, intersection: Intersection, seed: int | None = None) -> Scenario:
    sim_cfg = config.get("simulation", {}) or {}
    duration = float(sim_cfg.get("duration_s", 600.0))
    seed = int(seed if seed is not None else sim_cfg.get("seed", config.get("project.seed", 42)))
    rng = np.random.default_rng(seed)
    rates_cfg = sim_cfg.get("arrival_rate_per_lane_per_hour") or [420, 420, 380, 380]
    lane_names = intersection.lane_names
    if isinstance(rates_cfg, dict):
        rates = {lane: float(rates_cfg.get(lane, 400.0)) for lane in lane_names}
    else:
        rates = {lane: float(rates_cfg[i % len(rates_cfg)]) for i, lane in enumerate(lane_names)}
    normal_cross = float(sim_cfg.get("normal_crossing_s", 3.0))
    vehicles: List[SimVehicle] = []
    vid = 1
    for lane, rate in rates.items():
        if rate <= 0:
            continue
        mean_gap = 3600.0 / rate
        t = float(rng.exponential(mean_gap))
        while t < duration:
            vehicles.append(
                SimVehicle(
                    vehicle_id=vid,
                    lane=lane,
                    arrival_time=float(t),
                    class_name="normal_vehicle",
                    is_emergency=False,
                    crossing_time=normal_cross,
                )
            )
            vid += 1
            t += float(rng.exponential(mean_gap))
    schedule = sim_cfg.get("emergency_schedule") or []
    if schedule:
        for item in schedule:
            lane = str(item.get("lane", "north"))
            if lane not in intersection.lanes:
                continue
            arr = float(item.get("time_s", item.get("arrival_time", 60.0)))
            if arr > duration:
                continue
            vehicles.append(
                SimVehicle(
                    vehicle_id=vid,
                    lane=lane,
                    arrival_time=arr,
                    class_name=str(item.get("class", item.get("vehicle_class", "ambulance"))),
                    is_emergency=True,
                    crossing_time=float(sim_cfg.get("ev_crossing_s", 6.0)),
                    confirmation_delay=float(sim_cfg.get("ev_confirmation_s", 2.0)),
                )
            )
            vid += 1
    else:
        # Default demonstration schedule: three EV types in different approaches.
        ev_schedule = [
            (max(20.0, duration * 0.10), "north", "ambulance"),
            (max(40.0, duration * 0.35), "east", "fire_engine"),
            (max(60.0, duration * 0.60), "south", "police_vehicle"),
            (max(80.0, duration * 0.82), "west", "ambulance"),
        ]
        for t, lane, cls in ev_schedule:
            if lane not in intersection.lanes or t > duration:
                continue
            vehicles.append(
                SimVehicle(
                    vehicle_id=vid,
                    lane=lane,
                    arrival_time=float(t),
                    class_name=cls,
                    is_emergency=True,
                    crossing_time=float(sim_cfg.get("ev_crossing_s", 6.0)),
                    confirmation_delay=float(sim_cfg.get("ev_confirmation_s", 2.0)),
                )
            )
            vid += 1
    vehicles.sort(key=lambda v: (v.arrival_time, v.vehicle_id))
    return Scenario(vehicles, duration, seed, rates)


def _summarize_vehicles(vehicles: Iterable[SimVehicle], duration: float, label: str) -> Dict[str, Any]:
    vehicles = list(vehicles)
    ev = [v for v in vehicles if v.is_emergency]
    normal = [v for v in vehicles if not v.is_emergency]
    ev_waits = [v.wait_time for v in ev if v.wait_time is not None]
    normal_waits = [v.wait_time for v in normal if v.wait_time is not None]
    served = [v for v in vehicles if v.departure_time is not None]
    return {
        "controller": label,
        "duration_s": duration,
        "vehicles_total": len(vehicles),
        "vehicles_served": len(served),
        "emergency_total": len(ev),
        "emergency_served": sum(1 for v in ev if v.departure_time is not None),
        "emergency_waiting_time_s": float(np.mean(ev_waits)) if ev_waits else None,
        "emergency_max_wait_s": float(np.max(ev_waits)) if ev_waits else None,
        "normal_waiting_time_s": float(np.mean(normal_waits)) if normal_waits else None,
        "normal_max_wait_s": float(np.max(normal_waits)) if normal_waits else None,
        "throughput_vph": (len(served) / duration * 3600.0) if duration > 0 else None,
        "priority_events": 0,
    }


def simulate_baseline(config: AppConfig, intersection: Intersection, scenario: Scenario, dt: float = 0.25) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    baseline = FixedTimeController(config, intersection)
    vehicles = [SimVehicle(**{**v.__dict__}) for v in scenario.vehicles]
    by_lane: Dict[str, List[SimVehicle]] = {lane: [] for lane in intersection.lane_names}
    for v in vehicles:
        by_lane[v.lane].append(v)
    pending = sorted(vehicles, key=lambda v: v.arrival_time)
    arrived: List[SimVehicle] = []
    idx = 0
    now = 0.0
    crossing: List[SimVehicle] = []
    while now <= scenario.duration_s + 1e-9:
        while idx < len(pending) and pending[idx].arrival_time <= now:
            arrived.append(pending[idx])
            pending[idx].lane_was_green_on_arrival = baseline.lane_states(now).get(pending[idx].lane) == "GREEN"
            idx += 1
        states = baseline.lane_states(now)
        for v in arrived:
            if v.departure_time is not None or v.status == "crossing":
                continue
            if states.get(v.lane) == "GREEN":
                v.green_time = now
                v.wait_time = max(0.0, now - v.arrival_time)
                v.status = "crossing"
                v.departure_time = now
                crossing.append(v)
        # crossing lasts for crossing_time, then becomes served; throughput is
        # counted at green onset (a conservative crossing-start count).
        now += dt
    summary = _summarize_vehicles(vehicles, scenario.duration_s, "fixed_time_baseline")
    return summary, [v.to_dict() for v in vehicles]


def simulate_proposed(
    config: AppConfig,
    intersection: Intersection,
    scenario: Scenario,
    dt: float = 0.25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Reuse the exact same priority/signal controller as the vision pipeline.
    controller = SignalController(config, intersection)
    fixed = FixedTimeController(config, intersection)
    vehicles = [SimVehicle(**{**v.__dict__}) for v in scenario.vehicles]
    pending = sorted(vehicles, key=lambda v: v.arrival_time)
    arrived: List[SimVehicle] = []
    idx = 0
    now = 0.0
    priority_events: List[Dict[str, Any]] = []
    active_ev: Optional[SimVehicle] = None
    active_green_time: Optional[float] = None
    release_at: Optional[float] = None
    requested_ids: set[int] = set()
    baseline_wait_by_id: Dict[int, float] = {}
    max_wait = float(config.get("priority.cooldown_seconds", 10.0)) + float(config.get("signals.green_seconds", 20.0))
    while now <= scenario.duration_s + 1e-9:
        while idx < len(pending) and pending[idx].arrival_time <= now:
            v = pending[idx]
            v.lane_was_green_on_arrival = controller.signal_for(v.lane) == "GREEN"
            if v.is_emergency:
                v.confirmed_time = v.arrival_time + v.confirmation_delay
                baseline_wait_by_id[v.vehicle_id] = fixed.baseline_wait_seconds(v.lane, v.arrival_time)
                v.baseline_wait_time = baseline_wait_by_id[v.vehicle_id]
            arrived.append(v)
            idx += 1

        controller.update(now)
        # A priority request can only be made for the active EV after its
        # confirmation delay. Other EVs wait in arrival order (documented policy).
        if active_ev is None:
            candidates = [
                v for v in arrived
                if v.is_emergency and v.departure_time is None and v.status != "crossing"
                and v.confirmed_time is not None and now >= v.confirmed_time
                and v.vehicle_id not in requested_ids
            ]
            candidates.sort(key=lambda v: (v.arrival_time, v.vehicle_id))
            if candidates and controller.can_activate(now):
                ev = candidates[0]
                if controller.request_priority(ev.lane, now):
                    ev.priority_requested_time = now
                    requested_ids.add(ev.vehicle_id)
                    active_ev = ev
                    active_green_time = None
                    release_at = None
                    priority_events.append(
                        {
                            "event_id": len(priority_events) + 1,
                            "vehicle_id": ev.vehicle_id,
                            "vehicle_class": ev.class_name,
                            "lane": ev.lane,
                            "arrival_time": ev.arrival_time,
                            "requested_time": now,
                            "green_time": None,
                            "release_time": None,
                            "waiting_time_s": None,
                            "baseline_wait_s": ev.baseline_wait_time,
                            "priority_duration_s": None,
                        }
                    )

        # Normal vehicles and any already-green approaches cross when green.
        states = controller.lane_states
        for v in arrived:
            if v.departure_time is not None:
                continue
            if states.get(v.lane) == "GREEN":
                # An EV in the protected approach gets served first; normal
                # vehicles in the same approach may also cross (safe).
                v.green_time = now
                v.wait_time = max(0.0, now - v.arrival_time)
                v.departure_time = now
                v.status = "crossing"

        if active_ev is not None:
            states = controller.lane_states
            if states.get(active_ev.lane) == "GREEN" and active_green_time is None:
                active_green_time = now
                active_ev.green_time = now
                active_ev.wait_time = max(0.0, now - active_ev.arrival_time)
                for evt in reversed(priority_events):
                    if evt["vehicle_id"] == active_ev.vehicle_id:
                        evt["green_time"] = now
                        evt["waiting_time_s"] = active_ev.wait_time
                        break
                release_at = now + float(config.get("simulation.ev_crossing_s", 6.0))
            if release_at is not None and now >= release_at:
                controller.release_priority(now, "simulated emergency vehicle cleared")
                for evt in reversed(priority_events):
                    if evt["vehicle_id"] == active_ev.vehicle_id:
                        evt["release_time"] = now
                        evt["priority_duration_s"] = (now - active_green_time) if active_green_time else None
                        break
                active_ev.status = "priority_released"
                active_ev = None
                active_green_time = None
                release_at = None
        now += dt
    # Any EV that never received green is honestly marked unserved.
    for v in vehicles:
        if v.is_emergency and v.departure_time is None:
            v.status = "unserved"
            v.wait_time = None
    summary = _summarize_vehicles(vehicles, scenario.duration_s, "emergency_priority_proposed")
    summary["priority_events"] = len(priority_events)
    response_times = [
        e["green_time"] - e["requested_time"]
        for e in priority_events
        if e.get("green_time") is not None and e.get("requested_time") is not None
    ]
    summary["priority_response_time_s"] = float(np.mean(response_times)) if response_times else None
    return summary, [v.to_dict() for v in vehicles], priority_events


def simulate_comparison(
    config: AppConfig,
    intersection: Intersection,
    seed: int | None = None,
) -> Dict[str, Any]:
    sim_cfg = config.get("simulation", {}) or {}
    dt = float(sim_cfg.get("dt_s", 0.25))
    scenario = build_scenario(config, intersection, seed=seed)
    baseline, baseline_rows = simulate_baseline(config, intersection, scenario, dt=dt)
    proposed, proposed_rows, priority_events = simulate_proposed(config, intersection, scenario, dt=dt)
    baseline["emergency_total"] = scenario.to_dict()["emergency_vehicles"]
    proposed["emergency_total"] = scenario.to_dict()["emergency_vehicles"]
    return {
        "scenario": scenario.to_dict(),
        "baseline": baseline,
        "proposed": proposed,
        "priority_events": priority_events,
        "baseline_vehicles": baseline_rows,
        "proposed_vehicles": proposed_rows,
    }


def save_simulation_results(result: Dict[str, Any], directory: str | Path) -> Dict[str, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    (out / "simulation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    paths["summary"] = out / "simulation_summary.json"
    for key in ("baseline_vehicles", "proposed_vehicles", "priority_events"):
        rows = result.get(key, [])
        if not rows:
            continue
        path = out / f"{key}.csv"
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paths[key] = path
    return paths


def density_sweep(config: AppConfig, intersection: Intersection, seeds: Iterable[int] = (42, 43, 44)) -> List[Dict[str, Any]]:
    """Run the comparison at low/medium/high arrival rates (actual simulation)."""
    base_cfg = config.get("simulation.arrival_rate_per_lane_per_hour") or [420, 420, 380, 380]
    scenarios = {
        "low": [180, 180, 160, 160],
        "medium": list(base_cfg) if isinstance(base_cfg, list) else [420, 420, 380, 380],
        "high": [900, 900, 820, 820],
    }
    rows: List[Dict[str, Any]] = []
    for density, rates in scenarios.items():
        for seed in seeds:
            cfg = config.with_overrides(
                {
                    "simulation": {
                        "arrival_rate_per_lane_per_hour": rates,
                        "seed": int(seed),
                    }
                }
            )
            result = simulate_comparison(cfg, intersection, seed=int(seed))
            rows.append(
                {
                    "density": density,
                    "seed": seed,
                    "baseline_emergency_wait_s": result["baseline"]["emergency_waiting_time_s"],
                    "proposed_emergency_wait_s": result["proposed"]["emergency_waiting_time_s"],
                    "baseline_normal_wait_s": result["baseline"]["normal_waiting_time_s"],
                    "proposed_normal_wait_s": result["proposed"]["normal_waiting_time_s"],
                    "baseline_throughput_vph": result["baseline"]["throughput_vph"],
                    "proposed_throughput_vph": result["proposed"]["throughput_vph"],
                    "priority_events": result["proposed"]["priority_events"],
                }
            )
    return rows
