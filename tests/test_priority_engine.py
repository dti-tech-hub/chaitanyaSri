"""Tests for emergency confirmation, priority release and anti-false-trigger logic."""
from __future__ import annotations

from src.control.priority_engine import PriorityEngine
from src.control.signal_controller import FixedTimeController, SignalController
from src.detection.tracker import Track
from src.traffic.lane_detector import LaneDetector
from src.control.state_machine import SignalState


def _track(track_id, x, y, t, confidence=0.9, cls="ambulance"):
    track = Track(
        track_id=track_id,
        class_name=cls,
        raw_class_name=cls,
        bbox=(x - 30, y - 20, x + 30, y + 20),
        confidence=confidence,
        first_seen=0.0,
        last_seen=t,
        hit_count=5,
    )
    track.history = [(max(0.0, t - 0.2), x, y - 20), (t, x, y)]
    return track


def _engine(config, intersection, controller, baseline):
    detector = LaneDetector(config)
    return PriorityEngine(config, intersection, detector, controller, baseline_controller=baseline), detector


def test_confirmed_ev_requests_and_receives_priority(config, intersection):
    controller = SignalController(config, intersection)
    baseline = FixedTimeController(config, intersection)
    engine, detector = _engine(config, intersection, controller, baseline)
    for t, y in [(0.0, 100), (0.2, 130), (0.4, 160), (0.6, 190)]:
        track = _track(1, 640, y, t)
        lane_map = {1: detector.assign_track(track, 1280, 720)}
        controller.update(t)
        decision = engine.update([track], lane_map, t, 1280, 720, wall_time=t)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    assert decision.active_lane == "north"
    assert len(engine.events) == 1
    event = engine.events[0]
    assert event.green_time is not None
    assert event.waiting_time_s == 0.6
    assert event.arrival_time == 0.0
    assert event.baseline_wait_s is not None


def test_single_frame_or_low_confidence_ev_does_not_trigger(config, intersection):
    controller = SignalController(config, intersection)
    baseline = FixedTimeController(config, intersection)
    engine, detector = _engine(config, intersection, controller, baseline)
    # Two confident frames, then a low-confidence frame resets the streak.
    for t, y, conf in [(0.0, 100, 0.9), (0.2, 130, 0.9), (0.4, 160, 0.2), (0.6, 190, 0.9)]:
        track = _track(2, 640, y, t, confidence=conf)
        lane_map = {2: detector.assign_track(track, 1280, 720)}
        controller.update(t)
        engine.update([track], lane_map, t, 1280, 720, wall_time=t)
    assert controller.state == SignalState.NORMAL
    assert engine.events == []


def test_unknown_lane_never_requests_priority(config, intersection):
    controller = SignalController(config, intersection)
    baseline = FixedTimeController(config, intersection)
    engine, detector = _engine(config, intersection, controller, baseline)
    for t, y in [(0.0, 330), (0.2, 345), (0.4, 360), (0.6, 375)]:
        track = _track(3, 370, y, t)
        controller.update(t)
        engine.update([track], {3: None}, t, 1280, 720, wall_time=t)
    assert engine.events == []
    assert controller.state == SignalState.NORMAL


def test_exit_roi_holds_then_releases_and_restores(config, intersection):
    controller = SignalController(config, intersection)
    baseline = FixedTimeController(config, intersection)
    engine, detector = _engine(config, intersection, controller, baseline)
    for t, y in [(0.0, 100), (0.2, 130), (0.4, 160), (0.6, 190)]:
        track = _track(4, 640, y, t)
        lane_map = {4: detector.assign_track(track, 1280, 720)}
        controller.update(t)
        engine.update([track], lane_map, t, 1280, 720, wall_time=t)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    # EV reaches far-side exit; it must stay green for hold_after_exit seconds.
    for t in (0.8, 1.5, 2.7):
        track = _track(4, 640, 650, t)
        lane_map = {4: detector.assign_track(track, 1280, 720)}
        controller.update(t)
        engine.update([track], lane_map, t, 1280, 720, wall_time=t)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    track = _track(4, 640, 650, 2.85)
    controller.update(2.85)
    engine.update([track], {4: "north"}, 2.85, 1280, 720, wall_time=2.85)
    assert controller.state == SignalState.EMERGENCY_CLEARED
    controller.update(3.9)
    assert controller.state == SignalState.RETURNING_TO_NORMAL
    controller.update(4.0)
    assert controller.state == SignalState.NORMAL
    assert engine.events[0].status == "released"
    assert engine.events[0].priority_duration_s is not None


def test_lost_track_failsafe_release(config, intersection):
    controller = SignalController(config, intersection)
    baseline = FixedTimeController(config, intersection)
    engine, detector = _engine(config, intersection, controller, baseline)
    for t, y in [(0.0, 100), (0.2, 130), (0.4, 160), (0.6, 190)]:
        track = _track(5, 640, y, t)
        controller.update(t)
        engine.update([track], {5: detector.assign_track(track, 1280, 720)}, t, 1280, 720, wall_time=t)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    controller.update(1.0)
    engine.update([], {}, 1.0, 1280, 720, wall_time=1.0)
    assert controller.state == SignalState.PRIORITY_ACTIVE
    controller.update(4.2)
    engine.update([], {}, 4.2, 1280, 720, wall_time=4.2)
    assert controller.state == SignalState.EMERGENCY_CLEARED
    assert engine.events[0].status == "released"
