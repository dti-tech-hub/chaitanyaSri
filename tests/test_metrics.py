"""Tests for waiting-time, latency and density calculations."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.detection.tracker import Track
from src.evaluation.latency import LatencyRecorder, summarize_latencies
from src.evaluation.waiting_time import (
    compare_waiting_series,
    summarize_waiting_times,
    waiting_time,
    waiting_time_reduction,
)
from src.traffic.density import DensityEstimator
from src.traffic.lane_detector import LaneDetector


def _track(track_id=1, x=640, y=100, cls="ambulance", lane="north", emergency=True):
    return Track(
        track_id=track_id,
        class_name=cls,
        raw_class_name=cls,
        bbox=np.asarray([x - 30, y - 20, x + 30, y + 20], dtype=np.float32),
        confidence=0.9,
        first_seen=0.0,
        last_seen=1.0,
        lane=lane,
        is_emergency=emergency,
    )


def test_waiting_time_never_invents_values():
    assert waiting_time(10.0, 14.5) == pytest.approx(4.5)
    assert waiting_time(10.0, 9.0) == 0.0
    assert waiting_time(None, 10.0) is None
    assert waiting_time(10.0, None) is None


def test_waiting_summary_and_reduction():
    summary = summarize_waiting_times([3.0, None, 5.0, 4.0])
    assert summary["count"] == 3
    assert summary["mean_s"] == pytest.approx(4.0)
    assert summary["median_s"] == pytest.approx(4.0)
    assert summary["max_s"] == pytest.approx(5.0)
    reduction = waiting_time_reduction(10.0, 4.0)
    assert reduction == {"reduction_s": 6.0, "reduction_pct": 60.0}
    assert waiting_time_reduction(None, 4.0) is None


def test_comparison_series():
    result = compare_waiting_series([10, 20, 30], [2, 4, 6])
    assert result["baseline_mean_s"] == pytest.approx(20.0)
    assert result["proposed_mean_s"] == pytest.approx(4.0)
    assert result["reduction_pct"] == pytest.approx(80.0)
    empty = compare_waiting_series([], [1])
    assert empty["reduction_s"] is None


def test_latency_stats_and_stage_recorder():
    recorder = LatencyRecorder()
    with recorder.stage("detection"):
        time.sleep(0.01)
    assert recorder.stages["detection"] > 0
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats is not None
    assert stats.count == 5
    assert stats.mean_ms == pytest.approx(22.0)
    assert stats.p95_ms >= stats.median_ms
    assert summarize_latencies([]) is None


def test_density_counts_and_levels(config):
    detector = LaneDetector(config)
    density = DensityEstimator(config, detector)
    tracks = [_track(1, 640, 100, "normal_vehicle", "north", False)]
    for i in range(1, 6):
        tracks.append(_track(i + 1, 640, 120, "normal_vehicle", "north", False))
    state = density.update(tracks, 1280, 720)
    assert state["north"].count == 6
    assert state["north"].level == "MEDIUM"
    assert state["east"].count == 0
    # Occupancy is actual summed box area / ROI area, bounded to [0, 1].
    assert 0 <= state["north"].occupancy <= 1
    assert density.summary()["total_vehicles"] == 6


def test_density_high_level(config):
    detector = LaneDetector(config)
    density = DensityEstimator(config, detector)
    tracks = [_track(i, 640, 100, "normal_vehicle", "north", False) for i in range(1, 12)]
    state = density.update(tracks, 1280, 720)
    assert state["north"].level == "HIGH"
