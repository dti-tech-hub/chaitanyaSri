"""Tests for lane smoothing and tracker behaviour."""
from __future__ import annotations

import numpy as np

from src.detection.detector import Detection
from src.detection.tracker import IOUTracker
from src.traffic.lane_detector import LaneDetector


def _det(bbox, cls="ambulance", conf=0.9):
    return Detection(
        bbox=np.asarray(bbox, dtype=np.float32),
        confidence=conf,
        class_id=0,
        raw_class_name=cls,
        class_name=cls,
        is_emergency=cls != "normal_vehicle",
    )


def test_iou_tracker_rejects_single_frame_detection_after_miss():
    tracker = IOUTracker(iou_threshold=0.3, max_age_frames=5, min_hits=2)
    first = tracker.update([_det([10, 10, 60, 60])], 0.0)
    assert len(first) == 1
    # A tracker must not present a vanished detection as currently visible.
    after_miss = tracker.update([], 0.1)
    assert after_miss == []
    # Short gaps are retained for re-identification.
    again = tracker.update([_det([12, 12, 62, 62])], 0.2)
    assert len(again) == 1
    assert again[0].track_id == first[0].track_id


def test_iou_tracker_class_aware():
    tracker = IOUTracker(iou_threshold=0.3, class_aware=True)
    a = tracker.update([_det([10, 10, 60, 60], cls="ambulance")], 0.0)
    b = tracker.update([_det([10, 10, 60, 60], cls="police_vehicle")], 0.1)
    assert len(a) == 1 and len(b) == 1
    assert a[0].track_id != b[0].track_id


def test_iou_tracker_expires_after_max_age():
    tracker = IOUTracker(iou_threshold=0.3, max_age_frames=2)
    tracker.update([_det([10, 10, 60, 60])], 0.0)
    for i in range(1, 5):
        tracker.update([], float(i))
    assert tracker.tracks == {}


def test_lane_smoothing_and_center_retention(config, intersection):
    detector = LaneDetector(config)
    track = type("T", (), {})()
    track.track_id = 7
    track.center = (640, 100)
    track.lane_history = []
    assert detector.assign_track(track, 1280, 720) == "north"
    # Move into the central junction; retain previous approach while inside.
    track.center = (640, 360)
    assert detector.assign_track(track, 1280, 720) == "north"
    # Unknown position outside all ROIs and centre must not be guessed.
    track.center = (10, 700)
    assert detector.assign_track(track, 1280, 720) is None


def test_lane_history_majority_ties_newest(config):
    detector = LaneDetector(config)
    track = type("T", (), {})()
    track.track_id = 11
    track.lane_history = []
    for x, y in [(640, 100), (640, 100), (640, 100), (640, 100), (640, 100)]:
        track.center = (x, y)
        detector.assign_track(track, 1280, 720)
    assert track.lane == "north"
