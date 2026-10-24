"""Lane/direction identification using configurable polygons (never hard-coded)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional

import numpy as np

from src.detection.tracker import Track
from src.traffic.intersection import Intersection
from src.utils.config import AppConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger("lane_detector")


class LaneDetector:
    """Assign tracks to approaches using bounding-box centres and ROI polygons."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.intersection = Intersection(config)
        self.smoothing_frames = int(config.get("lanes.smoothing_frames", 5))
        self.unknown_lane_policy = str(config.get("lanes.unknown_lane_policy", "no_priority"))
        self._history: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=max(1, self.smoothing_frames)))
        self._last_lane: Dict[int, str] = {}
        self._exit_announced: set[int] = set()

    @property
    def lane_names(self) -> list[str]:
        return self.intersection.lane_names

    @property
    def groups(self) -> Dict[str, list[str]]:
        return self.intersection.groups

    def reset(self) -> None:
        self._history.clear()
        self._last_lane.clear()
        self._exit_announced.clear()

    def assign_center(self, x: float, y: float, width: int, height: int) -> Optional[str]:
        return self.intersection.lane_at(float(x), float(y), width, height)

    def assign_track(self, track: Track, width: int, height: int, smooth: bool = True) -> Optional[str]:
        x, y = track.center
        lane = self.assign_center(x, y, width, height)
        chosen: Optional[str] = None
        if lane is not None:
            self._history[track.track_id].append(lane)
            hist = self._history[track.track_id]
            # Majority vote over recent observations; ties resolve to the newest.
            counts: Dict[str, int] = {}
            newest_index: Dict[str, int] = {}
            for index, item in enumerate(hist):
                counts[item] = counts.get(item, 0) + 1
                newest_index[item] = index
            chosen = max(counts, key=lambda k: (counts[k], newest_index[k]))
        elif smooth and track.track_id in self._last_lane and self.intersection.in_center(x, y, width, height):
            # Retain the last approach only while the vehicle is actually in the
            # central junction; never guess a lane from image left/right.
            chosen = self._last_lane[track.track_id]
        if chosen is not None:
            self._last_lane[track.track_id] = chosen
        track.lane = chosen
        if chosen is not None and chosen not in track.lane_history:
            track.lane_history.append(chosen)
        return chosen

    def assign_tracks(self, tracks: Iterable[Track], width: int, height: int) -> Dict[int, Optional[str]]:
        out: Dict[int, Optional[str]] = {}
        for track in tracks:
            out[track.track_id] = self.assign_track(track, width, height)
        return out

    def has_exited(self, track: Track, lane: Optional[str], width: int, height: int) -> bool:
        """True when the track has reached the far-side exit ROI for its approach."""
        if not lane:
            return False
        x, y = track.center
        if self.intersection.in_exit(lane, x, y, width, height):
            return True
        # If the approach is not visible on the far side, use the central zone
        # plus the track moving beyond the centre. This remains geometry-based.
        return False

    def lane_group(self, lane: Optional[str]) -> Optional[str]:
        return self.intersection.group_of(lane)

    def distance_to_center(self, x: float, y: float, width: int, height: int) -> float:
        return self.intersection.normalized_distance_to_center(x, y, width, height)

    def in_center(self, x: float, y: float, width: int, height: int) -> bool:
        return self.intersection.in_center(x, y, width, height)

    def conflicting(self, lane: Optional[str]) -> list[str]:
        if not lane:
            return []
        return self.intersection.conflicting(lane)

    def non_conflicting(self, lane: Optional[str]) -> list[str]:
        if not lane:
            return []
        return self.intersection.non_conflicting(lane)

    def describe(self) -> Dict[str, object]:
        return self.intersection.describe()
