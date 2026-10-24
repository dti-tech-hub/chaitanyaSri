"""Traffic-density estimation from tracked vehicles and lane ROIs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from src.detection.tracker import Track
from src.traffic.lane_detector import LaneDetector
from src.utils.config import AppConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger("density")


@dataclass
class LaneDensity:
    lane: str
    count: int = 0
    emergency_count: int = 0
    occupancy: float = 0.0
    level: str = "LOW"
    smoothed_count: float = 0.0
    smoothed_occupancy: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "lane": self.lane,
            "count": int(self.count),
            "emergency_count": int(self.emergency_count),
            "occupancy": round(float(self.occupancy), 4),
            "smoothed_count": round(float(self.smoothed_count), 2),
            "smoothed_occupancy": round(float(self.smoothed_occupancy), 4),
            "level": self.level,
        }


class DensityEstimator:
    """Counts vehicles per lane and classifies LOW / MEDIUM / HIGH.

    Thresholds are prototype parameters, configurable in config.yaml:
    0..low_max -> LOW, low_max+1..medium_max -> MEDIUM, above -> HIGH.
    Occupancy is also computed but the level is based on vehicle count; the
    occupancy thresholds are reported for transparency.
    """

    def __init__(self, config: AppConfig, lane_detector: LaneDetector, width: int = 1280, height: int = 720) -> None:
        self.config = config
        self.lane_detector = lane_detector
        self.width = int(width)
        self.height = int(height)
        self.enabled = bool(config.get("density.enabled", True))
        self.low_max = int(config.get("density.low_max", 3))
        self.medium_max = int(config.get("density.medium_max", 8))
        self.occ_low = float(config.get("density.occupancy_low_max", 0.15))
        self.occ_med = float(config.get("density.occupancy_medium_max", 0.40))
        self.alpha = float(config.get("density.smoothing_alpha", 0.3))
        self.vehicle_classes = set(config.get("density.vehicle_classes", config.vehicle_classes))
        self._smoothed_count: Dict[str, float] = {name: 0.0 for name in lane_detector.lane_names}
        self._smoothed_occ: Dict[str, float] = {name: 0.0 for name in lane_detector.lane_names}
        self.last_state: Dict[str, LaneDensity] = {
            name: LaneDensity(name) for name in lane_detector.lane_names
        }
        self.lane_areas = {
            name: max(1.0, lane_detector.intersection.lane(name).area_pixels(self.width, self.height))
            for name in lane_detector.lane_names
        }

    def configure_frame(self, width: int, height: int) -> None:
        if int(width) == self.width and int(height) == self.height:
            return
        self.width, self.height = int(width), int(height)
        self.lane_areas = {
            name: max(1.0, self.lane_detector.intersection.lane(name).area_pixels(self.width, self.height))
            for name in self.lane_detector.lane_names
        }

    def classify(self, count: int) -> str:
        if count <= self.low_max:
            return "LOW"
        if count <= self.medium_max:
            return "MEDIUM"
        return "HIGH"

    def update(self, tracks: Iterable[Track], width: int | None = None, height: int | None = None) -> Dict[str, LaneDensity]:
        width = int(width or self.width)
        height = int(height or self.height)
        self.configure_frame(width, height)
        counts: Dict[str, int] = {name: 0 for name in self.lane_detector.lane_names}
        emergencies: Dict[str, int] = {name: 0 for name in self.lane_detector.lane_names}
        areas: Dict[str, float] = {name: 0.0 for name in self.lane_detector.lane_names}
        if self.enabled:
            for track in tracks:
                if track.lost or track.lane not in counts:
                    continue
                if track.class_name not in self.vehicle_classes:
                    continue
                counts[track.lane] += 1
                if track.is_emergency:
                    emergencies[track.lane] += 1
                areas[track.lane] += float(track.bbox[2] - track.bbox[0]) * float(track.bbox[3] - track.bbox[1])
        state: Dict[str, LaneDensity] = {}
        for lane in self.lane_detector.lane_names:
            occupancy = min(1.0, areas[lane] / max(self.lane_areas.get(lane, 1.0), 1.0))
            self._smoothed_count[lane] = (1 - self.alpha) * self._smoothed_count.get(lane, 0.0) + self.alpha * counts[lane]
            self._smoothed_occ[lane] = (1 - self.alpha) * self._smoothed_occ.get(lane, 0.0) + self.alpha * occupancy
            state[lane] = LaneDensity(
                lane=lane,
                count=counts[lane],
                emergency_count=emergencies[lane],
                occupancy=occupancy,
                smoothed_count=self._smoothed_count[lane],
                smoothed_occupancy=self._smoothed_occ[lane],
                level=self.classify(counts[lane]),
            )
        self.last_state = state
        return state

    def summary(self) -> Dict[str, object]:
        counts = {lane: s.count for lane, s in self.last_state.items()}
        total = sum(counts.values())
        dominant = max(counts, key=counts.get) if counts else None
        return {
            "total_vehicles": total,
            "dominant_lane": dominant,
            "lanes": {lane: s.to_dict() for lane, s in self.last_state.items()},
        }

    def describe(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "low_max": self.low_max,
            "medium_max": self.medium_max,
            "occupancy_low_max": self.occ_low,
            "occupancy_medium_max": self.occ_med,
            "state": self.summary(),
        }
