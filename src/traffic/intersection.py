"""Intersection geometry and conflict model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.config import AppConfig, ConfigError


@dataclass(frozen=True)
class Lane:
    name: str
    display: str
    group: str
    polygon: np.ndarray                 # normalised (N, 2)
    exit_polygon: Optional[np.ndarray]  # normalised (N, 2)

    def scaled_polygon(self, width: int, height: int) -> np.ndarray:
        return self.polygon * np.array([width, height], dtype=np.float32)

    def scaled_exit(self, width: int, height: int) -> Optional[np.ndarray]:
        if self.exit_polygon is None:
            return None
        return self.exit_polygon * np.array([width, height], dtype=np.float32)

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        import cv2

        poly = self.scaled_polygon(width, height)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def contains_exit(self, x: float, y: float, width: int, height: int) -> bool:
        import cv2

        poly = self.scaled_exit(width, height)
        if poly is None:
            return False
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def area_pixels(self, width: int, height: int) -> float:
        import cv2

        return float(abs(cv2.contourArea(self.scaled_polygon(width, height))))


class Intersection:
    """Four-approach intersection with configurable lanes and exit zones."""

    def __init__(self, config: AppConfig) -> None:
        spec = config.intersection.get("intersection", {}) or {}
        self.width = int(spec.get("width", 1280))
        self.height = int(spec.get("height", 720))
        self.coordinate_mode = str(spec.get("coordinate_mode", "normalized"))
        self.center = tuple(float(v) for v in spec.get("center", [0.5, 0.5]))
        center_zone = spec.get("center_zone")
        self.center_zone = np.asarray(center_zone, dtype=np.float32) if center_zone else None
        self.distance_scale = float(spec.get("approach_distance_scale", 0.707))
        lanes_cfg = config.intersection.get("lanes", {}) or {}
        exits_cfg = config.intersection.get("exit_zones", {}) or {}
        self.lanes: Dict[str, Lane] = {}
        for name, lane_spec in lanes_cfg.items():
            poly = np.asarray(lane_spec.get("polygon", []), dtype=np.float32)
            exit_poly = exits_cfg.get(name)
            self.lanes[str(name)] = Lane(
                name=str(name),
                display=str(lane_spec.get("display", name.upper())),
                group=str(lane_spec.get("group", name)),
                polygon=poly,
                exit_polygon=np.asarray(exit_poly, dtype=np.float32) if exit_poly else None,
            )
        self.groups: Dict[str, List[str]] = {}
        for lane in self.lanes.values():
            self.groups.setdefault(lane.group, []).append(lane.name)
        if len(self.groups) < 2:
            raise ConfigError("Intersection must have at least two conflict groups")

    @property
    def lane_names(self) -> List[str]:
        return list(self.lanes.keys())

    @property
    def group_names(self) -> List[str]:
        return list(self.groups.keys())

    def lane(self, name: str | None) -> Optional[Lane]:
        return self.lanes.get(str(name)) if name else None

    def display(self, name: str | None) -> str:
        lane = self.lane(name)
        return lane.display if lane else "UNKNOWN"

    def group_of(self, lane: str | None) -> Optional[str]:
        lane_obj = self.lane(lane)
        return lane_obj.group if lane_obj else None

    def conflicting(self, lane: str) -> List[str]:
        group = self.group_of(lane)
        if group is None:
            return []
        return [n for n in self.lane_names if self.lanes[n].group != group]

    def non_conflicting(self, lane: str) -> List[str]:
        group = self.group_of(lane)
        if group is None:
            return []
        return [n for n in self.lane_names if self.lanes[n].group == group]

    def conflicts(self, lane_a: str | None, lane_b: str | None) -> bool:
        ga = self.group_of(lane_a)
        gb = self.group_of(lane_b)
        return bool(ga and gb and ga != gb)

    def lane_at(self, x: float, y: float, width: int | None = None, height: int | None = None) -> Optional[str]:
        width = int(width or self.width)
        height = int(height or self.height)
        for name, lane in self.lanes.items():
            if lane.contains(x, y, width, height):
                return name
        return None

    def in_exit(self, lane: str, x: float, y: float, width: int | None = None, height: int | None = None) -> bool:
        lane_obj = self.lane(lane)
        if lane_obj is None:
            return False
        return lane_obj.contains_exit(x, y, int(width or self.width), int(height or self.height))

    def in_center(self, x: float, y: float, width: int | None = None, height: int | None = None) -> bool:
        if self.center_zone is None:
            return False
        import cv2

        poly = self.center_zone * np.array([int(width or self.width), int(height or self.height)], dtype=np.float32)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def normalized_distance_to_center(self, x: float, y: float, width: int, height: int) -> float:
        cx, cy = self.center
        dx = (float(x) / max(width, 1)) - cx
        dy = (float(y) / max(height, 1)) - cy
        return float(np.hypot(dx, dy) / max(self.distance_scale, 1e-6))

    def conflicting_green_pairs(self, greens: Iterable[str]) -> List[Tuple[str, str]]:
        green = [g for g in greens if g in self.lanes]
        bad: List[Tuple[str, str]] = []
        for i, a in enumerate(green):
            for b in green[i + 1:]:
                if self.conflicts(a, b):
                    bad.append((a, b))
        return bad

    def validate_signal_state(self, states: Dict[str, str]) -> None:
        greens = [lane for lane, state in states.items() if str(state).upper() == "GREEN"]
        bad = self.conflicting_green_pairs(greens)
        if bad:
            raise AssertionError(f"Conflicting green signal state: {bad}")

    def describe(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "center": self.center,
            "groups": self.groups,
            "lanes": {
                name: {
                    "display": lane.display,
                    "group": lane.group,
                    "polygon": lane.polygon.tolist(),
                    "exit_polygon": lane.exit_polygon.tolist() if lane.exit_polygon is not None else None,
                }
                for name, lane in self.lanes.items()
            },
        }
