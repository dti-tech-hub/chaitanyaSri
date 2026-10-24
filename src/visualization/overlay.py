"""OpenCV overlay for live detection and simulated signal status."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from src.detection.tracker import Track
from src.traffic.density import LaneDensity
from src.traffic.intersection import Intersection
from src.utils.config import AppConfig

EMERGENCY_COLORS = {
    "ambulance": (40, 40, 240),
    "fire_engine": (0, 90, 255),
    "police_vehicle": (255, 120, 0),
    "normal_vehicle": (0, 190, 90),
    "unknown": (160, 160, 160),
}
CLASS_LABELS = {
    "ambulance": "AMBULANCE",
    "fire_engine": "FIRE ENGINE",
    "police_vehicle": "POLICE VEHICLE",
    "normal_vehicle": "VEHICLE",
    "unknown": "UNKNOWN",
}


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class OverlayRenderer:
    def __init__(self, config: AppConfig, intersection: Intersection, show_lanes: bool = True) -> None:
        self.config = config
        self.intersection = intersection
        self.show_lanes = show_lanes
        self.emergency_classes = set(config.emergency_classes)

    def _lane_polygons(self, frame: np.ndarray, lane_states: Dict[str, str], active_lane: Optional[str]) -> None:
        if not self.show_lanes:
            return
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for lane in self.intersection.lanes.values():
            poly = lane.scaled_polygon(w, h).astype(np.int32)
            state = lane_states.get(lane.name, "RED")
            if lane.name == active_lane:
                color = (0, 220, 0)
                alpha = 0.25
            elif state == "GREEN":
                color = (0, 180, 0)
                alpha = 0.15
            elif state == "YELLOW":
                color = (0, 210, 230)
                alpha = 0.12
            else:
                color = (240, 140, 40)
                alpha = 0.07
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            x, y = poly.mean(axis=0).astype(int)
            _text(frame, lane.display, (int(x) - 28, int(y)), 0.55, (255, 255, 255), 1)
            cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_track(frame: np.ndarray, track: Track, color: tuple[int, int, int], priority: Optional[str]) -> None:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        thickness = 3 if track.is_emergency else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if track.is_emergency:
            # Pulsing corner accents to make the emergency vehicle obvious.
            length = max(10, min(28, (x2 - x1) // 5))
            for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(frame, (px, py), (px + dx * length, py), color, 4, cv2.LINE_AA)
                cv2.line(frame, (px, py), (px, py + dy * length), color, 4, cv2.LINE_AA)
        label = f"{CLASS_LABELS.get(track.class_name, track.class_name.upper())} {track.confidence:.2f}"
        lane = (track.lane or "UNKNOWN").upper()
        detail = f"ID {track.track_id} | LANE {lane}"
        if priority:
            detail += f" | {priority}"
        y = max(14, y1 - 22)
        _text(frame, label, (x1, y), 0.48, color, 2)
        _text(frame, detail, (x1, y + 18), 0.42, (245, 245, 245), 1)

    def _draw_hud(
        self,
        frame: np.ndarray,
        signal_snapshot: Dict[str, object],
        priority: Dict[str, object],
        metrics: Dict[str, object],
    ) -> None:
        h, w = frame.shape[:2]
        panel_w = min(430, int(w * 0.38))
        panel_h = 178
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (16, 20, 26), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        state = str(signal_snapshot.get("state", "NORMAL"))
        phase = str(signal_snapshot.get("phase", "-"))
        active_lane = signal_snapshot.get("priority_lane")
        priority_active = bool(signal_snapshot.get("priority_active"))
        colour = (0, 0, 255) if priority_active else ((0, 220, 255) if "PREPAR" in state else (0, 230, 80))
        _text(frame, "SIMULATED SIGNAL CONTROLLER", (22, 34), 0.62, (230, 230, 230), 2)
        _text(frame, f"STATE: {state}", (22, 56), 0.52, colour, 2)
        _text(frame, f"PHASE: {phase}", (22, 78), 0.46, (200, 210, 220), 1)
        if active_lane:
            _text(frame, f"PRIORITY LANE: {str(active_lane).upper()}", (22, 100), 0.50, (0, 255, 255), 2)
        else:
            _text(frame, "PRIORITY: STANDBY", (22, 100), 0.50, (180, 190, 200), 1)
        fps = float(metrics.get("fps", 0.0) or 0.0)
        infer = float(metrics.get("inference_ms", 0.0) or 0.0)
        total = float(metrics.get("total_ms", 0.0) or 0.0)
        _text(frame, f"FPS {fps:5.1f}  |  YOLO {infer:5.1f} ms  |  FRAME {total:5.1f} ms", (22, 124), 0.46, (230, 230, 230), 1)
        vehicles = int(metrics.get("vehicle_count", 0) or 0)
        emergencies = int(metrics.get("emergency_count", 0) or 0)
        _text(frame, f"VEHICLES {vehicles:2d}  |  EMERGENCY {emergencies:2d}", (22, 146), 0.46, (230, 230, 230), 1)
        msg = str(priority.get("message", ""))
        _text(frame, msg[:52], (22, 168), 0.40, (200, 210, 220), 1)

        # Signal head panel (top-right).
        lane_order = ["north", "east", "south", "west"]
        panel_w2, panel_h2 = 176, 160
        x0, y0 = w - panel_w2 - 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w2, y0 + panel_h2), (16, 20, 26), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        _text(frame, "APPROACH SIGNALS", (x0 + 10, y0 + 22), 0.42, (225, 225, 225), 1)
        lane_states = signal_snapshot.get("lane_states", {}) or {}
        for i, lane in enumerate(lane_order):
            if lane not in self.intersection.lanes:
                continue
            state_l = str(lane_states.get(lane, "RED")).upper()
            cx = x0 + 22 + (i % 2) * 78
            cy = y0 + 52 + (i // 2) * 54
            circle = {"GREEN": (0, 240, 40), "YELLOW": (0, 215, 255), "RED": (240, 30, 30)}.get(state_l, (100, 100, 100))
            cv2.circle(frame, (cx + 8, cy), 10, circle, -1, cv2.LINE_AA)
            if active_lane == lane:
                cv2.circle(frame, (cx + 8, cy), 15, (0, 255, 255), 2, cv2.LINE_AA)
            _text(frame, self.intersection.lanes[lane].display[:5], (cx - 2, cy + 27), 0.34, (220, 220, 220), 1)

        # Warning banner if custom classes are missing.
        warning = metrics.get("model_warning")
        if warning:
            banner = str(warning)
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bx = max(10, (w - tw) // 2)
            cv2.rectangle(frame, (bx - 10, h - 48), (bx + tw + 10, h - 10), (0, 0, 150), -1)
            _text(frame, banner, (bx, h - 24), 0.55, (255, 255, 255), 2)

    def render(
        self,
        frame: np.ndarray,
        tracks: Iterable[Track],
        density: Dict[str, LaneDensity],
        signal_snapshot: Dict[str, object],
        priority_decision,
        metrics: Dict[str, object],
        detector_info: Optional[Dict[str, object]] = None,
        priority_track_id: Optional[int] = None,
    ) -> np.ndarray:
        out = frame.copy()
        tracks = list(tracks)
        lane_states = signal_snapshot.get("lane_states", {}) or {}
        active_lane = signal_snapshot.get("priority_lane")
        self._lane_polygons(out, lane_states, active_lane)
        for track in tracks:
            color = EMERGENCY_COLORS.get(track.class_name, EMERGENCY_COLORS["unknown"])
            priority = None
            if track.track_id == priority_track_id:
                priority = "PRIORITY ACTIVE" if signal_snapshot.get("priority_active") else "PRIORITY PREPARING"
            elif track.is_emergency and track.lane == active_lane:
                priority = "PRIORITY"
            self._draw_track(out, track, color, priority)
            if track.is_emergency and len(track.history) > 2:
                pts = np.array([(int(x), int(y)) for _, x, y in track.history[-14:]], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)

        vehicles = [t for t in tracks if t.class_name in set(self.config.vehicle_classes)]
        emergencies = [t for t in vehicles if t.is_emergency]
        metrics = dict(metrics)
        metrics.setdefault("vehicle_count", len(vehicles))
        metrics.setdefault("emergency_count", len(emergencies))
        if detector_info and not detector_info.get("emergency_capable", True):
            metrics.setdefault(
                "model_warning",
                "WARNING: custom emergency-vehicle YOLO model required. Priority disabled.",
            )
        self._draw_hud(out, signal_snapshot, priority_decision.__dict__ if hasattr(priority_decision, "__dict__") else {}, metrics)

        # Density strip (bottom-left), compact.
        if density:
            y = out.shape[0] - 24
            x = 14
            for lane_name, d in density.items():
                colour = {"LOW": (0, 220, 80), "MEDIUM": (0, 200, 255), "HIGH": (0, 60, 240)}.get(d.level, (180, 180, 180))
                label = f"{lane_name.upper()}:{d.count}"
                _text(out, label, (x, y), 0.38, colour, 1)
                x += 105
        _text(out, "SIMULATED PROTOTYPE - NOT CONNECTED TO REAL TRAFFIC SIGNALS", (14, out.shape[0] - 6), 0.36, (180, 180, 180), 1)
        return out
