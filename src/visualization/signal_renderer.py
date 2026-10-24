"""Top-down simulated intersection renderer for the dashboard and reports."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from src.traffic.intersection import Intersection


_COLOR = {
    "GREEN": (40, 220, 40),
    "YELLOW": (0, 210, 245),
    "RED": (230, 40, 40),
    "OFF": (55, 60, 66),
}


def _text(img, text, org, scale=0.5, color=(235, 235, 235), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_signal_scene(
    snapshot: Dict[str, object],
    intersection: Intersection,
    size: int = 560,
    title: str = "SIMULATED INTERSECTION",
) -> np.ndarray:
    """Draw a schematic four-way intersection with four three-lamp heads."""
    img = np.full((size, size, 3), 18, dtype=np.uint8)
    road = (42, 46, 52)
    lane_line = (200, 200, 200)
    centre = size // 2
    road_w = int(size * 0.30)
    margin = int(size * 0.03)
    # Road arms.
    cv2.rectangle(img, (centre - road_w // 2, 0), (centre + road_w // 2, size), road, -1)
    cv2.rectangle(img, (0, centre - road_w // 2), (size, centre + road_w // 2), road, -1)
    # Lane markings.
    for x in (centre - road_w // 4, centre + road_w // 4):
        for y0, y1 in ((margin, centre - road_w // 2), (centre + road_w // 2, size - margin)):
            cv2.line(img, (x, y0), (x, y1), lane_line, 1, cv2.LINE_AA)
    for y in (centre - road_w // 4, centre + road_w // 4):
        for x0, x1 in ((margin, centre - road_w // 2), (centre + road_w // 2, size - margin)):
            cv2.line(img, (x0, y), (x1, y), lane_line, 1, cv2.LINE_AA)
    cv2.rectangle(
        img,
        (centre - road_w // 2, centre - road_w // 2),
        (centre + road_w // 2, centre + road_w // 2),
        (54, 58, 64),
        -1,
    )
    # Stop lines.
    cv2.line(img, (centre - road_w // 2, centre - road_w // 2), (centre + road_w // 2, centre - road_w // 2), (230, 230, 230), 3)
    cv2.line(img, (centre - road_w // 2, centre + road_w // 2), (centre + road_w // 2, centre + road_w // 2), (230, 230, 230), 3)
    cv2.line(img, (centre - road_w // 2, centre - road_w // 2), (centre - road_w // 2, centre + road_w // 2), (230, 230, 230), 3)
    cv2.line(img, (centre + road_w // 2, centre - road_w // 2), (centre + road_w // 2, centre + road_w // 2), (230, 230, 230), 3)

    states: Dict[str, str] = snapshot.get("lane_states", {}) or {}
    active_lane = snapshot.get("priority_lane")
    # Light heads at each approach.
    positions = {
        "north": (centre + road_w // 2 + 34, int(size * 0.20)),
        "south": (centre - road_w // 2 - 34, int(size * 0.80)),
        "west": (int(size * 0.20), centre - road_w // 2 - 34),
        "east": (int(size * 0.80), centre + road_w // 2 + 34),
    }
    for lane_name in intersection.lane_names:
        if lane_name not in positions:
            continue
        x, y = positions[lane_name]
        state = str(states.get(lane_name, "RED")).upper()
        # Small housing.
        cv2.rectangle(img, (x - 16, y - 32), (x + 16, y + 32), (28, 30, 34), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x - 16, y - 32), (x + 16, y + 32), (90, 94, 100), 1, cv2.LINE_AA)
        for i, colour_name in enumerate(("RED", "YELLOW", "GREEN")):
            cy = y - 19 + i * 19
            lit = state == colour_name
            colour = _COLOR[colour_name] if lit else _COLOR["OFF"]
            cv2.circle(img, (x, cy), 6, colour, -1, cv2.LINE_AA)
            if lit:
                cv2.circle(img, (x, cy), 9, colour, 1, cv2.LINE_AA)
        label = intersection.lanes[lane_name].display
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        _text(img, label, (x - tw // 2, y + 50), 0.44, (230, 230, 230), 1)
        if active_lane == lane_name:
            cv2.circle(img, (x, y), 39, (0, 255, 255), 2, cv2.LINE_AA)

    # Centre status.
    status = str(snapshot.get("state", "NORMAL"))
    phase = str(snapshot.get("phase", "-"))
    priority = bool(snapshot.get("priority_active"))
    colour = (0, 255, 255) if priority else (200, 210, 220)
    _text(img, title, (18, 28), 0.52, (235, 235, 235), 2)
    _text(img, f"STATE: {status}", (18, 50), 0.42, colour, 1)
    _text(img, f"PHASE: {phase}", (18, 70), 0.38, (190, 200, 210), 1)
    remaining = snapshot.get("phase_remaining_s")
    if remaining is not None:
        _text(img, f"PHASE REMAINING: {float(remaining):5.1f}s", (18, 90), 0.38, (190, 200, 210), 1)
    if priority:
        lane = str(active_lane or "").upper()
        cv2.rectangle(img, (18, size - 54), (size - 18, size - 18), (0, 80, 120), -1)
        _text(img, f"EMERGENCY PRIORITY ACTIVE - {lane} GREEN", (30, size - 28), 0.46, (0, 255, 255), 2)
    else:
        _text(img, "NORMAL FIXED-SEQUENCE SIMULATION", (18, size - 24), 0.38, (170, 180, 190), 1)
    return img
