"""Lightweight IoU tracker plus an optional Ultralytics ByteTrack adapter.

The default tracker is dependency-free (no `lap` requirement) and is sufficient
for the prototype. ByteTrack/BoT-SORT can be enabled through Ultralytics when
the optional `lap` package is installed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.detection.detector import Detection
from src.utils.logger import setup_logger

LOGGER = setup_logger("tracker")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    raw_class_name: str
    bbox: np.ndarray
    confidence: float
    first_seen: float
    last_seen: float
    hit_count: int = 1
    consecutive_hits: int = 1
    age: int = 1
    lost_frames: int = 0
    lane: Optional[str] = None
    lane_history: List[str] = field(default_factory=list)
    history: List[tuple[float, float, float]] = field(default_factory=list)
    lane_distance: float = 1.0
    is_emergency: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (float((self.bbox[0] + self.bbox[2]) / 2), float((self.bbox[1] + self.bbox[3]) / 2))

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def confirmed(self) -> bool:
        return self.hit_count >= 2 and self.consecutive_hits >= 1

    @property
    def lost(self) -> bool:
        return self.lost_frames > 0

    def speed_px_s(self) -> float:
        if len(self.history) < 2:
            return 0.0
        t0, x0, y0 = self.history[0]
        t1, x1, y1 = self.history[-1]
        dt = max(t1 - t0, 1e-6)
        return math.hypot(x1 - x0, y1 - y0) / dt

    def displacement_over(self, seconds: float) -> float:
        if len(self.history) < 2:
            return 0.0
        t_end = self.history[-1][0]
        for t, x, y in self.history:
            if t_end - t <= seconds:
                return math.hypot(self.history[-1][1] - x, self.history[-1][2] - y)
        return 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "first_seen": round(float(self.first_seen), 3),
            "last_seen": round(float(self.last_seen), 3),
            "duration": round(float(self.duration), 3),
            "hits": int(self.hit_count),
            "lane": self.lane,
            "lost": self.lost,
        }


class IOUTracker:
    """Class-aware greedy IoU tracker with short-term re-identification."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_frames: int = 30,
        min_hits: int = 2,
        class_aware: bool = True,
        history_length: int = 40,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age_frames = int(max_age_frames)
        self.min_hits = int(min_hits)
        self.class_aware = bool(class_aware)
        self.history_length = int(history_length)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.last_tracking_ms = 0.0
        self.total_updates = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def _new_track(self, det: Detection, timestamp: float) -> Track:
        tid = self._next_id
        self._next_id += 1
        track = Track(
            track_id=tid,
            class_name=det.class_name,
            raw_class_name=det.raw_class_name,
            bbox=np.array(det.bbox, dtype=np.float32),
            confidence=float(det.confidence),
            first_seen=float(timestamp),
            last_seen=float(timestamp),
            is_emergency=bool(det.is_emergency),
        )
        track.history.append((timestamp, *track.center))
        self.tracks[tid] = track
        return track

    def update(self, detections: Iterable[Detection], timestamp: float) -> List[Track]:
        start = time.perf_counter()
        dets = list(detections)
        active = [t for t in self.tracks.values() if t.lost_frames <= self.max_age_frames]
        pairs: List[Tuple[float, int, int]] = []
        for ti, track in enumerate(active):
            for di, det in enumerate(dets):
                if self.class_aware and det.class_name != track.class_name:
                    continue
                if det.class_name == "unknown" and track.class_name == "unknown":
                    pass
                score = _iou(track.bbox, det.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            track = active[ti]
            det = dets[di]
            track.bbox = np.array(det.bbox, dtype=np.float32)
            track.confidence = float(det.confidence)
            track.class_name = det.class_name
            track.raw_class_name = det.raw_class_name
            track.is_emergency = bool(det.is_emergency)
            track.last_seen = float(timestamp)
            track.hit_count += 1
            track.consecutive_hits += 1
            track.age += 1
            track.lost_frames = 0
            track.history.append((timestamp, *track.center))
            if len(track.history) > self.history_length:
                track.history.pop(0)
            det.track_id = track.track_id
            used_tracks.add(ti)
            used_dets.add(di)

        for ti, track in enumerate(active):
            if ti not in used_tracks:
                track.age += 1
                track.consecutive_hits = 0
                # An unmatched track is immediately marked lost. It is retained
                # for max_age_frames so a short detection gap can be re-linked,
                # but callers never treat a lost track as currently visible.
                track.lost_frames += 1

        for di, det in enumerate(dets):
            if di in used_dets:
                continue
            track = self._new_track(det, timestamp)
            det.track_id = track.track_id

        # Expire stale tracks, but retain a short history for diagnostics.
        expired = [
            tid for tid, t in self.tracks.items()
            if t.lost_frames > self.max_age_frames
        ]
        for tid in expired:
            del self.tracks[tid]

        self.total_updates += 1
        self.last_tracking_ms = (time.perf_counter() - start) * 1000.0
        return sorted(
            [t for t in self.tracks.values() if not t.lost],
            key=lambda t: t.track_id,
        )

    def get(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(int(track_id))

    def active_emergency(self, emergency_classes: Iterable[str]) -> List[Track]:
        wanted = set(emergency_classes)
        return [t for t in self.tracks.values() if not t.lost and t.class_name in wanted]


class ByteTrackAdapter:
    """Optional ByteTrack/BoT-SORT adapter backed by Ultralytics.

    Ultralytics trackers require the optional `lap` package. If it is missing,
    constructing this adapter raises ImportError and the pipeline falls back to
    IOUTracker unless the user explicitly requires ByteTrack.
    """

    def __init__(self, algorithm: str = "bytetrack", config: Dict | None = None) -> None:
        from ultralytics.trackers import BYTETracker, BOTSORT
        from types import SimpleNamespace

        cfg = {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "with_reid": False,
        }
        if config:
            cfg.update(config)
        args = SimpleNamespace(**cfg)
        self.tracker = BOTSORT(args) if algorithm == "botsort" else BYTETracker(args)
        self.last_tracking_ms = 0.0

    def update(self, detections: Iterable[Detection], timestamp: float) -> List[Track]:
        """Not implemented: use Ultralytics model.track directly.

        This adapter exists so the interface is explicit. The main pipeline uses
        the dependency-free IOUTracker; ByteTrack remains an optional deployment
        choice documented in the README.
        """
        raise NotImplementedError(
            "Use YOLO().track(..., tracker='bytetrack.yaml') for ByteTrack; "
            "the prototype's measurable tracker is IOUTracker."
        )


def create_tracker(config) -> IOUTracker:
    algorithm = str(config.get("tracking.algorithm", "iou")).lower()
    if algorithm in {"bytetrack", "botsort"}:
        try:
            import lap  # noqa: F401
        except Exception as exc:
            LOGGER.warning(
                "Tracker '%s' requested but optional `lap` is unavailable (%s). "
                "Falling back to the built-in IoU tracker.",
                algorithm,
                exc,
            )
            algorithm = "iou"
        else:
            LOGGER.info(
                "Ultralytics %s is available; this build measures the built-in IoU tracker. "
                "See README for the ByteTrack integration note.",
                algorithm,
            )
            algorithm = "iou"
    return IOUTracker(
        iou_threshold=float(config.get("tracking.iou_threshold", 0.3)),
        max_age_frames=int(config.get("tracking.max_age_frames", 30)),
        min_hits=int(config.get("tracking.min_hits", 2)),
        class_aware=bool(config.get("tracking.class_aware", True)),
        history_length=int(config.get("tracking.history_length", 40)),
    )
