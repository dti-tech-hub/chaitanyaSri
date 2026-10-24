"""Latency and frame-performance measurement."""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from src.utils.config import AppConfig, resolve_path
from src.utils.logger import setup_logger

LOGGER = setup_logger("latency")

PERFORMANCE_FIELDS = [
    "frame_index",
    "timestamp_wall",
    "sim_time",
    "fps",
    "detection_ms",
    "tracking_ms",
    "lane_ms",
    "decision_ms",
    "render_ms",
    "total_ms",
    "vehicle_count",
    "emergency_count",
    "signal_state",
    "priority_lane",
    "signal_changed",
    "e2e_priority_latency_ms",
]


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": int(self.count),
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
        }


def summarize_latencies(values: List[float]) -> Optional[LatencyStats]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return LatencyStats(
        count=int(arr.size),
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


class LatencyRecorder:
    """Tiny stage timer that records wall-clock milliseconds."""

    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        start = self._starts.pop(stage, None)
        if start is None:
            return 0.0
        ms = (time.perf_counter() - start) * 1000.0
        self.stages[stage] = self.stages.get(stage, 0.0) + ms
        return ms

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def reset(self) -> None:
        self.stages.clear()
        self._starts.clear()


class PerformanceRecorder:
    """Writes per-frame performance measurements to CSV."""

    def __init__(self, config: AppConfig, path: str | Path | None = None, reset: bool = True) -> None:
        self.config = config
        self.path = resolve_path(path or config.get("paths.performance_csv", "results/metrics/performance.csv"), config.root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=PERFORMANCE_FIELDS)
                writer.writeheader()
        self.rows: List[Dict[str, Any]] = []
        self.fps_values: List[float] = []
        self.total_values: List[float] = []
        self.detection_values: List[float] = []
        self.tracking_values: List[float] = []
        self.decision_values: List[float] = []
        self.start_time = time.perf_counter()
        self.last_signal_state: Optional[str] = None

    def record(self, row: Dict[str, Any]) -> None:
        full = {key: row.get(key, "") for key in PERFORMANCE_FIELDS}
        self.rows.append(full)
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=PERFORMANCE_FIELDS)
            writer.writerow(full)
        try:
            self.fps_values.append(float(row.get("fps", 0.0) or 0.0))
            self.total_values.append(float(row.get("total_ms", 0.0) or 0.0))
            self.detection_values.append(float(row.get("detection_ms", 0.0) or 0.0))
            self.tracking_values.append(float(row.get("tracking_ms", 0.0) or 0.0))
            self.decision_values.append(float(row.get("decision_ms", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass

    def summary(self) -> Dict[str, Any]:
        stats = summarize_latencies(self.total_values)
        detection = summarize_latencies(self.detection_values)
        tracking = summarize_latencies(self.tracking_values)
        decision = summarize_latencies(self.decision_values)
        fps = self.fps_values
        e2e = []
        for row in self.rows:
            try:
                if row.get("e2e_priority_latency_ms") not in ("", None):
                    e2e.append(float(row["e2e_priority_latency_ms"]))
            except (TypeError, ValueError):
                pass
        e2e_stats = summarize_latencies(e2e)
        return {
            "frames": len(self.rows),
            "duration_s": round(max(0.0, time.perf_counter() - self.start_time), 3),
            "fps_mean": round(float(np.mean(fps)), 3) if fps else None,
            "fps_min": round(float(np.min(fps)), 3) if fps else None,
            "fps_max": round(float(np.max(fps)), 3) if fps else None,
            "total_processing_ms": stats.to_dict() if stats else None,
            "detection_ms": detection.to_dict() if detection else None,
            "tracking_ms": tracking.to_dict() if tracking else None,
            "decision_ms": decision.to_dict() if decision else None,
            "e2e_priority_latency_ms": e2e_stats.to_dict() if e2e_stats else None,
        }

    def save_summary(self, path: str | Path | None = None) -> Path:
        import json

        target = resolve_path(path or "results/metrics/performance_summary.json", self.config.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return target

    def save_figures(self) -> None:
        if not self.rows:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_dir = self.config.path("paths.figures_dir")
            fig_dir.mkdir(parents=True, exist_ok=True)
            if self.fps_values:
                fig, ax = plt.subplots(figsize=(8, 3.8))
                ax.plot(self.fps_values, color="#1d3557", lw=1.2)
                ax.axhline(float(np.mean(self.fps_values)), color="#e63946", ls="--", lw=1, label=f"mean {np.mean(self.fps_values):.1f} FPS")
                ax.set_xlabel("Processed frame")
                ax.set_ylabel("FPS")
                ax.set_title("Real-time processing FPS (CPU)")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(fig_dir / "fps.png", dpi=160)
                plt.close(fig)
            if self.total_values:
                fig, ax = plt.subplots(figsize=(8, 3.8))
                ax.plot(self.total_values, color="#457b9d", lw=1.0, label="total")
                if self.detection_values:
                    ax.plot(self.detection_values, color="#e76f51", lw=0.9, alpha=0.8, label="YOLO inference")
                ax.set_xlabel("Processed frame")
                ax.set_ylabel("Milliseconds")
                ax.set_title("Frame processing latency")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(fig_dir / "latency.png", dpi=160)
                plt.close(fig)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not save performance figures: %s", exc)


def read_performance_csv(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
