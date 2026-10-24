"""End-to-end processing pipeline shared by CLI, demo and dashboard."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.control.priority_engine import PriorityDecision, PriorityEngine
from src.control.signal_controller import FixedTimeController, SignalController
from src.detection.detector import YOLODetector
from src.detection.tracker import create_tracker
from src.evaluation.latency import LatencyRecorder, PerformanceRecorder
from src.traffic.density import DensityEstimator
from src.traffic.lane_detector import LaneDetector
from src.utils.config import AppConfig
from src.utils.events import EventLogger
from src.utils.logger import setup_logger
from src.utils.video import Frame, VideoSource
from src.visualization.overlay import OverlayRenderer

LOGGER = setup_logger("pipeline")


@dataclass
class FrameResult:
    annotated: np.ndarray
    raw: np.ndarray
    sim_time: float
    tracks: list
    lane_map: Dict[int, Optional[str]]
    density: Dict[str, Any]
    signal: Dict[str, Any]
    priority: PriorityDecision
    metrics: Dict[str, Any]
    detector_info: Dict[str, Any]
    baseline_signal: Dict[str, Any]


class EmergencySignalPipeline:
    """Wires detection -> tracking -> lanes -> density -> priority -> signal."""

    def __init__(
        self,
        config: AppConfig,
        model_path: str | Path | None = None,
        reset_events: bool = True,
        display_lanes: bool = True,
    ) -> None:
        self.config = config
        self.config.ensure_directories()
        self.log = setup_logger(
            "ev_smart_signal",
            log_file=config.path("paths.logs_dir") / "pipeline.log",
        )
        self.detector = YOLODetector(config, model_path=model_path)
        self.tracker = create_tracker(config)
        self.lane_detector = LaneDetector(config)
        self.density = DensityEstimator(config, self.lane_detector)
        self.signal = SignalController(config, self.lane_detector.intersection, on_event=self._on_controller_event)
        self.baseline = FixedTimeController(config, self.lane_detector.intersection)
        self.priority = PriorityEngine(
            config,
            self.lane_detector.intersection,
            self.lane_detector,
            self.signal,
            baseline_controller=self.baseline,
            on_event=self._on_priority_event,
        )
        self.events = EventLogger(
            csv_path=config.path("paths.events_csv"),
            jsonl_path=config.path("paths.events_dir") / "events.jsonl",
            reset=reset_events,
        )
        self.performance = PerformanceRecorder(config, reset=reset_events)
        self.overlay = OverlayRenderer(config, self.lane_detector.intersection, show_lanes=display_lanes)
        self._latency = LatencyRecorder()
        self._start_wall = time.perf_counter()
        self._last_wall = self._start_wall
        self._fps_ema = 0.0
        self._loop_offset = 0.0
        self._last_source_time = -1.0
        self._last_result: Optional[FrameResult] = None
        self._frame_counter = 0
        self._fail_safe_reason: Optional[str] = None
        self._signal_changed_last = False
        self._priority_green_seen = False
        self._last_e2e_latency_ms: Optional[float] = None
        self._session_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.events.log(
            "session_start",
            sim_time=0.0,
            message="Simulated traffic-signal prototype started",
            details={
                "model": self.detector.info(),
                "config": {
                    "priority": config.get("priority"),
                    "signals": config.get("signals"),
                    "density": config.get("density"),
                },
                "disclaimer": config.get("project.disclaimer"),
            },
        )
        if not self.detector.emergency_capable:
            self.events.log(
                "model_warning",
                sim_time=0.0,
                message="Loaded model lacks emergency classes; priority activation is disabled.",
                details={"missing": self.detector.missing_emergency, "model": str(self.detector.model_path)},
            )

    # ------------------------------------------------------------------ timing
    def _timeline(self, frame: Frame) -> float:
        if frame.is_image:
            return float(self._frame_counter)
        if self.source_is_live:
            return time.perf_counter() - self._start_wall
        # Looping video: keep a monotonically increasing simulation clock.
        if frame.source_time + 1e-6 < self._last_source_time:
            self._loop_offset += self._last_source_time + (1.0 / max(frame.source_fps, 1.0))
        self._last_source_time = float(frame.source_time)
        return self._loop_offset + float(frame.source_time)

    @property
    def source_is_live(self) -> bool:
        # Set by the CLI/demo/dashboard before processing a live webcam.
        return bool(getattr(self, "_live_source", False))

    def set_live_source(self, live: bool) -> None:
        self._live_source = bool(live)

    # ------------------------------------------------------------------ events
    def _on_controller_event(self, event_type: str, payload: Dict[str, object]) -> None:
        self.events.log(
            "controller_" + event_type,
            sim_time=payload.get("sim_time", payload.get("activated_at", 0.0)),
            **{
                k: v
                for k, v in payload.items()
                if k
                in {
                    "phase",
                    "signal_state",
                    "lane",
                    "lane_states",
                    "reason",
                    "duration_s",
                    "waiting_time_s",
                    "from_state",
                    "to_state",
                    "cooldown_remaining",
                    "prepare_seconds",
                    "immediate",
                    "error",
                }
            },
            message=event_type.replace("_", " "),
        )

    def _on_priority_event(self, event_type: str, payload: Dict[str, object]) -> None:
        known = {
            "track_id",
            "vehicle_class",
            "vehicle",
            "confidence",
            "lane",
            "signal_state",
            "priority_state",
            "waiting_time_s",
            "baseline_wait_s",
            "duration_s",
            "latency_ms",
            "event_id",
            "reason",
            "status",
            "arrival_time",
            "green_time",
            "requested_time",
            "wall_time",
            "policy",
            "count",
            "selected_policy",
            "selected_track_id",
            "tracks",
        }
        fields = {k: v for k, v in payload.items() if k in known}
        if "vehicle_class" not in fields and "vehicle" in fields:
            fields["vehicle_class"] = fields.pop("vehicle")
        self.events.log(
            event_type,
            sim_time=payload.get("green_time", payload.get("requested_time", payload.get("exit_time", 0.0))),
            message=event_type.replace("_", " "),
            **fields,
        )

    # ------------------------------------------------------------------ core
    def process(self, frame: Frame, live: bool = False) -> FrameResult:
        self.set_live_source(live)
        now = self._timeline(frame)
        wall_now = time.perf_counter()
        self._frame_counter += 1
        self._latency.reset()
        total_start = wall_now
        h, w = frame.image.shape[:2]

        # Always advance the controller clock, even when detection is skipped.
        before_snapshot = self.signal.snapshot(now)
        try:
            self.signal.update(now)
        except Exception as exc:
            self._fail_safe(now, "signal controller failure", str(exc))

        if not self.detector.emergency_capable:
            # Still detect normal vehicles if the model has them; never fabricate
            # emergency detections and never activate priority.
            pass

        detections = []
        tracks = []
        lane_map: Dict[int, Optional[str]] = {}
        try:
            with self._latency.stage("detection"):
                detections = self.detector.detect(frame.image)
            with self._latency.stage("tracking"):
                tracks = self.tracker.update(detections, now)
            with self._latency.stage("lanes"):
                lane_map = self.lane_detector.assign_tracks(tracks, w, h)
        except Exception as exc:
            self._fail_safe(now, "detection/tracking failure", str(exc))
            LOGGER.exception("Detection/tracking failure")
            annotated = frame.image.copy()
            cv2.putText(
                annotated,
                "FAIL-SAFE: detection unavailable - priority disabled",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            signal_snapshot = self.signal.snapshot(now)
            decision = PriorityDecision("FAIL_SAFE", "detection unavailable", controller_state="FAIL_SAFE", lane_states=signal_snapshot.get("lane_states", {}))
            metrics = {
                "fps": self._fps_ema,
                "inference_ms": 0.0,
                "tracking_ms": 0.0,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
                "vehicle_count": 0,
                "emergency_count": 0,
                "fail_safe": True,
            }
            result = FrameResult(
                annotated=annotated,
                raw=frame.image,
                sim_time=now,
                tracks=[],
                lane_map={},
                density={},
                signal=signal_snapshot,
                priority=decision,
                metrics=metrics,
                detector_info=self.detector.info(),
                baseline_signal=self.baseline.snapshot(now),
            )
            self._last_result = result
            return result

        density = self.density.update(tracks, w, h)

        with self._latency.stage("decision"):
            decision = self.priority.update(
                tracks,
                lane_map,
                now,
                w,
                h,
                wall_time=wall_now,
            )
        self._maybe_capture_e2e(decision, wall_now)

        after_snapshot = self.signal.snapshot(now)
        self._signal_changed_last = (
            before_snapshot.get("phase") != after_snapshot.get("phase")
            or before_snapshot.get("state") != after_snapshot.get("state")
        )

        # FPS / frame pacing metrics.
        dt = max(1e-6, wall_now - self._last_wall)
        self._last_wall = wall_now
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema <= 0 else (0.9 * self._fps_ema + 0.1 * inst_fps)

        metrics = {
            "fps": self._fps_ema,
            "inference_ms": self.detector.last_inference_ms,
            "tracking_ms": self.tracker.last_tracking_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
            "vehicle_count": sum(1 for t in tracks if t.class_name in set(self.config.vehicle_classes)),
            "emergency_count": sum(1 for t in tracks if t.is_emergency),
            "detections": len(detections),
            "e2e_priority_latency_ms": self._last_e2e_latency_ms,
            "fail_safe": self.signal.state.value == "FAIL_SAFE",
        }

        with self._latency.stage("render"):
            annotated = self.overlay.render(
                frame.image,
                tracks,
                density,
                after_snapshot,
                decision,
                metrics,
                detector_info=self.detector.info(),
                priority_track_id=decision.active_track_id,
            )
        metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        metrics["signal_changed"] = self._signal_changed_last
        metrics["latency_stages_ms"] = dict(self._latency.stages)

        self.performance.record(
            {
                "frame_index": self._frame_counter,
                "timestamp_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "sim_time": round(now, 3),
                "fps": round(self._fps_ema, 3),
                "detection_ms": round(self.detector.last_inference_ms, 3),
                "tracking_ms": round(self.tracker.last_tracking_ms, 3),
                "lane_ms": round(self._latency.stages.get("lanes", 0.0), 3),
                "decision_ms": round(self._latency.stages.get("decision", 0.0), 3),
                "render_ms": round(self._latency.stages.get("render", 0.0), 3),
                "total_ms": round(metrics["total_ms"], 3),
                "vehicle_count": metrics["vehicle_count"],
                "emergency_count": metrics["emergency_count"],
                "signal_state": after_snapshot.get("state"),
                "priority_lane": after_snapshot.get("priority_lane") or "",
                "signal_changed": bool(self._signal_changed_last),
                "e2e_priority_latency_ms": self._last_e2e_latency_ms,
            }
        )
        result = FrameResult(
            annotated=annotated,
            raw=frame.image,
            sim_time=now,
            tracks=tracks,
            lane_map=lane_map,
            density=density,
            signal=after_snapshot,
            priority=decision,
            metrics=metrics,
            detector_info=self.detector.info(),
            baseline_signal=self.baseline.snapshot(now),
        )
        self._last_result = result
        return result

    def _maybe_capture_e2e(self, decision: PriorityDecision, wall_now: float) -> None:
        event = self.priority._event(self.priority._active_event_id)
        if event is None:
            return
        if event.green_time is not None and event.e2e_latency_ms is None:
            arrival_wall = getattr(event, "arrival_wall_time", None)
            if arrival_wall is not None:
                event.e2e_latency_ms = max(0.0, (wall_now - float(arrival_wall)) * 1000.0)
                self._last_e2e_latency_ms = event.e2e_latency_ms
                self.events.log(
                    "priority_e2e_latency",
                    sim_time=event.green_time,
                    lane=event.lane,
                    track_id=event.track_id,
                    vehicle_class=event.vehicle_class,
                    latency_ms=event.e2e_latency_ms,
                    message="first lane-assigned EV detection to priority green (wall clock)",
                )

    def _fail_safe(self, now: float, reason: str, error: Optional[str] = None) -> None:
        self._fail_safe_reason = reason
        try:
            self.signal.force_fail_safe(now, reason, error=error)
        except Exception:
            LOGGER.exception("Could not set fail-safe state")
        self.events.log("fail_safe", sim_time=now, reason=reason, message=error or reason)

    def summary(self) -> Dict[str, Any]:
        now = self._last_result.sim_time if self._last_result else 0.0
        return {
            "session_started": self._session_started,
            "frames": self._frame_counter,
            "model": self.detector.info(),
            "signal": self.signal.snapshot(now),
            "priority": self.priority.waiting_summary(),
            "priority_events": self.priority.event_rows(),
            "performance": self.performance.summary(),
            "density": self.density.summary(),
            "events_csv": str(self.events.csv_path),
        }

    def close(self) -> None:
        self.performance.save_summary()
        self.performance.save_figures()
        self.events.log(
            "session_end",
            sim_time=self._last_result.sim_time if self._last_result else 0.0,
            message="session ended",
            details=self.priority.waiting_summary(),
        )
        import json

        summary_path = self.config.path("paths.metrics_dir") / "pipeline_summary.json"
        summary_path.write_text(json.dumps(self.summary(), indent=2, default=str), encoding="utf-8")
