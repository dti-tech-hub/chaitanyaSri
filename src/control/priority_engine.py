"""Emergency-priority decision engine with confirmation and anti-false-trigger logic."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from src.control.signal_controller import FixedTimeController, SignalController
from src.control.state_machine import SignalState
from src.detection.tracker import Track
from src.traffic.lane_detector import LaneDetector
from src.utils.config import AppConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger("priority_engine")

EventCallback = Callable[[str, Dict[str, object]], None]


@dataclass
class TrackEmergencyState:
    track_id: int
    class_name: str
    first_seen: float
    first_confident: Optional[float] = None
    arrival_time: Optional[float] = None
    arrival_wall_time: Optional[float] = None
    arrival_lane: Optional[str] = None
    arrival_confidence: float = 0.0
    consecutive: int = 0
    last_seen: float = 0.0
    last_confident: Optional[float] = None
    last_confidence: float = 0.0
    last_lane: Optional[str] = None
    lane_changes: int = 0
    exit_seen_at: Optional[float] = None
    latest_distance: float = 1.0
    distance_history: list[tuple[float, float]] = field(default_factory=list)
    distance_decreasing: bool = False
    confirmed: bool = False
    priority_granted: bool = False
    served: bool = False


@dataclass
class PriorityEvent:
    event_id: int
    track_id: int
    vehicle_class: str
    lane: str
    arrival_time: float
    arrival_confidence: float
    arrival_wall_time: Optional[float] = None
    confirmed_time: Optional[float] = None
    requested_time: Optional[float] = None
    green_time: Optional[float] = None
    release_time: Optional[float] = None
    waiting_time_s: Optional[float] = None
    baseline_wait_s: Optional[float] = None
    priority_duration_s: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    exit_seen_at: Optional[float] = None
    status: str = "candidate"
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "lane": self.lane,
            "arrival_time": None if self.arrival_time is None else round(float(self.arrival_time), 3),
            "arrival_confidence": round(float(self.arrival_confidence), 4),
            "confirmed_time": None if self.confirmed_time is None else round(float(self.confirmed_time), 3),
            "requested_time": None if self.requested_time is None else round(float(self.requested_time), 3),
            "green_time": None if self.green_time is None else round(float(self.green_time), 3),
            "release_time": None if self.release_time is None else round(float(self.release_time), 3),
            "waiting_time_s": None if self.waiting_time_s is None else round(float(self.waiting_time_s), 3),
            "baseline_wait_s": None if self.baseline_wait_s is None else round(float(self.baseline_wait_s), 3),
            "priority_duration_s": None if self.priority_duration_s is None else round(float(self.priority_duration_s), 3),
            "e2e_latency_ms": None if self.e2e_latency_ms is None else round(float(self.e2e_latency_ms), 3),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass
class PriorityDecision:
    state: str
    message: str
    active_lane: Optional[str] = None
    active_track_id: Optional[int] = None
    candidate_track_id: Optional[int] = None
    candidate_class: Optional[str] = None
    confidence: Optional[float] = None
    waiting_time_s: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    should_activate: bool = False
    controller_state: str = "NORMAL"
    lane_states: Dict[str, str] = field(default_factory=dict)


class PriorityEngine:
    """Decides whether and when to request a simulated priority green.

    Confirmation is deliberately conservative:
      * model confidence >= priority.confidence_threshold
      * at least priority.confirmation_frames consecutive confident frames
      * track alive for priority.min_tracking_seconds
      * lane known from a calibrated ROI (never guessed)
      * approaching motion OR stopped near the stop area
    Priority is never activated from a single weak detection.
    """

    def __init__(
        self,
        config: AppConfig,
        intersection,
        lane_detector: LaneDetector,
        controller: SignalController,
        baseline_controller: Optional[FixedTimeController] = None,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        self.config = config
        self.intersection = intersection
        self.lane_detector = lane_detector
        self.controller = controller
        self.baseline = baseline_controller
        self.on_event = on_event
        self.confidence_threshold = float(config.get("priority.confidence_threshold", 0.6))
        self.confirmation_frames = int(config.get("priority.confirmation_frames", 3))
        self.min_tracking_seconds = float(config.get("priority.min_tracking_seconds", 0.5))
        self.cooldown_seconds = float(config.get("priority.cooldown_seconds", 10.0))
        self.hold_after_exit = float(config.get("priority.hold_after_exit_seconds", 2.0))
        self.lost_timeout = float(config.get("priority.lost_track_timeout_seconds", 3.0))
        self.max_priority = float(config.get("priority.max_priority_seconds", 45.0))
        self.motion_window = float(config.get("priority.approach_motion_window", 8))
        self.min_motion_px = float(config.get("priority.approach_min_displacement_px", 4.0))
        self.min_distance_delta = float(config.get("priority.approach_min_distance_delta", 0.01))
        self.stationary_distance = float(config.get("priority.approach_stationary_distance", 0.42))
        self.policy = str(config.get("priority.policy", "closest")).lower()
        self.exit_mode = str(config.get("priority.exit_mode", "roi")).lower()
        self.emergency_classes = set(config.emergency_classes)
        self.states: Dict[int, TrackEmergencyState] = {}
        self.events: List[PriorityEvent] = []
        self._active_track_id: Optional[int] = None
        self._active_event_id: Optional[int] = None
        self._lost_since: Optional[float] = None
        self._last_release_time: Optional[float] = None
        self._next_event_id = 1
        self._multiple_logged: set[tuple[int, ...]] = set()
        self.last_decision = PriorityDecision("NORMAL", "initialising")
        self.activation_count = 0
        self.max_simultaneous = 0

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.states.clear()
        self.events.clear()
        self._active_track_id = None
        self._active_event_id = None
        self._lost_since = None
        self._last_release_time = None
        self._next_event_id = 1
        self.last_decision = PriorityDecision("NORMAL", "reset")
        self.activation_count = 0

    def _emit(self, event_type: str, payload: Dict[str, object]) -> None:
        if self.on_event:
            self.on_event(event_type, payload)

    def _event(self, event_id: Optional[int]) -> Optional[PriorityEvent]:
        if event_id is None:
            return None
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def _get_state(
        self,
        track: Track,
        now: float,
        lane: Optional[str],
        confidence_ok: bool,
        wall_time: Optional[float] = None,
    ) -> TrackEmergencyState:
        state = self.states.get(track.track_id)
        if state is None:
            state = TrackEmergencyState(
                track_id=track.track_id,
                class_name=track.class_name,
                first_seen=track.first_seen,
                last_seen=now,
                last_lane=lane,
            )
            self.states[track.track_id] = state
        state.last_seen = now
        state.class_name = track.class_name
        if lane is not None:
            if state.arrival_lane is None and confidence_ok:
                # Arrival is defined as the first confident detection that can
                # be assigned to a calibrated approach lane.
                state.arrival_lane = lane
                state.arrival_time = now
                state.arrival_wall_time = wall_time
                state.arrival_confidence = float(track.confidence)
                state.first_confident = now
            elif lane != state.last_lane and state.last_lane is not None:
                state.lane_changes += 1
            state.last_lane = lane
        if confidence_ok:
            state.consecutive += 1
            state.last_confident = now
            state.last_confidence = float(track.confidence)
            if state.first_confident is None:
                state.first_confident = now
            if state.arrival_lane is None and lane is not None:
                state.arrival_lane = lane
                state.arrival_time = now
                state.arrival_wall_time = wall_time
                state.arrival_confidence = float(track.confidence)
        else:
            state.consecutive = 0
        return state

    def _is_approaching(self, track: Track, state: TrackEmergencyState) -> bool:
        """Approach is inferred from decreasing distance to the junction centre.

        Using distance-to-centre (rather than a fixed x-direction) is valid for
        all four approaches, including cameras where an approach moves upwards
        or leftwards. A vehicle stopped close to the junction is also accepted,
        because it is already queued at the stop line.
        """
        distance = float(track.lane_distance)
        if distance <= self.stationary_distance:
            return True
        if len(state.distance_history) >= 2:
            window_start = track.last_seen - self.motion_window
            samples = [(t, d) for t, d in state.distance_history if t >= window_start]
            if len(samples) >= 2:
                decrease = samples[0][1] - samples[-1][1]
                if decrease >= self.min_distance_delta:
                    return True
        # Pixel displacement is used only as a secondary signal when the
        # normalized distance trend is not yet available.
        displacement = track.displacement_over(self.motion_window)
        return displacement >= self.min_motion_px and track.hit_count >= 2

    def _candidate_score(self, track: Track, state: TrackEmergencyState) -> tuple:
        distance = float(getattr(track, "lane_distance", 1.0))
        if self.policy == "confidence":
            return (float(track.confidence), -distance)
        if self.policy == "arrival":
            return (-float(state.arrival_time or 0.0), -distance)
        return (-distance, float(track.confidence))

    def update(
        self,
        tracks: Iterable[Track],
        lane_map: Dict[int, Optional[str]],
        now: float,
        frame_width: int,
        frame_height: int,
        wall_time: Optional[float] = None,
    ) -> PriorityDecision:
        now = float(now)
        tracks = list(tracks)
        vehicle_tracks = {t.track_id: t for t in tracks if not t.lost}
        emergency_tracks = [t for t in vehicle_tracks.values() if t.class_name in self.emergency_classes and t.confidence >= self.confidence_threshold]
        self.max_simultaneous = max(self.max_simultaneous, len(emergency_tracks))

        # Update per-track confirmation state for every emergency-class track,
        # including low-confidence frames (which reset the consecutive count).
        for track in vehicle_tracks.values():
            if track.class_name not in self.emergency_classes:
                continue
            lane = lane_map.get(track.track_id)
            track.lane_distance = self.lane_detector.distance_to_center(*track.center, frame_width, frame_height) if track.center else 1.0
            confidence_ok = track.confidence >= self.confidence_threshold
            state = self._get_state(track, now, lane, confidence_ok, wall_time=wall_time)
            if lane is not None and state.arrival_lane is not None:
                # Keep the current lane only when the ROI supports it.
                state.last_lane = lane
            state.latest_distance = float(track.lane_distance)
            state.distance_history.append((now, state.latest_distance))
            cutoff = now - max(2.0 * self.motion_window, 1.0)
            while state.distance_history and state.distance_history[0][0] < cutoff:
                state.distance_history.pop(0)
            if len(state.distance_history) >= 2:
                state.distance_decreasing = state.distance_history[-1][1] < state.distance_history[0][1]
            state.confirmed = (
                confidence_ok
                and state.consecutive >= self.confirmation_frames
                and track.duration >= self.min_tracking_seconds
                and lane is not None
                and self._is_approaching(track, state)
            )
            if state.confirmed and not state.priority_granted:
                if self.on_event:
                    self._emit(
                        "emergency_confirmed",
                        {
                            "track_id": track.track_id,
                            "vehicle_class": track.class_name,
                            "confidence": float(track.confidence),
                            "lane": lane,
                            "track_duration_s": float(track.duration),
                            "consecutive_frames": int(state.consecutive),
                        },
                    )

        # Active-priority handling.
        if self._active_track_id is not None:
            active_track = vehicle_tracks.get(self._active_track_id)
            event = self._event(self._active_event_id)
            active_lane = event.lane if event else None
            if event is not None and event.green_time is None and self.controller.signal_for(event.lane) == "GREEN":
                event.green_time = now
                event.waiting_time_s = max(0.0, now - event.arrival_time)
                event.status = "active"
                baseline_wait = None
                if self.baseline is not None and event.arrival_time is not None:
                    baseline_wait = self.baseline.baseline_wait_seconds(event.lane, event.arrival_time)
                    event.baseline_wait_s = baseline_wait
                self._emit(
                    "emergency_waiting_measured",
                    {
                        "event_id": event.event_id,
                        "track_id": event.track_id,
                        "lane": event.lane,
                        "arrival_time": event.arrival_time,
                        "green_time": event.green_time,
                        "waiting_time_s": event.waiting_time_s,
                        "baseline_wait_s": baseline_wait,
                        "confidence": event.arrival_confidence,
                        "vehicle_class": event.vehicle_class,
                    },
                )
                LOGGER.info(
                    "EMERGENCY GREEN | %s lane=%s arrival=%.2fs green=%.2fs waiting=%.2fs baseline=%.2fs",
                    event.vehicle_class,
                    event.lane,
                    event.arrival_time,
                    event.green_time,
                    event.waiting_time_s,
                    baseline_wait if baseline_wait is not None else -1,
                )

            if active_track is not None:
                self._lost_since = None
                exited = False
                if self.exit_mode in {"roi", "roi_or_lost"}:
                    exited = self.lane_detector.has_exited(active_track, active_lane, frame_width, frame_height)
                if exited and event is not None and event.exit_seen_at is None:
                    event.exit_seen_at = now
                    self._emit(
                        "emergency_passed",
                        {
                            "event_id": event.event_id,
                            "track_id": event.track_id,
                            "lane": event.lane,
                            "exit_time": now,
                        },
                    )
            else:
                if self._lost_since is None:
                    self._lost_since = now
                    self._emit(
                        "emergency_temporarily_lost",
                        {
                            "track_id": self._active_track_id,
                            "lost_at": now,
                            "tolerance_s": self.lost_timeout,
                        },
                    )

            release_reason: Optional[str] = None
            if event is not None:
                exit_seen = event.exit_seen_at
                if exit_seen is not None and now - exit_seen >= self.hold_after_exit:
                    release_reason = "vehicle cleared exit ROI"
                elif active_track is None and self._lost_since is not None:
                    if self.exit_mode in {"lost", "roi_or_lost"} and now - self._lost_since >= self.lost_timeout:
                        release_reason = "track lost after priority"
                    elif self.exit_mode == "roi" and now - self._lost_since >= self.lost_timeout:
                        # Even in ROI mode a permanently missing track must not
                        # hold a green forever; this is the fail-safe release.
                        release_reason = "track lost after priority (fail-safe)"
                if event.requested_time is not None and now - event.requested_time >= self.max_priority:
                    release_reason = "max priority duration reached"
            if release_reason and self.controller.state in {SignalState.PRIORITY_ACTIVE, SignalState.PRIORITY_PREPARING}:
                self.controller.release_priority(now, release_reason)
                if event is not None:
                    event.release_time = now
                    event.priority_duration_s = (
                        max(0.0, now - float(event.green_time))
                        if event.green_time is not None
                        else max(0.0, now - float(event.requested_time or event.arrival_time))
                    )
                    event.status = "released"
                    event.reason = release_reason
                    self._emit(
                        "priority_event_complete",
                        {
                            **event.to_dict(),
                            "wall_time": wall_time,
                        },
                    )
                self._last_release_time = now
                self._active_track_id = None
                self._active_event_id = None
                self._lost_since = None
                self._emit("priority_released", {"reason": release_reason, "lane": active_lane})

        # Reconcile if the controller released priority on its own (for example
        # max_priority_seconds). Without this, the engine could remain attached
        # to an event that no longer owns a green.
        if (
            self._active_track_id is not None
            and self.controller.state not in {SignalState.PRIORITY_ACTIVE, SignalState.PRIORITY_PREPARING}
        ):
            event = self._event(self._active_event_id)
            reason = f"controller released priority ({self.controller.state.value})"
            if event is not None and event.release_time is None:
                event.release_time = now
                event.priority_duration_s = (
                    max(0.0, now - float(event.green_time))
                    if event.green_time is not None
                    else max(0.0, now - float(event.requested_time or event.arrival_time))
                )
                event.status = "released"
                event.reason = reason
                self._emit("priority_event_complete", {**event.to_dict(), "wall_time": wall_time})
            self._active_track_id = None
            self._active_event_id = None
            self._lost_since = None

        # If controller is priority-active but engine has no tracked event,
        # do not leave an orphaned green. Release after the safety hold.
        if self._active_track_id is None and self.controller.state in {SignalState.PRIORITY_ACTIVE, SignalState.PRIORITY_PREPARING}:
            if self._last_release_time is None:
                self._last_release_time = now
            elif now - self._last_release_time >= self.hold_after_exit:
                self.controller.release_priority(now, "no tracked emergency vehicle")
                self._last_release_time = None

        # Select a candidate only if the controller can accept one.
        candidates = [t for t in emergency_tracks if self.states.get(t.track_id) and self.states[t.track_id].confirmed]
        chosen: Optional[Track] = None
        if candidates:
            chosen = max(candidates, key=lambda t: self._candidate_score(t, self.states[t.track_id]))
        could_activate = self.controller.can_activate(now) and self._active_track_id is None
        if chosen is not None and could_activate:
            state = self.states[chosen.track_id]
            lane = lane_map.get(chosen.track_id) or state.arrival_lane
            if lane is not None:
                requested = self.controller.request_priority(lane, now)
                if requested:
                    event = PriorityEvent(
                        event_id=self._next_event_id,
                        track_id=chosen.track_id,
                        vehicle_class=chosen.class_name,
                        lane=lane,
                        arrival_time=float(state.arrival_time if state.arrival_time is not None else now),
                        arrival_confidence=float(state.arrival_confidence or chosen.confidence),
                        arrival_wall_time=state.arrival_wall_time,
                        confirmed_time=now,
                        requested_time=now,
                        status="requested",
                    )
                    self._next_event_id += 1
                    self.events.append(event)
                    self._active_track_id = chosen.track_id
                    self._active_event_id = event.event_id
                    self._lost_since = None
                    state.priority_granted = True
                    self.activation_count += 1
                    if self.controller.state == SignalState.PRIORITY_ACTIVE and self.controller.signal_for(lane) == "GREEN":
                        event.green_time = now
                        event.waiting_time_s = max(0.0, now - event.arrival_time)
                        event.status = "active"
                        if self.baseline is not None and event.arrival_time is not None:
                            event.baseline_wait_s = self.baseline.baseline_wait_seconds(event.lane, event.arrival_time)
                    self._emit(
                        "priority_requested",
                        {
                            "event_id": event.event_id,
                            "track_id": chosen.track_id,
                            "vehicle_class": chosen.class_name,
                            "confidence": event.arrival_confidence,
                            "lane": lane,
                            "arrival_time": event.arrival_time,
                            "requested_time": now,
                            "policy": self.policy,
                            "wall_time": wall_time,
                        },
                    )
                    LOGGER.info(
                        "PRIORITY REQUEST | %s track=%d lane=%s conf=%.2f",
                        chosen.class_name,
                        chosen.track_id,
                        lane,
                        chosen.confidence,
                    )
        elif len(candidates) > 1:
            key = tuple(sorted(t.track_id for t in candidates))
            if key not in self._multiple_logged:
                self._multiple_logged.add(key)
                self._emit(
                    "multiple_emergency_detected",
                    {
                        "count": len(candidates),
                        "tracks": [
                            {
                                "track_id": t.track_id,
                                "class": t.class_name,
                                "lane": lane_map.get(t.track_id),
                                "distance": float(getattr(t, "lane_distance", 1.0)),
                                "confidence": float(t.confidence),
                            }
                            for t in candidates
                        ],
                        "selected_policy": self.policy,
                        "selected_track_id": chosen.track_id if chosen is not None else None,
                    },
                )

        active_lane = None
        active_event = self._event(self._active_event_id)
        if active_event is not None:
            active_lane = active_event.lane
        message = "monitoring"
        if self.controller.state == SignalState.FAIL_SAFE:
            message = "FAIL-SAFE: all-red, priority disabled"
        elif active_event is not None:
            if active_event.green_time is not None:
                message = f"priority active for {active_event.lane}"
            else:
                message = f"priority preparing for {active_event.lane}"
        elif chosen is not None:
            message = f"emergency candidate in {lane_map.get(chosen.track_id)}"
        elif len(candidates) > 1:
            message = f"{len(candidates)} emergency candidates; policy={self.policy}"
        elif self.controller.state in {SignalState.EMERGENCY_CLEARED, SignalState.RETURNING_TO_NORMAL}:
            message = "returning to normal sequence"
        cand_state = self.states.get(chosen.track_id) if chosen is not None else None
        decision = PriorityDecision(
            state=self.controller.state.value,
            message=message,
            active_lane=active_lane,
            active_track_id=self._active_track_id,
            candidate_track_id=chosen.track_id if chosen is not None else (active_event.track_id if active_event else None),
            candidate_class=(
                chosen.class_name if chosen is not None else (active_event.vehicle_class if active_event else None)
            ),
            confidence=(
                chosen.confidence if chosen is not None else (active_event.arrival_confidence if active_event else None)
            ),
            waiting_time_s=(active_event.waiting_time_s if active_event else None),
            e2e_latency_ms=(active_event.e2e_latency_ms if active_event else None),
            should_activate=bool(self._active_track_id is not None),
            controller_state=self.controller.state.value,
            lane_states=self.controller.lane_states,
        )
        self.last_decision = decision
        return decision

    def event_rows(self) -> List[Dict[str, object]]:
        return [event.to_dict() for event in self.events]

    def waiting_summary(self) -> Dict[str, object]:
        waits = [e.waiting_time_s for e in self.events if e.waiting_time_s is not None]
        baselines = [e.baseline_wait_s for e in self.events if e.baseline_wait_s is not None]
        durations = [e.priority_duration_s for e in self.events if e.priority_duration_s is not None]
        reduction = None
        pct = None
        if waits and baselines:
            avg_w = sum(waits) / len(waits)
            avg_b = sum(baselines) / len(baselines)
            reduction = avg_b - avg_w
            pct = (reduction / avg_b * 100.0) if avg_b > 0 else 0.0
        return {
            "priority_events": len(self.events),
            "priority_activations": sum(1 for e in self.events if e.green_time is not None),
            "avg_emergency_wait_s": (sum(waits) / len(waits)) if waits else None,
            "max_emergency_wait_s": max(waits) if waits else None,
            "avg_priority_duration_s": (sum(durations) / len(durations)) if durations else None,
            "avg_baseline_wait_s": (sum(baselines) / len(baselines)) if baselines else None,
            "waiting_time_reduction_s": reduction,
            "waiting_time_reduction_pct": pct,
            "multiple_emergency_max": self.max_simultaneous,
        }
