"""Streamlit dashboard for the simulated emergency-vehicle signal prototype.

Run:
    streamlit run dashboard/app.py

Live processing uses the same EmergencySignalPipeline as main.py. Evaluation and
simulation tabs read generated result files; when a file is absent the dashboard
shows "Not evaluated" instead of inventing values.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import EmergencySignalPipeline
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.video import VideoSource

st.set_page_config(
    page_title="Emergency Vehicle Smart Signal (Simulated)",
    page_icon="🚦",
    layout="wide",
)

DEMO_VIDEOS = {
    "Aerial fire engine + ambulance (9784628)": "data/raw/ambu-test/9784628-uhd_2560_1440_24fps.mp4",
    "Aerial intersection emergency response (9784626)": "data/raw/ambu-test/9784626-uhd_2560_1440_24fps.mp4",
    "Street response compilation": "data/raw/ambu-test/Police Cars Fire Trucks And Ambulances Responding Compilation Part 12.mp4",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    .sim-warning {background:#fff3cd;color:#5c3c00;padding:12px 16px;border-radius:8px;
                  border-left:6px solid #f0ad4e;font-weight:600;margin-bottom:12px;}
    .metric-note {color:#666;font-size:0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🚑 Emergency Vehicle Detection & Smart Traffic Signal")
st.markdown(
    '<div class="sim-warning">SIMULATED PROTOTYPE — the signal controller is not connected to, '
    "nor validated on, real public-road infrastructure. All priority decisions are rendered in software only.</div>",
    unsafe_allow_html=True,
)

config = load_config(PROJECT_ROOT / "config" / "config.yaml")


def _resolve(relative: str) -> Path:
    return PROJECT_ROOT / relative


def _format(value, digits=2) -> str:
    if value is None:
        return "Not evaluated"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _read_json(relative: str):
    path = _resolve(relative)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _empty_message(what: str, command: str) -> None:
    st.warning(f"Not evaluated — {what} unavailable. Run: `{command}`")


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Live processing")
    source_kind = st.selectbox(
        "Source",
        ["Demo video", "Local video path", "Upload video", "Webcam", "Image"],
        index=0,
    )
    source_path: Path | None = None
    camera_index = 0
    is_live = False
    is_image = False
    if source_kind == "Demo video":
        choice = st.selectbox("Real emergency footage", list(DEMO_VIDEOS))
        source_path = _resolve(DEMO_VIDEOS[choice])
        if not source_path.is_file():
            st.error(f"Demo footage missing: {source_path}")
            source_path = None
    elif source_kind == "Local video path":
        raw = st.text_input("Path relative to project root", "data/raw/ambu-test/9784628-uhd_2560_1440_24fps.mp4")
        candidate = Path(raw)
        source_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if raw and not source_path.is_file():
            st.error(f"Video not found: {source_path}")
    elif source_kind == "Upload video":
        upload = st.file_uploader("Upload MP4/AVI/MOV", type=["mp4", "avi", "mov", "mkv"])
        if upload is not None:
            suffix = Path(upload.name).suffix or ".mp4"
            temp = Path(tempfile.gettempdir()) / "ev_smart_signal_uploads"
            temp.mkdir(parents=True, exist_ok=True)
            target = temp / f"upload_{int(time.time())}{suffix}"
            target.write_bytes(upload.getbuffer())
            source_path = target
    elif source_kind == "Webcam":
        camera_index = int(st.number_input("Camera index", min_value=0, max_value=8, value=0, step=1))
        source_path = None
        is_live = True
    else:
        raw = st.text_input("Image path relative to project root", "data/samples/ambulance_1.jpg")
        candidate = Path(raw)
        source_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        is_image = True
        if raw and not source_path.is_file():
            st.error(f"Image not found: {source_path}")

    max_frames = st.slider("Maximum frames", 10, 900, 240, 10)
    run_seconds = st.slider("Maximum wall-clock seconds", 10, 180, 45, 5)
    max_width = st.slider("Process width (px)", 640, 1600, 1100, 80)
    show_lanes = st.checkbox("Show calibrated lane ROIs", True)
    model_path = st.text_input("Model path", "models/yolo/best.pt")
    st.caption("The controller clock advances with video time. Webcam mode uses wall time.")
    start = st.button("▶ Start live simulation", type="primary", width="stretch")
    st.divider()
    st.caption(f"Working directory: {PROJECT_ROOT}")
    st.caption("No data leaves this machine.")

model_file = _resolve(model_path)
if model_file.is_file():
    st.sidebar.success("Model weights found")
else:
    st.sidebar.error("Model weights missing — train or point to a valid .pt file")
    st.sidebar.code("python scripts/train_yolo.py --epochs 30", language="powershell")

tab_live, tab_detection, tab_simulation, tab_dataset, tab_about = st.tabs(
    ["🎥 Live simulation", "🎯 Detection results", "🚦 Signal comparison", "🗂 Dataset audit", "ℹ️ Method"]
)

# ----------------------------------------------------------------------- live
with tab_live:
    left, right = st.columns([2.15, 1.0])
    frame_placeholder = left.empty()
    status_placeholder = left.empty()
    metrics_placeholder = right.empty()
    signal_placeholder = right.empty()

    if "last_summary" not in st.session_state:
        st.session_state["last_summary"] = None
        st.session_state["last_events"] = []
        st.session_state["last_latency"] = []

    if start:
        if source_path is None and not is_live:
            st.error("Choose a valid source first.")
        elif not model_file.is_file():
            st.error("Model weights not found. Train the custom detector before running live mode.")
        else:
            display_width = int(max_width)
            try:
                if is_live:
                    source = VideoSource("webcam", webcam_index=camera_index)
                elif is_image:
                    source = VideoSource("image", path=source_path)
                else:
                    source = VideoSource("video", path=source_path, loop=True)
                pipeline = EmergencySignalPipeline(
                    config,
                    model_path=model_file,
                    reset_events=True,
                    display_lanes=show_lanes,
                )
                pipeline.set_live_source(is_live)
                events = []
                latencies = []
                started = time.perf_counter()
                frames = 0
                progress = st.progress(0.0, text="Starting detection…")
                try:
                    for frame in source:
                        if frame.image.shape[1] > display_width:
                            scale = display_width / frame.image.shape[1]
                            frame.image = cv2.resize(
                                frame.image,
                                (display_width, int(frame.image.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        result = pipeline.process(frame, live=is_live)
                        frames += 1
                        elapsed = time.perf_counter() - started
                        if frames % 2 == 0 or frames == 1:
                            rgb = cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb, channels="RGB", width="stretch")
                            metrics = result.metrics
                            metrics_placeholder.metric("FPS", f"{metrics.get('fps', 0):.1f}")
                            metrics_placeholder.metric("YOLO latency", f"{metrics.get('inference_ms', 0):.0f} ms")
                            metrics_placeholder.metric("Frame latency", f"{metrics.get('total_ms', 0):.0f} ms")
                            metrics_placeholder.metric("Emergency tracks", metrics.get("emergency_count", 0))
                            status_placeholder.info(
                                f"Sim time {result.sim_time:6.1f}s | state **{result.signal.get('state')}** | "
                                f"phase `{result.signal.get('phase')}` | {result.priority.message}"
                            )
                            snap = pipeline.signal.snapshot(result.sim_time)
                            signal_placeholder.json(
                                {
                                    "state": snap.get("state"),
                                    "phase": snap.get("phase"),
                                    "lane_states": snap.get("lane_states"),
                                    "priority_lane": snap.get("priority_lane"),
                                    "cooldown_remaining_s": round(float(snap.get("cooldown_remaining_s") or 0), 2),
                                }
                            )
                        events = pipeline.priority.event_rows()
                        if metrics.get("e2e_priority_latency_ms") is not None:
                            latencies.append(float(metrics["e2e_priority_latency_ms"]))
                        progress.progress(
                            min(1.0, frames / max_frames),
                            text=f"Frame {frames} / {max_frames} | {elapsed:.0f}s",
                        )
                        if frames >= max_frames or elapsed >= run_seconds:
                            break
                finally:
                    source.release()
                    pipeline.close()
                st.session_state["last_summary"] = pipeline.summary()
                st.session_state["last_events"] = events
                st.session_state["last_latency"] = latencies
                progress.empty()
                st.success(f"Finished: {frames} frames, {elapsed:.1f}s wall time.")
            except Exception as exc:
                st.exception(exc)

    summary = st.session_state.get("last_summary")
    if summary:
        priority = summary.get("priority", {})
        perf = summary.get("performance", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Priority events", priority.get("priority_events", 0))
        c2.metric("Avg EV wait", _format(priority.get("avg_emergency_wait_s"), 1) + (" s" if priority.get("avg_emergency_wait_s") is not None else ""))
        c3.metric("Avg baseline wait", _format(priority.get("avg_baseline_wait_s"), 1) + (" s" if priority.get("avg_baseline_wait_s") is not None else ""))
        c4.metric("Mean FPS", _format(perf.get("fps_mean"), 1))
        c5.metric("Frames", perf.get("frames", 0))
        if summary.get("model", {}).get("emergency_capable") is False:
            st.error("Loaded model has no emergency classes. Priority activation is disabled by design.")
        st.subheader("Measured priority events")
        events = st.session_state.get("last_events") or summary.get("priority_events") or []
        if events:
            df = pd.DataFrame(events)
            st.dataframe(df, width="stretch", hide_index=True)
            waits = [float(e["waiting_time_s"]) for e in events if e.get("waiting_time_s") is not None]
            bases = [float(e["baseline_wait_s"]) for e in events if e.get("baseline_wait_s") is not None]
            if waits and bases:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7.2, 3.4))
                ax.bar(["Measured EV wait", "Fixed-time baseline"], [np.mean(waits), np.mean(bases)], color=["#2a9d8f", "#8d99ae"])
                ax.set_ylabel("Seconds")
                ax.set_title("Actual events in this session (not a benchmark)")
                for i, val in enumerate([np.mean(waits), np.mean(bases)]):
                    ax.text(i, val + 0.15, f"{val:.2f}s", ha="center")
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No emergency-priority events were measured in this run. This is not a result of zero waiting time.")
        with st.expander("Session performance summary"):
            st.json(perf)
            st.caption("Latency is wall-clock on this machine; it excludes display and video writing.")

# ------------------------------------------------------------------ detection
with tab_detection:
    data = _read_json("results/metrics/detection_metrics.json")
    if not data or data.get("status") != "evaluated":
        _empty_message("detection metrics", "python scripts/evaluate.py")
    else:
        overall = data.get("overall", {})
        st.subheader("Held-out test split")
        st.caption(f"Model: `{data.get('model')}` | Dataset: `{data.get('dataset')}` | Split: `{data.get('split')}`")
        cols = st.columns(5)
        cols[0].metric("Precision", _format(overall.get("precision"), 3))
        cols[1].metric("Recall", _format(overall.get("recall"), 3))
        cols[2].metric("F1", _format(overall.get("f1"), 3))
        cols[3].metric("mAP@50", _format(overall.get("mAP50"), 3))
        cols[4].metric("mAP@50-95", _format(overall.get("mAP50-95"), 3))
        per_class = data.get("per_class", [])
        if per_class:
            st.dataframe(pd.DataFrame(per_class), width="stretch", hide_index=True)
        op = data.get("operating_point", {})
        if op.get("available"):
            st.markdown(
                f"**Operating point:** conf={op.get('conf_threshold')} | IoU≥{op.get('iou_match')} | "
                f"TP={op.get('tp')} FP={op.get('fp')} FN={op.get('fn')} | "
                f"precision={_format(op.get('precision'), 3)} recall={_format(op.get('recall'), 3)}"
            )
        speed = data.get("speed_ms_per_image")
        if speed:
            st.caption("Validation speed (ms/image): " + ", ".join(f"{k}={v}" for k, v in speed.items()))
        fig_path = _resolve("results/figures/detection_metrics.png")
        if fig_path.is_file():
            st.image(str(fig_path), width=720)
        roc = op.get("roc", {}) if op else {}
        if roc.get("available"):
            roc_path = _resolve("results/figures/roc_curve.png")
            if roc_path.is_file():
                st.image(str(roc_path), width=520, caption=f"Auxiliary image-level ROC (AUC={roc.get('auc'):.3f})")
        else:
            st.caption("Auxiliary ROC not available: the test split has no negative images or model outputs.")

# ----------------------------------------------------------------- simulation
with tab_simulation:
    sim = _read_json("results/metrics/comparison.json")
    if not sim:
        _empty_message("signal comparison", "python simulation.py")
    else:
        b = sim.get("baseline", {})
        p = sim.get("proposed", {})
        st.subheader("Fixed-time baseline vs emergency priority")
        st.caption(
            f"Deterministic signal-level simulation | seed={sim.get('scenario', {}).get('seed')} | "
            f"duration={sim.get('scenario', {}).get('duration_s')}s | "
            f"{sim.get('scenario', {}).get('vehicles')} vehicles"
        )
        table_rows = [
            ("Emergency mean wait (s)", b.get("emergency_waiting_time_s"), p.get("emergency_waiting_time_s")),
            ("Emergency max wait (s)", b.get("emergency_max_wait_s"), p.get("emergency_max_wait_s")),
            ("Normal mean wait (s)", b.get("normal_waiting_time_s"), p.get("normal_waiting_time_s")),
            ("Throughput (veh/h)", b.get("throughput_vph"), p.get("throughput_vph")),
            ("Priority activations", b.get("priority_events"), p.get("priority_events")),
        ]
        st.dataframe(
            pd.DataFrame(table_rows, columns=["Metric", "Fixed-time baseline", "Proposed priority"]),
            width="stretch",
            hide_index=True,
        )
        fig_path = _resolve("results/figures/simulation_waiting_time.png")
        if fig_path.is_file():
            st.image(str(fig_path), width=760)
        st.info(
            "The simulation is microscopic and signal-level: it models arrivals and green onsets, "
            "not queue spillback, pedestrians, or road safety."
        )
        dist_path = _resolve("results/figures/simulation_wait_distribution.png")
        if dist_path.is_file():
            st.image(str(dist_path), width=620)
        events = sim.get("priority_events", [])
        if events:
            st.subheader("Priority events")
            st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
        sweep_path = _resolve("results/simulation/density_sweep.csv")
        if sweep_path.is_file():
            st.subheader("Arrival-rate sweep")
            st.dataframe(pd.read_csv(sweep_path), width="stretch", hide_index=True)

# -------------------------------------------------------------------- dataset
with tab_dataset:
    report = _read_json("results/metrics/dataset_validation.json")
    if not report:
        _empty_message("dataset audit", "python scripts/validate_dataset.py")
    else:
        st.subheader("Dataset audit (real labelled data)")
        if report.get("status") == "ok":
            st.success("Structural validation passed")
        else:
            st.error(f"Validation failed with {len(report.get('errors', []))} error(s)")
            for error in report.get("errors", [])[:20]:
                st.write("•", error)
        rows = []
        for split, s in report.get("splits", {}).items():
            rows.append({"split": split, **{k: v for k, v in s.items() if k != "class_counts"}, **{f"class_{k}": v for k, v in s.get("class_counts", {}).items()}})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        sheet = _resolve("results/figures/dataset_samples.png")
        if sheet.is_file():
            st.image(str(sheet), caption="One annotated example per class from the training split", width=760)
        st.caption("Source datasets and licenses: docs/DATASETS.md. Fire-engine labels are scarce; see the report limitations.")

# ---------------------------------------------------------------------- about
with tab_about:
    st.subheader("How the decision works")
    st.markdown(
        """
        1. A custom YOLO model detects `ambulance`, `fire_engine`, `police_vehicle` and `normal_vehicle`.
        2. A dependency-free IoU tracker assigns persistent IDs; lane assignment is made only when the
           detection centre is inside a calibrated approach polygon.
        3. An emergency request requires **confidence ≥ threshold**, **N consecutive frames**,
           **minimum track duration**, a **known lane**, and **approaching/queued motion**.
        4. The simulated controller transitions to all-red, then a protected priority green.
           When the EV reaches the far-side exit ROI (after a short hold), it clears, serves an all-red
           clearance, and restores the normal sequence. A cooldown prevents immediate re-triggering.
        5. Invalid signal-state jumps raise `InvalidTransition`; detection failure forces all-red.
        """
    )
    diagram = _resolve("docs/architecture.png")
    if diagram.is_file():
        st.image(str(diagram), caption="System architecture", width=900)
    st.subheader("Configuration snapshot")
    st.json(
        {
            "model_path": config.get("model.path"),
            "model_confidence": config.get("model.confidence"),
            "priority_confidence": config.get("priority.confidence_threshold"),
            "confirmation_frames": config.get("priority.confirmation_frames"),
            "cooldown_seconds": config.get("priority.cooldown_seconds"),
            "hold_after_exit_seconds": config.get("priority.hold_after_exit_seconds"),
            "green_seconds": config.get("signals.green_seconds"),
            "yellow_seconds": config.get("signals.yellow_seconds"),
            "all_red_seconds": config.get("signals.all_red_seconds"),
        }
    )
    st.warning("This is a simulated controller. Do not deploy it to real traffic infrastructure.")
