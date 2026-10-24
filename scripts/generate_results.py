"""Derive report tables strictly from result files produced by actual runs.

Nothing in this script invents a metric. When a result file is absent, the
corresponding table is emitted with an explicit "Not evaluated" note so the
report cannot silently substitute placeholder numbers.

Outputs
-------
    results/report_tables.md
    results/metrics/report_tables.json

Run:
    python scripts/generate_results.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "results"
METRICS = RESULTS / "metrics"
SIMULATION = RESULTS / "simulation"
TRAINING = RESULTS / "training"


# --------------------------------------------------------------------- helpers
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "Not evaluated"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "Not evaluated"
    return f"{number:.{digits}f}"


def _int(value: Any) -> str:
    if value is None or value == "":
        return "Not evaluated"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _mean_sd(values: List[float]) -> str:
    if not values:
        return "Not evaluated"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return f"{mean:.2f} ± {math.sqrt(variance):.2f}"


# ------------------------------------------------------------------- sections
def dataset_section() -> Dict[str, Any]:
    report = _read_json(METRICS / "dataset_validation.json")
    payload: Dict[str, Any] = {"available": bool(report), "splits": {}, "status": None}
    if not report:
        return payload
    payload["status"] = report.get("status")
    payload["errors"] = report.get("errors", [])
    payload["classes"] = report.get("classes", [])
    for split, data in (report.get("splits") or {}).items():
        payload["splits"][split] = {
            "images": data.get("images"),
            "boxes": data.get("boxes"),
            "class_counts": data.get("class_counts", {}),
        }
    return payload


def training_section() -> Dict[str, Any]:
    metadata = _read_json(TRAINING / "training_metadata.json")
    rows = _read_csv(TRAINING / "results.csv")
    payload: Dict[str, Any] = {
        "available": bool(metadata),
        "metadata": metadata,
        "epochs_recorded": len(rows),
        "best_epoch": None,
        "best": None,
    }
    if rows:
        parsed: List[Dict[str, float]] = []
        for row in rows:
            clean: Dict[str, float] = {}
            for key, value in row.items():
                try:
                    clean[key.strip()] = float(str(value).strip())
                except (TypeError, ValueError):
                    continue
            if clean:
                parsed.append(clean)
        if parsed:
            map_key = next((k for k in parsed[-1] if "mAP50-95" in k and "metrics" in k), None)
            if map_key is not None:
                best_row = max(parsed, key=lambda r: r.get(map_key, -1.0))
                # Ultralytics writes 1-based epochs in results.csv; do not offset.
                payload["best_epoch"] = int(best_row.get("epoch", 0))
                payload["best"] = {
                    "precision": best_row.get("metrics/precision(B)"),
                    "recall": best_row.get("metrics/recall(B)"),
                    "mAP50": best_row.get("metrics/mAP50(B)"),
                    "mAP50-95": best_row.get(map_key),
                }
            payload["final"] = {
                "precision": parsed[-1].get("metrics/precision(B)"),
                "recall": parsed[-1].get("metrics/recall(B)"),
                "mAP50": parsed[-1].get("metrics/mAP50(B)"),
                "mAP50-95": parsed[-1].get(map_key) if map_key else None,
            }
    return payload


def detection_section() -> Dict[str, Any]:
    data = _read_json(METRICS / "detection_metrics.json")
    if not data:
        return {"available": False}
    op = data.get("operating_point") or {}
    return {
        "available": data.get("status") == "evaluated",
        "model": data.get("model"),
        "dataset": data.get("dataset"),
        "split": data.get("split"),
        "overall": data.get("overall", {}),
        "per_class": data.get("per_class", []),
        "operating_point": {
            "tp": op.get("tp"),
            "fp": op.get("fp"),
            "fn": op.get("fn"),
            "precision": op.get("precision"),
            "recall": op.get("recall"),
            "f1": op.get("f1"),
            "conf": op.get("conf_threshold"),
            "iou": op.get("iou_match"),
            "images": op.get("images"),
            "roc": op.get("roc", {}),
        },
        "speed_ms_per_image": data.get("speed_ms_per_image", {}),
    }


def simulation_section() -> Dict[str, Any]:
    data = _read_json(METRICS / "comparison.json")
    sweep = _read_csv(SIMULATION / "density_sweep.csv")
    payload: Dict[str, Any] = {"available": bool(data), "sweep_available": bool(sweep)}
    if data:
        payload["scenario"] = data.get("scenario", {})
        payload["baseline"] = data.get("baseline", {})
        payload["proposed"] = data.get("proposed", {})
        payload["priority_events"] = data.get("priority_events", [])
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in sweep:
        density = row.get("density", "unknown")
        bucket = grouped.setdefault(
            density,
            {
                "baseline_ev": [],
                "proposed_ev": [],
                "baseline_normal": [],
                "proposed_normal": [],
                "baseline_throughput": [],
                "proposed_throughput": [],
                "priority_events": [],
            },
        )
        for key, target in (
            ("baseline_emergency_wait_s", "baseline_ev"),
            ("proposed_emergency_wait_s", "proposed_ev"),
            ("baseline_normal_wait_s", "baseline_normal"),
            ("proposed_normal_wait_s", "proposed_normal"),
            ("baseline_throughput_vph", "baseline_throughput"),
            ("proposed_throughput_vph", "proposed_throughput"),
            ("priority_events", "priority_events"),
        ):
            try:
                bucket[target].append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
    payload["sweep_grouped"] = grouped
    return payload


def pipeline_section() -> Dict[str, Any]:
    summary = _read_json(METRICS / "pipeline_summary.json")
    performance = _read_json(METRICS / "performance_summary.json")
    if not summary:
        return {"available": False}
    priority = summary.get("priority", {}) or {}
    return {
        "available": True,
        "frames": summary.get("frames"),
        "model": summary.get("model", {}),
        "priority": priority,
        "events": summary.get("priority_events", []),
        "performance": performance or summary.get("performance", {}),
    }


# ------------------------------------------------------------------- markdown
def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Generated result tables",
        "",
        f"Generated from files under `results/` at {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        "> Every value below is read from an actual run artifact. `Not evaluated` means the "
        "corresponding experiment did not produce a value; it is never replaced with an estimate.",
        "",
    ]

    # Dataset.
    dataset = payload["dataset"]
    lines += ["## 1. Dataset audit", ""]
    if not dataset.get("available"):
        lines += ["Not evaluated — run `python scripts/validate_dataset.py`.", ""]
    else:
        lines += [
            f"Validation status: **{dataset.get('status')}** · classes: "
            + ", ".join(dataset.get("classes", [])),
            "",
        ]
        classes = dataset.get("classes", [])
        rows = []
        for split in ("train", "val", "test"):
            data = dataset.get("splits", {}).get(split)
            if not data:
                continue
            counts = data.get("class_counts", {})
            rows.append([split, _int(data.get("images")), _int(data.get("boxes"))] + [_int(counts.get(c)) for c in classes])
        lines += _table(["Split", "Images", "Boxes"] + classes, rows)
        lines += ["", f"Errors: {len(dataset.get('errors', []))}.", ""]

    # Training.
    training = payload["training"]
    lines += ["## 2. Training run", ""]
    if not training.get("available"):
        lines += ["Not evaluated — no `results/training/training_metadata.json` found.", ""]
    else:
        meta = training.get("metadata") or {}
        lines += _table(
            ["Field", "Value"],
            [
                ["Device", meta.get("device")],
                ["Image size", meta.get("imgsz")],
                ["Batch", meta.get("batch")],
                ["Epochs requested", meta.get("epochs_requested")],
                ["Elapsed", meta.get("elapsed_human")],
                ["Recorded epochs", training.get("epochs_recorded")],
                ["Best epoch", training.get("best_epoch")],
            ],
        )
        if training.get("best"):
            best = training["best"]
            lines += [
                "",
                "Best validation checkpoint (epoch "
                f"{training.get('best_epoch')}): precision {_num(best.get('precision'), 3)}, "
                f"recall {_num(best.get('recall'), 3)}, mAP@50 {_num(best.get('mAP50'), 3)}, "
                f"mAP@50-95 {_num(best.get('mAP50-95'), 3)}.",
                "",
            ]
        else:
            lines += ["", "Per-epoch metrics not copied yet; the held-out test results below take precedence.", ""]

    # Detection.
    detection = payload["detection"]
    lines += ["## 3. Held-out detector evaluation", ""]
    if not detection.get("available"):
        lines += ["Not evaluated — run `python scripts/evaluate.py` after training.", ""]
    else:
        lines += [
            f"Model: `{detection.get('model')}` · split: `{detection.get('split')}` · "
            f"dataset: `{detection.get('dataset')}`",
            "",
        ]
        overall = detection.get("overall", {})
        lines += _table(
            ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"],
            [[_num(overall.get(k), 3) for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")]],
        )
        if detection.get("per_class"):
            lines += ["", "### Per-class test metrics", ""]
            lines += _table(
                ["Class", "Instances", "Precision", "Recall", "F1", "mAP@50", "mAP@50-95"],
                [
                    [
                        row.get("class"),
                        _int(row.get("instances")),
                        _num(row.get("precision"), 3),
                        _num(row.get("recall"), 3),
                        _num(row.get("f1"), 3),
                        _num(row.get("mAP50"), 3),
                        _num(row.get("mAP50-95"), 3),
                    ]
                    for row in detection["per_class"]
                ],
            )
        op = detection.get("operating_point", {})
        lines += ["", "### Operating point", ""]
        lines += _table(
            ["Conf", "IoU", "Images", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            [
                [
                    _num(op.get("conf"), 2),
                    _num(op.get("iou"), 2),
                    _int(op.get("images")),
                    _int(op.get("tp")),
                    _int(op.get("fp")),
                    _int(op.get("fn")),
                    _num(op.get("precision"), 3),
                    _num(op.get("recall"), 3),
                    _num(op.get("f1"), 3),
                ]
            ],
        )
        roc = op.get("roc", {})
        if roc.get("available"):
            lines += ["", f"Auxiliary image-level ROC AUC: **{_num(roc.get('auc'), 3)}** "
                      "(image-level presence, not a detection mAP metric).", ""]
        else:
            lines += ["", "Auxiliary ROC: **Not evaluated** — " + str(roc.get("reason", "unavailable")) + ".", ""]
        speed = detection.get("speed_ms_per_image", {})
        if speed:
            lines += ["Validation speed (ms/image): " + ", ".join(f"{k}={_num(v, 1)}" for k, v in speed.items()) + ".", ""]

    # Simulation.
    simulation = payload["simulation"]
    lines += ["## 4. Simulated fixed-time vs emergency-priority control", ""]
    if not simulation.get("available"):
        lines += ["Not evaluated — run `python simulation.py`."]
    else:
        scenario = simulation.get("scenario", {})
        lines += [
            f"Seed {scenario.get('seed')} · duration {scenario.get('duration_s')} s · "
            f"{_int(scenario.get('vehicles'))} vehicles · {_int(scenario.get('emergency_vehicles'))} emergency vehicles. "
            "Signal-level simulation only.",
            "",
        ]
        b = simulation.get("baseline", {})
        p = simulation.get("proposed", {})
        lines += _table(
            ["Metric", "Fixed-time baseline", "Emergency priority", "Change"],
            [
                [
                    "Emergency mean wait (s)",
                    _num(b.get("emergency_waiting_time_s")),
                    _num(p.get("emergency_waiting_time_s")),
                    _num(_diff(b.get("emergency_waiting_time_s"), p.get("emergency_waiting_time_s"))),
                ],
                [
                    "Emergency max wait (s)",
                    _num(b.get("emergency_max_wait_s")),
                    _num(p.get("emergency_max_wait_s")),
                    _num(_diff(b.get("emergency_max_wait_s"), p.get("emergency_max_wait_s"))),
                ],
                [
                    "Normal mean wait (s)",
                    _num(b.get("normal_waiting_time_s")),
                    _num(p.get("normal_waiting_time_s")),
                    _num(_diff(b.get("normal_waiting_time_s"), p.get("normal_waiting_time_s"))),
                ],
                [
                    "Normal max wait (s)",
                    _num(b.get("normal_max_wait_s")),
                    _num(p.get("normal_max_wait_s")),
                    _num(_diff(b.get("normal_max_wait_s"), p.get("normal_max_wait_s"))),
                ],
                [
                    "Emergency vehicles served",
                    f"{_int(b.get('emergency_served'))} / {_int(b.get('emergency_total'))}",
                    f"{_int(p.get('emergency_served'))} / {_int(p.get('emergency_total'))}",
                    "—",
                ],
                [
                    "Throughput (veh/h)",
                    _num(b.get("throughput_vph"), 0),
                    _num(p.get("throughput_vph"), 0),
                    _num(_diff(b.get("throughput_vph"), p.get("throughput_vph")), 0),
                ],
                [
                    "Priority activations",
                    _int(b.get("priority_events")),
                    _int(p.get("priority_events")),
                    "—",
                ],
                [
                    "Priority response (s)",
                    "Not applicable",
                    _num(p.get("priority_response_time_s")),
                    "—",
                ],
            ],
        )
        events = simulation.get("priority_events", [])
        if events:
            lines += ["", "### Measured priority events", ""]
            lines += _table(
                ["#", "Class", "Lane", "Arrival (s)", "Green (s)", "Wait (s)", "Baseline wait (s)", "Duration (s)"],
                [
                    [
                        e.get("event_id"),
                        e.get("vehicle_class"),
                        e.get("lane"),
                        _num(e.get("arrival_time")),
                        _num(e.get("green_time")),
                        _num(e.get("waiting_time_s")),
                        _num(e.get("baseline_wait_s")),
                        _num(e.get("priority_duration_s")),
                    ]
                    for e in events
                ],
            )
        lines += [
            "",
            "The normal-vehicle maximum wait can rise even when its mean falls: serving an EV "
            "out of sequence necessarily delays at least one conflicting movement.",
        ]

    # Sweep.
    lines += ["", "## 5. Arrival-rate sweep", ""]
    grouped = simulation.get("sweep_grouped", {})
    if not grouped:
        lines += ["Not evaluated — run `python simulation.py --sweep`."]
    else:
        rows = []
        for density in ("low", "medium", "high"):
            data = grouped.get(density)
            if not data:
                continue
            rows.append(
                [
                    density,
                    _mean_sd(data.get("baseline_ev", [])),
                    _mean_sd(data.get("proposed_ev", [])),
                    _mean_sd(data.get("baseline_normal", [])),
                    _mean_sd(data.get("proposed_normal", [])),
                    _mean_sd(data.get("baseline_throughput", [])),
                    _mean_sd(data.get("proposed_throughput", [])),
                    _int(sum(data.get("priority_events", []))),
                ]
            )
        lines += _table(
            [
                "Density",
                "Baseline EV wait (s)",
                "Proposed EV wait (s)",
                "Baseline normal wait (s)",
                "Proposed normal wait (s)",
                "Baseline throughput",
                "Proposed throughput",
                "Priority events (all seeds)",
            ],
            rows,
        )
        seeds = sorted({int(r["seed"]) for r in _read_csv(SIMULATION / "density_sweep.csv") if r.get("seed", "").isdigit()}) if simulation.get("sweep_available") else []
        lines += ["", f"Mean ± sample standard deviation across seeds {seeds}.", ""]

    # Live pipeline.
    pipeline = payload["pipeline"]
    lines += ["## 6. Live computer-vision pipeline", ""]
    if not pipeline.get("available"):
        lines += ["Not evaluated — run `python main.py --source video --input <video> --save-video`."]
    else:
        priority = pipeline.get("priority", {})
        perf = pipeline.get("performance", {})
        total = perf.get("total_processing_ms") or {}
        detection = perf.get("detection_ms") or {}
        lines += _table(
            ["Metric", "Measured value"],
            [
                ["Frames processed", _int(pipeline.get("frames"))],
                ["Priority events", _int(priority.get("priority_events"))],
                ["Priority activations", _int(priority.get("priority_activations"))],
                ["Average measured EV wait (s)", _num(priority.get("avg_emergency_wait_s"))],
                ["Average fixed-time baseline wait (s)", _num(priority.get("avg_baseline_wait_s"))],
                ["Measured reduction (s)", _num(priority.get("waiting_time_reduction_s"))],
                ["Measured reduction (%)", _num(priority.get("waiting_time_reduction_pct"))],
                ["Mean FPS", _num(perf.get("fps_mean"), 2)],
                ["Mean frame latency (ms)", _num(total.get("mean_ms"), 1)],
                ["P95 frame latency (ms)", _num(total.get("p95_ms"), 1)],
                ["Mean YOLO inference (ms)", _num(detection.get("mean_ms"), 1)],
            ],
        )
        lines += ["", "Live values come from one session; they are not a benchmark across hardware.", ""]

    return "\n".join(lines).strip() + "\n"


def _diff(baseline: Any, proposed: Any) -> Optional[float]:
    try:
        if baseline is None or proposed is None:
            return None
        return float(proposed) - float(baseline)
    except (TypeError, ValueError):
        return None


def main() -> int:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_section(),
        "training": training_section(),
        "detection": detection_section(),
        "simulation": simulation_section(),
        "pipeline": pipeline_section(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    json_path = METRICS / "report_tables.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = RESULTS / "report_tables.md"
    markdown = build_markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
