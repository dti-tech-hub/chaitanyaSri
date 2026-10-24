"""Baseline vs proposed controller comparison helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "Not evaluated" if value is None else f"{value:.{digits}f}"


def comparison_rows(baseline: Dict[str, float], proposed: Dict[str, float]) -> List[Dict[str, object]]:
    definitions = [
        ("emergency_waiting_time_s", "Average emergency waiting time (s)", "lower"),
        ("emergency_max_wait_s", "Maximum emergency waiting time (s)", "lower"),
        ("normal_waiting_time_s", "Average normal vehicle waiting time (s)", "lower"),
        ("priority_response_time_s", "Priority response time (s)", "lower"),
        ("throughput_vph", "Vehicle throughput (vehicles/hour)", "higher"),
        ("priority_events", "Priority activations", "higher"),
    ]
    rows: List[Dict[str, object]] = []
    for key, label, direction in definitions:
        b = baseline.get(key)
        p = proposed.get(key)
        change = None
        if b is not None and p is not None:
            change = p - b
        rows.append(
            {
                "metric": label,
                "key": key,
                "baseline": b,
                "proposed": p,
                "change": change,
                "better": direction,
                "available": b is not None and p is not None,
            }
        )
    return rows


def save_comparison_csv(rows: Iterable[Dict[str, object]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "key", "baseline", "proposed", "change", "better", "available"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def save_comparison_json(payload: Dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def comparison_markdown(rows: Iterable[Dict[str, object]]) -> str:
    lines = [
        "| Metric | Fixed-time baseline | Proposed priority | Change |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        b = row.get("baseline")
        p = row.get("proposed")
        change = row.get("change")
        lines.append(
            f"| {row['metric']} | {_fmt(b) if b is not None else 'Not evaluated'} | "
            f"{_fmt(p) if p is not None else 'Not evaluated'} | "
            f"{('+' if isinstance(change, (int, float)) and change > 0 else '') + _fmt(change) if change is not None else '-'} |"
        )
    return "\n".join(lines)
