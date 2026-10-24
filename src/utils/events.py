"""Event logging to CSV/JSONL for the simulated signal prototype."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

FIELDS = [
    "timestamp_wall",
    "sim_time",
    "event_type",
    "vehicle_class",
    "confidence",
    "lane",
    "track_id",
    "signal_state",
    "phase",
    "priority_state",
    "waiting_time_s",
    "baseline_wait_s",
    "duration_s",
    "latency_ms",
    "message",
    "details",
]


class EventLogger:
    """Append-only event recorder. Keeps a human-readable CSV and JSONL mirror."""

    def __init__(
        self,
        csv_path: str | Path = "results/events/events.csv",
        jsonl_path: str | Path | None = None,
        reset: bool = True,
        archive: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else self.csv_path.with_suffix(".jsonl")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and archive and self.csv_path.is_file() and self.csv_path.stat().st_size > 0:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_dir = self.csv_path.parent / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.csv_path, archive_dir / f"{self.csv_path.stem}_{stamp}{self.csv_path.suffix}")
        self._first_write = True
        if reset:
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                writer.writeheader()
            with self.jsonl_path.open("w", encoding="utf-8"):
                pass
            self._first_write = False

    @staticmethod
    def _coerce(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, float):
            return round(value, 6)
        return value

    def log(self, event_type: str, sim_time: Optional[float] = None, **fields: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {field: "" for field in FIELDS}
        row.update(
            {
                "timestamp_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "sim_time": "" if sim_time is None else round(float(sim_time), 3),
                "event_type": event_type,
            }
        )
        for key in FIELDS:
            if key in fields:
                row[key] = self._coerce(fields[key])
        known = set(FIELDS)
        extras = {k: v for k, v in fields.items() if k not in known}
        if extras:
            existing = {}
            if row.get("details"):
                try:
                    existing = json.loads(str(row["details"]))
                except json.JSONDecodeError:
                    existing = {"raw": row["details"]}
            existing.update(extras)
            row["details"] = json.dumps(existing, ensure_ascii=False, default=str)
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if self._first_write:
                writer.writeheader()
                self._first_write = False
            writer.writerow(row)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def priority_started(self, sim_time: float, **fields: Any) -> None:
        self.log("priority_started", sim_time=sim_time, **fields)

    def priority_released(self, sim_time: float, **fields: Any) -> None:
        self.log("priority_released", sim_time=sim_time, **fields)

    def read_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.is_file():
            return []
        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


class CompositeEventLogger:
    """Broadcast one event to multiple loggers (e.g., run log + session log)."""

    def __init__(self, loggers: Iterable[EventLogger]) -> None:
        self.loggers = list(loggers)

    def log(self, event_type: str, sim_time: Optional[float] = None, **fields: Any) -> None:
        for logger in self.loggers:
            logger.log(event_type, sim_time=sim_time, **fields)
