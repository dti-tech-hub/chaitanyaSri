"""Configuration loading and validation for the emergency-vehicle prototype."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when configuration is missing or inconsistent."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "path": "models/yolo/best.pt",
        "confidence": 0.55,
        "iou": 0.45,
        "imgsz": 640,
        "device": "auto",
    },
    "priority": {
        "confidence_threshold": 0.60,
        "confirmation_frames": 3,
        "cooldown_seconds": 10.0,
    },
    "signals": {
        "green_seconds": 20,
        "yellow_seconds": 3,
        "all_red_seconds": 1,
    },
    "density": {"enabled": True, "low_max": 3, "medium_max": 8},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root."""
    p = Path(os.path.expandvars(str(path))).expanduser()
    if not p.is_absolute():
        p = (root or PROJECT_ROOT) / p
    return p


def _require(mapping: Dict[str, Any], dotted: str, expected_type: type | tuple[type, ...] | None = None) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"Missing required configuration key: {dotted}")
        cur = cur[part]
    if expected_type is not None and not isinstance(cur, expected_type):
        raise ConfigError(f"Configuration key {dotted} must be {expected_type}, got {type(cur)}")
    return cur


class AppConfig:
    """Thin, validated wrapper around the project YAML files.

    The object is intentionally simple so experiments can override values at
    runtime (the Streamlit dashboard does this) without mutating global state.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        intersection: Dict[str, Any] | None = None,
        root: Path | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.data = data
        self.intersection = intersection or {}
        self.root = Path(root or PROJECT_ROOT)
        self.source_path = source_path
        self.validate()

    @classmethod
    def load(
        cls,
        path: str | Path = "config/config.yaml",
        intersection_path: str | Path | None = None,
        overrides: Dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> "AppConfig":
        root = Path(root or PROJECT_ROOT)
        config_path = resolve_path(path, root)
        if not config_path.is_file():
            raise ConfigError(f"Configuration file not found: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - depends on malformed input
            raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
        data = _deep_merge(DEFAULT_CONFIG, raw)
        if overrides:
            data = _deep_merge(data, overrides)

        inter_path = resolve_path(
            intersection_path or data.get("lanes", {}).get("config_path", "config/intersection.yaml"),
            root,
        )
        if not inter_path.is_file():
            raise ConfigError(f"Intersection geometry file not found: {inter_path}")
        try:
            intersection = yaml.safe_load(inter_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ConfigError(f"Invalid YAML in {inter_path}: {exc}") from exc
        return cls(data, intersection, root=root, source_path=config_path)

    # ------------------------------------------------------------------ access
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, dotted: str, default: str | None = None) -> Path:
        value = self.get(dotted, default)
        if value is None:
            raise ConfigError(f"No path configured at {dotted}")
        return resolve_path(value, self.root)

    @property
    def emergency_classes(self) -> list[str]:
        return list(self.get("emergency_classes", []) or [])

    @property
    def vehicle_classes(self) -> list[str]:
        return list(self.get("vehicle_classes", []) or [])

    def ensure_directories(self) -> None:
        for dotted in (
            "paths.results_dir",
            "paths.metrics_dir",
            "paths.figures_dir",
            "paths.events_dir",
            "paths.logs_dir",
            "paths.simulation_dir",
        ):
            value = self.get(dotted)
            if value:
                self.path(dotted).mkdir(parents=True, exist_ok=True)

    def effective_device(self) -> str:
        import torch

        requested = str(self.get("model.device", "auto")).lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ConfigError("CUDA requested but torch.cuda.is_available() is False")
        return requested

    def save_effective(self, path: str | Path | None = None) -> Path:
        target = resolve_path(path or "results/logs/effective_config.yaml", self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.data, "intersection": self.intersection}
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return target

    # ---------------------------------------------------------------- validate
    def validate(self) -> None:
        _require(self.data, "model.confidence", (int, float))
        _require(self.data, "model.iou", (int, float))
        _require(self.data, "priority.confidence_threshold", (int, float))
        _require(self.data, "priority.confirmation_frames", int)
        _require(self.data, "signals.green_seconds", (int, float))
        _require(self.data, "signals.yellow_seconds", (int, float))
        _require(self.data, "signals.all_red_seconds", (int, float))
        if not 0 <= float(self.get("model.confidence")) <= 1:
            raise ConfigError("model.confidence must be within [0, 1]")
        if not 0 <= float(self.get("model.iou")) <= 1:
            raise ConfigError("model.iou must be within [0, 1]")
        if int(self.get("priority.confirmation_frames")) < 1:
            raise ConfigError("priority.confirmation_frames must be >= 1")
        if float(self.get("signals.green_seconds")) <= 0:
            raise ConfigError("signals.green_seconds must be > 0")
        if float(self.get("signals.yellow_seconds")) < 0 or float(self.get("signals.all_red_seconds")) < 0:
            raise ConfigError("signal yellow/all-red durations cannot be negative")
        if self.get("density.enabled", True):
            low = int(self.get("density.low_max"))
            med = int(self.get("density.medium_max"))
            if low >= med:
                raise ConfigError("density.low_max must be smaller than density.medium_max")

        lanes = self.intersection.get("lanes")
        if not isinstance(lanes, dict) or not lanes:
            raise ConfigError("intersection.yaml must define at least one lane")
        for name, spec in lanes.items():
            poly = spec.get("polygon")
            if not isinstance(poly, list) or len(poly) < 3:
                raise ConfigError(f"lane {name} needs a polygon with at least 3 points")
            for pt in poly:
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    raise ConfigError(f"lane {name} polygon points must be [x, y]")
        groups = set()
        for name, spec in lanes.items():
            group = str(spec.get("group", name))
            groups.add(group)
        if len(groups) < 2:
            raise ConfigError("intersection must define at least two conflicting signal groups")

    # ------------------------------------------------------------- overrides
    def with_overrides(self, overrides: Dict[str, Any]) -> "AppConfig":
        return AppConfig(
            _deep_merge(self.data, overrides),
            copy.deepcopy(self.intersection),
            root=self.root,
            source_path=self.source_path,
        )


def load_config(path: str | Path = "config/config.yaml", **kwargs: Any) -> AppConfig:
    return AppConfig.load(path, **kwargs)
