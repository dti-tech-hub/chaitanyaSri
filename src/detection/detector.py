"""YOLO detector with safe mapping from dataset class names to project classes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.utils.config import AppConfig, ConfigError, PROJECT_ROOT, resolve_path
from src.utils.logger import setup_logger

LOGGER = setup_logger("detector")


@dataclass
class Detection:
    bbox: np.ndarray                    # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    raw_class_name: str
    class_name: str                     # canonical: ambulance/fire_engine/...
    is_emergency: bool
    track_id: Optional[int] = None
    lane: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "raw_class_name": self.raw_class_name,
            "class_name": self.class_name,
            "is_emergency": bool(self.is_emergency),
            "track_id": self.track_id,
            "lane": self.lane,
        }


def _norm(name: str) -> str:
    return " ".join(
        str(name).lower().replace("_", " ").replace("-", " ").replace(".", " ").split()
    )


class YOLODetector:
    """Wrapper around Ultralytics YOLO.

    The detector never invents an emergency class. If the loaded weights do not
    contain any emergency class, ``emergency_capable`` is False and callers must
    refuse to activate priority.
    """

    def __init__(self, config: AppConfig, model_path: str | Path | None = None, warmup: bool = True) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - environment failure
            raise RuntimeError(f"Ultralytics could not be imported: {exc}") from exc

        self.config = config
        configured = model_path or config.get("model.path", "models/yolo/best.pt")
        path = resolve_path(configured, config.root)
        fallback = resolve_path(config.get("model.fallback_path", "models/yolo/yolov8n.pt"), config.root)
        self.requested_path = path
        if not path.is_file():
            if fallback.is_file() and path != fallback:
                LOGGER.warning("Custom model not found at %s; using fallback %s", path, fallback)
                path = fallback
            else:
                raise FileNotFoundError(
                    "YOLO weights not found. Train a custom model with "
                    "`python scripts/train_yolo.py` or set model.path in config/config.yaml. "
                    f"Tried: {path}"
                )
        self.model_path = path
        self.model = YOLO(str(path))
        self.model_names: Dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self.alias_map = self._build_alias_map(config.get("model.class_aliases", {}) or {})
        self.canonical_by_id: Dict[int, str] = {
            cid: self.alias_map.get(_norm(name), "unknown") for cid, name in self.model_names.items()
        }
        self.available_classes = sorted({v for v in self.canonical_by_id.values() if v != "unknown"})
        self.emergency_capable = bool(set(self.available_classes) & set(config.emergency_classes))
        self.missing_emergency = sorted(set(config.emergency_classes) - set(self.available_classes))
        self.device = config.effective_device()
        self.imgsz = int(config.get("model.imgsz", 640))
        self.conf = float(config.get("model.confidence", 0.55))
        self.iou = float(config.get("model.iou", 0.45))
        self.max_det = int(config.get("model.max_detections", 60))
        self.half = bool(config.get("model.half", False)) and self.device.startswith("cuda")
        self.verbose = bool(config.get("model.verbose", False))
        self.last_inference_ms = 0.0
        self.inference_times_ms: List[float] = []

        if not self.emergency_capable:
            LOGGER.warning(
                "The loaded model contains no canonical emergency classes. "
                "Emergency priority will remain disabled. Missing: %s",
                ", ".join(self.missing_emergency),
            )
        LOGGER.info("YOLO model: %s | classes=%s | device=%s", self.model_path, self.model_names, self.device)

        if warmup:
            self._warmup()

    @staticmethod
    def _build_alias_map(aliases: Dict[str, Iterable[str]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for canonical, names in aliases.items():
            for name in names:
                out[_norm(name)] = canonical
        return out

    def _warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on model/device
            LOGGER.warning("Warm-up inference failed: %s", exc)

    def detect(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf if conf is None else float(conf),
            iou=self.iou if iou is None else float(iou),
            device=self.device,
            max_det=self.max_det,
            half=self.half,
            verbose=self.verbose,
            classes=classes,
        )
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        self.inference_times_ms.append(self.last_inference_ms)
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)
        for bbox, confidence, class_id in zip(xyxy, confs, clss):
            canonical = self.canonical_by_id.get(int(class_id), "unknown")
            detections.append(
                Detection(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    raw_class_name=self.model_names.get(int(class_id), str(class_id)),
                    class_name=canonical,
                    is_emergency=canonical in self.config.emergency_classes,
                )
            )
        return detections

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0.0

    def info(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_names": self.model_names,
            "canonical_classes": self.canonical_by_id,
            "available_classes": self.available_classes,
            "emergency_capable": self.emergency_capable,
            "missing_emergency_classes": self.missing_emergency,
            "device": self.device,
            "imgsz": self.imgsz,
        }
