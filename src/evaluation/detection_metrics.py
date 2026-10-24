"""Detection evaluation: mAP, precision/recall, FP/FN and auxiliary ROC."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from src.utils.config import AppConfig, resolve_path
from src.utils.logger import setup_logger

LOGGER = setup_logger("detection_metrics")


def _metric_row(metric: str, value: Any, class_name: str = "all", note: str = "") -> Dict[str, Any]:
    return {"metric": metric, "class": class_name, "value": value, "note": note}


def _read_dataset_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_split_path(data_yaml: Path, split: str) -> Optional[Path]:
    data = _read_dataset_yaml(data_yaml)
    value = data.get(split)
    if not value:
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = (data_yaml.parent / p).resolve()
    return p


def _to_list(obj) -> list:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj) if obj is not None else []


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = a + b - inter
    return inter / union if union > 1e-9 else 0.0


def _load_gt(label_path: Path, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    classes: List[int] = []
    boxes: List[List[float]] = []
    if not label_path.is_file():
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        classes.append(c)
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def count_fp_fn(
    model,
    data_yaml: Path,
    split: str,
    conf: float,
    iou_match: float = 0.5,
    imgsz: int = 640,
    device: str = "cpu",
    max_images: int = 0,
) -> Dict[str, Any]:
    split_path = _resolve_split_path(data_yaml, split)
    if split_path is None or not split_path.exists():
        return {"available": False, "reason": f"split {split} not found"}
    images = sorted(
        [p for p in split_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )
    if max_images:
        images = images[:max_images]
    labels_dir = data_yaml.parent / "labels" / split
    tp = fp = fn = 0
    total_gt = 0
    images_with_emergency = 0
    emergency_scores: List[float] = []
    emergency_labels: List[int] = []
    per_class_tp: Dict[int, int] = {}
    per_class_fp: Dict[int, int] = {}
    per_class_fn: Dict[int, int] = {}
    processed = 0
    for image_path in images:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        processed += 1
        h, w = image.shape[:2]
        gt_boxes, gt_classes = _load_gt(labels_dir / f"{image_path.stem}.txt", w, h)
        results = model.predict(
            image,
            imgsz=imgsz,
            conf=float(conf),
            iou=0.45,
            device=device,
            verbose=False,
            max_det=100,
        )
        pred_boxes = np.empty((0, 4), dtype=np.float32)
        pred_classes = np.empty((0,), dtype=np.int64)
        pred_conf = np.empty((0,), dtype=np.float32)
        if results and results[0].boxes is not None and len(results[0].boxes):
            b = results[0].boxes
            pred_boxes = b.xyxy.detach().cpu().numpy()
            pred_classes = b.cls.detach().cpu().numpy().astype(np.int64)
            pred_conf = b.conf.detach().cpu().numpy()
        has_gt_emergency = any(int(c) in {0, 1, 2} for c in gt_classes.tolist())
        if has_gt_emergency:
            images_with_emergency += 1
        emergency = pred_conf[np.isin(pred_classes, [0, 1, 2])]
        emergency_scores.append(float(emergency.max()) if len(emergency) else 0.0)
        emergency_labels.append(1 if has_gt_emergency else 0)
        total_gt += len(gt_classes)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        candidates: List[Tuple[float, int, int]] = []
        for pi in range(len(pred_boxes)):
            for gi in range(len(gt_boxes)):
                if int(pred_classes[pi]) != int(gt_classes[gi]):
                    continue
                score = _iou(pred_boxes[pi], gt_boxes[gi])
                if score >= iou_match:
                    candidates.append((score, pi, gi))
        candidates.sort(reverse=True, key=lambda x: x[0])
        for score, pi, gi in candidates:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
        for pi in range(len(pred_boxes)):
            cls = int(pred_classes[pi])
            if pi in matched_pred:
                tp += 1
                per_class_tp[cls] = per_class_tp.get(cls, 0) + 1
            else:
                fp += 1
                per_class_fp[cls] = per_class_fp.get(cls, 0) + 1
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn += 1
                cls = int(gt_classes[gi])
                per_class_fn[cls] = per_class_fn.get(cls, 0) + 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and precision + recall > 0 else None
    roc = {"available": False, "reason": "requires both positive and negative images"}
    labels_arr = np.asarray(emergency_labels, dtype=np.int64)
    scores_arr = np.asarray(emergency_scores, dtype=np.float32)
    if len(np.unique(labels_arr)) == 2:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve

            fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
            auc = float(roc_auc_score(labels_arr, scores_arr))
            roc = {
                "available": True,
                "auc": auc,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "note": "Auxiliary image-level emergency-presence ROC, not an object-detection mAP metric.",
            }
        except Exception as exc:  # pragma: no cover
            roc = {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "images": processed,
        "images_with_emergency": images_with_emergency,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_threshold": float(conf),
        "iou_match": float(iou_match),
        "per_class": {
            "tp": per_class_tp,
            "fp": per_class_fp,
            "fn": per_class_fn,
        },
        "roc": roc,
    }


def run_detection_evaluation(
    config: AppConfig,
    model_path: str | Path | None = None,
    data_yaml: str | Path | None = None,
    split: str | None = None,
    plots: bool = True,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    data_path = resolve_path(data_yaml or "data/dataset/data.yaml", config.root)
    model_file = resolve_path(model_path or config.get("model.path", "models/yolo/best.pt"), config.root)
    if not data_path.is_file():
        result = {
            "status": "not_evaluated",
            "reason": f"dataset yaml not found: {data_path}",
            "note": "Not evaluated - dataset unavailable.",
        }
        _write_unavailable(config, result)
        return result
    if not model_file.is_file():
        result = {
            "status": "not_evaluated",
            "reason": f"model not found: {model_file}",
            "note": "Not evaluated - model unavailable.",
        }
        _write_unavailable(config, result)
        return result
    split = split or str(config.get("evaluation.test_split", "test"))
    device = config.effective_device()
    imgsz = int(config.get("model.imgsz", 640))
    project_dir = config.path("paths.metrics_dir")
    plots_dir = config.path("paths.figures_dir")
    run_name = "yolo_val_test"
    LOGGER.info("Running YOLO validation: model=%s data=%s split=%s", model_file, data_path, split)
    model = YOLO(str(model_file))
    val_kwargs = dict(
        data=str(data_path),
        split=split,
        imgsz=imgsz,
        conf=float(config.get("evaluation.conf_threshold", 0.001)),
        iou=float(config.get("evaluation.iou_threshold", 0.6)),
        device=device,
        plots=bool(plots),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    metrics = model.val(**val_kwargs)
    box = metrics.box
    names = {int(k): str(v) for k, v in getattr(metrics, "names", {}).items()}
    rows: List[Dict[str, Any]] = []
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": (2 * float(box.mp) * float(box.mr) / (float(box.mp) + float(box.mr))) if (box.mp + box.mr) > 0 else 0.0,
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
    for key, value in overall.items():
        rows.append(_metric_row(key, round(value, 6), "all", "Ultralytics validation"))
    # Safely handle ap_class_index - convert numpy arrays to plain Python lists
    ap_class_index_raw = getattr(box, "ap_class_index", None)
    if ap_class_index_raw is not None:
        ap_class_index = _to_list(ap_class_index_raw)
    else:
        ap_class_index = []
    maps_raw = getattr(box, "maps", None)
    if maps_raw is not None:
        maps = _to_list(maps_raw)
    else:
        maps = []
    per_class: List[Dict[str, Any]] = []
    try:
        summary_rows = metrics.summary()
    except Exception:
        summary_rows = []
    for row in summary_rows:
        class_name = str(row.get("Class", ""))
        payload = {
            "class": class_name,
            "images": int(row.get("Images", 0)),
            "instances": int(row.get("Instances", 0)),
            "precision": float(row.get("Box-P", 0.0)),
            "recall": float(row.get("Box-R", 0.0)),
            "f1": float(row.get("Box-F1", 0.0)),
            "mAP50": float(row.get("mAP50", 0.0)),
            "mAP50-95": float(row.get("mAP50-95", 0.0)),
        }
        per_class.append(payload)
        for metric_name in ("precision", "recall", "f1", "mAP50", "mAP50-95"):
            rows.append(_metric_row(metric_name, round(payload[metric_name], 6), class_name, "per-class validation"))
        rows.append(_metric_row("instances", payload["instances"], class_name, "test split ground truth"))
        rows.append(_metric_row("images_with_class", payload["images"], class_name, "test split ground truth"))
    if not per_class:
        for idx, class_id in enumerate(ap_class_index):
            ap = float(maps[idx]) if idx < len(maps) else None
            if ap is None:
                continue
            per_class.append({"class": names.get(int(class_id), str(class_id)), "mAP50-95": ap})
            rows.append(_metric_row("mAP50-95", round(ap, 6), names.get(int(class_id), str(class_id)), "per-class"))
    op = count_fp_fn(
        model,
        data_path,
        split=split,
        conf=float(config.get("model.confidence", 0.55)),
        imgsz=imgsz,
        device=device,
    )
    if op.get("available"):
        for key in ("tp", "fp", "fn"):
            rows.append(_metric_row(key, int(op[key]), "all", f"conf={op['conf_threshold']} IoU>={op['iou_match']}"))
        for key in ("precision", "recall", "f1"):
            if op.get(key) is not None:
                rows.append(_metric_row(f"operating_{key}", round(float(op[key]), 6), "all", "operating point"))
        rows.append(_metric_row("images_evaluated", int(op["images"]), "all", "operating point"))
        for cls_id, cls_name in names.items():
            tp_c = int(op.get("per_class", {}).get("tp", {}).get(str(cls_id), op.get("per_class", {}).get("tp", {}).get(cls_id, 0)))
            fp_c = int(op.get("per_class", {}).get("fp", {}).get(str(cls_id), op.get("per_class", {}).get("fp", {}).get(cls_id, 0)))
            fn_c = int(op.get("per_class", {}).get("fn", {}).get(str(cls_id), op.get("per_class", {}).get("fn", {}).get(cls_id, 0)))
            if tp_c + fp_c + fn_c == 0:
                continue
            rows.append(_metric_row("operating_tp", tp_c, cls_name, "operating point"))
            rows.append(_metric_row("operating_fp", fp_c, cls_name, "operating point"))
            rows.append(_metric_row("operating_fn", fn_c, cls_name, "operating point"))
            precision_c = tp_c / (tp_c + fp_c) if tp_c + fp_c else None
            recall_c = tp_c / (tp_c + fn_c) if tp_c + fn_c else None
            if precision_c is not None:
                rows.append(_metric_row("operating_precision", round(precision_c, 6), cls_name, "operating point"))
            if recall_c is not None:
                rows.append(_metric_row("operating_recall", round(recall_c, 6), cls_name, "operating point"))
    rows.insert(0, _metric_row("evaluated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"), "all", ""))
    rows.insert(1, _metric_row("model", str(model_file), "all", ""))
    rows.insert(2, _metric_row("dataset", str(data_path), "all", ""))
    rows.insert(3, _metric_row("split", split, "all", ""))
    csv_path = config.path("paths.detection_metrics_csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "class", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)
    speed = {k: round(float(v), 4) for k, v in dict(getattr(metrics, "speed", {}) or {}).items()}
    payload = {
        "status": "evaluated",
        "model": str(model_file),
        "dataset": str(data_path),
        "split": split,
        "overall": {k: round(v, 6) for k, v in overall.items()},
        "per_class": per_class,
        "operating_point": op,
        "speed_ms_per_image": speed,
        "val_dir": str(project_dir / run_name),
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _copy_plots(project_dir / run_name, plots_dir)
    _plot_operating_curves(overall, plots_dir)
    if op.get("available") and op.get("roc", {}).get("available"):
        _plot_roc(op["roc"], plots_dir)
    LOGGER.info("Detection metrics written to %s", csv_path)
    return payload


def _copy_plots(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    mapping = {
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "PR_curve.png": "precision_recall_curve.png",
        "F1_curve.png": "f1_curve.png",
        "P_curve.png": "precision_curve.png",
        "R_curve.png": "recall_curve.png",
        "results.png": "training_results.png",
    }
    if not source.exists():
        return
    for src_name, dst_name in mapping.items():
        src = source / src_name
        if src.is_file():
            shutil.copy2(src, target / dst_name)


def _plot_operating_curves(overall: Dict[str, float], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["precision", "recall", "f1", "mAP50", "mAP50-95"]
        values = [overall.get(m, 0.0) for m in metrics]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(metrics, values, color=["#2a9d8f", "#457b9d", "#e9c46a", "#f4a261", "#e76f51"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Object-detection evaluation (actual test split)")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(target / "detection_metrics.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not plot detection metrics: %s", exc)


def _plot_roc(roc: Dict[str, Any], target: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr = np.asarray(roc["fpr"])
        tpr = np.asarray(roc["tpr"])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot(fpr, tpr, label=f"AUC = {roc['auc']:.3f}", color="#d62828", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Image-level emergency presence ROC\n(auxiliary; not object-detection mAP)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(target / "roc_curve.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not plot ROC: %s", exc)


def _write_unavailable(config: AppConfig, payload: Dict[str, Any]) -> None:
    path = config.path("paths.detection_metrics_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "class", "value", "note"])
        writer.writeheader()
        writer.writerow(
            {
                "metric": "status",
                "class": "all",
                "value": "not_evaluated",
                "note": payload.get("note", "Not evaluated - dataset/model unavailable."),
            }
        )
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")