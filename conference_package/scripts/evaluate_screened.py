"""Leakage-screened detector evaluation.

Two perceptual-hash audits are combined: the committed list in
paper/provenance/near_duplicate_audit.json (77 flagged) and the reproducible
dHash pass written by audit_near_duplicates.py (81 flagged). Their union (87 of
143 test images) is removed, and the same checkpoint is re-evaluated on the
remaining 56 images. Using the union is the conservative choice: the two hash
variants disagree on a handful of borderline frames.

Writes only inside conference_package/experiments/leakage_screened/ so the
project's primary results/ artifacts are left untouched.

Run: python conference_package/scripts/evaluate_screened.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.evaluation.detection_metrics import count_fp_fn  # noqa: E402
from src.utils.config import load_config  # noqa: E402

PKG = ROOT / "conference_package"
EXP = PKG / "experiments" / "leakage_screened"
COMMITTED_AUDIT = ROOT / "paper" / "provenance" / "near_duplicate_audit.json"
RECOMPUTED_AUDIT = PKG / "provenance" / "near_duplicate_audit_recomputed.json"
DATASET = ROOT / "data" / "dataset"
MODEL = ROOT / "models" / "yolo" / "best.pt"


def build_split() -> dict:
    committed = {row["test_image"] for row in json.loads(COMMITTED_AUDIT.read_text(encoding="utf-8"))}
    recomputed = {row["test_image"] for row in json.loads(RECOMPUTED_AUDIT.read_text(encoding="utf-8"))["pairs"]}
    flagged = committed | recomputed
    image_dir = DATASET / "images" / "test"
    label_dir = DATASET / "labels" / "test"
    # Standard YOLO layout: <root>/images/test and <root>/labels/test, so both
    # Ultralytics and src.evaluation.detection_metrics resolve the labels.
    dataset_root = EXP / "dataset"
    out_images = dataset_root / "images" / "test"
    out_labels = dataset_root / "labels" / "test"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    # Rebuild from scratch so a stricter screen cannot leave stale copies behind.
    for directory in (out_images, out_labels):
        for stale in directory.iterdir():
            if stale.is_file():
                stale.unlink()

    kept, removed = [], []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if image.name in flagged:
            removed.append(image.name)
            continue
        shutil.copy2(image, out_images / image.name)
        label = label_dir / f"{image.stem}.txt"
        if label.is_file():
            shutil.copy2(label, out_labels / label.name)
        kept.append(image.name)

    data = {
        "path": str(dataset_root),
        "train": str(DATASET / "images" / "train"),
        "val": str(DATASET / "images" / "val"),
        "test": "images/test",
        "names": {0: "ambulance", 1: "fire_engine", 2: "police_vehicle", 3: "normal_vehicle"},
        "nc": 4,
    }
    # Ultralytics resolves relative image roots against the yaml directory.
    data_path = dataset_root / "data_clean.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {
        "images_total": len(kept) + len(removed),
        "images_flagged": len(removed),
        "images_kept": len(kept),
        "audit_committed_flagged": len(committed),
        "audit_recomputed_flagged": len(recomputed),
        "audit_union_flagged": len(flagged),
        "data_yaml": str(data_path),
        "flagged_files": removed,
    }


def main() -> int:
    from ultralytics import YOLO

    config = load_config("config/config.yaml")
    imgsz = int(config.get("model.imgsz", 448))
    device = config.effective_device()
    split_info = build_split()
    started = time.perf_counter()
    model = YOLO(str(MODEL))

    metrics = model.val(
        data=split_info["data_yaml"],
        split="test",
        imgsz=imgsz,
        conf=float(config.get("evaluation.conf_threshold", 0.001)),
        iou=float(config.get("evaluation.iou_threshold", 0.6)),
        device=device,
        plots=True,
        project=str(EXP),
        name="yolo_val_clean",
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": (2 * float(box.mp) * float(box.mr) / (float(box.mp) + float(box.mr)))
        if (box.mp + box.mr) > 0
        else 0.0,
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
    per_class = []
    for row in metrics.summary():
        per_class.append(
            {
                "class": str(row.get("Class", "")),
                "images": int(row.get("Images", 0)),
                "instances": int(row.get("Instances", 0)),
                "precision": float(row.get("Box-P", 0.0)),
                "recall": float(row.get("Box-R", 0.0)),
                "f1": float(row.get("Box-F1", 0.0)),
                "mAP50": float(row.get("mAP50", 0.0)),
                "mAP50-95": float(row.get("mAP50-95", 0.0)),
            }
        )
    operating = count_fp_fn(
        model,
        Path(split_info["data_yaml"]),
        split="test",
        conf=float(config.get("model.confidence", 0.55)),
        imgsz=imgsz,
        device=device,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "status": "evaluated",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "purpose": "Leakage-screened re-evaluation",
            "screen": "union of paper/provenance/near_duplicate_audit.json and the reproducible dHash pass (threshold <= 6); every flagged test image removed",
            "model": "models/yolo/best.pt",
            "imgsz": imgsz,
            "mAP_conf": float(config.get("evaluation.conf_threshold", 0.001)),
            "mAP_iou": float(config.get("evaluation.iou_threshold", 0.6)),
            "operating_conf": float(config.get("model.confidence", 0.55)),
            "device": device,
            "elapsed_s": round(elapsed, 1),
        },
        "split": {k: v for k, v in split_info.items() if k != "flagged_files"},
        "overall": {k: round(v, 6) for k, v in overall.items()},
        "per_class": per_class,
        "operating_point": operating,
        "speed_ms_per_image": {k: round(float(v), 4) for k, v in dict(getattr(metrics, "speed", {}) or {}).items()},
    }
    (EXP / "screened_metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    with (EXP / "screened_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scope", "metric", "value"])
        for key, value in payload["overall"].items():
            writer.writerow(["overall", key, value])
        for row in per_class:
            for key in ("precision", "recall", "f1", "mAP50", "mAP50-95", "instances"):
                writer.writerow([row["class"], key, row[key]])
        for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
            if key in payload["operating_point"]:
                writer.writerow(["operating_point", key, payload["operating_point"][key]])

    print(json.dumps({"split": payload["split"], "overall": payload["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
