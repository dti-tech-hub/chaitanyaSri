"""Audit the prepared YOLO dataset and fail on structural errors.

Checks:
  * every image has a label file (empty labels are allowed and counted)
  * every label class id is declared in data.yaml
  * every box is finite, normalised, positive-area and inside [0, 1]
  * image files decode and are unique (SHA-1)
  * class/split distribution is reported, including empty labels
  * a contact sheet of one annotated image per class is generated if possible
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_value: str, root: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (root / p).resolve()


def validate(data_yaml: Path, contact_sheet: Path | None = None) -> Dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = spec.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        raise ValueError("data.yaml has no class names")
    root = Path(spec.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    report: Dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "root": str(root),
        "classes": list(names),
        "status": "ok",
        "errors": [],
        "warnings": [],
        "splits": {},
        "duplicates": 0,
    }
    errors: List[str] = report["errors"]
    warnings: List[str] = report["warnings"]
    hashes: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        rel = spec.get(split)
        if not rel:
            errors.append(f"data.yaml has no '{split}' split")
            continue
        image_dir = _resolve(str(rel), root)
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"missing image directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"missing label directory: {label_dir}")
            continue
        class_counts: Counter[int] = Counter()
        images = 0
        empty = 0
        boxes = 0
        bad_boxes = 0
        corrupt = 0
        missing_labels = 0
        unknown_classes = 0
        samples: Dict[int, Path] = {}
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            images += 1
            digest = _sha1(image_path)
            if digest in hashes:
                report["duplicates"] += 1
                warnings.append(f"duplicate image: {image_path} == {hashes[digest]}")
            hashes[digest] = str(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels += 1
                errors.append(f"missing label: {label_path}")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                corrupt += 1
                errors.append(f"could not decode image: {image_path}")
                continue
            h, w = image.shape[:2]
            lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for parts in lines:
                boxes += 1
                if len(parts) < 5:
                    bad_boxes += 1
                    errors.append(f"malformed label line in {label_path}")
                    continue
                try:
                    class_id = int(float(parts[0]))
                    xc, yc, bw, bh = (float(v) for v in parts[1:5])
                except ValueError:
                    bad_boxes += 1
                    errors.append(f"non-numeric label in {label_path}")
                    continue
                if not 0 <= class_id < len(names):
                    unknown_classes += 1
                    errors.append(f"class id {class_id} outside data.yaml in {label_path}")
                    continue
                if not all(np.isfinite([xc, yc, bw, bh])) or not (0 < bw <= 1 and 0 < bh <= 1):
                    bad_boxes += 1
                    errors.append(f"invalid box size in {label_path}: {parts}")
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1):
                    bad_boxes += 1
                    errors.append(f"invalid box centre in {label_path}: {parts}")
                    continue
                class_counts[class_id] += 1
                samples.setdefault(class_id, image_path)
        report["splits"][split] = {
            "images": images,
            "boxes": boxes,
            "empty_labels": empty,
            "missing_labels": missing_labels,
            "corrupt_images": corrupt,
            "bad_boxes": bad_boxes,
            "unknown_class_ids": unknown_classes,
            "class_counts": {names[k]: int(class_counts[k]) for k in sorted(class_counts)},
        }
        if images == 0:
            errors.append(f"split {split} has no images")
        if split == "train" and not class_counts:
            errors.append("training split has no labelled boxes")
        if contact_sheet is not None and split == "train" and samples:
            _make_contact_sheet(samples, names, label_dir, contact_sheet)
            report["contact_sheet"] = str(contact_sheet)

    if errors:
        report["status"] = "failed"
    return report


def _make_contact_sheet(
    samples: Dict[int, Path],
    names: List[str],
    label_dir: Path,
    target: Path,
    tile: int = 320,
) -> None:
    tiles = []
    colors = [(40, 40, 240), (0, 120, 255), (255, 120, 0), (40, 200, 80), (200, 40, 200)]
    for class_id in sorted(samples):
        image = cv2.imread(str(samples[class_id]))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = label_dir / f"{samples[class_id].stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if cid != class_id:
                continue
            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            color = colors[class_id % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, names[class_id], (max(2, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        scale = tile / max(w, h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
        y0 = (tile - resized.shape[0]) // 2
        x0 = (tile - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(canvas, names[class_id].upper(), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(canvas)
    if tiles:
        # Pad to a 2-column grid.
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
        sheet = np.vstack(rows)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset/data.yaml")
    parser.add_argument("--output", default="results/metrics/dataset_validation.json")
    parser.add_argument("--contact-sheet", default="results/figures/dataset_samples.png")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    sheet = Path(args.contact_sheet)
    if not sheet.is_absolute():
        sheet = PROJECT_ROOT / sheet
    report = validate(data_path, contact_sheet=sheet)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
