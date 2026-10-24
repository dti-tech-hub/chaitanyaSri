"""Merge the two labelled source datasets into one auditable four-class dataset.

Sources
-------
1. Roboflow "emergency-vehicle-crigt" (CC BY 4.0):
   classes ambulance=0, fire brigade=1, police=2
2. Roboflow "vehicle-detc." (CC BY 4.0):
   classes ambulance=0, fire-truck=1, vehicle=2

Canonical output classes
------------------------
0 ambulance, 1 fire_engine, 2 police_vehicle, 3 normal_vehicle

The script copies images and rewrites every label line. It does not synthesise
images or labels. It also writes source metadata and a machine-readable report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_NAMES = ["ambulance", "fire_engine", "police_vehicle", "normal_vehicle"]

SOURCES = [
    {
        "name": "emergency_vehicle_crigt",
        "root": "data/raw/repos/AI-Traffic-Management-System/project/datasets/emergency_vehicles",
        "prefix": "rf",
        "license": "CC BY 4.0",
        "class_map": {0: 0, 1: 1, 2: 2},
        "source_names": {0: "ambulance", 1: "fire brigade", 2: "police"},
    },
    {
        "name": "vehicle_detc",
        "root": "data/raw/repos/Emergency-Vehicle-and-Density-Detection/vehicle-detc",
        "prefix": "vd",
        "license": "CC BY 4.0",
        "class_map": {0: 0, 1: 1, 2: 3},
        "source_names": {0: "ambulance", 1: "fire-truck", 2: "vehicle"},
    },
]

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_label(
    parts: List[str], class_map: Dict[int, int]
) -> Tuple[int, float, float, float, float] | None:
    if len(parts) < 5:
        return None
    try:
        source_class = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    if source_class not in class_map:
        return None
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        return None
    if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        return None
    x1, y1 = xc - bw / 2, yc - bh / 2
    x2, y2 = xc + bw / 2, yc + bh / 2
    # Reject boxes completely outside the image. Clamp mild float spill-over.
    if x2 <= 0 or y2 <= 0 or x1 >= 1 or y1 >= 1:
        return None
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    bw = min(1.0, bw)
    bh = min(1.0, bh)
    return class_map[source_class], xc, yc, bw, bh


def prepare(
    output_root: Path,
    include_oid: bool = False,
    clean: bool = False,
) -> Dict[str, object]:
    if clean and output_root.exists():
        # Refuse to delete anything but the generated dataset directory.
        resolved = output_root.resolve()
        if resolved.name != "dataset" or PROJECT_ROOT not in resolved.parents:
            raise ValueError(f"Refusing to clean unexpected directory: {resolved}")
        shutil.rmtree(resolved)

    for split in SPLIT_MAP.values():
        for kind in ("images", "labels"):
            (output_root / kind / split).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(output_root),
        "classes": CANONICAL_NAMES,
        "sources": [],
        "splits": {},
        "excluded": {},
    }
    metadata_rows: List[Dict[str, object]] = []
    seen_hashes: Dict[str, str] = {}
    duplicate_images = 0
    invalid_boxes = 0

    for source in SOURCES:
        source_root = PROJECT_ROOT / source["root"]
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"Source dataset missing: {source_root}. "
                "See docs/DATASETS.md for acquisition instructions."
            )
        source_stats = {
            "name": source["name"],
            "root": str(source_root.relative_to(PROJECT_ROOT)),
            "license": source["license"],
            "class_map": source["class_map"],
            "source_names": source["source_names"],
            "splits": {},
        }
        for src_split, dst_split in SPLIT_MAP.items():
            image_dir = source_root / src_split / "images"
            label_dir = source_root / src_split / "labels"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing split images: {image_dir}")
            class_counts: Counter[int] = Counter()
            split_stats = {
                "images": 0,
                "boxes": 0,
                "empty_labels": 0,
                "missing_labels": 0,
                "invalid_boxes": 0,
                "duplicates_skipped": 0,
                "class_counts": {},
            }
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                digest = _sha1(image_path)
                if digest in seen_hashes:
                    duplicate_images += 1
                    split_stats["duplicates_skipped"] += 1
                    continue
                seen_hashes[digest] = str(image_path)
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    split_stats["missing_labels"] += 1
                    # Do not copy an image whose annotations are absent: this
                    # would silently turn an object into background.
                    continue
                new_lines: List[str] = []
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    validated = _validate_label(parts, source["class_map"])
                    if validated is None:
                        invalid_boxes += 1
                        split_stats["invalid_boxes"] += 1
                        continue
                    cls, xc, yc, bw, bh = validated
                    new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    class_counts[cls] += 1

                dst_name = f"{source['prefix']}_{image_path.name}"
                dst_image = output_root / "images" / dst_split / dst_name
                dst_label = output_root / "labels" / dst_split / f"{dst_name.rsplit('.', 1)[0]}.txt"
                shutil.copy2(image_path, dst_image)
                dst_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
                split_stats["images"] += 1
                split_stats["boxes"] += len(new_lines)
                if not new_lines:
                    split_stats["empty_labels"] += 1
                metadata_rows.append(
                    {
                        "split": dst_split,
                        "source": source["name"],
                        "source_image": str(image_path.relative_to(PROJECT_ROOT)),
                        "output_image": str(dst_image.relative_to(PROJECT_ROOT)),
                        "sha1": digest,
                        "boxes": len(new_lines),
                        "empty_label": str(not new_lines).lower(),
                    }
                )
            split_stats["class_counts"] = {
                CANONICAL_NAMES[k]: int(v) for k, v in sorted(class_counts.items())
            }
            source_stats["splits"][dst_split] = split_stats
        stats["sources"].append(source_stats)

    oid_stats = None
    if include_oid:
        oid_stats = _include_oid(output_root, metadata_rows, seen_hashes)
        stats["sources"].append(oid_stats)

    # Aggregate split statistics from the copied dataset.
    for split in SPLIT_MAP.values():
        class_counts: Counter[int] = Counter()
        images = boxes = empty = 0
        for label_path in sorted((output_root / "labels" / split).glob("*.txt")):
            images += 1
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            boxes += len(lines)
            if not lines:
                empty += 1
            for line in lines:
                class_counts[int(line.split()[0])] += 1
        stats["splits"][split] = {
            "images": images,
            "boxes": boxes,
            "empty_labels": empty,
            "class_counts": {CANONICAL_NAMES[k]: int(v) for k, v in sorted(class_counts.items())},
        }

    stats["duplicates_skipped"] = duplicate_images
    stats["invalid_boxes_dropped"] = invalid_boxes

    # No absolute `path`: Ultralytics then resolves train/val/test against the
    # YAML file's own directory, so the dataset stays portable across machines.
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CANONICAL_NAMES),
        "names": CANONICAL_NAMES,
        "license": "Source datasets: CC BY 4.0 (see docs/DATASETS.md)",
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    report_path = output_root / "dataset_report.json"
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _include_oid(output_root: Path, metadata_rows: List[Dict[str, object]], seen_hashes: Dict[str, str]) -> Dict[str, object]:
    """Optionally add OID ambulance crops/images.

    The OID annotations available in this workspace contain only ambulance
    boxes. They are *not* used by default because the CSV path does not carry a
    parked/held-out image mapping that can be verified here. This function is
    deliberately conservative and refuses incomplete inputs.
    """
    raise SystemExit(
        "OID inclusion is intentionally disabled: the downloaded OID CSV "
        "contains only ambulance boxes and no reliable fire/police labels. "
        "Use the two labelled Roboflow sources instead."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "dataset"),
        help="Output dataset directory (default: data/dataset)",
    )
    parser.add_argument("--include-oid", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--clean", action="store_true", help="Remove the existing generated dataset first")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    stats = prepare(output, include_oid=args.include_oid, clean=args.clean)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
