"""Reproducible perceptual-hash audit for cross-split near-duplicates.

Method
------
* grayscale, resize to 9 x 8 with LANCZOS, 64-bit difference hash (dHash):
  one bit per horizontally adjacent pixel comparison;
* for every test image, nearest Hamming distance to any train/val image;
* a test image is flagged when that distance is <= 6.

Writes conference_package/provenance/near_duplicate_audit_recomputed.json and
prints a comparison with the committed audit in
paper/provenance/near_duplicate_audit.json (whose generating script is not in
the repository).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "conference_package"
DATASET = ROOT / "data" / "dataset" / "images"
THRESHOLD = 6


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        pixels = im.tobytes()
    bits = 0
    row = size + 1
    for r in range(size):
        base = r * row
        for c in range(size):
            bits = (bits << 1) | (1 if pixels[base + c] > pixels[base + c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main() -> int:
    reference: dict[str, tuple[int, str]] = {}
    for split in ("train", "val"):
        directory = DATASET / split
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                reference[path.name] = (dhash(path), split)

    rows = []
    for path in sorted((DATASET / "test").iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        h = dhash(path)
        best_name, best_split, best_dist = "", "", 999
        for name, (rh, split) in reference.items():
            dist = hamming(h, rh)
            if dist < best_dist:
                best_name, best_split, best_dist = name, split, dist
        if best_dist <= THRESHOLD:
            rows.append(
                {
                    "test_image": path.name,
                    "near_duplicate": best_name,
                    "near_duplicate_split": best_split,
                    "hamming": best_dist,
                }
            )

    out_dir = PKG / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "dHash 64-bit, grayscale 9x8 LANCZOS, horizontal neighbour comparison, nearest train/val image, threshold <= 6",
        "threshold": THRESHOLD,
        "test_images": 143,
        "flagged": len(rows),
        "pairs": rows,
    }
    (out_dir / "near_duplicate_audit_recomputed.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    committed_path = ROOT / "paper" / "provenance" / "near_duplicate_audit.json"
    committed = {row["test_image"] for row in json.loads(committed_path.read_text(encoding="utf-8"))}
    recomputed = {row["test_image"] for row in rows}
    summary = {
        "committed_flagged": len(committed),
        "recomputed_flagged": len(recomputed),
        "overlap": len(committed & recomputed),
        "committed_only": len(committed - recomputed),
        "recomputed_only": len(recomputed - committed),
        "union": len(committed | recomputed),
    }
    (out_dir / "near_duplicate_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
