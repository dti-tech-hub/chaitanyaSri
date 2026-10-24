"""Train the custom four-class emergency-vehicle YOLO model.

This script is intentionally explicit: it refuses to run if the prepared
dataset is absent, uses CPU when no GPU is available, and records every
training parameter and the resulting weight paths.

Example
-------
    python scripts/train_yolo.py --epochs 40 --imgsz 512 --batch 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_weights() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "yolo" / "yolov8n.pt",
        PROJECT_ROOT / "data" / "raw" / "repos" / "AI-Traffic-Management-System" / "yolov8n.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args: argparse.Namespace) -> Dict[str, Any]:
    from ultralytics import YOLO

    data_path = (PROJECT_ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Run `python scripts/prepare_dataset.py` first."
        )
    weights = Path(args.model)
    if not weights.is_absolute():
        weights = (PROJECT_ROOT / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(
            f"Initial weights not found: {weights}. Place yolov8n.pt in models/yolo/."
        )
    device = args.device or _device()
    run_name = args.name
    project_dir = (PROJECT_ROOT / args.project).resolve()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model = YOLO(str(weights))
    train_kwargs = {
        "data": str(data_path),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": device,
        "workers": int(args.workers),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "deterministic": True,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": args.optimizer,
        "lr0": float(args.lr0),
        "lrf": 0.01,
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "close_mosaic": 10,
        "amp": device != "cpu",
        "cache": args.cache,
        "plots": True,
        "save": True,
        "save_period": max(1, int(args.epochs) // 2),
        "verbose": True,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.05,
    }
    print("Training parameters:")
    print(json.dumps({**train_kwargs, "weights": str(weights)}, indent=2))
    result = model.train(**train_kwargs)
    elapsed = time.perf_counter() - started
    run_dir = Path(getattr(result, "save_dir", project_dir / run_name))
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    model_dir = PROJECT_ROOT / "models" / "yolo"
    model_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    if best.is_file():
        shutil.copy2(best, model_dir / "best.pt")
        copied["best"] = str((model_dir / "best.pt").relative_to(PROJECT_ROOT))
    if last.is_file():
        shutil.copy2(last, model_dir / "last.pt")
        copied["last"] = str((model_dir / "last.pt").relative_to(PROJECT_ROOT))

    metadata = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1),
        "elapsed_human": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "dataset": str(data_path),
        "initial_weights": str(weights),
        "run_dir": str(run_dir),
        "device": device,
        "epochs_requested": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "copied_weights": copied,
        "command_parameters": {k: v for k, v in train_kwargs.items() if k not in {"project", "data"}},
    }
    metadata_dir = PROJECT_ROOT / "results" / "training"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, metadata_dir / "results.csv")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset/data.yaml")
    parser.add_argument("--model", default=str(_default_weights()))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, 0, cuda:0, mps (default: auto)")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM (may use several GB)")
    parser.add_argument("--project", default="results/training")
    parser.add_argument("--name", default="ev_yolo")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
