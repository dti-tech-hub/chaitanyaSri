"""Evaluate the trained detector on the held-out test split.

Produces:
  * results/metrics/detection_metrics.{csv,json}
  * results/figures/detection_metrics.png, PR/F1 curves, confusion matrix, ROC if available
  * results/metrics/evaluation_run.json

Run only after `scripts/train_yolo.py` has produced models/yolo/best.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import run_detection_evaluation
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo/best.pt")
    parser.add_argument("--data", default="data/dataset/data.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config("config/config.yaml")
    started = time.perf_counter()
    result = run_detection_evaluation(
        config,
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        plots=not args.no_plots,
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1),
        "command": {
            "model": args.model,
            "data": args.data,
            "split": args.split,
            "plots": not args.no_plots,
        },
        "status": result.get("status"),
        "result": result,
    }
    out = PROJECT_ROOT / "results" / "metrics" / "evaluation_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if result.get("status") == "evaluated":
        print(json.dumps(result["overall"], indent=2))
        print(f"Evaluation complete in {elapsed:.1f}s -> {out}")
        return 0
    print(json.dumps(result, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
