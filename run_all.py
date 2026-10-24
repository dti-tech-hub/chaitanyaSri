"""Reproduce the complete project pipeline, in order, with explicit statuses.

Default behaviour (no retraining):
  1. validate the prepared dataset
  2. evaluate the existing model on the held-out test split
  3. rerun the fixed-time vs priority simulation and density sweep
  4. run the pytest suite
  5. regenerate results/report_tables.md from real artifacts
  6. optionally record an annotated demo video and smoke-test Streamlit

Examples
--------
    python run_all.py
    python run_all.py --demo --check-dashboard
    python run_all.py --train --epochs 30 --imgsz 448 --batch 16

Training is never started implicitly: `--train` must be passed because it can
take hours on CPU. If the model file is missing, evaluation is reported as
"Not evaluated - model unavailable" rather than skipped silently.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS = PROJECT_ROOT / "results"
LOGS = RESULTS / "logs"
MODEL = PROJECT_ROOT / "models" / "yolo" / "best.pt"


def _run_step(name: str, command: List[str], timeout: int | None = None) -> Dict[str, Any]:
    print(f"\n=== {name} ===")
    print("$ " + " ".join(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT, timeout=timeout)
    elapsed = round(time.perf_counter() - started, 2)
    status = "ok" if result.returncode == 0 else "failed"
    print(f"--- {name}: {status} ({elapsed:.1f}s) ---", flush=True)
    return {
        "name": name,
        "command": command,
        "returncode": int(result.returncode),
        "status": status,
        "elapsed_s": elapsed,
    }


def _check_dashboard(timeout: int = 45) -> Dict[str, Any]:
    """Start Streamlit headless, confirm its health endpoint, then terminate it."""
    import urllib.request

    print("\n=== dashboard smoke test ===")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.headless",
        "true",
        "--server.port",
        "8599",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("$ " + " ".join(command), flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    status = "failed"
    health = ""
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen("http://127.0.0.1:8599/_stcore/health", timeout=2) as response:
                    health = response.read().decode("utf-8", errors="replace").strip()
                if health.lower() == "ok":
                    status = "ok"
                    break
            except Exception:
                time.sleep(0.75)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        output = process.stdout.read()[-4000:] if process.stdout is not None else ""
    elapsed = round(time.perf_counter() - started, 2)
    print(f"--- dashboard smoke test: {status} (health={health or 'no response'}, {elapsed:.1f}s) ---")
    if status != "ok" and output.strip():
        print(output.strip())
    script_test = _run_step(
        "dashboard script test",
        [sys.executable, "scripts/check_dashboard.py"],
        timeout=180,
    )
    if status == "ok" and script_test.get("status") != "ok":
        status = "failed"
    return {
        "name": "dashboard smoke test",
        "command": command,
        "status": status,
        "health": health,
        "elapsed_s": elapsed,
        "script_test": script_test,
        "output_tail": output,
    }


def _not_evaluated(name: str, reason: str) -> Dict[str, Any]:
    return {"name": name, "status": "not_evaluated", "reason": reason}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="Run scripts/train_yolo.py before evaluation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Record an annotated demo video")
    parser.add_argument("--demo-frames", type=int, default=400)
    parser.add_argument("--check-dashboard", action="store_true", help="Start and stop the Streamlit dashboard")
    args = parser.parse_args(argv)

    LOGS.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    steps: List[Dict[str, Any]] = []
    overall = 0

    # 1. Dataset validation is cheap and must pass before any training claim.
    steps.append(
        _run_step(
            "dataset validation",
            [sys.executable, "scripts/validate_dataset.py"],
            timeout=900,
        )
    )

    # 2. Optional training. The explicit flag prevents accidental multi-hour runs.
    if args.train:
        steps.append(
            _run_step(
                "YOLO training",
                [
                    sys.executable,
                    "scripts/train_yolo.py",
                    "--epochs",
                    str(args.epochs),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--workers",
                    str(args.workers),
                    "--patience",
                    str(args.patience),
                ],
                timeout=None,
            )
        )
    else:
        steps.append(_not_evaluated("YOLO training", "not requested; pass --train to retrain"))

    # 3. Evaluation only when weights exist.
    if args.skip_evaluate:
        steps.append(_not_evaluated("held-out evaluation", "skipped by --skip-evaluate"))
    elif not MODEL.is_file():
        steps.append(_not_evaluated("held-out evaluation", f"model unavailable: {MODEL}"))
    else:
        steps.append(
            _run_step(
                "held-out evaluation",
                [sys.executable, "scripts/evaluate.py"],
                timeout=None,
            )
        )

    # 4. Simulation and pre-registered density sweep.
    if args.skip_simulation:
        steps.append(_not_evaluated("simulation", "skipped by --skip-simulation"))
    else:
        steps.append(
            _run_step(
                "simulation + sweep",
                [sys.executable, "simulation.py", "--sweep", "--seeds", "42,43,44", "--quiet"],
                timeout=1800,
            )
        )

    # 5. Tests.
    if args.skip_tests:
        steps.append(_not_evaluated("pytest", "skipped by --skip-tests"))
    else:
        steps.append(
            _run_step(
                "pytest",
                [sys.executable, "-m", "pytest", "-q"],
                timeout=1800,
            )
        )

    # 6. Optional live pipeline evidence.
    if args.demo:
        demo_output = RESULTS / "videos" / "demo_annotated.mp4"
        steps.append(
            _run_step(
                "annotated demo",
                [
                    sys.executable,
                    "main.py",
                    "--source",
                    "demo",
                    "--max-frames",
                    str(args.demo_frames),
                    "--no-display",
                    "--save-video",
                    "--output",
                    str(demo_output),
                ],
                timeout=None,
            )
        )
    else:
        steps.append(_not_evaluated("annotated demo", "not requested; pass --demo"))

    # 7. Optional Streamlit smoke test.
    if args.check_dashboard:
        steps.append(_check_dashboard())
    else:
        steps.append(_not_evaluated("dashboard smoke test", "not requested; pass --check-dashboard"))

    # 8. Tables always regenerate from whatever artifacts exist.
    steps.append(
        _run_step(
            "report tables",
            [sys.executable, "scripts/generate_results.py"],
            timeout=300,
        )
    )

    for step in steps:
        if step.get("status") == "failed":
            overall = 1

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "project_root": str(PROJECT_ROOT),
        "arguments": vars(args),
        "steps": steps,
        "status": "ok" if overall == 0 else "failed",
    }
    summary_path = LOGS / "run_all_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== RUN_ALL SUMMARY ===")
    for step in steps:
        print(f"{step.get('status', 'unknown'):>13} | {step.get('name')}")
    print(f"\nSummary: {summary_path}")
    if overall:
        print("One or more steps failed. See the summary for exact commands and statuses.")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
