"""Run the reproducible fixed-time vs emergency-priority simulation.

This is a signal-level simulation, not a traffic-flow or road-safety claim.
It uses the same SignalController as `main.py`.

Examples
--------
    python simulation.py
    python simulation.py --duration 300 --seed 7
    python simulation.py --sweep --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from src.control.simulation import density_sweep, save_simulation_results, simulate_comparison
from src.evaluation.comparison import comparison_markdown, comparison_rows, save_comparison_csv, save_comparison_json
from src.traffic.intersection import Intersection
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import setup_logger

LOGGER = setup_logger("simulation_cli")


def _plot_summary(result: dict, target: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        b = result["baseline"]
        p = result["proposed"]
        metrics = [
            ("Emergency mean", "emergency_waiting_time_s"),
            ("Emergency max", "emergency_max_wait_s"),
            ("Normal mean", "normal_waiting_time_s"),
        ]
        labels = [m[0] for m in metrics]
        baseline_values = [b.get(m[1]) or 0.0 for m in metrics]
        proposed_values = [p.get(m[1]) or 0.0 for m in metrics]
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        bars_b = ax.bar(x - width / 2, baseline_values, width, label="Fixed-time baseline", color="#8d99ae")
        bars_p = ax.bar(x + width / 2, proposed_values, width, label="Emergency priority", color="#2a9d8f")
        for bars in (bars_b, bars_p):
            for bar in bars:
                value = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Waiting time (s)")
        ax.set_title("Simulated signal-level comparison (actual run)")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("Could not plot simulation summary: %s", exc)


def _plot_wait_distribution(result: dict, target: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        b = [v["wait_time_s"] for v in result["baseline_vehicles"] if v["is_emergency"] and v["wait_time_s"] is not None]
        p = [v["wait_time_s"] for v in result["proposed_vehicles"] if v["is_emergency"] and v["wait_time_s"] is not None]
        if not b or not p:
            return
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.boxplot([b, p], tick_labels=["Fixed-time baseline", "Emergency priority"], patch_artist=True, boxprops={"facecolor": "#c9d6df"})
        ax.set_ylabel("Emergency-vehicle waiting time (s)")
        ax.set_title(f"Simulated EV waits (n={len(b)} baseline, {len(p)} priority)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("Could not plot waiting-time distribution: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="Run low/medium/high arrival-rate sweep")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-transition controller logs")
    args = parser.parse_args()

    if args.quiet:
        for logger_name in ("simulation_cli", "signal_controller", "simulation", "priority_engine"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    config = load_config(args.config)
    overrides = {}
    if args.duration is not None:
        overrides.setdefault("simulation", {})["duration_s"] = float(args.duration)
    if args.dt is not None:
        overrides.setdefault("simulation", {})["dt_s"] = float(args.dt)
    if overrides:
        config = config.with_overrides(overrides)
    config.ensure_directories()
    intersection = Intersection(config)
    result = simulate_comparison(config, intersection, seed=args.seed)
    out_dir = config.path("paths.simulation_dir")
    paths = save_simulation_results(result, out_dir)
    rows = comparison_rows(result["baseline"], result["proposed"])
    comparison_csv = config.path("paths.metrics_dir") / "comparison.csv"
    save_comparison_csv(rows, comparison_csv)
    comparison_json = config.path("paths.metrics_dir") / "comparison.json"
    save_comparison_json(result, comparison_json)
    markdown = comparison_markdown(rows)
    (out_dir / "comparison.md").write_text(
        "# Simulated fixed-time vs emergency-priority comparison\n\n"
        f"Seed: {result['scenario']['seed']} | Duration: {result['scenario']['duration_s']} s | "
        f"Vehicles: {result['scenario']['vehicles']} | Emergency vehicles: {result['scenario']['emergency_vehicles']}\n\n"
        + markdown
        + "\n\nThis is a deterministic signal-level simulation. It does not model queue spillback, "
        "driver behaviour, pedestrian phases, or real-world safety.\n",
        encoding="utf-8",
    )
    if not args.no_plots:
        _plot_summary(result, config.path("paths.figures_dir") / "simulation_waiting_time.png")
        _plot_wait_distribution(result, config.path("paths.figures_dir") / "simulation_wait_distribution.png")

    if args.sweep:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        sweep_rows = density_sweep(config, intersection, seeds=seeds)
        sweep_path = out_dir / "density_sweep.csv"
        with sweep_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        (out_dir / "density_sweep.json").write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")
        print(f"Density sweep: {sweep_path}")

    print(markdown)
    print(f"\nSimulation summary: {paths['summary']}")
    print(f"Comparison CSV: {comparison_csv}")
    print("SIMULATED SIGNAL-LEVEL RESULTS. NOT REAL-WORLD TRAFFIC MEASUREMENTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
