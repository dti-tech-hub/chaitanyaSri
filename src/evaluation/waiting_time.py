"""Waiting-time metrics for emergency and normal vehicles."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def waiting_time(arrival_time: Optional[float], green_time: Optional[float]) -> Optional[float]:
    """Waiting time = green onset - arrival at the approach/queue zone.

    Returns None when either timestamp is unavailable; never invents a value.
    """
    if arrival_time is None or green_time is None:
        return None
    return max(0.0, float(green_time) - float(arrival_time))


def waiting_time_reduction(baseline_wait: Optional[float], smart_wait: Optional[float]) -> Optional[Dict[str, float]]:
    if baseline_wait is None or smart_wait is None:
        return None
    reduction = float(baseline_wait) - float(smart_wait)
    pct = (reduction / float(baseline_wait) * 100.0) if baseline_wait > 0 else 0.0
    return {"reduction_s": reduction, "reduction_pct": pct}


def summarize_waiting_times(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean_s": None, "max_s": None, "min_s": None, "median_s": None}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    median = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
    return {
        "count": n,
        "mean_s": sum(vals) / n,
        "max_s": max(vals),
        "min_s": min(vals),
        "median_s": median,
    }


def compare_waiting_series(
    baseline_waits: Iterable[Optional[float]],
    proposed_waits: Iterable[Optional[float]],
) -> Dict[str, Optional[float]]:
    b = [float(v) for v in baseline_waits if v is not None]
    p = [float(v) for v in proposed_waits if v is not None]
    if not b or not p:
        return {
            "baseline_mean_s": None,
            "proposed_mean_s": None,
            "reduction_s": None,
            "reduction_pct": None,
            "baseline_max_s": max(b) if b else None,
            "proposed_max_s": max(p) if p else None,
            "n_baseline": len(b),
            "n_proposed": len(p),
        }
    bm = sum(b) / len(b)
    pm = sum(p) / len(p)
    return {
        "baseline_mean_s": bm,
        "proposed_mean_s": pm,
        "reduction_s": bm - pm,
        "reduction_pct": ((bm - pm) / bm * 100.0) if bm > 0 else 0.0,
        "baseline_max_s": max(b),
        "proposed_max_s": max(p),
        "n_baseline": len(b),
        "n_proposed": len(p),
    }


def plot_waiting_time_comparison(
    baseline_mean: float,
    proposed_mean: float,
    target_path: str,
    title: str = "Emergency vehicle waiting time",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        bars = ax.bar(["Fixed-time\nbaseline", "Emergency\npriority"], [baseline_mean, proposed_mean], color=["#8d99ae", "#2a9d8f"])
        ax.set_ylabel("Mean waiting time (s)")
        ax.set_title(title)
        for bar, val in zip(bars, [baseline_mean, proposed_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}s", ha="center")
        reduction = baseline_mean - proposed_mean
        if baseline_mean > 0:
            ax.text(0.5, max(baseline_mean, proposed_mean) * 0.55, f"reduction: {reduction:.1f}s\n({reduction / baseline_mean * 100:.0f}%)", ha="center", color="#1d3557")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(target_path, dpi=160)
        plt.close(fig)
    except Exception:
        pass
