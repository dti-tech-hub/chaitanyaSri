"""Render the system architecture diagram used in the README and report."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#12263a"
MUTED = "#5c677d"
BLUE = "#dce9f5"
GREEN = "#dcf2e3"
AMBER = "#fff1cc"
RED = "#fde2e1"
GREY = "#eef1f4"


def box(ax, x, y, w, h, title, lines=(), fc=GREY, ec=INK, title_size=10.5, body_size=8.6):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=0.9",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 1.7, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    if lines:
        ax.text(
            x + w / 2,
            y + h - 4.4,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=body_size,
            color=MUTED,
            linespacing=1.45,
            zorder=3,
        )


def arrow(ax, start, end, color=INK, style="-|>", lw=1.4, rad=0.0, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle="--" if dashed else "-",
            zorder=1,
        )
    )


def main() -> int:
    fig, ax = plt.subplots(figsize=(15.2, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(2, 96.2, "Emergency Vehicle Detection & Simulated Smart Traffic Signal", fontsize=17, fontweight="bold", color=INK)
    ax.text(
        2,
        92.6,
        "End-to-end prototype: real detector, real footage, deterministic simulated controller. No connection to road infrastructure.",
        fontsize=9.5,
        color=MUTED,
    )

    # Sources.
    box(ax, 2, 69, 20, 17, "1. Real inputs", ["Webcam", "Video file", "Still image", "Demo footage"], fc=BLUE)
    # Offline ML.
    box(
        ax,
        2,
        16,
        20,
        44,
        "Offline model pipeline",
        [
            "prepare_dataset.py",
            "2 labelled Roboflow sets",
            "998 / 285 / 143 images",
            "4 canonical classes",
            "",
            "train_yolo.py",
            "YOLOv8n, 448 px, CPU",
            "",
            "evaluate.py",
            "held-out test split",
            "P / R / F1 / mAP",
            "FP / FN at operating point",
        ],
        fc=GREY,
    )
    # Perception.
    box(
        ax,
        26,
        63,
        22,
        23,
        "2. Perception (CPU)",
        ["YOLO detector", "confidence + NMS", "→", "IoU tracker", "persistent IDs, max age", "→", "Lane ROI assignment", "normalised polygons"],
        fc=GREEN,
    )
    # Priority.
    box(
        ax,
        52,
        63,
        22,
        23,
        "3. Priority decision",
        ["conf ≥ 0.60", "≥ 3 consecutive frames", "≥ 0.5 s track age", "known lane only", "approaching / queued", "→", "closest EV policy", "cooldown 10 s"],
        fc=AMBER,
    )
    # Controller.
    box(
        ax,
        78,
        63,
        20,
        23,
        "4. Signal FSM",
        ["NORMAL cycle", "EMERGENCY_DETECTED", "PRIORITY_PREPARING", "all-red 1.5 s", "PRIORITY_ACTIVE", "EMERGENCY_CLEARED", "RETURNING_TO_NORMAL", "cooldown → NORMAL"],
        fc=RED,
    )
    # Safety.
    box(
        ax,
        78,
        44,
        20,
        13,
        "Safety interlocks",
        ["lane polygons validated", "conflict groups", "max green 45 s", "forced all-red on failure"],
        fc=RED,
        title_size=9.8,
        body_size=8.0,
    )
    # Baseline.
    box(
        ax,
        52,
        44,
        22,
        13,
        "Fixed-time baseline",
        ["same arrivals", "same green/yellow/all-red", "no priority", "measured counterfactual wait"],
        fc=GREY,
        title_size=9.8,
        body_size=8.0,
    )
    # Outputs.
    box(
        ax,
        2,
        2,
        46,
        11,
        "5. Evidence and outputs",
        ["annotated video · event log CSV/JSONL · performance CSV · report tables", "Streamlit dashboard · pytest suite · reproducible run_all.py"],
        fc=BLUE,
        title_size=10.2,
        body_size=8.2,
    )
    box(
        ax,
        52,
        2,
        46,
        11,
        "6. Experiments (measured, not fabricated)",
        ["fixed-time vs priority simulation · low/medium/high arrival sweep", "held-out detector metrics · operating-point FP/FN · CPU latency/FPS"],
        fc=GREEN,
        title_size=10.2,
        body_size=8.2,
    )

    # Main flow.
    arrow(ax, (22.3, 77.5), (25.7, 77.5))
    arrow(ax, (48.3, 74.5), (51.7, 74.5))
    arrow(ax, (74.3, 74.5), (77.7, 74.5))
    # Perception -> outputs.
    arrow(ax, (37, 62.7), (37, 13.3), color=BLUE, dashed=True, lw=1.2)
    # Priority <-> baseline.
    arrow(ax, (63, 62.7), (63, 57.3), color=MUTED, style="<|-|>", lw=1.2)
    # Controller -> safety.
    arrow(ax, (88, 62.7), (88, 57.3), color=RED)
    # Offline ML -> detector.
    arrow(ax, (22.3, 45), (25.7, 66), color=MUTED, dashed=True, rad=-0.2)

    # Exit / restoration callout.
    ax.text(
        52,
        40.6,
        "EV clears far-side exit ROI → 2 s hold →\nall-red clearance → normal sequence",
        fontsize=7.8,
        color=MUTED,
        va="top",
    )
    ax.text(
        52,
        15.2,
        "SIMULATED CONTROLLER — no real signal hardware is accessed.",
        fontsize=8.0,
        color="#a4161a",
        fontweight="bold",
        va="bottom",
    )

    target = PROJECT_ROOT / "docs" / "architecture.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
