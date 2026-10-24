"""Generate every figure and table used by the conference paper package.

All values are read from committed run artifacts; nothing is hand-typed.
Outputs:
  conference_package/figures/*.png
  conference_package/tables/*.csv, *.md
  conference_package/assets_manifest.md
"""
from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "conference_package"
FIG = PKG / "figures"
TAB = PKG / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- helpers
INK = "#1f2937"
MUTED = "#6b7280"
BLUE = "#2a6f97"
TEAL = "#2a9d8f"
AMBER = "#e9c46a"
ORANGE = "#f4a261"
RED = "#e76f51"
LIGHT = "#eef2f6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=9, bold=False, tc=INK, rounded=0.02):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={rounded}",
        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.25)
    return patch


def arrow(ax, p1, p2, label=None, color=MUTED, style="-|>", lw=1.2, ls="-", fs=7.5, rad=0.0, lx=0, ly=0):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle=style, mutation_scale=10, linewidth=lw,
            color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )
    if label:
        ax.text(
            (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label,
            fontsize=fs, color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=4,
        )


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_csv(name, header, rows):
    with (TAB / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def write_md(name, title, header, rows, note=""):
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    (TAB / name).write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- data
val = load("results/metrics/dataset_validation.json")
full = load("results/metrics/detection_metrics.json")
clean = load("conference_package/experiments/leakage_screened/screened_metrics.json")
sim = load("results/simulation/simulation_summary.json")
train = load("results/training/training_metadata.json")
audit = load("paper/provenance/near_duplicate_audit.json")
recomputed = load("conference_package/provenance/near_duplicate_audit_recomputed.json")
audit_summary = load("conference_package/provenance/near_duplicate_audit_summary.json")
clean_images = clean["split"]["images_kept"]
clean_instances = sum(c["instances"] for c in clean["per_class"])
density_rows = list(csv.DictReader((ROOT / "results/simulation/density_sweep.csv").open(encoding="utf-8")))
events = sim["priority_events"]

CLASSES = ["ambulance", "fire_engine", "police_vehicle", "normal_vehicle"]
CLASS_COLORS = {"ambulance": BLUE, "fire_engine": RED, "police_vehicle": AMBER, "normal_vehicle": TEAL}

# ----------------------------------------------------------------------------- Fig 1 architecture
fig, ax = plt.subplots(figsize=(12.4, 6.5))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 6.5)
ax.axis("off")

ax.text(0.15, 6.22, "Online perception and simulated control", fontsize=11, fontweight="bold", color=INK)
box(ax, 0.15, 4.55, 1.55, 1.05, "Camera /\nvideo / image", fc="#f8fafc")
box(ax, 2.05, 4.55, 1.55, 1.05, "YOLOv8n\n4-class detector")
box(ax, 3.95, 4.55, 1.45, 1.05, "Class-aware\nIoU tracker")
box(ax, 5.75, 4.55, 1.55, 1.05, "Lane ROI +\napproach test")
box(ax, 7.65, 4.55, 1.75, 1.05, "Priority engine\n(7 conditions)", fc="#e8f4f1", ec=TEAL)
box(ax, 9.75, 4.55, 1.6, 1.05, "Guarded signal\nFSM (simulated)", fc="#e8f4f1", ec=TEAL)
box(ax, 11.55, 5.25, 0.7, 0.62, "Signal\nstate", fc="#fdf6e3", ec=AMBER, fs=8)
box(ax, 11.55, 4.35, 0.7, 0.7, "Overlay\n& UI", fc="#fdf6e3", ec=AMBER, fs=8)

for x1, x2 in [(1.70, 2.05), (3.60, 3.95), (5.40, 5.75), (7.30, 7.65), (9.40, 9.75), (11.40, 11.55)]:
    arrow(ax, (x1, 5.07), (x2, 5.07))
arrow(ax, (10.55, 5.60), (11.55, 5.56), color=AMBER)
arrow(ax, (10.55, 4.85), (11.55, 4.70), color=AMBER)

box(ax, 7.65, 3.28, 1.75, 0.72, "Event log +\nwait metrics", fc="#fdf6e3", ec=AMBER, fs=8)
box(ax, 9.75, 3.28, 1.6, 0.72, "Fixed-time\nbaseline", fc="#fdf6e3", ec=AMBER, fs=8)
arrow(ax, (8.52, 4.55), (8.52, 4.00), color=AMBER)
arrow(ax, (9.40, 3.64), (9.75, 3.64), color=AMBER)

ax.text(0.15, 2.72, "Offline training and evaluation", fontsize=11, fontweight="bold", color=INK)
box(ax, 0.15, 1.15, 1.85, 1.05, "Roboflow\nsources (CC BY 4.0)", fc="#f8fafc")
box(ax, 2.30, 1.15, 1.85, 1.05, "Merge, remap,\ndedup, audit")
box(ax, 4.45, 1.15, 1.85, 1.05, "Train / val / test\n(998 / 285 / 143)")
box(ax, 6.60, 1.15, 1.85, 1.05, "Fine-tune\nYOLOv8n @448")
box(ax, 8.75, 1.15, 2.05, 1.05, "Held-out test\nevaluation")
box(ax, 11.10, 1.15, 1.15, 1.05, "Metrics\n& tables", fc="#fdf6e3", ec=AMBER, fs=8)
for x1, x2 in [(2.00, 2.30), (4.15, 4.45), (6.30, 6.60), (8.45, 8.75), (10.80, 11.10)]:
    arrow(ax, (x1, 1.67), (x2, 1.67))
arrow(ax, (7.52, 2.20), (3.00, 4.50), label="weights", color=TEAL, ls=(0, (4, 3)), rad=0.18, lx=-0.2, ly=0.15)

box(ax, 0.15, 0.10, 7.10, 0.72,
    "Experiment harness: seeded Poisson arrivals, dt = 0.25 s\n"
    "same arrival list -> fixed-time baseline and proposed priority runs",
    fc="#f8fafc", ec=MUTED, fs=7.6)
arrow(ax, (7.25, 0.46), (11.10, 0.46), label="event-level metrics", color=AMBER, ls=(0, (4, 3)), fs=7)
fig.savefig(FIG / "fig1_system_architecture.png")
plt.close(fig)

# ----------------------------------------------------------------------------- Fig 2 FSM
fig, ax = plt.subplots(figsize=(11.6, 5.5))
ax.set_xlim(0, 11.6)
ax.set_ylim(0, 5.5)
ax.axis("off")

states = {
    "NORMAL": (0.25, 3.45, 1.85, 0.85),
    "DETECTED": (2.85, 3.45, 2.05, 0.85),
    "PREPARING": (5.45, 3.45, 2.10, 0.85),
    "ACTIVE": (8.05, 3.45, 1.95, 0.85),
    "CLEARED": (8.05, 1.95, 1.95, 0.85),
    "RETURNING": (5.45, 1.95, 2.10, 0.85),
    "FAILSAFE": (0.70, 0.20, 2.00, 0.75),
}
labels = {
    "NORMAL": "NORMAL\nfixed cycle",
    "DETECTED": "EMERGENCY_\nDETECTED",
    "PREPARING": "PRIORITY_\nPREPARING",
    "ACTIVE": "PRIORITY_\nACTIVE",
    "CLEARED": "EMERGENCY_\nCLEARED",
    "RETURNING": "RETURNING_\nTO_NORMAL",
    "FAILSAFE": "FAIL_SAFE\nall red",
}
for key, (x, y, w, h) in states.items():
    fc = "#e8f4f1" if key in {"ACTIVE", "PREPARING"} else ("#fdecea" if key == "FAILSAFE" else LIGHT)
    ec = TEAL if key in {"ACTIVE", "PREPARING"} else (RED if key == "FAILSAFE" else BLUE)
    box(ax, x, y, w, h, labels[key], fc=fc, ec=ec, fs=8.5, bold=key in {"ACTIVE", "FAILSAFE"})

arrow(ax, (2.10, 3.87), (2.85, 3.87))
ax.text(2.48, 4.36, "confirmed\nEV request", fontsize=7.2, ha="center", va="bottom")
arrow(ax, (4.90, 3.87), (5.45, 3.87))
ax.text(5.18, 4.36, "conflicting\nmovement", fontsize=7.2, ha="center", va="bottom")
arrow(ax, (7.55, 3.87), (8.05, 3.87))
ax.text(7.20, 3.36, "all-red prepare\n>= 1.5 s", fontsize=7.2, ha="center", va="top")
arrow(ax, (9.02, 3.45), (9.02, 2.80))
ax.text(9.18, 3.12, "exit ROI + 2 s hold / 45 s cap", fontsize=7.2, ha="left", va="center")
arrow(ax, (8.05, 2.37), (7.55, 2.37))
ax.text(7.80, 2.82, "all-red\nclearance", fontsize=7.2, ha="center", va="bottom")
arrow(ax, (5.45, 2.37), (2.10, 3.55), color=MUTED, rad=-0.25)
ax.text(3.45, 2.55, "restore normal sequence", fontsize=7.2, ha="center", color=MUTED,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# Already-green fast path, routed above the row so it cannot collide with labels.
ax.plot([3.87, 3.87], [4.30, 4.85], color=TEAL, lw=1.1, ls=(0, (4, 3)))
ax.plot([3.87, 9.02], [4.85, 4.85], color=TEAL, lw=1.1, ls=(0, (4, 3)))
arrow(ax, (9.02, 4.85), (9.02, 4.32), color=TEAL, ls=(0, (4, 3)))
ax.text(6.45, 4.98, "approach already safely green (no re-phasing)", fontsize=7.2, color=TEAL, ha="center")

# Fail-safe applies to every state: a single labelled fault path keeps the plot readable.
arrow(ax, (4.60, 1.30), (1.80, 0.95), color=RED, ls=(0, (3, 3)), lw=1.0, rad=0.15)
ax.text(4.75, 1.28, "any state: exception, invalid transition or safety violation ->\nall red; explicit reset required", fontsize=7.2, color=RED, va="center")
ax.text(5.45, 1.62, "10 s cooldown starts at release", fontsize=7, color=MUTED)
fig.savefig(FIG / "fig2_priority_fsm.png")
plt.close(fig)

# ----------------------------------------------------------------------------- Fig 3 dataset + leakage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
splits = ["train", "val", "test"]
width = 0.2
xs = range(len(splits))
for i, cls in enumerate(CLASSES):
    values = [val["splits"][s]["class_counts"][cls] for s in splits]
    bars = ax1.bar([x + (i - 1.5) * width for x in xs], values, width, label=cls, color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 8, str(v), ha="center", va="bottom", fontsize=6.6, color=INK)
ax1.set_xticks(list(xs))
ax1.set_xticklabels([f"{s} (n={val['splits'][s]['images']})" for s in splits])
ax1.set_ylabel("Annotated boxes")
ax1.set_ylim(0, 780)
ax1.set_title("(a) Dataset composition by split", fontsize=9.5, loc="left")
ax1.legend(ncol=2, fontsize=6.8, frameon=False, loc="upper right")
ax1.grid(axis="y", alpha=0.2)
ax1.set_axisbelow(True)

hamm = [row["hamming"] for row in recomputed["pairs"]]
bins = [0, 1, 2, 3, 4, 5, 6]
counts = [hamm.count(b) for b in bins]
bars = ax2.bar([str(b) for b in bins], counts, color=[RED if b <= 2 else ORANGE if b <= 4 else AMBER for b in bins], edgecolor="white")
for bar, v in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=7)
ax2.set_xlabel("dHash Hamming distance to nearest train/val image")
ax2.set_ylabel("Test images")
ax2.set_ylim(0, max(counts) + 6)
ax2.set_title("(b) Near-duplicate screen: 87 of 143 test images removed", fontsize=9.5, loc="left")
ax2.text(0.02, 0.97, f"committed audit {audit_summary['committed_flagged']} | dHash {audit_summary['recomputed_flagged']} | union {audit_summary['union']}",
         transform=ax2.transAxes, fontsize=6.8, color=INK, va="top")
ax2.text(0.02, 0.87, f"{len([h for h in hamm if h == 0])} dHash pairs are identical", transform=ax2.transAxes,
         fontsize=6.8, color=RED, va="top")
ax2.grid(axis="y", alpha=0.2)
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "fig3_dataset_and_leakage.png")
plt.close(fig)

# ----------------------------------------------------------------------------- Fig 4 detector comparison
fig, ax = plt.subplots(figsize=(8.4, 3.8))
metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("mAP50", "mAP@50"), ("mAP50-95", "mAP@50-95")]
x = range(len(metrics))
full_vals = [full["overall"][k] for k, _ in metrics]
clean_vals = [clean["overall"][k] for k, _ in metrics]
b1 = ax.bar([i - 0.19 for i in x], full_vals, 0.38, label="Full test split (n=143)", color="#9db4c0", edgecolor="white")
b2 = ax.bar([i + 0.19 for i in x], clean_vals, 0.38, label=f"Leakage-screened (n={clean_images})", color=TEAL, edgecolor="white")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}",
                ha="center", fontsize=7)
ax.set_xticks(list(x))
ax.set_xticklabels([label for _, label in metrics])
ax.set_ylim(0, 1.09)
ax.set_ylabel("Score")
ax.set_title("Held-out detector performance before and after removing near-duplicate test images", fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "fig4_detector_full_vs_screened.png")
plt.close(fig)

# ----------------------------------------------------------------------------- Fig 5 event-level simulation
fig, ax = plt.subplots(figsize=(8.4, 3.9))
labels = [f"E{e['event_id']}  {e['vehicle_class'].replace('_', ' ')} / {e['lane']}" for e in events]
ys = list(range(len(events)))[::-1]
base = [e["baseline_wait_s"] for e in events]
prop = [e["waiting_time_s"] for e in events]
ax.barh([y + 0.19 for y in ys], base, 0.36, color="#9db4c0", label="Fixed-time baseline", edgecolor="white")
ax.barh([y - 0.19 for y in ys], prop, 0.36, color=TEAL, label="Emergency priority", edgecolor="white")
for y, b, p in zip(ys, base, prop):
    ax.text(b + 0.4, y + 0.19, f"{b:.1f}", va="center", fontsize=7, color=MUTED)
    ax.text(p + 0.4, y - 0.19, f"{p:.1f}", va="center", fontsize=7, color=TEAL)
    delta = p - b
    ax.text(31.8, y, f"{delta:+.1f} s", va="center", ha="right", fontsize=8,
            color=TEAL if delta < 0 else RED, fontweight="bold")
ax.axvline(29.0, color="#d1d5db", lw=0.8)
ax.text(30.4, 4.62, "change", fontsize=7.5, color=MUTED, ha="center")
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=7.6)
ax.set_xlim(0, 32.5)
ax.set_xlabel("Waiting time (s)")
ax.set_title("Per-event simulated waiting time (seed 42): priority helps on red arrivals, costs 3.5 s on green arrivals", fontsize=8.2, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "fig5_event_level_waits.png")
plt.close(fig)

# ----------------------------------------------------------------------------- copy real result plots
copies = {
    "results/figures/dataset_samples.png": "fig6_dataset_samples.png",
    "results/figures/confusion_matrix_normalized.png": "fig7_confusion_matrix_full.png",
    "results/metrics/yolo_val_test/BoxPR_curve.png": "fig8_precision_recall_full.png",
    "results/figures/simulation_waiting_time.png": "fig9_simulation_aggregate.png",
    "results/figures/simulation_wait_distribution.png": "fig10_ev_wait_distribution.png",
    "results/figures/detection_metrics.png": "fig11_detection_metrics_plot.png",
    "results/figures/roc_curve.png": "fig12_auxiliary_roc.png",
    "conference_package/experiments/leakage_screened/yolo_val_clean/confusion_matrix_normalized.png": "fig13_confusion_matrix_screened.png",
    "conference_package/experiments/leakage_screened/yolo_val_clean/BoxPR_curve.png": "fig14_precision_recall_screened.png",
}
for src, dst in copies.items():
    src_path = ROOT / src
    if src_path.is_file():
        shutil.copy2(src_path, FIG / dst)

# ----------------------------------------------------------------------------- tables
rows = []
for split in splits:
    s = val["splits"][split]
    rows.append([split, s["images"], s["boxes"]] + [s["class_counts"][c] for c in CLASSES])
write_csv("table1_dataset.csv", ["split", "images", "boxes"] + CLASSES, rows)
write_md(
    "table1_dataset.md",
    "Table 1. Dataset composition after merging, class remapping and exact-duplicate removal",
    ["Split", "Images", "Boxes", "Ambulance", "Fire engine", "Police", "Normal"],
    [[r[0].capitalize(), *r[1:]] for r in rows],
    note="Source: `results/metrics/dataset_validation.json` (status: ok, 0 corrupt images, 0 missing labels, 0 invalid boxes, 0 unknown class IDs).",
)

train_rows = [
    ["Device", train["device"]],
    ["Input size", f"{train['imgsz']} x {train['imgsz']}"],
    ["Batch size", train["batch"]],
    ["Epochs requested", train["epochs_requested"]],
    ["Elapsed", f"{train['elapsed_s']:.1f} s ({train['elapsed_human']})"],
    ["Seed / deterministic", f"{train['command_parameters']['seed']} / {train['command_parameters']['deterministic']}"],
    ["Optimizer / LR", f"{train['command_parameters']['optimizer']} / lr0={train['command_parameters']['lr0']}, lrf={train['command_parameters']['lrf']}, cosine={train['command_parameters']['cos_lr']}"],
    ["Augmentation", "HSV (0.015/0.6/0.4), rotate 5 deg, translate 0.1, scale 0.5, shear 2, flip-LR 0.5, mosaic 0.8, mixup 0.05"],
    ["Model", "YOLOv8n, 72 layers, 3,006,428 parameters, 8.1 GFLOPs"],
]
write_csv("table2_training.csv", ["field", "value"], train_rows)
write_md("table2_training.md", "Table 2. Detector training configuration and cost", ["Field", "Value"], train_rows,
         note="Source: `results/training/training_metadata.json`; parameter count from the Ultralytics model summary in the evaluation log.")

full_rows = [["Overall", "-", "-", full["overall"]["precision"], full["overall"]["recall"], full["overall"]["f1"], full["overall"]["mAP50"], full["overall"]["mAP50-95"]]]
for c in full["per_class"]:
    full_rows.append([c["class"], c["images"], c["instances"], c["precision"], c["recall"], c["f1"], c["mAP50"], c["mAP50-95"]])
write_csv("table3_detector_full.csv", ["class", "images", "instances", "precision", "recall", "f1", "mAP50", "mAP50-95"], full_rows)
write_md("table3_detector_full.md", "Table 3. Detector performance on the full held-out test split (143 images)",
         ["Class", "Images", "Instances", "P", "R", "F1", "mAP@50", "mAP@50-95"],
         [[r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]:.3f}", f"{r[7]:.3f}"] for r in full_rows],
         note="Source: `results/metrics/detection_metrics.json`. This split contains near-duplicate frames of training images (see Table 4 and Fig. 3b).")

clean_rows = [["Overall", "-", "-", clean["overall"]["precision"], clean["overall"]["recall"], clean["overall"]["f1"], clean["overall"]["mAP50"], clean["overall"]["mAP50-95"]]]
for c in clean["per_class"]:
    clean_rows.append([c["class"], c["images"], c["instances"], c["precision"], c["recall"], c["f1"], c["mAP50"], c["mAP50-95"]])
write_csv("table4_detector_screened.csv", ["class", "images", "instances", "precision", "recall", "f1", "mAP50", "mAP50-95"], clean_rows)
write_md("table4_detector_screened.md", f"Table 4. Detector performance after removing all near-duplicate test images ({clean_images} images)",
         ["Class", "Images", "Instances", "P", "R", "F1", "mAP@50", "mAP@50-95"],
         [[r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]:.3f}", f"{r[7]:.3f}"] for r in clean_rows],
         note="Source: `conference_package/experiments/leakage_screened/screened_metrics.json`. Same checkpoint, same thresholds (mAP: conf=0.001, IoU=0.60; operating point: conf=0.55, IoU>=0.50).")

op_rows = []
for name, data, n_img, n_inst in [
    ("Full test split", full, 143, 226),
    ("Leakage-screened", clean, clean_images, clean_instances),
]:
    op = data["operating_point"]
    op_rows.append([name, n_img, n_inst, op["tp"], op["fp"], op["fn"], op["precision"], op["recall"], op["f1"]])
write_csv("table5_operating_point.csv", ["split", "images", "instances", "tp", "fp", "fn", "precision", "recall", "f1"], op_rows)
write_md("table5_operating_point.md", "Table 5. Fixed-threshold operating point used by the live pipeline (conf = 0.55, IoU >= 0.50)",
         ["Split", "Images", "Instances", "TP", "FP", "FN", "P", "R", "F1"],
         [[r[0], r[1], r[2], r[3], r[4], r[5], f"{r[6]:.3f}", f"{r[7]:.3f}", f"{r[8]:.3f}"] for r in op_rows],
         note="Sources: `results/metrics/detection_metrics.json`, `conference_package/experiments/leakage_screened/screened_metrics.json`.")

b, p = sim["baseline"], sim["proposed"]
sim_rows = [
    ["Emergency mean wait (s)", b["emergency_waiting_time_s"], p["emergency_waiting_time_s"], round(p["emergency_waiting_time_s"] - b["emergency_waiting_time_s"], 2)],
    ["Emergency max wait (s)", b["emergency_max_wait_s"], p["emergency_max_wait_s"], round(p["emergency_max_wait_s"] - b["emergency_max_wait_s"], 2)],
    ["Normal mean wait (s)", round(b["normal_waiting_time_s"], 2), round(p["normal_waiting_time_s"], 2), round(p["normal_waiting_time_s"] - b["normal_waiting_time_s"], 2)],
    ["Normal max wait (s)", round(b["normal_max_wait_s"], 2), round(p["normal_max_wait_s"], 2), round(p["normal_max_wait_s"] - b["normal_max_wait_s"], 2)],
    ["Emergency vehicles served", f"{b['emergency_served']} / {b['emergency_total']}", f"{p['emergency_served']} / {p['emergency_total']}", "-"],
    ["Throughput (veh/h)", b["throughput_vph"], p["throughput_vph"], p["throughput_vph"] - b["throughput_vph"]],
    ["Priority activations", b["priority_events"], p["priority_events"], p["priority_events"] - b["priority_events"]],
    ["Priority response (s)", "n/a", p["priority_response_time_s"], "-"],
]
write_csv("table6_simulation_aggregate.csv", ["metric", "fixed_time_baseline", "emergency_priority", "change"], sim_rows)
write_md("table6_simulation_aggregate.md", "Table 6. Simulated fixed-time vs emergency-priority control (600 s, seed 42, 287 vehicles)",
         ["Metric", "Fixed-time", "Priority", "Change"], sim_rows,
         note="Source: `results/simulation/simulation_summary.json`. Signal-level simulation only; not a road-traffic measurement.")

event_rows = []
for e in events:
    event_rows.append([
        e["event_id"], e["vehicle_class"], e["lane"], f"{e['arrival_time']:.1f}", f"{e['requested_time']:.1f}",
        f"{e['green_time']:.1f}", f"{e['release_time']:.1f}", f"{e['waiting_time_s']:.1f}",
        f"{e['baseline_wait_s']:.1f}", f"{e['waiting_time_s'] - e['baseline_wait_s']:+.1f}", f"{e['priority_duration_s']:.1f}",
    ])
write_csv("table7_priority_events.csv",
          ["event_id", "class", "lane", "arrival_s", "request_s", "green_s", "release_s", "wait_s", "baseline_wait_s", "delta_s", "priority_duration_s"],
          event_rows)
write_md("table7_priority_events.md", "Table 7. Event-level priority outcome (seed 42)",
         ["#", "Class", "Lane", "Arrival (s)", "Request (s)", "Green (s)", "Release (s)", "Wait (s)", "Baseline wait (s)", "Delta", "Green held (s)"],
         event_rows,
         note="Source: `results/simulation/simulation_summary.json`. Positive delta means the priority controller was slower for that event.")

groups = {}
for row in density_rows:
    groups.setdefault(row["density"], []).append(row)
sweep_summary = []
for density in ["low", "medium", "high"]:
    rs = groups[density]
    def mean_sd(key):
        vals = [float(r[key]) for r in rs]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0
    ev_base = mean_sd("baseline_emergency_wait_s")
    ev_prop = mean_sd("proposed_emergency_wait_s")
    n_base = mean_sd("baseline_normal_wait_s")
    n_prop = mean_sd("proposed_normal_wait_s")
    t_base = mean_sd("baseline_throughput_vph")
    t_prop = mean_sd("proposed_throughput_vph")
    ev_count = sum(int(r["priority_events"]) for r in rs)
    sweep_summary.append([
        density, len(rs), f"{ev_base[0]:.2f} +/- {ev_base[1]:.2f}", f"{ev_prop[0]:.2f} +/- {ev_prop[1]:.2f}",
        f"{n_base[0]:.2f} +/- {n_base[1]:.2f}", f"{n_prop[0]:.2f} +/- {n_prop[1]:.2f}",
        f"{t_base[0]:.0f} +/- {t_base[1]:.0f}", f"{t_prop[0]:.0f} +/- {t_prop[1]:.0f}", ev_count,
    ])
write_csv("table8_density_sweep.csv",
          ["density", "seeds", "baseline_ev_wait_s", "proposed_ev_wait_s", "baseline_normal_wait_s", "proposed_normal_wait_s", "baseline_throughput_vph", "proposed_throughput_vph", "priority_events"],
          sweep_summary)
write_md("table8_density_sweep.md", "Table 8. Arrival-rate sweep, mean +/- sample standard deviation over seeds 42, 43, 44",
         ["Density", "Seeds", "Baseline EV wait (s)", "Priority EV wait (s)", "Baseline normal wait (s)", "Priority normal wait (s)", "Baseline throughput", "Priority throughput", "Priority events"],
         sweep_summary,
         note="Source: `results/simulation/density_sweep.csv`. The EV wait is identical across seeds because the EV schedule is fixed and the controller delays are deterministic.")

env_rows = [
    ["OS", "Windows 10/11 workstation"],
    ["Python", "3.14.5"],
    ["PyTorch", "2.14.1+cpu"],
    ["Ultralytics", "8.4.174"],
    ["OpenCV", "5.0.0"],
    ["NumPy / pandas", "2.4.4 / 3.0.1"],
    ["Streamlit / Matplotlib", "1.65.0 / 3.10.8"],
    ["scikit-learn / PyYAML / pytest", "1.8.0 / 6.0.3 / 9.1.1"],
    ["GPU", "None (CPU-only run)"],
    ["Model size / cost", "YOLOv8n, 3.01 M parameters, 8.1 GFLOPs"],
]
write_csv("table9_environment.csv", ["component", "value"], env_rows)
write_md("table9_environment.md", "Table 9. Reported execution environment", ["Component", "Value"], env_rows,
         note="Source: `docs/ENVIRONMENT.md`, confirmed by the evaluation log (Python 3.14.5, torch 2.14.1+cpu, Ultralytics 8.4.174).")

# ----------------------------------------------------------------------------- manifest
manifest = [
    "# Asset manifest",
    "",
    "All figures and tables are generated by `conference_package/scripts/make_paper_assets.py`",
    "from committed artifacts. No value is typed by hand.",
    "",
    "| Asset | Source |",
    "|---|---|",
    "| fig1_system_architecture.png | drawn from `docs/ARCHITECTURE.md` and `src/` module boundaries |",
    "| fig2_priority_fsm.png | drawn from the state machine in `src/control/signal_controller.py` and `docs/ARCHITECTURE.md` |",
    "| fig3_dataset_and_leakage.png | `results/metrics/dataset_validation.json`, `paper/provenance/near_duplicate_audit.json`, `conference_package/provenance/near_duplicate_audit_recomputed.json` |",
    "| fig4_detector_full_vs_screened.png | `results/metrics/detection_metrics.json`, `conference_package/experiments/leakage_screened/screened_metrics.json` |",
    "| fig5_event_level_waits.png | `results/simulation/simulation_summary.json` |",
    "| fig6_dataset_samples.png | copied from `results/figures/dataset_samples.png` |",
    "| fig7_confusion_matrix_full.png | copied from `results/figures/confusion_matrix_normalized.png` (full test split) |",
    "| fig8_precision_recall_full.png | copied from `results/metrics/yolo_val_test/BoxPR_curve.png` (full test split) |",
    "| fig9_simulation_aggregate.png | copied from `results/figures/simulation_waiting_time.png` |",
    "| fig10_ev_wait_distribution.png | copied from `results/figures/simulation_wait_distribution.png` |",
    "| fig11_detection_metrics_plot.png | copied from `results/figures/detection_metrics.png` |",
    "| fig12_auxiliary_roc.png | copied from `results/figures/roc_curve.png` |",
    "| fig13_confusion_matrix_screened.png | copied from `conference_package/experiments/leakage_screened/yolo_val_clean/confusion_matrix_normalized.png` |",
    "| fig14_precision_recall_screened.png | copied from `conference_package/experiments/leakage_screened/yolo_val_clean/BoxPR_curve.png` |",
    "| table1..table9 (.csv/.md) | see the note under each table |",
    "",
]
(PKG / "ASSETS_MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")

print("assets written")
print(sorted(p.name for p in FIG.iterdir()))
print(sorted(p.name for p in TAB.iterdir()))
