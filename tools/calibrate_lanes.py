"""Interactive lane/exit polygon calibration for a camera view.

Controls
--------
1-4       select lane (north, south, east, west)
e         toggle editing approach polygon / exit polygon
left click  add polygon vertex
u         undo last vertex
c         clear current polygon draft
s         save current draft to config/intersection.yaml (backup created)
q / ESC   quit without saving remaining drafts

The defaults are only a schematic. A wrong approach ROI can suppress or
mis-trigger simulated priority, so recalibrate for every camera.

Example
-------
    python tools/calibrate_lanes.py --input data/raw/ambu-test/9784628-uhd_2560_1440_24fps.mp4
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_reference(path: str, max_width: int) -> tuple[np.ndarray, str]:
    p = Path(path)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(p))
        if image is None:
            raise RuntimeError(f"Could not read image: {p}")
        cap_info = "image"
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {p}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.35)))
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not extract a frame from: {p}")
        cap_info = f"video frame ~35% ({total} frames)"
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, int(h * scale)))
    return image, cap_info


def calibrate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    lanes = spec.setdefault("lanes", {})
    exits = spec.setdefault("exit_zones", {})
    if not lanes:
        raise ValueError("No lanes in intersection config")
    names = list(lanes.keys())
    image, info = _load_reference(args.input, args.max_width)
    h, w = image.shape[:2]
    mode = "lane"
    selected = names[0]
    points: list[tuple[int, int]] = []
    saved = {}

    def to_norm(pts):
        return [[round(x / w, 6), round(y / h, 6)] for x, y in pts]

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    window = "Lane calibration - 1-4 lane | e approach/exit | click add | u undo | s save | q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(f"Reference: {args.input} [{info}] {w}x{h}")
    print("Defaults are schematic. Draw quadrilaterals that follow the visible approach lanes.")
    while True:
        canvas = image.copy()
        # Existing geometry, thin lines.
        for name, lane in lanes.items():
            poly = np.asarray(lane.get("polygon", []), dtype=np.float32)
            if len(poly) >= 3:
                pts = (poly * np.array([w, h])).astype(np.int32)
                colour = (60, 200, 60) if name == selected and mode == "lane" else (120, 120, 120)
                cv2.polylines(canvas, [pts], True, colour, 1, cv2.LINE_AA)
                cv2.putText(canvas, str(lane.get("display", name)), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
            if name in exits:
                epoly = np.asarray(exits[name], dtype=np.float32)
                if len(epoly) >= 3:
                    epts = (epoly * np.array([w, h])).astype(np.int32)
                    colour = (0, 200, 255) if name == selected and mode == "exit" else (90, 90, 90)
                    cv2.polylines(canvas, [epts], True, colour, 1, cv2.LINE_AA)
                    cv2.putText(canvas, f"{name} exit", tuple(epts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        if points:
            arr = np.asarray(points, dtype=np.int32)
            for i in range(len(points) - 1):
                cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (0, 80, 255), 2)
            if len(points) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [arr], (0, 120, 255))
                cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
            for i, pt in enumerate(points):
                cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
                cv2.putText(canvas, str(i + 1), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(canvas, (8, 8), (min(w - 8, 670), 96), (20, 20, 20), -1)
        cv2.putText(canvas, f"EDIT {mode.upper()} | LANE {selected.upper()} | vertices {len(points)}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(canvas, "1-4 lane  e approach/exit  left-click add  u undo  c clear  s save  q quit", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(canvas, "Use the visible stop line and lane boundaries. Do not guess.", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in {27, ord("q")}:
            break
        if ord("1") <= key <= ord("4"):
            index = key - ord("1")
            if index < len(names):
                selected = names[index]
                points = []
                print(f"Selected {selected} ({mode})")
        elif key == ord("e"):
            mode = "exit" if mode == "lane" else "lane"
            points = []
            print(f"Editing {mode}")
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points = []
        elif key == ord("s"):
            if len(points) < 3:
                print("Need at least 3 vertices before saving.")
                continue
            norm = to_norm(points)
            if mode == "lane":
                lanes.setdefault(selected, {})["polygon"] = norm
            else:
                exits[selected] = norm
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = config_path.with_name(f"{config_path.stem}.backup_{stamp}{config_path.suffix}")
            shutil.copy2(config_path, backup)
            config_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            saved[selected] = mode
            print(f"Saved {selected} {mode} polygon -> {config_path} (backup: {backup.name})")
            points = []
    cv2.destroyAllWindows()
    if saved:
        print(f"Saved changes for: {saved}")
    else:
        print("No changes saved.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Video or image used as calibration reference")
    parser.add_argument("--config", default="config/intersection.yaml")
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()
    try:
        return calibrate(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
