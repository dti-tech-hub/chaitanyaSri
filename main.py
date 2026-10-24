"""Command-line runner for the emergency-vehicle smart-signal prototype.

Examples
--------
    python main.py --source demo
    python main.py --source video --input data/raw/ambu-test/9784628-uhd_2560_1440_24fps.mp4
    python main.py --source image --input data/samples/ambulance_1.jpg
    python main.py --source webcam --camera 0

The controller is simulated. No real traffic hardware is accessed.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from src.pipeline import EmergencySignalPipeline
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import setup_logger
from src.utils.video import VideoSource, VideoWriter

LOGGER = setup_logger("main")


def _default_demo_video() -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "ambu-test" / "9784628-uhd_2560_1440_24fps.mp4",
        PROJECT_ROOT / "data" / "raw" / "ambu-test" / "9784626-uhd_2560_1440_24fps.mp4",
        PROJECT_ROOT / "data" / "raw" / "ambu-test" / "Police Cars Fire Trucks And Ambulances Responding Compilation Part 12.mp4",
    ]
    for path in candidates:
        if path.is_file():
            return path
    videos = sorted((PROJECT_ROOT / "data" / "raw" / "ambu-test").glob("*.mp4"))
    return videos[0] if videos else None


def _resize_max_width(image, max_width: int):
    if max_width <= 0:
        return image
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _build_source(args: argparse.Namespace) -> tuple[VideoSource, bool]:
    source = args.source.lower()
    if source == "demo":
        path = Path(args.input) if args.input else _default_demo_video()
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "No demo video found. Download the real emergency-response clips described "
                "in docs/DATASETS.md or pass --input path/to/video.mp4."
            )
        LOGGER.info("DEMO SOURCE (real footage): %s", path)
        return VideoSource("video", path=path, loop=args.loop), False
    if source == "webcam":
        return VideoSource("webcam", webcam_index=args.camera, loop=False), True
    if source == "video":
        if not args.input:
            raise ValueError("--input is required for --source video")
        return VideoSource("video", path=args.input, loop=args.loop), False
    if source == "image":
        if not args.input:
            raise ValueError("--input is required for --source image")
        return VideoSource("image", path=args.input), False
    raise ValueError(f"Unsupported source: {source}")


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.confidence is not None:
        config = config.with_overrides({"model": {"confidence": float(args.confidence)}})
    source, live = _build_source(args)
    pipeline = EmergencySignalPipeline(
        config,
        model_path=args.model,
        reset_events=not args.keep_events,
        display_lanes=not args.hide_lanes,
    )
    pipeline.set_live_source(live)
    writer = None
    display_ok = bool(args.display)
    window = str(config.get("realtime.window_name", "Emergency Vehicle Smart Signal - Simulated Prototype"))
    max_frames = int(args.max_frames or config.get("realtime.max_frames", 0) or 0)
    frame_delay = max(1, int(1000.0 / max(1.0, float(config.get("realtime.target_fps", 15)))))
    processed = 0
    started = time.perf_counter()
    try:
        for frame in source:
            if args.max_width and frame.image.shape[1] > args.max_width:
                frame.image = _resize_max_width(frame.image, args.max_width)
            result = pipeline.process(frame, live=live)
            if args.save_video and writer is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = PROJECT_ROOT / "results" / "videos"
                out_dir.mkdir(parents=True, exist_ok=True)
                target = Path(args.output) if args.output else out_dir / f"demo_{stamp}.mp4"
                writer = VideoWriter(target, fps=float(config.get("realtime.target_fps", 15)))
            if writer is not None:
                writer.write(result.annotated)
            processed += 1
            if processed % 100 == 0:
                LOGGER.info(
                    "frames=%d | sim=%.1fs | fps=%.1f | state=%s | priority_lane=%s",
                    processed,
                    result.sim_time,
                    result.metrics.get("fps", 0.0),
                    result.signal.get("state"),
                    result.signal.get("priority_lane"),
                )
            if display_ok:
                try:
                    view = _resize_max_width(result.annotated, args.max_width or 1280)
                    cv2.imshow(window, view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        LOGGER.info("Stopped by user")
                        break
                except cv2.error as exc:
                    LOGGER.warning("Display unavailable (%s); continuing without window", exc)
                    display_ok = False
            elif not live:
                # Preserve approximate source pace only when processing is faster
                # than the target; never slow down slower-than-realtime processing.
                elapsed = time.perf_counter() - started
                expected = processed / max(1.0, float(config.get("realtime.target_fps", 15)))
                if expected > elapsed:
                    time.sleep(min(0.05, expected - elapsed))
            if max_frames and processed >= max_frames:
                LOGGER.info("Reached max frames (%d)", max_frames)
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        if writer is not None:
            writer.release()
        source.release()
        if display_ok:
            cv2.destroyAllWindows()
        pipeline.close()

    summary = pipeline.summary()
    elapsed = time.perf_counter() - started
    print("\n=== RUN SUMMARY ===")
    print(f"Frames processed : {processed}")
    print(f"Wall time        : {elapsed:.1f}s ({processed / max(elapsed, 1e-9):.2f} FPS end-to-end)")
    print(f"Model            : {summary['model']['model_path']}")
    print(f"Emergency-capable: {summary['model']['emergency_capable']}")
    print(f"Priority events  : {summary['priority'].get('priority_events', 0)}")
    avg_wait = summary["priority"].get("avg_emergency_wait_s")
    avg_base = summary["priority"].get("avg_baseline_wait_s")
    print(f"EV wait (measured): {avg_wait if avg_wait is not None else 'not measured'}")
    print(f"Baseline wait     : {avg_base if avg_base is not None else 'not available'}")
    print(f"Events CSV        : {summary['events_csv']}")
    print("SIMULATED PROTOTYPE - NOT CONNECTED TO REAL TRAFFIC SIGNALS")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--source", choices=["demo", "video", "image", "webcam"], default="demo")
    parser.add_argument("--input", default=None, help="Video/image path for --source video/image")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--model", default=None, help="Override YOLO weights path")
    parser.add_argument("--confidence", type=float, default=None, help="Model detection confidence")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize frames to this width (0 = native)")
    parser.add_argument("--loop", action="store_true", help="Loop video files")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window")
    parser.add_argument("--hide-lanes", action="store_true", help="Hide lane ROI overlays")
    parser.add_argument("--save-video", action="store_true", help="Write an annotated MP4")
    parser.add_argument("--output", default=None, help="Annotated output MP4 path")
    parser.add_argument("--keep-events", action="store_true", help="Append to existing events CSV")
    args = parser.parse_args(argv)
    args.display = not args.no_display
    try:
        return run(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
