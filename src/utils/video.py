"""Video, webcam and image input helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    index: int
    source_time: float      # seconds since source start
    wall_time: float        # monotonic wall clock (seconds)
    source_fps: float
    is_image: bool = False


class VideoSource:
    """Uniform frame source for webcam, video files and still images.

    Videos loop by default so the dashboard/demo can run indefinitely. Images
    yield a single frame and then stop (loop is ignored).
    """

    def __init__(
        self,
        source: str = "webcam",
        path: str | Path | None = None,
        loop: bool = True,
        webcam_index: int = 0,
        resize: Tuple[int, int] | None = None,
    ) -> None:
        self.kind = self._normalise_kind(source)
        self.loop = loop
        self.webcam_index = webcam_index
        self.resize = resize
        self.path = Path(path) if path else None
        self.cap: cv2.VideoCapture | None = None
        self.source_fps = 0.0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self._frame_idx = 0
        self._image: np.ndarray | None = None
        self._image_emitted = False
        self._reset_count = 0
        self._open()

    @staticmethod
    def _normalise_kind(source: str) -> str:
        s = str(source).lower().strip()
        if s in {"webcam", "camera", "cam", "0"}:
            return "webcam"
        if s in {"image", "photo"}:
            return "image"
        if s in {"video", "file", "demo"}:
            return "video"
        if s.isdigit():
            return "webcam"
        raise ValueError(f"Unsupported source: {source}. Use webcam, video or image.")

    def _open(self) -> None:
        if self.kind == "image":
            if not self.path or not self.path.is_file():
                raise FileNotFoundError(f"Image not found: {self.path}")
            self._image = cv2.imread(str(self.path))
            if self._image is None:
                raise RuntimeError(f"Could not decode image: {self.path}")
            self.height, self.width = self._image.shape[:2]
            self.source_fps = 1.0
            self.frame_count = 1
            return

        target: int | str = self.webcam_index if self.kind == "webcam" else str(self.path)
        if self.kind == "video" and (not self.path or not self.path.is_file()):
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.path or self.webcam_index}")
        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.kind == "webcam":
            # Webcam frame counts are unreliable/meaningless.
            self.frame_count = 0

    @property
    def is_live(self) -> bool:
        return self.kind == "webcam"

    @property
    def duration(self) -> float:
        if self.kind == "image":
            return 0.0
        if self.source_fps > 0 and self.frame_count > 0:
            return self.frame_count / self.source_fps
        return 0.0

    def read(self) -> tuple[bool, Frame | None]:
        """Read one frame. Returns (ok, Frame). Rewinds automatically if looping."""
        if self.kind == "image":
            if self._image_emitted:
                return False, None
            self._image_emitted = True
            image = self._image
            if self.resize is not None:
                image = cv2.resize(image, self.resize)
            return True, Frame(image, 0, 0.0, time.monotonic(), 1.0, True)

        assert self.cap is not None
        ok, image = self.cap.read()
        if not ok:
            if self.loop and self.kind == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._reset_count += 1
                self._frame_idx = 0
                ok, image = self.cap.read()
                if not ok:
                    return False, None
            else:
                return False, None

        if self.resize is not None:
            image = cv2.resize(image, self.resize)
        source_time = self._frame_idx / max(self.source_fps, 1e-6)
        frame = Frame(image, self._frame_idx, source_time, time.monotonic(), self.source_fps)
        self._frame_idx += 1
        return True, frame

    def __iter__(self) -> Iterator[Frame]:
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class VideoWriter:
    """Small wrapper that opens the writer lazily and clamps frame sizes."""

    def __init__(self, path: str | Path, fps: float = 20.0, size: Tuple[int, int] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(float(fps), 1.0)
        self.size = size
        self.writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray) -> None:
        if self.writer is None:
            h, w = image.shape[:2]
            self.size = self.size or (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self.path}")
        if image.shape[1] != self.size[0] or image.shape[0] != self.size[1]:
            image = cv2.resize(image, self.size)
        self.writer.write(image)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def enumerate_video_files(directory: str | Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in exts)


def get_video_info(path: str | Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frames": count,
        "duration": count / fps if fps else 0.0,
    }
    cap.release()
    return info


def grab_thumbnail(path: str | Path, fraction: float = 0.5, width: int = 640) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * fraction)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(width * h / max(w, 1))))
