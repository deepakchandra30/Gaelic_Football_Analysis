from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterator
import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def get_video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def read_all_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def sliding_windows(total: int, window: int, stride: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) frame indices [start, end) for sliding windows."""
    i = 0
    while i + window <= total:
        yield i, i + window
        i += stride
    # Optionally include a tail window if remainder is significant
    remainder = total - i
    if remainder >= window // 2 and total > window:
        yield total - window, total


def batch_frames_to_clip(frames: List[np.ndarray], start: int, end: int) -> np.ndarray:
    """
    Stack frames[start:end] into array (T, H, W, C) in RGB uint8.
    """
    clip = np.stack(frames[start:end], axis=0)
    return clip
