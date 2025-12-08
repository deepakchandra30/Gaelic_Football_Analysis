from typing import Tuple
import numpy as np
import torch

# Basic preprocessing pipeline compatible with torchvision R3D pretrained weights.
# Expect input clip shape (T, H, W, C) RGB uint8.

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_short_side(frames: np.ndarray, short_size: int = 128) -> np.ndarray:
    # frames: (T, H, W, C)
    import cv2
    T, H, W, C = frames.shape
    if min(H, W) == short_size:
        return frames
    scale = short_size / min(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    resized = np.empty((T, new_h, new_w, C), dtype=frames.dtype)
    for i in range(T):
        resized[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def center_crop(frames: np.ndarray, size: int = 112) -> np.ndarray:
    T, H, W, C = frames.shape
    if H < size or W < size:
        raise ValueError("Frames smaller than crop size.")
    top = (H - size) // 2
    left = (W - size) // 2
    return frames[:, top:top+size, left:left+size, :]


def to_tensor_normalized(frames: np.ndarray) -> torch.Tensor:
    # (T, H, W, C) -> (C, T, H, W) float32 normalized
    frames_f = frames.astype(np.float32) / 255.0
    frames_f = (frames_f - IMAGENET_MEAN) / IMAGENET_STD
    # reorder
    frames_f = np.transpose(frames_f, (3, 0, 1, 2))
    return torch.from_numpy(frames_f)


def preprocess_clip(frames: np.ndarray, short_side: int = 128, crop_size: int = 112) -> torch.Tensor:
    frames = resize_short_side(frames, short_size=short_side)
    frames = center_crop(frames, size=crop_size)
    tensor = to_tensor_normalized(frames)
    return tensor
