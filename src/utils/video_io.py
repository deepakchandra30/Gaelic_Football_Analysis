import cv2
import numpy as np

def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def sample_indices(num_frames, clip_len):
    if num_frames <= clip_len:
        indices = list(range(num_frames))
        while len(indices) < clip_len:
            indices.append(num_frames - 1)
        return indices
    step = (num_frames - 1) / (clip_len - 1)
    return [int(i * step) for i in range(clip_len)]
