"""Minimal smoke test for the action inference pipeline without a real video.
Generates random frames and runs through model to ensure shapes work.
"""
import torch
import numpy as np
from src.models.r3d_factory import build_model
from src.utils.transforms_3d import preprocess_clip


def main():
    num_classes = 6
    model, _ = build_model(num_classes=num_classes, pretrained_backbone=False)
    model.eval()
    # Create synthetic clip of 16 frames 112x112 RGB
    T = 16
    H = W = 112
    clip = (np.random.rand(T, H, W, 3) * 255).astype('uint8')
    tensor = preprocess_clip(clip)  # (C, T, H, W)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    print('Logits shape:', out.shape)
    print('Sample logits:', out[0])

if __name__ == '__main__':
    main()
