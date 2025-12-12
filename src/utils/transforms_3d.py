import torch
import numpy as np

def preprocess_clip(clip_np, augment=False):
    """
    Preprocess video clip with optional data augmentation
    Args:
        clip_np: numpy array of shape (T, H, W, C)
        augment: whether to apply random augmentations (for training)
    """
    clip_np = clip_np.astype(np.float32) / 255.0
    
    # Data augmentation for training
    if augment:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            clip_np = np.flip(clip_np, axis=2).copy()
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        clip_np = np.clip(clip_np * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)
        clip_np = np.clip((clip_np - 0.5) * contrast_factor + 0.5, 0, 1)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    clip_np = (clip_np - mean) / std
    clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2)
    return clip_tensor
