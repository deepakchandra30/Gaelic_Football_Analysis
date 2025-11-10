from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class ActionModel(nn.Module):
    """
    Wrapper for R3D-18 with configurable num_classes and checkpoint loading.
    Input shape: (B, C, T, H, W) with C=3, T typically 16, H=W=112.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.backbone = r3d_18(pretrained=pretrained)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(num_classes: int = 6,
                pretrained_backbone: bool = True,
                checkpoint_path: Optional[str] = None,
                map_location: Optional[str] = None) -> Tuple[nn.Module, Optional[dict]]:
    """
    Build model and optionally load a fine-tuned checkpoint.

    Returns (model, checkpoint_dict) where checkpoint_dict may be None.
    """
    model = ActionModel(num_classes=num_classes, pretrained=pretrained_backbone)
    ckpt = None
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=map_location or 'cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        # Support common keys like 'model.' prefix
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state[k[len('model.'):]] = v
            elif k.startswith('backbone.'):
                new_state[k[len('backbone.'):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)
    return model, ckpt
