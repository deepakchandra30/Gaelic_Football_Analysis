import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import pandas as pd

from src.models.r3d_factory import build_model
from src.utils.video_io import read_all_frames, sliding_windows, batch_frames_to_clip, get_video_info
from src.utils.transforms_3d import preprocess_clip


def load_class_map(path: str) -> Dict[int, str]:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    classes = data['classes']
    # ensure int keys
    return {int(k): v for k, v in classes.items()}


def predict_video(model: torch.nn.Module,
                  frames: List,
                  window_size: int = 16,
                  stride: int = 8,
                  device: str = 'cpu') -> List[Dict]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start, end in sliding_windows(len(frames), window=window_size, stride=stride):
            clip_np = batch_frames_to_clip(frames, start, end)
            clip_tensor = preprocess_clip(clip_np)  # (C, T, H, W)
            clip_tensor = clip_tensor.unsqueeze(0).to(device)
            logits = model(clip_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_class = int(probs.argmax())
            preds.append({
                'start_frame': start,
                'end_frame': end,
                'pred_class': pred_class,
                'probabilities': probs.tolist()
            })
    return preds


def segment_actions(preds: List[Dict], fps: float) -> List[Dict]:
    if not preds:
        return []
    segments = []
    current = preds[0].copy()
    for p in preds[1:]:
        # Merge consecutive clips with the same class
        if p['pred_class'] == current['pred_class']:
            current['end_frame'] = p['end_frame']
        else:
            segments.append(current)
            current = p.copy()
    segments.append(current)
    # Add timestamps
    for s in segments:
        s['start_time_sec'] = round(s['start_frame'] / fps, 3)
        s['end_time_sec'] = round(s['end_frame'] / fps, 3)
    return segments


def main():
    parser = argparse.ArgumentParser(description='Sliding-window Gaelic Football action inference')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--class-map', default='configs/class_map.yaml', help='YAML class map path')
    parser.add_argument('--checkpoint', default=None, help='Optional fine-tuned checkpoint')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--window', type=int, default=16, help='Frames per clip')
    parser.add_argument('--stride', type=int, default=8, help='Stride between windows')
    parser.add_argument('--output-dir', default='outputs', help='Directory to save results')
    args = parser.parse_args()

    class_map = load_class_map(args.class_map)
    num_classes = len(class_map)
    model, _ = build_model(num_classes=num_classes, pretrained_backbone=True, checkpoint_path=args.checkpoint)
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model.to(device)

    info = get_video_info(args.video)
    frames = read_all_frames(args.video)
    raw_preds = predict_video(model, frames, window_size=args.window, stride=args.stride, device=device)
    segments = segment_actions(raw_preds, fps=info.fps)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Save raw predictions CSV
    df = pd.DataFrame(raw_preds)
    df.to_csv(Path(args.output_dir) / 'clip_predictions.csv', index=False)

    # Map class ids to names for segments
    for s in segments:
        s['class_name'] = class_map.get(s['pred_class'], 'unknown')
    with open(Path(args.output_dir) / 'segments.json', 'w') as f:
        json.dump(segments, f, indent=2)

    print('Saved clip predictions to outputs/clip_predictions.csv')
    print('Saved segments to outputs/segments.json')
    print('Top 5 segments preview:')
    for s in segments[:5]:
        print(s)

if __name__ == '__main__':
    main()
