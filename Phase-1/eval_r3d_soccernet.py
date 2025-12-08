"""
Evaluate a saved checkpoint and write predictions + metrics.

Usage:
python examples/eval_r3d_soccernet.py --val_csv data/val.csv --ckpt outputs/soccernet_r3d/best.pth --num_classes 4 --out outputs/soccernet_eval
"""
import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
sys.path.append(os.path.abspath('.'))

from src.models.r3d_factory import build_model
from src.utils.video_io import read_all_frames, sample_indices
from src.utils.transforms_3d import preprocess_clip

class ClipCsvDataset:
    def __init__(self, csv_file, clip_len=16):
        import csv as _csv
        self.rows = []
        with open(csv_file, 'r') as f:
            r = _csv.DictReader(f)
            for row in r:
                self.rows.append((row['path'], int(row['label'])))
        self.clip_len = clip_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        frames = read_all_frames(path)
        num_frames = len(frames)
        indices = sample_indices(num_frames or 1, self.clip_len)
        sampled = [frames[i] if i < num_frames else frames[-1] for i in indices]
        clip_np = np.stack(sampled, axis=0)
        tensor = preprocess_clip(clip_np)
        return tensor, label, path

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return xs, ys, paths

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--val_csv', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--clip_len', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='outputs/soccernet_eval')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = ClipCsvDataset(args.val_csv, clip_len=args.clip_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model, _ = build_model(num_classes=args.num_classes, pretrained_backbone=True, checkpoint_path=None)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(args.device)
    model.eval()

    all_preds, all_trues, rows = [], [], []
    with torch.no_grad():
        for X, y, paths in loader:
            X = X.to(args.device)
            out = model(X)
            probs = torch.nn.functional.softmax(out, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(y.numpy().tolist())
            for pth, pr, pb in zip(paths, preds, probs.cpu().numpy().tolist()):
                rows.append({'path': pth, 'pred': int(pr), 'probs': pb})

    acc = accuracy_score(all_trues, all_preds)
    report = classification_report(all_trues, all_preds, digits=4)
    cm = confusion_matrix(all_trues, all_preds).tolist()
    print("Accuracy:", acc)
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)

    with open(os.path.join(args.out, 'predictions.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump({'accuracy': acc, 'report': report, 'confusion_matrix': cm}, f, indent=2)

if __name__ == '__main__':
    main()