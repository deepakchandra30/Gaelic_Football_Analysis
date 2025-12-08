"""
Fine-tune R3D-18 on clips extracted from SoccerNet.

Assumes:
- data/train.csv and data/val.csv exist (path,label)
- configs/soccernet_class_map.yaml or data/class_map.txt exists (for mapping id->name)
- uses src/models/r3d_factory.build_model and src/utils/transforms_3d.preprocess_clip

Example:
  python examples/train_r3d_soccernet.py --train_csv data/train.csv --val_csv data/val.csv --num_classes 4 --epochs 6 --batch_size 6 --output outputs/soccernet_r3d

Notes:
- For small datasets keep backbone frozen (default), only train final fc.
- For larger datasets set --freeze False to fine-tune whole model.
"""
import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
sys.path.append(os.path.abspath('.'))

from src.models.r3d_factory import build_model
from src.utils.video_io import read_all_frames, sample_indices
from src.utils.transforms_3d import preprocess_clip

class ClipCsvDataset(Dataset):
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
        frames = read_all_frames(path)  # list of RGB uint8 frames
        num_frames = len(frames)
        indices = sample_indices(num_frames or 1, self.clip_len)
        sampled = [frames[i] if i < num_frames else frames[-1] for i in indices]
        clip_np = np.stack(sampled, axis=0)
        tensor = preprocess_clip(clip_np)  # (C, T, H, W)
        return tensor, label

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--val_csv', required=True)
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--clip_len', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--output', default='outputs/soccernet_r3d')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--freeze', action='store_true', help='Freeze backbone and only train final fc')
    return p.parse_args()

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    losses = []
    preds = []
    trues = []
    pbar = tqdm(loader, desc='Train')
    for X, y in pbar:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
        trues.extend(y.cpu().numpy().tolist())
        losses.append(loss.item())
        pbar.set_postfix(loss=np.mean(losses), acc=100.0*accuracy_score(trues, preds))
    return np.mean(losses), accuracy_score(trues, preds)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds = []
    trues = []
    for X, y in tqdm(loader, desc='Val'):
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
        trues.extend(y.cpu().numpy().tolist())
    return accuracy_score(trues, preds)

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = args.device

    train_ds = ClipCsvDataset(args.train_csv, clip_len=args.clip_len)
    val_ds = ClipCsvDataset(args.val_csv, clip_len=args.clip_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model, _ = build_model(num_classes=args.num_classes, pretrained_backbone=True, checkpoint_path=None)
    model = model.to(device)

    # Freeze backbone if requested (faster for small datasets)
    if args.freeze:
        for name, p in model.named_parameters():
            if 'backbone.fc' not in name:
                p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = eval_epoch(model, val_loader, device)
        print(f"Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Val acc {val_acc*100:.2f}%")
        ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'val_acc': val_acc}
        torch.save(ckpt, os.path.join(args.output, f'ckpt_epoch{epoch}.pth'))
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, os.path.join(args.output, 'best.pth'))
    print("Training complete. Best val acc:", best_val)

if __name__ == '__main__':
    main()