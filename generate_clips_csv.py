import os
import csv
from pathlib import Path
import random

# Path to clips directory
CLIPS_DIR = Path("data/soccernet_mini/clips")
OUTPUT_DIR = Path("data/soccernet_mini")

# Collect all clips with their labels
clips = []
for label_dir in CLIPS_DIR.iterdir():
    if not label_dir.is_dir():
        continue
    
    label_name = label_dir.name
    for clip_file in label_dir.glob("*.mp4"):
        clips.append({
            'path': str(clip_file),
            'label': label_name
        })

print(f"Found {len(clips)} clips across {len(list(CLIPS_DIR.iterdir()))} classes")

# Shuffle and split into train/val
random.seed(42)
random.shuffle(clips)

split_idx = int(len(clips) * 0.8)
train_clips = clips[:split_idx]
val_clips = clips[split_idx:]

# Write train CSV
with open(OUTPUT_DIR / "train_clips.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'label'])
    writer.writeheader()
    writer.writerows(train_clips)

# Write val CSV
with open(OUTPUT_DIR / "val_clips.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'label'])
    writer.writeheader()
    writer.writerows(val_clips)

print(f"Created train_clips.csv with {len(train_clips)} samples")
print(f"Created val_clips.csv with {len(val_clips)} samples")
