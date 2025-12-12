import os
import sys
import json
import csv
from pathlib import Path
import subprocess

try:
    from SoccerNet.Downloader import SoccerNetDownloader
except ImportError:
    print("Installing SoccerNet...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "SoccerNet"])
    from SoccerNet.Downloader import SoccerNetDownloader

OUTPUT_DIR = "data/soccernet_mini"
PASSWORD = "s0cc3rn3t"

# Hardcode the first 2 games from the SoccerNet 'train' split to be explicit.
GAMES_TO_DOWNLOAD = [
    "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
    "england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

downloader = SoccerNetDownloader(LocalDirectory=OUTPUT_DIR)
downloader.password = PASSWORD

# Step 1: Download the videos and annotations for only the 2 specified games.
print(f"Downloading {len(GAMES_TO_DOWNLOAD)} specific games...")
for game in GAMES_TO_DOWNLOAD:
    print(f"  Downloading {game}...")
    downloader.downloadGame(
        game=game,
        files=["Labels-v2.json", "1_224p.mkv", "2_224p.mkv"]
    )

# Step 2: Process the annotations for the games we just downloaded.
print("\nConverting annotations to CSV...")
events = []
for game_name in GAMES_TO_DOWNLOAD:
    game_dir = Path(OUTPUT_DIR) / game_name
    json_file = game_dir / "Labels-v2.json"
    
    if not json_file.exists():
        print(f"Warning: Annotation file not found for {game_name}")
        continue

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for ann in data.get('annotations', []):
        video_half = ann.get('half', '1')
        video_path = game_dir / f"{video_half}_224p.mkv"
        if video_path.exists():
            position = ann.get('position', 0)
            # Convert position to float if it's a string
            if isinstance(position, str):
                position = int(position)
            events.append({
                'video_path': str(video_path),
                'timestamp_sec': position / 1000.0,
                'label': ann.get('label', ''),
            })

# Step 3: Create train/val CSVs from the collected events.
if not events:
    print("\n❌ No events found. Cannot create CSV files. Please check the download.")
    sys.exit(1)

train_events = events[:int(len(events) * 0.8)]
val_events = events[int(len(events) * 0.8):]

with open(Path(OUTPUT_DIR) / "train_events.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video_path', 'timestamp_sec', 'label'])
    writer.writeheader()
    writer.writerows(train_events)

with open(Path(OUTPUT_DIR) / "val_events.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video_path', 'timestamp_sec', 'label'])
    writer.writeheader()
    writer.writerows(val_events)

print(f"\n✅ Done. {len(events)} events from {len(GAMES_TO_DOWNLOAD)} matches are ready.")
print(f"   Train: {len(train_events)} events | Val: {len(val_events)} events")
print(f"   Data is in: {OUTPUT_DIR}")