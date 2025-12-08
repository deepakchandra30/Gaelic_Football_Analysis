"""
Prepare short labelled clips from SoccerNet-style event list.

Input: events CSV with columns:
  video_path,timestamp_sec,label
  - video_path : path to the full-match mp4 file (local)
  - timestamp_sec : float, timestamp in seconds where the event occurs
  - label : string label (e.g., 'pass', 'shot', 'foul')

Output: data/clips/<label>/*.mp4
Usage:
  python examples/prepare_soccernet_clips.py --events_csv events.csv --out_dir data/clips --clip_len 4.0 --pad 2.0

Notes:
- You can generate events.csv from SoccerNet annotations (see docs/soccernet_pipeline.md for a short snippet).
- Requires ffmpeg on PATH.
"""
import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Tuple

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--events_csv', required=True, help='CSV with columns: video_path,timestamp_sec,label')
    p.add_argument('--out_dir', default='data/clips', help='Output root for clips (will create subfolders per label)')
    p.add_argument('--clip_len', type=float, default=4.0, help='Duration (s) of extracted clip')
    p.add_argument('--pad', type=float, default=2.0, help='Seconds before event timestamp to start clip (so clip centers on event)')
    p.add_argument('--reencode', action='store_true', help='Re-encode clips with ffmpeg (safer) instead of stream-copy')
    return p.parse_args()

def safe_makedirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def ffmpeg_extract(src: str, dst: str, start: float, duration: float, reencode: bool = False) -> Tuple[bool, str]:
    """
    Extracts a clip using ffmpeg. Returns (success, stdout/stderr).
    """
    safe_makedirs(os.path.dirname(dst))
    if reencode:
        cmd = [
            'ffmpeg', '-y', '-ss', f'{start:.3f}', '-i', src,
            '-t', f'{duration:.3f}', '-vf', 'scale=640:-2', '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '96k', dst
        ]
    else:
        # stream copy (fast) but can fail with some codecs
        cmd = ['ffmpeg', '-y', '-ss', f'{start:.3f}', '-i', src, '-t', f'{duration:.3f}', '-c', 'copy', dst]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True, proc.stderr.decode('utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode('utf-8', errors='replace')

def main():
    args = parse_args()
    out_root = args.out_dir
    clip_len = args.clip_len
    pad = args.pad
    if not os.path.exists(args.events_csv):
        raise FileNotFoundError(f"events_csv not found: {args.events_csv}")

    with open(args.events_csv, 'r') as f:
        reader = csv.DictReader(f)
        i = 0
        failures = 0
        for row in reader:
            video = row['video_path']
            t = float(row['timestamp_sec'])
            label = row['label'].strip().replace(' ', '_')
            if not os.path.exists(video):
                print(f"[WARN] video not found: {video} -> skipping")
                failures += 1
                continue
            start = max(0.0, t - pad)
            dst_dir = os.path.join(out_root, label)
            safe_makedirs(dst_dir)
            dst_name = f"{Path(video).stem}_{int(t)}_{i:04d}.mp4"
            dst = os.path.join(dst_dir, dst_name)
            ok, out = ffmpeg_extract(video, dst, start=start, duration=clip_len, reencode=args.reencode)
            if ok:
                print(f"[OK] wrote {dst} (event {label} @ {t:.2f}s)")
            else:
                print(f"[ERR] failed to write {dst}: {out}")
                failures += 1
            i += 1
    print("Done. Failures:", failures)

if __name__ == '__main__':
    main()