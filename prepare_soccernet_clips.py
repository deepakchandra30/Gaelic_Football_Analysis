import argparse
import csv
import os
import subprocess
from pathlib import Path

def safe_makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def ffmpeg_extract(src, dst, start, duration, reencode=False):
    safe_makedirs(os.path.dirname(dst))
    if reencode:
        cmd = [
            'ffmpeg', '-y', '-ss', f'{start:.3f}', '-i', src,
            '-t', f'{duration:.3f}', '-vf', 'scale=640:-2', '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '96k', dst
        ]
    else:
        cmd = ['ffmpeg', '-y', '-ss', f'{start:.3f}', '-i', src, '-t', f'{duration:.3f}', '-c', 'copy', dst]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True, proc.stderr.decode('utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode('utf-8', errors='replace')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events_csv', required=True)
    p.add_argument('--out_dir', default='data/clips')
    p.add_argument('--clip_len', type=float, default=4.0)
    p.add_argument('--pad', type=float, default=2.0)
    p.add_argument('--reencode', action='store_true')
    args = p.parse_args()

    failures = 0
    with open(args.events_csv, 'r') as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            video = row['video_path']
            if not os.path.exists(video):
                print(f"[WARN] missing video: {video} -> skipping")
                failures += 1
                continue
            try:
                t = float(row['timestamp_sec'])
            except:
                print(f"[WARN] bad timestamp: {row['timestamp_sec']} -> skipping")
                failures += 1
                continue
            label = str(row['label']).strip().replace(' ', '_')
            start = max(0.0, t - args.pad)
            dst_dir = os.path.join(args.out_dir, label)
            Path(dst_dir).mkdir(parents=True, exist_ok=True)
            dst_name = f"{Path(video).stem}_{int(t)}_{i:04d}.mp4"
            dst = os.path.join(dst_dir, dst_name)
            ok, out = ffmpeg_extract(video, dst, start, args.clip_len, reencode=args.reencode)
            if ok:
                print(f"[OK] {dst}")
            else:
                print(f"[ERR] {dst}: {out}")
                failures += 1
            i += 1
    print("Done. Failures:", failures)

if __name__ == '__main__':
    main()