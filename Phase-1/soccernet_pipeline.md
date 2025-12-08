```markdown
SoccerNet pipeline (quick guide) — use this for midpoint

1) Obtain SoccerNet videos & annotations
- Visit https://github.com/SoccerNet/SoccerNet and follow their README to download videos and annotation files.
- Alternatively, if access is slow, prepare a small set of allowed soccer clips (UCF101 soccer clips or CC soccer clips) as proof-of-concept.

2) Convert SoccerNet annotations to events.csv
- SoccerNet annotations are match-level files. Create a CSV with columns:
    video_path,timestamp_sec,label
  where video_path is the local path to the full match mp4 and timestamp_sec is the event time (in seconds).
- Example small script (pseudo):
  - parse the SoccerNet JSON annotations for events and write each event row with the match video local path.

3) Extract short clips around events
- Run:
  python examples/prepare_soccernet_clips.py --events_csv events.csv --out_dir data/clips --clip_len 4.0 --pad 2.0 --reencode
- This will create data/clips/<label>/*.mp4

4) Prepare train/val CSVs
- Use the existing helper:
  python examples/prepare_video_csv.py --clips_root data/clips --out_dir data --val_frac 0.2
- This writes data/train.csv, data/val.csv, data/class_map.txt

5) Fine-tune R3D
- Small run for midpoint:
  python examples/train_r3d_soccernet.py --train_csv data/train.csv --val_csv data/val.csv --num_classes <N> --epochs 5 --batch_size 4 --freeze --output outputs/soccernet_r3d --device cpu
- For GPU use --device cuda

6) Evaluate
  python examples/eval_r3d_soccernet.py --val_csv data/val.csv --ckpt outputs/soccernet_r3d/best.pth --num_classes <N> --out outputs/soccernet_eval

7) Inference on a full match (once you have permitted full-match video and ethics)
- Use your repo inference script:
  python scripts/infer_actions.py --video path/to/match.mp4 --class-map configs/soccernet_class_map.yaml --checkpoint outputs/soccernet_r3d/best.pth --device cpu --window 16 --stride 8 --output-dir outputs/match_analysis

Notes & tips
- For small datasets freeze the backbone (use --freeze) to avoid overfitting.
- Use data augmentation: temporal jitter (different pad values), horizontal flip, color jitter.
- Record all provenance (SoccerNet version, annotation files used, download timestamps) and include in your midpoint appendix.
- Cite SoccerNet and the R3D/Kinetics pretraining in your report.

References to cite:
- SoccerNet: https://github.com/SoccerNet/SoccerNet
- R3D and Kinetics (torchvision models): https://pytorch.org/vision/stable/models.html#video-classification
- Carreira & Zisserman (I3D) and other video model papers if you mention them.
```