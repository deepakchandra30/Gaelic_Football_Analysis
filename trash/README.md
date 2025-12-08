# Gaelic Football Multi-Action Recognition

This repo scaffolds a minimal pipeline to run sliding-window action recognition on a video using a 3D CNN (R3D-18) in PyTorch.

Current focus: inference over a single video to produce action segments for 6 classes:

- 0: kick
- 1: hand_pass
- 2: catch
- 3: solo
- 4: tackle
- 5: idle

> Note: Until you fine-tune the model on your Gaelic dataset, predictions will be random. This pipeline is ready for your checkpoint.

## Quickstart

### 1) Create environment and install dependencies (Windows cmd)

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run a smoke test (no video required)

This confirms the model and transforms run end-to-end on synthetic data:

```
python scripts\smoke_test.py
```

Expected: prints logits shape like `[1, 6]`.

### 3) Run inference on a video

```
python scripts\infer_actions.py --video path\to\your_video.mp4 --class-map configs\class_map.yaml --checkpoint path\to\your_finetuned.pt --device cpu --window 16 --stride 8 --output-dir outputs
```

Artifacts:
- `outputs/clip_predictions.csv` – one row per clip with predicted class and probabilities
- `outputs/segments.json` – merged segments of consecutive clips with the same predicted class

If you don't have a checkpoint yet, omit `--checkpoint` to validate the pipeline. Results will be random.

## How it works

- Video is read frame-by-frame (RGB) via OpenCV.
- Sliding windows of 16 frames with stride 8 are preprocessed to 112x112 and normalized.
- R3D-18 predicts per-clip logits -> softmax probabilities.
- Consecutive clips with the same predicted class are merged into coarse action segments.

## Next steps (training outline)

1. Prepare dataset folders:
```
dataset/
  kick/       # 3–5s mp4 clips
  hand_pass/
  catch/
  solo/
  tackle/
  idle/
```
2. Write a PyTorch Dataset to load (T=16) clips with temporal sampling and augmentations.
3. Fine-tune R3D-18 from torchvision (pretrained on Kinetics-400) to 6 classes.
4. Export checkpoint (`.pt` or `.pth`) compatible with `src/models/r3d_factory.py`.
5. Re-run inference using your checkpoint to get meaningful segments.

## Notes

- Default preprocessing uses ImageNet mean/std; for best results during training match the Kinetics video normalization used by torchvision weights.
- You can switch to R(2+1)D or add temporal smoothing (e.g., HMM/Viterbi) once you have first trained results.
- Annotated video overlay can be added later once class predictions are stable.
