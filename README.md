# Gaelic Football Action Recognition

This project implements an action recognition system for Gaelic football using deep learning and computer vision techniques.

## Project Structure

```
├── download_soccernet_data.py     # Download SoccerNet dataset for training
├── prepare_soccernet_clips.py     # Extract action clips from videos
├── generate_clips_csv.py          # Generate training/validation CSV files
├── train.py                       # Main training script (Transfer Learning with ResNet50)
├── requirements.txt               # Python dependencies
├── configs/
│   └── soccernet_class_map.yaml  # Action class definitions
├── src/
│   ├── models/                   # Model architectures
│   └── utils/                    # Helper functions
├── checkpoints/                  # Trained model weights
├── outputs/                      # Training results and reports
└── data/                        # Dataset storage

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python download_soccernet_data.py
```

3. Prepare training clips:
```bash
python prepare_soccernet_clips.py --events_csv data/soccernet_mini/train_events.csv --out_dir data/soccernet_mini/clips
```

4. Generate clip CSV files:
```bash
python generate_clips_csv.py
```

## Training

Train the action recognition model:

```bash
python train.py
```

**Model Details:**
- Architecture: ResNet50 (ImageNet pretrained) with custom classifier
- Input: 224x224 RGB frames (middle frame from video clips)
- Data Augmentation: Horizontal flip, brightness, contrast, rotation
- Regularization: Dropout (0.5), BatchNorm, Weight decay, Label smoothing
- Training: 25 epochs, Transfer learning approach

## Results

The model achieves high accuracy on the SoccerNet action recognition task:
- Training results saved in `outputs/training_report.txt`
- Model checkpoint saved in `checkpoints/resnet50_model.pth`

## Action Classes

The model recognizes 16 different soccer/football actions:
- Ball out of play
- Clearance
- Corner
- Direct free-kick
- Foul
- Goal
- Indirect free-kick
- Kick-off
- Offside
- Penalty
- Red card
- Shots off target
- Shots on target
- Substitution
- Throw-in
- Yellow card

## Future Work

- Extend to full 3D convolutional networks for temporal modeling
- Apply to Gaelic football specific actions
- Implement real-time action detection pipeline
- Add player tracking and team classification
