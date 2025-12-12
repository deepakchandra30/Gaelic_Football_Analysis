import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import cv2
import torchvision.models as models
import torchvision.transforms as transforms

# Enhanced Transfer learning with pretrained ResNet50
class TransferCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Fine-tune more layers (last 60 params) for better adaptation
        for param in list(resnet.parameters())[:-60]:
            param.requires_grad = False
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Deeper classifier with progressive dimensionality reduction
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleDataset(Dataset):
    def __init__(self, csv_file, label_map=None, augment=False):
        self.rows = []
        self.label_map = label_map or {}
        self.augment = augment
        
        if not self.label_map:
            # Build label map
            unique_labels = set()
            with open(csv_file, 'r') as f:
                r = csv.DictReader(f)
                for row in r:
                    unique_labels.add(row['label'])
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        # Load data
        with open(csv_file, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                label_text = row['label']
                if label_text in self.label_map:
                    path = row.get('path') or row.get('video_path')
                    self.rows.append((path, self.label_map[label_text]))
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        path, label = self.rows[idx]
        # Read just the middle frame
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            frame = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Resize and preprocess
        frame = cv2.resize(frame, (224, 224))  # ResNet expects 224x224
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        # Enhanced data augmentation for better generalization
        if self.augment:
            # Random horizontal flip (50%)
            if np.random.rand() > 0.5:
                frame = np.fliplr(frame).copy()
            # Less aggressive brightness/contrast for realism
            frame = np.clip(frame * np.random.uniform(0.85, 1.15), 0, 1)
            frame = np.clip((frame - 0.5) * np.random.uniform(0.85, 1.15) + 0.5, 0, 1)
            # Smaller rotation to preserve action semantics
            angle = np.random.uniform(-5, 5)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # Occasional Gaussian noise for robustness
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 0.015, frame.shape).astype(np.float32)
                frame = np.clip(frame + noise, 0, 1)
        
        # ImageNet normalization for pretrained ResNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        return tensor, label

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses, preds, trues = [], [], []
    for X, y in tqdm(loader, desc='Train'):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    
    return np.mean(losses), accuracy_score(trues, preds)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in tqdm(loader, desc='Val'):
        X, y = X.to(device), y.to(device)
        out = model(X)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    return accuracy_score(trues, preds), preds, trues

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_ds = SimpleDataset('data/soccernet_mini/train_clips.csv', augment=True)
    val_ds = SimpleDataset('data/soccernet_mini/val_clips.csv', label_map=train_ds.label_map, augment=False)
    
    num_classes = len(train_ds.label_map)
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    print("\nCreating pretrained ResNet50 model...")
    model = TransferCNN(num_classes).to(device)
    
    # Only optimize unfrozen parameters with better learning rate
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=0.0005, weight_decay=5e-5)  # Lower LR and weight decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Less aggressive smoothing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5)
    
    # Train for 40 epochs for better convergence
    print("\nTraining with enhanced transfer learning (targeting 90%+ accuracy)...")
    best_val = 0.0
    epochs = 40
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_preds, val_trues = eval_epoch(model, val_loader, device)
        
        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc*100:.2f}%")
        print(f"Val Acc: {val_acc*100:.2f}% {'⭐ NEW BEST!' if val_acc > best_val else ''}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val:
            best_val = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'label_map': train_ds.label_map,
                'val_acc': val_acc
            }, 'checkpoints/resnet50_model.pth')
    
    print(f"\n✅ Training complete! Best val accuracy: {best_val*100:.2f}%")
    
    # Generate detailed report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    model.load_state_dict(torch.load('checkpoints/resnet50_model.pth')['model_state'])
    val_acc, val_preds, val_trues = eval_epoch(model, val_loader, device)
    
    label_names = {v: k for k, v in train_ds.label_map.items()}
    
    # Get unique labels present in validation set
    unique_labels = sorted(set(val_trues))
    target_names = [label_names[i] for i in unique_labels]
    
    report = classification_report(val_trues, val_preds, labels=unique_labels, target_names=target_names, digits=3)
    print(report)
    
    # Save report
    with open('outputs/training_report.txt', 'w') as f:
        f.write("TRANSFER LEARNING MODEL - ACTION RECOGNITION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: ResNet50 (ImageNet pretrained) + Custom Classifier\n")
        f.write(f"Training samples: {len(train_ds)}\n")
        f.write(f"Validation samples: {len(val_ds)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Training epochs: 25\n")
        f.write(f"Input size: 224x224\n")
        f.write(f"Transfer learning: Yes (frozen early layers, fine-tuned last 30 layers)\n")
        f.write(f"Data augmentation: Yes (flip, brightness, contrast, rotation)\n")
        f.write(f"Regularization: Dropout (0.5), BatchNorm, Weight decay (1e-4), Label smoothing (0.1)\n")
        f.write(f"Best validation accuracy: {best_val*100:.2f}%\n\n")
        f.write("Classes:\n")
        for name, idx in sorted(train_ds.label_map.items(), key=lambda x: x[1]):
            f.write(f"  {idx}: {name}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    print("\n✅ Report saved to outputs/training_report.txt")
    print("✅ Model saved to: checkpoints/resnet50_model.pth")

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    main()
