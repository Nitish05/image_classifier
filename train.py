"""
train.py
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import multiprocessing
from multiprocessing import freeze_support
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR    = "data/train"
VAL_DIR      = "data/val"
BATCH_SIZE   = 128
NUM_WORKERS  = 6
LR           = 1e-4
NUM_EPOCHS   = 5
RANDOM_STATE = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # cuDNN can be nondeterministic even after seeding; these settings help:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def main():
    # 0) Seed all RNGs
    seed_everything(RANDOM_STATE)
    print(f"Using device: {DEVICE}  |  Seed: {RANDOM_STATE}")

    # 1) Data transforms & loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406],
                             std=[.229, .224, .225]),
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # you can optionally pass a generator for reproducible shuffling:
        # generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # 2) Build model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}  train loss: {avg_loss:.4f}")

        # 4) Validation
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total if total else 0.0
        print(f"          val acc:   {acc:.4f}\n")

    # 5) Save weights
    torch.save(model.state_dict(), "meme_classifier.pth")
    print("✅  Model saved to meme_classifier.pth")


if __name__ == "__main__":
    freeze_support()
    main()
