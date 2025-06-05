"""
classify_photos.py

Classify images in your photo library as “meme” vs. “non_meme” using a trained ResNet‑18.

Usage:
  python classify_photos.py \
    --src_dir "D:/MyPhotos" \
    --dst_meme "D:/MyPhotos/Memes" \
    --dst_nonmeme "D:/MyPhotos/Keep" \
    --model_path "D:/image_classifier/meme_classifier.pth"

Requirements:
  pip install torch torchvision Pillow
"""

import os
import shutil
import argparse

from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms

# Allow PIL to load truncated/corrupt images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_model(model_path, device):
    # Build the same architecture
    model = models.resnet18(pretrained=False)
    # Two classes: 0=meme, 1=non_meme
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Load weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def classify_image(img_path, model, device, transform):
    try:
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            pred = logits.argmax(dim=1).item()
        return pred  # 0 or 1
    except Exception as e:
        print(f"⚠️  Failed to process {img_path}: {e}")
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir",     required=True, help="Root folder of your photos")
    p.add_argument("--dst_meme",    required=True, help="Where to move memes")
    p.add_argument("--dst_nonmeme", required=True, help="Where to move non‑memes")
    p.add_argument("--model_path",  default="meme_classifier.pth",
                   help="Path to your trained .pth file")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model_path, device)

    # Same preprocessing as training
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406],
                             std=[.229,.224,.225]),
    ])

    # Make sure output dirs exist
    os.makedirs(args.dst_meme,    exist_ok=True)
    os.makedirs(args.dst_nonmeme, exist_ok=True)

    # Walk through all images
    exts = (".jpg",".jpeg",".png")
    src_root = os.path.abspath(args.src_dir)
    dst_meme = os.path.abspath(args.dst_meme)
    dst_nonmeme = os.path.abspath(args.dst_nonmeme)

    def is_within(path, directory):
        """Return True if *path* is inside *directory* (handles cross-drive paths)."""
        try:
            return os.path.commonpath([path, directory]) == directory
        except ValueError:
            return False

    for root, _, files in os.walk(src_root):
        for fn in files:
            if not fn.lower().endswith(exts):
                continue

            src_path = os.path.join(root, fn)
            src_path_abs = os.path.abspath(src_path)
            # Skip if already in one of the destination folders
            if is_within(src_path_abs, dst_meme) or is_within(src_path_abs, dst_nonmeme):
                continue

            cls = classify_image(src_path, model, device, transform)
            if cls is None:
                continue

            if cls == 0:
                dst_path = os.path.join(args.dst_meme, fn)
            else:
                dst_path = os.path.join(args.dst_nonmeme, fn)

            shutil.move(src_path, dst_path)
            print(f"{fn} → {'Meme' if cls==0 else 'Keep'}")

    print("✅  Classification complete!")

if __name__ == "__main__":
    main()
