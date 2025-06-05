
"""
preprocess.py

Combine the Kaggle “6992 Meme Images” (archive/) and Flickr8k (flickr8k/Images)
into a binary “meme” vs. “non_meme” train/val folder structure.

Usage:
    python preprocess.py

Requirements:
    pip install pandas scikit-learn
"""

import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MEME_LABELS_CSV   = os.path.join("archive",   "labels.csv")
MEME_IMAGES_ROOT  = os.path.join("archive",   "images", "images")
FLICKR_IMAGES_DIR = os.path.join("flickr8k",  "Images")
OUTPUT_DIR        = "data"           # will create data/train/... & data/val/...
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load meme labels
    df_meme = pd.read_csv(MEME_LABELS_CSV)
    df_meme = df_meme.rename(columns={"image_name": "filename"})
    df_meme["label"] = "meme"

    # 2) Gather non‑meme (Flickr8k) filenames
    flickr_files = [
        f for f in os.listdir(FLICKR_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    df_non = pd.DataFrame({
        "filename": flickr_files,
        "label":    ["non_meme"] * len(flickr_files)
    })

    # 3) Combine into one DataFrame
    df_all = pd.concat(
        [df_meme[["filename", "label"]], df_non],
        ignore_index=True
    )

    # 4) Stratified train/val split
    train_df, val_df = train_test_split(
        df_all,
        test_size=TEST_SIZE,
        stratify=df_all["label"],
        random_state=RANDOM_STATE
    )

    # 5) Build a lookup for meme image paths (handles nested folders)
    meme_paths = {}
    for root, _, files in os.walk(MEME_IMAGES_ROOT):
        for fname in files:
            meme_paths[fname] = os.path.join(root, fname)

    # 6) Copy files into data/{train,val}/{meme,non_meme}/
    for subset, subset_df in [("train", train_df), ("val", val_df)]:
        for _, row in subset_df.iterrows():
            fn = row["filename"]
            lbl = row["label"]
            if lbl == "meme":
                src = meme_paths.get(fn)
                if src is None:
                    print(f"⚠️  Meme file not found, skipping: {fn}")
                    continue
            else:
                src = os.path.join(FLICKR_IMAGES_DIR, fn)
                if not os.path.isfile(src):
                    print(f"⚠️  Flickr file not found, skipping: {fn}")
                    continue

            dst = os.path.join(OUTPUT_DIR, subset, lbl, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    # 7) Report
    print("✅ Preprocessing complete!")
    print(f"   • train: {len(train_df)} images")
    print(f"   • val:   {len(val_df)} images")


if __name__ == "__main__":
    main()
