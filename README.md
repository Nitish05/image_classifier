# Image Classifier

This repository contains simple utilities for building and using a binary "meme" vs. "non_meme" image classifier based on ResNet‑18.

## Contents

- `preprocess.py` – prepares a training dataset by combining the Kaggle "6992 Meme Images" archive with the Flickr8k dataset.
- `train.py` – trains a ResNet‑18 model on the prepared dataset and saves the weights to `meme_classifier.pth`.
- `classify_photos.py` – uses the trained model to sort images in your photo collection into meme and non‑meme folders.

The repository also includes a pre‑trained weight file (`meme_classifier.pth`).

## Installation

1. Install Python 3.8 or later.
2. Create a virtual environment (optional) and install the required packages:

```bash
pip install torch torchvision pandas scikit-learn Pillow
```

## Data Preparation

1. Download the following datasets (not included in this repo):
   - **6992 Meme Images** from Kaggle
   - **Flickr8k** image dataset
2. Unzip the archives so that the folder structure matches what `preprocess.py` expects:
   - `archive/` should contain a `labels.csv` and an `images/` directory with meme images.
   - `flickr8k/Images/` should contain the Flickr images.
3. Run the preprocessing script to create the `data/` directory used for training:

```bash
python preprocess.py
```

This will create `data/train/` and `data/val/` folders with `meme` and `non_meme` subfolders.

## Training

Train the classifier using the prepared dataset:

```bash
python train.py
```

The script trains a ResNet‑18 for a few epochs and saves the model weights to `meme_classifier.pth`.

## Classifying Your Photos

You can use the trained model to sort your personal photos. Example usage:

```bash
python classify_photos.py \
    --src_dir "/path/to/Photos" \
    --dst_meme "/path/to/Photos/Memes" \
    --dst_nonmeme "/path/to/Photos/Keep" \
    --model_path meme_classifier.pth
```

Images found in `src_dir` are moved into the destination folders based on the model prediction.

## Notes

- The scripts assume a CUDA‑enabled GPU is available but will fall back to CPU if necessary.
- The provided weight file was trained on a small dataset and is intended as a demonstration.

