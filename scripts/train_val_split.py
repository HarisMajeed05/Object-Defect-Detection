r"""
scripts/train_val_split.py
============================
Takes a flat folder of images + YOLO-format label .txt files (the layout you
get from Label Studio, LabelImg, Roboflow exports, etc, an "images" folder
plus a "labels" folder) and splits it into the train/val structure Ultralytics
requires:

    data/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

Usage:
    python scripts/train_val_split.py --datapath "C:\path\to\my_dataset" --train-pct 0.8

For the Coin Detection Dataset specifically, --datapath should point at the
folder that contains its images/ and labels/ subfolders after you unzip it.
"""

import argparse
import os
import random
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(images_dir, labels_dir):
    """Matches each image to its label file by filename stem. Images with no
    matching label are skipped with a warning, since an unlabeled image would
    silently teach the model nothing, better to know about it up front."""
    pairs = []
    skipped = []
    for filename in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(filename)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(label_path):
            pairs.append((os.path.join(images_dir, filename), label_path))
        else:
            skipped.append(filename)
    return pairs, skipped


def copy_split(pairs, dest_root, split_name):
    images_out = os.path.join(dest_root, split_name, "images")
    labels_out = os.path.join(dest_root, split_name, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_out)
        shutil.copy2(label_path, labels_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", required=True,
                     help="Folder containing images/ and labels/ subfolders")
    ap.add_argument("--train-pct", type=float, default=0.8,
                     help="Fraction of data used for training, remainder goes to validation")
    ap.add_argument("--out", default="data",
                     help="Output root folder (default: ./data)")
    ap.add_argument("--seed", type=int, default=42,
                     help="Random seed, fixed by default so the split is reproducible")
    args = ap.parse_args()

    images_dir = os.path.join(args.datapath, "images")
    labels_dir = os.path.join(args.datapath, "labels")
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise SystemExit(
            f"Expected {images_dir} and {labels_dir} to both exist. "
            f"Point --datapath at the folder that directly contains images/ and labels/."
        )

    pairs, skipped = find_pairs(images_dir, labels_dir)
    if not pairs:
        raise SystemExit("No image/label pairs found, nothing to split.")

    random.seed(args.seed)
    random.shuffle(pairs)

    split_index = int(len(pairs) * args.train_pct)
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    copy_split(train_pairs, args.out, "train")
    copy_split(val_pairs, args.out, "val")

    print(f"Split {len(pairs)} labeled images: {len(train_pairs)} train, {len(val_pairs)} val")
    print(f"Written to: {os.path.abspath(args.out)}")
    if skipped:
        print(f"\nSkipped {len(skipped)} image(s) with no matching label file:")
        for name in skipped[:10]:
            print(f"  - {name}")
        if len(skipped) > 10:
            print(f"  ...and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
