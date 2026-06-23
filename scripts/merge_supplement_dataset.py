#!/usr/bin/env python3
"""Merge supplementary data into an existing Ultralytics YOLO dataset.

Each supplement folder must use the flat layout:

    supplement/
        images/   *.jpg
        labels/   *.txt

The script auto-discovers all such folders under --supplements (default: datas),
splits each by --train-ratio (default: 0.8), and copies into the target dataset.
Existing files with the same names are skipped (no overwrite).

Usage:
    python scripts/merge_supplement_dataset.py
    python scripts/merge_supplement_dataset.py --supplements datas --target datasets
    python scripts/merge_supplement_dataset.py --train-ratio 0.7 --seed 123
"""

import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _image_label_pairs(images_dir, labels_dir):
    """Yield (stem, image_path, label_path) for matched pairs."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_stems = {}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_stems[p.stem] = p

    for stem, img_path in image_stems.items():
        lbl_path = labels_dir / f"{stem}.txt"
        if lbl_path.exists():
            yield stem, img_path, lbl_path


def _copy_pair(img_src, lbl_src, dst_img_dir, dst_lbl_dir):
    """Copy one image+label pair into target dirs. Skip if either already exists."""
    dst_img = Path(dst_img_dir) / img_src.name
    dst_lbl = Path(dst_lbl_dir) / lbl_src.name
    if dst_img.exists() or dst_lbl.exists():
        return False
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, dst_img)
    shutil.copy2(lbl_src, dst_lbl)
    return True


def _discover_supplements(parent_dir):
    """Find all supplement subdirs (those with images/ + labels/) under parent."""
    parent = Path(parent_dir)
    if not parent.is_dir():
        return []
    return sorted(
        p for p in parent.iterdir()
        if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()
    )


def merge_supplement_dataset(supplements, target, train_ratio=0.8, seed=42):
    """Merge supplement sources into the target dataset, splitting each 8:2."""
    target = Path(target)
    for required in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (target / required).is_dir():
            raise FileNotFoundError(
                f"Target missing: {target / required}. Run prepare_yolo_dataset.py first."
            )

    if not (0 < train_ratio <= 1):
        raise ValueError(f"train_ratio must be in (0, 1], got: {train_ratio}")

    def _count(split):
        p = target / "images" / split
        return len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]) if p.exists() else 0

    train_before = _count("train")
    val_before = _count("val")
    print(f"Before merge:  train={train_before}  val={val_before}\n")

    total_train = total_val = total_skipped = 0

    for i, source in enumerate(supplements):
        print(f"[{i + 1}/{len(supplements)}] {source}")
        pairs = list(_image_label_pairs(Path(source) / "images", Path(source) / "labels"))
        if not pairs:
            print(f"  Warning: no matched image/label pairs")
            continue

        shuffled = list(pairs)
        random.Random(seed + i).shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_ratio)) if shuffled else 0
        train_pairs = shuffled[:n_train]
        val_pairs = shuffled[n_train:]

        skipped = 0
        for _, img, lbl in train_pairs:
            if not _copy_pair(img, lbl, target / "images" / "train", target / "labels" / "train"):
                skipped += 1
        for _, img, lbl in val_pairs:
            if not _copy_pair(img, lbl, target / "images" / "val", target / "labels" / "val"):
                skipped += 1

        total_train += len(train_pairs)
        total_val += len(val_pairs)
        total_skipped += skipped
        info = f"  train={len(train_pairs)} val={len(val_pairs)}"
        if skipped:
            info += f" (skipped {skipped})"
        print(info)

    print()
    train_after = _count("train")
    val_after = _count("val")
    print(f"After merge:   train={train_after}  val={val_after}")
    total = train_after + val_after
    if total:
        print(f"Train ratio:   {train_after}/{total} = {train_after / total:.1%}")
    print(f"Added:         train={total_train}  val={total_val}")
    if total_skipped:
        print(f"Skipped (duplicates): {total_skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge supplement data into an existing YOLO dataset layout."
    )
    parser.add_argument(
        "--supplements", default="datas",
        help="Parent directory with supplement subdirs (default: datas).",
    )
    parser.add_argument(
        "--target", default="datasets",
        help="Target YOLO dataset directory (default: datasets).",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    supplements = _discover_supplements(args.supplements)
    if not supplements:
        print(f"No supplement subdirectories found under {args.supplements}")
        return
    print(f"Found {len(supplements)} supplement(s) under {args.supplements}:")
    for d in supplements:
        print(f"  {d}")
    print()

    merge_supplement_dataset(
        supplements, args.target,
        train_ratio=args.train_ratio, seed=args.seed,
    )


if __name__ == "__main__":
    main()
