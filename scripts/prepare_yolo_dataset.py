#!/usr/bin/env python3
"""Prepare an Ultralytics YOLO dataset from `dataset/labels/*.txt` sources.

Input layout:
    dataset/labels/<image_name>.txt
    dataset/<class-or-folder>/<image_name>.jpg
    dataset/<class-or-folder>/<image_name>.json

This script uses the `labels/` files as the authoritative annotation source,
finds the matching image by filename, optionally samples a percentage of the
full set, splits into train/val, and writes:
    datasets/images/train|val/<image_name>.<ext>
    datasets/labels/train|val/<image_name>.txt
    datasets/datasets.yaml

The json files are left in the source tree untouched for visualization/editing.
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_image_files(source_dir):
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return []
    return [
        path for path in sorted(source_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "labels" not in path.relative_to(source_dir).parts
    ]


def discover_label_files(labels_dir):
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return []
    return [
        path for path in sorted(labels_dir.glob("*.txt"))
        if path.is_file()
    ]


def read_classes(classes_path):
    classes_path = Path(classes_path)
    if classes_path.exists():
        names = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not names:
            raise ValueError(f"No class names found in {classes_path}")
        return names

    return None


def validate_class_id(raw_class_id, class_count, label_path, line_number):
    try:
        class_id = int(raw_class_id)
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number} has invalid class id: {raw_class_id}") from exc
    if class_id < 0 or class_id >= class_count:
        raise ValueError(
            f"{label_path}:{line_number} class id {class_id} is outside 0..{class_count - 1}"
        )


def validate_normalized_coords(values, label_path, line_number):
    for value in values:
        try:
            coord = float(value)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} has invalid coordinate: {value}") from exc
        if coord < 0.0 or coord > 1.0:
            raise ValueError(
                f"{label_path}:{line_number} coordinate {coord} is outside normalized range [0, 1]"
            )


def validate_segmentation_label(label_path, class_count):
    """Validate YOLO segmentation rows: class_id x1 y1 x2 y2 x3 y3 ..."""
    label_path = Path(label_path)
    if not label_path.exists():
        return

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ValueError(
                f"{label_path}:{line_number} is not YOLO segmentation format "
                f"(expected class_id plus at least 3 x/y pairs)"
            )
        validate_class_id(parts[0], class_count, label_path, line_number)
        validate_normalized_coords(parts[1:], label_path, line_number)


def validate_detection_label(label_path, class_count):
    """Validate YOLO detection rows: class_id x_center y_center width height."""
    label_path = Path(label_path)
    if not label_path.exists():
        return

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number} is not YOLO detection format "
                f"(expected class_id x_center y_center width height)"
            )
        validate_class_id(parts[0], class_count, label_path, line_number)
        validate_normalized_coords(parts[1:], label_path, line_number)


def validate_yolo_label(label_path, class_count, task):
    if task == "segment":
        validate_segmentation_label(label_path, class_count)
    elif task == "detect":
        validate_detection_label(label_path, class_count)
    else:
        raise ValueError(f"Unsupported task: {task}")


def clamp01(value):
    return min(1.0, max(0.0, value))


def reset_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def sample_items(items, sample_percent, split, seed):
    if sample_percent is None or sample_percent <= 0 or sample_percent >= 100:
        return items
    sample_size = max(1, int(len(items) * sample_percent / 100))
    sampled = random.Random(seed).sample(items, sample_size)
    print(f"  Sampling {sample_percent}% ({sample_size}/{len(items)}) from {split}")
    return sampled


def sample_items_fixed_count(label_files, sample_count, background_count, split, seed):
    """Sample fixed count per class - useful for imbalanced datasets."""
    class_groups = {}
    background_labels = []
    for label_file in label_files:
        class_id = get_label_class_id(label_file)
        if class_id is None:
            background_labels.append(label_file)
        else:
            class_groups.setdefault(class_id, []).append(label_file)

    # Sample fixed count from each class
    if background_count is not None and background_count >= 0 and len(background_labels) > background_count:
        background_labels = random.Random(seed).sample(background_labels, background_count)
    sampled = list(background_labels)
    if background_labels:
        print(f"  Keeping background labels: {len(background_labels)}")

    if sample_count is None or sample_count <= 0:
        for items in class_groups.values():
            sampled.extend(items)
        return sampled

    # Sample fixed count from each class.
    for class_id, items in class_groups.items():
        actual_count = min(sample_count, len(items))
        class_sampled = random.Random(seed + class_id).sample(items, actual_count)
        sampled.extend(class_sampled)
        print(f"  Sampling class {class_id}: {actual_count}/{len(items)} (requested: {sample_count})")

    print(f"  Total sampled from {split}: {len(sampled)}/{len(label_files)}")
    return sampled


def split_items(items, train_ratio, seed):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    if n_train == 0 and train_ratio > 0 and shuffled:
        n_train = 1
    return shuffled[:n_train], shuffled[n_train:]


def build_image_index(image_files):
    """Build index mapping stem to list of image paths (handles duplicates across classes)."""
    index = {}
    for image_path in image_files:
        stem = image_path.stem
        if stem not in index:
            index[stem] = []
        index[stem].append(image_path)
    return index


def get_label_class_id(label_path):
    """Extract the class ID from the first line of a YOLO label file."""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        return int(parts[0])
    except Exception:
        pass
    return None


def copy_labelled_items(source_dir, labels_dir, output_dir, split, label_files, image_index, names, task, seed):
    copied = 0
    missing_images = []

    for label_src in label_files:
        validate_yolo_label(label_src, len(names), task)
        
        # Get class ID from label file
        class_id = get_label_class_id(label_src)
        
        # Find matching image
        image_candidates = image_index.get(label_src.stem, [])
        image_src = None
        
        if len(image_candidates) == 1:
            # Only one candidate, use it
            image_src = image_candidates[0]
        elif len(image_candidates) > 1:
            # Multiple candidates, choose by class
            if class_id is not None and class_id < len(names):
                class_name = names[class_id]
                for candidate in image_candidates:
                    if class_name in str(candidate.parent):
                        image_src = candidate
                        break
        
        if image_src is None:
            missing_images.append(label_src.stem)
            continue

        image_dst = Path(output_dir) / "images" / split / image_src.name
        label_dst = Path(output_dir) / "labels" / split / label_src.name
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)
        copied += 1

    return copied, missing_images


def write_dataset_yaml(yaml_path, output_dir, names):
    data = {
        "path": str(Path(output_dir).resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {idx: name for idx, name in enumerate(names)},
    }
    Path(yaml_path).write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def prepare_yolo_dataset(
    source_dir,
    output_dir,
    yaml_path=None,
    sample_percent=None,
    sample_count=None,
    background_count=None,
    task="segment",
    train_ratio=0.8,
    seed=42,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if yaml_path is None:
        yaml_path = output_dir / "datasets.yaml"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if task not in {"segment", "detect"}:
        raise ValueError(f"task must be 'segment' or 'detect', got: {task}")
    if train_ratio < 0.0 or train_ratio > 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got: {train_ratio}")

    labels_dir = source_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    names = read_classes(source_dir / "classes.txt")
    if names is None:
        class_dirs = [p for p in sorted(source_dir.iterdir()) if p.is_dir() and p.name != "labels"]
        if not class_dirs:
            raise FileNotFoundError(f"Missing classes file and class folders: {source_dir}")
        names = [path.name for path in class_dirs]

    label_files = discover_label_files(labels_dir)
    if not label_files:
        raise ValueError(f"No label files found in {labels_dir}")

    image_index = build_image_index(discover_image_files(source_dir))
    reset_output_dir(output_dir)

    sampled_labels = sample_items(label_files, sample_percent, "all", seed)
    sampled_labels = sample_items_fixed_count(sampled_labels, sample_count, background_count, "all", seed)
    train_labels, val_labels = split_items(sampled_labels, train_ratio, seed)
    train_count, train_missing = copy_labelled_items(
        source_dir, labels_dir, output_dir, "train", train_labels, image_index, names, task, seed
    )
    val_count, val_missing = copy_labelled_items(
        source_dir, labels_dir, output_dir, "val", val_labels, image_index, names, task, seed
    )
    missing = train_missing + val_missing
    write_dataset_yaml(yaml_path, output_dir, names)

    print(f"Prepared YOLO {task} dataset: {output_dir}")
    print(f"  train: {train_count} images (with labels)")
    print(f"  val:   {val_count} images (with labels)")
    print(f"  yaml:  {yaml_path}")
    if missing:
        print(f"  warning: {len(missing)} items have no matching image/json, skipped:")
        for stem in missing[:10]:
            print(f"    - {stem}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    return {
        "train": train_count,
        "val": val_count,
        "missing_images": missing,
        "classes": names,
        "task": task,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert edited X-AnyLabeling/YOLO labels into an Ultralytics YOLO layout."
    )
    parser.add_argument("--source", default="dataset", help="Edited dataset directory (default: dataset).")
    parser.add_argument("--output", default="datasets", help="Output YOLO dataset directory (default: datasets).")
    parser.add_argument(
        "--yaml",
        default=None,
        help="Output dataset YAML path (default: <output>/datasets.yaml).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Fixed number of samples per class. Useful for imbalanced datasets. E.g., --sample-count 500 means 500 samples from each class.",
    )
    parser.add_argument(
        "--background-count",
        type=int,
        default=0,
        help="Maximum number of empty-label background samples to keep. Default keeps all background samples.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for mirrored X-AnyLabeling sources without train/val folders (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mirrored-source train/val split (default: 42).",
    )
    parser.add_argument(
        "--task",
        choices=["segment", "detect"],
        default="detect",
        help="YOLO task label format to prepare: segment polygons (default) or detect boxes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_yolo_dataset(
        args.source,
        args.output,
        args.yaml,
        sample_count=args.sample_count,
        background_count=args.background_count,
        task=args.task,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
