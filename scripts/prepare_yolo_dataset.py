#!/usr/bin/env python3
"""Prepare an Ultralytics YOLO segmentation dataset from edited labels.

Input layout:
    dataset/
      classes.txt
      labels/<stem>.txt
      train/<image files and optional json files>
      val/<image files and optional json files>

Output layout:
    datasets/
      images/train
      images/val
      labels/train
      labels/val
    datasets/datasets.yaml
"""

import argparse
import shutil
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return [
        path for path in sorted(split_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_classes(classes_path):
    classes_path = Path(classes_path)
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")

    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"No class names found in {classes_path}")
    return names


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
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} has invalid class id: {parts[0]}") from exc
        if class_id < 0 or class_id >= class_count:
            raise ValueError(
                f"{label_path}:{line_number} class id {class_id} is outside 0..{class_count - 1}"
            )
        for value in parts[1:]:
            try:
                coord = float(value)
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_number} has invalid coordinate: {value}") from exc
            if coord < 0.0 or coord > 1.0:
                raise ValueError(
                    f"{label_path}:{line_number} coordinate {coord} is outside normalized range [0, 1]"
                )


def reset_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_split(source_dir, labels_dir, output_dir, split, class_count):
    images = discover_images(Path(source_dir) / split)
    copied = 0
    missing_labels = []

    for image_path in images:
        label_src = Path(labels_dir) / f"{image_path.stem}.txt"
        validate_segmentation_label(label_src, class_count)

        image_dst = Path(output_dir) / "images" / split / image_path.name
        label_dst = Path(output_dir) / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_dst)

        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            label_dst.write_text("", encoding="utf-8")
            missing_labels.append(image_path.name)

        copied += 1

    return copied, missing_labels


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


def prepare_yolo_dataset(source_dir="dataset", output_dir="datasets", yaml_path=None):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if yaml_path is None:
        yaml_path = output_dir / "datasets.yaml"
    labels_dir = source_dir / "labels"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Edited labels directory does not exist: {labels_dir}")

    names = read_classes(source_dir / "classes.txt")
    reset_output_dir(output_dir)

    train_count, train_missing = copy_split(source_dir, labels_dir, output_dir, "train", len(names))
    val_count, val_missing = copy_split(source_dir, labels_dir, output_dir, "val", len(names))
    write_dataset_yaml(yaml_path, output_dir, names)

    print(f"Prepared YOLO segmentation dataset: {output_dir}")
    print(f"  train: {train_count} images")
    print(f"  val:   {val_count} images")
    print(f"  yaml:  {yaml_path}")
    missing = train_missing + val_missing
    if missing:
        print(f"  warning: created {len(missing)} empty label files for images without edited labels")

    return {
        "train": train_count,
        "val": val_count,
        "missing_labels": missing,
        "classes": names,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert edited dataset/train,val + dataset/labels into Ultralytics YOLO segmentation layout."
    )
    parser.add_argument("--source", default="dataset", help="Edited dataset directory (default: dataset).")
    parser.add_argument("--output", default="datasets", help="Output YOLO dataset directory (default: datasets).")
    parser.add_argument(
        "--yaml",
        default=None,
        help="Output dataset YAML path (default: <output>/datasets.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_yolo_dataset(args.source, args.output, args.yaml)


if __name__ == "__main__":
    main()
