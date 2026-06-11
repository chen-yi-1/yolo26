#!/usr/bin/env python3
"""Prepare an Ultralytics YOLO dataset from edited labels.

Supported sources:
- dataset/labels/<stem>.txt with existing YOLO labels.
- dataset/{train,val}/<stem>.json from X-AnyLabeling/Labelme.

For task="segment", polygon shapes become YOLO segmentation rows:
    class_id x1 y1 x2 y2 x3 y3 ...

For task="detect", rectangle shapes become YOLO detection rows:
    class_id x_center y_center width height
"""

import argparse
import json
import random
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


def class_id_from_shape(shape, names, json_path):
    label = shape.get("label")
    if label not in names:
        raise ValueError(f"{json_path}: unknown label {label!r}; expected one of {names}")
    return names.index(label)


def shape_to_yolo_row(shape, names, image_width, image_height, task, json_path):
    shape_type = shape.get("shape_type")
    points = shape.get("points") or []
    class_id = class_id_from_shape(shape, names, json_path)

    if task == "segment":
        if shape_type != "polygon" or len(points) < 3:
            return None
        coords = []
        for x, y in points:
            coords.extend([clamp01(float(x) / image_width), clamp01(float(y) / image_height)])
        return " ".join([str(class_id)] + [f"{coord:.6f}" for coord in coords])

    if task == "detect":
        if shape_type != "rectangle" or len(points) < 2:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        coords = [
            ((x_min + x_max) / 2.0) / image_width,
            ((y_min + y_max) / 2.0) / image_height,
            (x_max - x_min) / image_width,
            (y_max - y_min) / image_height,
        ]
        return " ".join([str(class_id)] + [f"{clamp01(coord):.6f}" for coord in coords])

    raise ValueError(f"Unsupported task: {task}")


def write_label_from_json(json_path, label_dst, names, task):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")
    if not image_width or not image_height:
        raise ValueError(f"{json_path}: missing imageWidth/imageHeight")

    rows = []
    for shape in data.get("shapes", []):
        row = shape_to_yolo_row(shape, names, image_width, image_height, task, json_path)
        if row is not None:
            rows.append(row)

    label_dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def reset_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_image_for_label(images_dir, image_stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def sample_items(items, sample_percent, split):
    if sample_percent is None or sample_percent <= 0 or sample_percent >= 100:
        return items
    sample_size = max(1, int(len(items) * sample_percent / 100))
    sampled = random.sample(items, sample_size)
    print(f"  Sampling {sample_percent}% ({sample_size}/{len(items)}) from {split}")
    return sampled


def copy_split_from_txt(source_dir, labels_dir, output_dir, split, names, task, sample_percent=None):
    images_dir = Path(source_dir) / split
    label_files = sample_items(sorted(Path(labels_dir).glob("*.txt")), sample_percent, split)
    copied = 0
    missing_images = []

    for label_src in label_files:
        image_path = find_image_for_label(images_dir, label_src.stem)
        validate_yolo_label(label_src, len(names), task)

        if image_path is None:
            missing_images.append(label_src.stem)
            continue

        image_dst = Path(output_dir) / "images" / split / image_path.name
        label_dst = Path(output_dir) / "labels" / split / label_src.name
        shutil.copy2(image_path, image_dst)
        shutil.copy2(label_src, label_dst)
        copied += 1

    return copied, missing_images


def copy_split_from_json(source_dir, output_dir, split, names, task, sample_percent=None):
    image_paths = sample_items(discover_images(Path(source_dir) / split), sample_percent, split)
    copied = 0
    missing_labels = []

    for image_path in image_paths:
        json_src = image_path.with_suffix(".json")
        if not json_src.exists():
            missing_labels.append(image_path.stem)
            continue

        image_dst = Path(output_dir) / "images" / split / image_path.name
        label_dst = Path(output_dir) / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_dst)
        write_label_from_json(json_src, label_dst, names, task)
        validate_yolo_label(label_dst, len(names), task)
        copied += 1

    return copied, missing_labels


def copy_split(source_dir, labels_dir, output_dir, split, names, task, sample_percent=None):
    labels_dir = Path(labels_dir)
    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
        return copy_split_from_txt(source_dir, labels_dir, output_dir, split, names, task, sample_percent)
    return copy_split_from_json(source_dir, output_dir, split, names, task, sample_percent)


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


def prepare_yolo_dataset(source_dir, output_dir, yaml_path=None, sample_percent=None, task="segment"):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if yaml_path is None:
        yaml_path = output_dir / "datasets.yaml"
    labels_dir = source_dir / "labels"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if task not in {"segment", "detect"}:
        raise ValueError(f"task must be 'segment' or 'detect', got: {task}")

    names = read_classes(source_dir / "classes.txt")
    reset_output_dir(output_dir)

    train_count, train_missing = copy_split(source_dir, labels_dir, output_dir, "train", names, task, sample_percent)
    val_count, val_missing = copy_split(source_dir, labels_dir, output_dir, "val", names, task, sample_percent)
    write_dataset_yaml(yaml_path, output_dir, names)

    print(f"Prepared YOLO {task} dataset: {output_dir}")
    print(f"  train: {train_count} images (with labels)")
    print(f"  val:   {val_count} images (with labels)")
    print(f"  yaml:  {yaml_path}")
    missing = train_missing + val_missing
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
        description="Convert edited dataset/train,val into an Ultralytics YOLO layout."
    )
    parser.add_argument("--source", default="dataset", help="Edited dataset directory (default: dataset).")
    parser.add_argument("--output", default="datasets", help="Output YOLO dataset directory (default: datasets).")
    parser.add_argument(
        "--yaml",
        default=None,
        help="Output dataset YAML path (default: <output>/datasets.yaml).",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Random sample percentage (0-100) from train and val sets. E.g., --sample 10 for 10%%.",
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
    prepare_yolo_dataset(args.source, args.output, args.yaml, args.sample, task=args.task)


if __name__ == "__main__":
    main()
