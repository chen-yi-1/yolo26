#!/usr/bin/env python3
"""
Auto-annotate seedling images for X-AnyLabeling editing using RGB vegetation indices.

Assumes one seedling per image (one pot, large in frame).
- Bbox: bounding rectangle of ExG vegetation mask
- Health classification: batch-relative ranking of green/red/coverage indices

Input:  raw_datas/       — directory of seedling images
Output: dataset/          — X-AnyLabeling dataset (train/ and val/ image+json pairs)

Usage:
    python scripts/rgb_yolo_annotate.py --input raw_datas --output dataset
    python scripts/rgb_yolo_annotate.py --input raw_datas --output dataset --train-ratio 0.8 --exg-threshold 0.1
    python scripts/rgb_yolo_annotate.py --convert-yolo-dataset --input dataset --output dataset_xanylabeling
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_NAMES = {
    0: "healthy",
    1: "subhealthy",
    2: "unhealthy",
}

# Indices used for green-health scoring (higher = greener/healthier)
GREEN_FEATURES = ("ExG", "GLI", "NGRDI", "VARI")

# ---------------------------------------------------------------------------
# RGB index helpers
# ---------------------------------------------------------------------------


def safe_divide(numerator, denominator):
    """Element-wise division, returning 0 where denominator ≈ 0."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > EPSILON,
    )


def calculate_indices(rgb):
    """Compute 7 RGB vegetation indices for a [0,1] normalized RGB image.

    Args:
        rgb: float32 ndarray of shape (H, W, 3), values in [0, 1].

    Returns:
        dict mapping index name → (H, W) float64 ndarray.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g

    return {
        "ExG": exg,
        "ExR": exr,
        "ExGR": exg - exr,
        "NGRDI": safe_divide(g - r, g + r),
        "GLI": safe_divide(2.0 * g - r - b, 2.0 * g + r + b),
        "VARI": safe_divide(g - r, g + r - b),
        "CIVE": 0.441 * r - 0.811 * g + 0.385 * b,
    }


def create_vegetation_mask(indices, threshold):
    """Binary vegetation mask: True where ExG exceeds threshold."""
    return indices["ExG"] > threshold


def load_rgb_float(image_path):
    """Load image and convert to float32 RGB in [0, 1]."""
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Bbox extraction
# ---------------------------------------------------------------------------


def extract_yolo_bbox(mask):
    """Extract YOLO-format bbox from a binary mask.

    Args:
        mask: (H, W) boolean ndarray.

    Returns:
        (cx, cy, w, h) all normalized to [0, 1], or None if mask is empty.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    h, w = mask.shape
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # YOLO format: center-x, center-y, width, height — all normalized
    # Pixel indices to coordinates: pixel (i,j) center is at (i+0.5)/w, (j+0.5)/h
    cx = ((x1 + x2 + 1) / 2.0) / w
    cy = ((y1 + y2 + 1) / 2.0) / h
    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h

    return (float(cx), float(cy), float(bw), float(bh))


# ---------------------------------------------------------------------------
# Per-image statistics (for classification)
# ---------------------------------------------------------------------------


def compute_image_stats(indices, mask):
    """Compute per-image summary statistics from vegetation indices.

    Returns dict with:
        vegetation_coverage: fraction of pixels in mask
        {name}_veg_mean: mean of index over vegetation pixels
        {name}_veg_std: std of index over vegetation pixels
    """
    stats = {"vegetation_coverage": float(np.mean(mask))}

    for name, arr in indices.items():
        veg_values = arr[mask]
        if veg_values.size > 0:
            stats[f"{name}_veg_mean"] = float(np.mean(veg_values))
            stats[f"{name}_veg_std"] = float(np.std(veg_values))
        else:
            stats[f"{name}_veg_mean"] = 0.0
            stats[f"{name}_veg_std"] = 0.0

    return stats


# ---------------------------------------------------------------------------
# Batch-relative classification
# ---------------------------------------------------------------------------


def percentile_ranks(values):
    """Compute percentile rank [0, 1] for each element in a 1-D array.

    Ties receive the average rank of their group.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    if values.size == 1:
        return np.array([1.0], dtype=np.float64)

    order = np.argsort(values)
    sorted_vals = values[order]
    raw_ranks = np.arange(values.size, dtype=np.float64)
    ranks = np.empty(values.size, dtype=np.float64)

    i = 0
    while i < values.size:
        j = i
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = np.mean(raw_ranks[i:j])
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks / (values.size - 1)


def compute_batch_scores(records):
    """Compute batch-relative scores for a list of per-image stat dicts.

    Each record is updated in-place with:
        green_score: mean percentile rank across ExG/GLI/NGRDI/VARI veg means
        coverage_rank: percentile rank of vegetation_coverage
        red_rank: percentile rank of ExR_veg_mean
    """
    if not records:
        return

    coverage = np.array([r["stats"]["vegetation_coverage"] for r in records])
    exr = np.array([r["stats"]["ExR_veg_mean"] for r in records])

    green_ranks = []
    for feat in GREEN_FEATURES:
        key = f"{feat}_veg_mean"
        vals = np.array([r["stats"][key] for r in records])
        green_ranks.append(percentile_ranks(vals))

    green_scores = np.mean(np.vstack(green_ranks), axis=0)
    coverage_ranks = percentile_ranks(coverage)
    red_ranks = percentile_ranks(exr)

    for i, rec in enumerate(records):
        rec["green_score"] = float(green_scores[i])
        rec["coverage_rank"] = float(coverage_ranks[i])
        rec["red_rank"] = float(red_ranks[i])


def classify_record(rec):
    """Assign health class_id and confidence using batch-relative scores.

    Reads green_score, coverage_rank, red_rank from rec; writes class_id,
    confidence, and reason back.

    Classification rules (one seedling per image, large in frame):
        healthy:     good coverage + strong green + low red
        subhealthy:  intermediate pattern (mild yellowing, pale, slow growth)
        unhealthy:   weak green + elevated red + low coverage
                     (wilted, yellowing, dead, rotten — all merged)
    """
    cov_rank = rec["coverage_rank"]
    green = rec["green_score"]
    red = rec["red_rank"]

    if cov_rank >= 0.35 and green >= 0.55 and red <= 0.75:
        class_id = 0  # healthy
        confidence = (cov_rank + green + (1.0 - red)) / 3.0
        reason = "good coverage, strong green, low red/brown signal"
    elif green <= 0.35 and red >= 0.45:
        class_id = 2  # unhealthy
        confidence = (1.0 - green + red) / 2.0
        reason = "weak green indices with elevated red/brown signal"
    elif cov_rank <= 0.20:
        class_id = 2  # unhealthy
        confidence = 1.0 - cov_rank
        reason = "very low vegetation coverage"
    else:
        class_id = 1  # subhealthy
        confidence = 0.55
        reason = "intermediate RGB pattern; review manually"

    rec["class_id"] = class_id
    rec["confidence"] = round(float(confidence), 4)
    rec["reason"] = reason

    # Override: no vegetation pixels → unhealthy
    if rec["stats"]["vegetation_coverage"] <= 0.0:
        rec["class_id"] = 2
        rec["confidence"] = 1.0
        rec["reason"] = "no vegetation pixels detected"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_images(input_dir, recursive):
    """Yield Path objects for all image files under input_dir."""
    input_dir = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    paths = []
    for p in sorted(input_dir.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# X-AnyLabeling output
# ---------------------------------------------------------------------------


def write_classes_txt(output_dir, class_names):
    """Write class names for reference/editing workflows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [class_names[idx] for idx in sorted(class_names)]
    (output_dir / "classes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def yolo_bbox_to_rectangle_points(bbox, image_width, image_height):
    """Convert normalized YOLO bbox to X-AnyLabeling rectangle corner points."""
    cx, cy, bw, bh = bbox
    x1 = max(0.0, (cx - bw / 2.0) * image_width)
    y1 = max(0.0, (cy - bh / 2.0) * image_height)
    x2 = min(float(image_width), (cx + bw / 2.0) * image_width)
    y2 = min(float(image_height), (cy + bh / 2.0) * image_height)
    return [
        [round(x1, 6), round(y1, 6)],
        [round(x2, 6), round(y1, 6)],
        [round(x2, 6), round(y2, 6)],
        [round(x1, 6), round(y2, 6)],
    ]


def build_xanylabeling_data(image_name, image_width, image_height, shapes):
    """Build the common X-AnyLabeling/Labelme-style JSON envelope."""
    return {
        "version": "4.0.0-beta.7",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": "",
    }


def build_xanylabeling_json(rec, image_name):
    """Build an X-AnyLabeling/Labelme-style JSON annotation dict."""
    shapes = []
    if rec["bbox"] is not None:
        shapes.append(
            {
                "label": CLASS_NAMES[rec["class_id"]],
                "score": rec["confidence"],
                "points": yolo_bbox_to_rectangle_points(
                    rec["bbox"],
                    rec["image_width"],
                    rec["image_height"],
                ),
                "group_id": None,
                "description": None,
                "difficult": False,
                "shape_type": "rectangle",
                "flags": None,
                "attributes": {},
                "kie_linking": [],
            }
        )

    return build_xanylabeling_data(image_name, rec["image_width"], rec["image_height"], shapes)


def write_xanylabeling_json(json_path, rec, image_name):
    """Write an X-AnyLabeling JSON file beside its image."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_xanylabeling_json(rec, image_name)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_yolo_class_names(dataset_dir):
    """Load class names from classes.txt, falling back to built-in names."""
    classes_path = Path(dataset_dir) / "classes.txt"
    if not classes_path.exists():
        return CLASS_NAMES.copy()

    names = {}
    for idx, line in enumerate(classes_path.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            names[idx] = name
    return names or CLASS_NAMES.copy()


def read_yolo_label_file(label_path):
    """Read YOLO txt annotations as (class_id, bbox) pairs."""
    annotations = []
    if not label_path.exists():
        return annotations

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_no}")
        class_id = int(parts[0])
        bbox = tuple(float(v) for v in parts[1:])
        annotations.append((class_id, bbox))
    return annotations


def convert_yolo_dataset_to_xanylabeling(input_dir, output_dir, copy_mode="copy"):
    """Convert an existing YOLO dataset to X-AnyLabeling train/val folders.

    Input layout:
        dataset/images/train/*.jpg
        dataset/labels/train/*.txt
        dataset/images/val/*.jpg
        dataset/labels/val/*.txt

    Output layout:
        dataset_xany/train/*.jpg + *.json
        dataset_xany/val/*.jpg + *.json
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")

    class_names = load_yolo_class_names(input_dir)
    copy_fn = shutil.copy2 if copy_mode == "copy" else lambda src, dst: Path(dst).symlink_to(Path(src).resolve())
    records = []

    for split in ("train", "val"):
        image_dir = input_dir / "images" / split
        label_dir = input_dir / "labels" / split
        if not image_dir.exists():
            continue

        for image_path in discover_images(image_dir, recursive=False):
            dst_image = output_dir / split / image_path.name
            dst_json = output_dir / split / f"{image_path.stem}.json"
            dst_image.parent.mkdir(parents=True, exist_ok=True)

            copy_fn(image_path, dst_image)

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            shapes = []
            for class_id, bbox in read_yolo_label_file(label_dir / f"{image_path.stem}.txt"):
                shapes.append(
                    {
                        "label": class_names.get(class_id, f"class_{class_id}"),
                        "score": None,
                        "points": yolo_bbox_to_rectangle_points(bbox, image_width, image_height),
                        "group_id": None,
                        "description": None,
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": None,
                        "attributes": {},
                        "kie_linking": [],
                    }
                )

            data = build_xanylabeling_data(image_path.name, image_width, image_height, shapes)
            dst_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append({"image_path": image_path, "json_path": dst_json, "split": split, "shape_count": len(shapes)})

    if not records:
        raise ValueError(f"No YOLO images found in {input_dir}")

    write_classes_txt(output_dir, class_names)
    print(f"Done: converted {len(records)} images → {output_dir}")
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def annotate(
    input_dir,
    output_dir,
    train_ratio=0.8,
    exg_threshold=0.1,
    recursive=False,
    seed=42,
    copy_mode="copy",
):
    """Run the full annotation pipeline.

    Args:
        input_dir: directory of seedling images.
        output_dir: directory for X-AnyLabeling image/json pairs.
        train_ratio: fraction of images for training (rest for val).
        exg_threshold: ExG threshold for vegetation mask.
        recursive: whether to search input_dir recursively.
        seed: random seed for reproducible train/val split.
        copy_mode: "copy" to duplicate images, "symlink" for symlinks.

    Returns:
        list of annotation records (dicts).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # ---- 1. Discover images ----
    image_paths = discover_images(input_dir, recursive)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Found {len(image_paths)} images in {input_dir}")

    # ---- 2. Per-image: indices + bbox + stats ----
    records = []
    for img_path in image_paths:
        try:
            rgb = load_rgb_float(img_path)
        except Exception as exc:
            print(f"  Warning: skipped {img_path.name}: {exc}")
            continue

        indices = calculate_indices(rgb)
        mask = create_vegetation_mask(indices, exg_threshold)
        bbox = extract_yolo_bbox(mask)
        stats = compute_image_stats(indices, mask)

        records.append(
            {
                "image_path": img_path,
                "stem": img_path.stem,
                "suffix": img_path.suffix,
                "image_height": rgb.shape[0],
                "image_width": rgb.shape[1],
                "bbox": bbox,
                "stats": stats,
            }
        )

    if not records:
        raise ValueError("No images could be processed")

    # ---- 3. Batch-relative classification ----
    compute_batch_scores(records)
    for rec in records:
        classify_record(rec)

    # ---- 4. Train/val split ----
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_train = int(len(records) * train_ratio)
    if n_train == 0 and train_ratio > 0:
        n_train = 1  # ensure at least 1 training image when split is requested
    train_indices = set(indices[:n_train])

    for i, rec in enumerate(records):
        rec["split"] = "train" if i in train_indices else "val"

    # ---- 5. Write output ----
    # dataset/{train,val}/ with image + JSON pairs, plus dataset/classes.txt

    copy_fn = shutil.copy2 if copy_mode == "copy" else lambda src, dst: Path(dst).symlink_to(Path(src).resolve())

    train_count = 0
    val_count = 0
    no_bbox_count = 0

    for rec in records:
        split = rec["split"]
        img_dst = output_dir / split / f"{rec['stem']}{rec['suffix']}"
        label_dst = output_dir / split / f"{rec['stem']}.json"

        img_dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy image
        try:
            copy_fn(rec["image_path"], img_dst)
        except Exception as exc:
            print(f"  Warning: failed to copy {rec['image_path'].name}: {exc}")
            continue

        # Write label
        write_xanylabeling_json(label_dst, rec, img_dst.name)
        if rec["bbox"] is None:
            no_bbox_count += 1

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    # ---- 6. Generate classes.txt ----
    write_classes_txt(output_dir, CLASS_NAMES)

    # ---- 7. Summary ----
    print(f"\nDone: {train_count} train + {val_count} val images → {output_dir}")
    if no_bbox_count:
        print(f"  {no_bbox_count} images had no vegetation bbox (empty labels)")

    # Class distribution
    class_counts = {}
    for rec in records:
        cid = rec["class_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1

    print("\nClass distribution:")
    for cid in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        print(f"  {cid} ({name}): {class_counts[cid]}")

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-annotate seedling images for X-AnyLabeling editing using RGB vegetation indices."
    )
    parser.add_argument(
        "--input",
        default="raw_datas",
        help="Directory of seedling images (default: raw_datas).",
    )
    parser.add_argument(
        "--output",
        default="dataset",
        help="Output X-AnyLabeling dataset directory (default: dataset).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images for training split (default: 0.8).",
    )
    parser.add_argument(
        "--exg-threshold",
        type=float,
        default=0.1,
        help="ExG threshold for vegetation mask (default: 0.1).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to place images in output: copy (default) or symlink.",
    )
    parser.add_argument(
        "--convert-yolo-dataset",
        action="store_true",
        help="Convert an existing YOLO dataset into X-AnyLabeling train/val image+json folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.convert_yolo_dataset:
        convert_yolo_dataset_to_xanylabeling(
            input_dir=args.input,
            output_dir=args.output,
            copy_mode=args.copy_mode,
        )
        return

    annotate(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        exg_threshold=args.exg_threshold,
        recursive=args.recursive,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
