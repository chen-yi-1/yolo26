#!/usr/bin/env python3
"""
Auto-annotate seedling images for X-AnyLabeling editing using RGB vegetation indices.

Assumes one seedling per image (one pot, large in frame).
- Polygon: external contour of ExG vegetation mask
- Rectangle: bounding box around each extracted vegetation instance
- Label: every generated shape is marked as healthy for manual review/editing

Input:  raw_data/        directory of seedling images
Output: dataset/         X-AnyLabeling dataset (train/ and val/ image+json pairs)

Usage:
    python scripts/rgb_yolo_annotate.py --input raw_data --output dataset
"""

import argparse
import concurrent.futures
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_NAMES = {
    0: "healthy",
    1: "abnormal",
}

# Pixel-mask cleanup defaults for auto-generated segmentation labels.
DEFAULT_MIN_COMPONENT_EXG_MEAN = 0.12
DEFAULT_MAX_INSTANCES = 12
DEFAULT_MAX_POLYGON_POINTS = 96

# ---------------------------------------------------------------------------
# RGB index helpers
# ---------------------------------------------------------------------------


def safe_divide(numerator, denominator):
    """Element-wise division, returning 0 where denominator 鈮?0."""
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
        dict mapping index name 鈫?(H, W) float64 ndarray.
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


def create_vegetation_mask(
    indices,
    threshold,
    rgb=None,
    min_saturation=0.08,
    min_value=0.04,
    max_value=1.0,
    hue_ranges=((18, 95), (105, 155)),
):
    """Create a vegetation mask from ExG plus optional color sanity filters.

    ExG alone is too permissive for pale trays, pink side walls, and bright substrate.
    When RGB is provided, HSV saturation/value and plant-like hue ranges suppress
    those non-vegetation regions while keeping green, yellow-green, and purple leaves.
    """
    mask = indices["ExG"] > threshold
    if rgb is None:
        return mask

    try:
        import cv2
    except ImportError:
        return mask

    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0

    hue_mask = np.zeros(mask.shape, dtype=bool)
    for lower, upper in hue_ranges:
        hue_mask |= (hue >= lower) & (hue <= upper)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    green_or_yellow = (g >= 0.72 * r) & (g >= 0.72 * b)
    purple_leaf = (hue >= 105) & (hue <= 155) & (saturation >= min_saturation)

    color_mask = (
        hue_mask
        & (saturation >= min_saturation)
        & (value >= min_value)
        & (value <= max_value)
        & (green_or_yellow | purple_leaf)
    )
    return mask & color_mask


def load_rgb_float(image_path):
    """Load image and convert to float32 RGB in [0, 1]."""
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Mask geometry extraction
# ---------------------------------------------------------------------------


def convex_hull(points):
    """Compute a 2D convex hull for fallback polygon extraction."""
    points = sorted(set((int(x), int(y)) for x, y in points))
    if len(points) <= 1:
        return points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def extract_boundary_points(mask):
    """Return [x, y] points on the 4-connected boundary of a binary mask."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = center & ~interior
    ys, xs = np.where(boundary)
    return list(zip(xs, ys))


def extract_mask_polygon_fallback(mask):
    """Extract a polygon without OpenCV by taking the convex hull of mask boundary pixels."""
    boundary_points = extract_boundary_points(mask)
    if len(boundary_points) < 3:
        return None

    hull = convex_hull(boundary_points)
    if len(hull) < 3:
        return None

    return [[float(x), float(y)] for x, y in hull]


def odd_kernel_size(image_shape, ratio, minimum=3):
    """Scale a morphology kernel to image size and force it to be odd."""
    if ratio <= 0:
        return 0
    size = int(round(min(image_shape[:2]) * ratio))
    size = max(minimum, size)
    if size % 2 == 0:
        size += 1
    return size


def refine_mask(
    mask,
    close_kernel_ratio=0.015,
    open_kernel_ratio=0.003,
    min_area_ratio=0.0005,
    keep_largest=False,
):
    """Clean a vegetation mask before contour extraction.

    The order is chosen for annotation quality:
    close gaps -> remove speckles -> fill holes -> remove tiny components.
    """
    try:
        import cv2
    except ImportError:
        return mask

    h, w = mask.shape
    mask_uint8 = mask.astype(np.uint8) * 255

    close_size = odd_kernel_size(mask.shape, close_kernel_ratio)
    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    open_size = odd_kernel_size(mask.shape, open_kernel_ratio)
    if open_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    min_area = max(1, int(h * w * min_area_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if num_labels <= 1:
        return filled > 0

    component_ids = [
        idx for idx in range(1, num_labels)
        if stats[idx, cv2.CC_STAT_AREA] >= min_area
    ]
    if not component_ids:
        component_ids = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    if keep_largest:
        component_ids = [max(component_ids, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])]

    refined = np.isin(labels, component_ids)
    return refined


def simplify_contour_adaptive(contour, epsilon_ratio=0.003, min_points=12, max_points=300):
    """Simplify a contour while keeping a useful number of polygon vertices."""
    try:
        import cv2
    except ImportError:
        return contour

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return contour

    best = contour
    ratio = max(0.0005, epsilon_ratio)
    for _ in range(10):
        epsilon = max(0.5, ratio * perimeter)
        candidate = cv2.approxPolyDP(contour, epsilon, True)
        if min_points <= len(candidate) <= max_points:
            return candidate
        best = candidate
        if len(candidate) < min_points:
            ratio *= 0.5
        else:
            ratio *= 1.5

    if len(best) < 3:
        best = cv2.convexHull(contour)
    return best


def contour_to_polygon_points(contour, image_shape):
    h, w = image_shape[:2]
    points = []
    previous = None
    for point in contour.reshape(-1, 2):
        x = min(w - 1, max(0, int(point[0])))
        y = min(h - 1, max(0, int(point[1])))
        current = [float(x), float(y)]
        if current != previous:
            points.append(current)
            previous = current
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points if len(points) >= 3 else None


def mask_to_rectangle_points(mask):
    """Return X-AnyLabeling rectangle points [[x_min, y_min], [x_max, y_max]]."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [
        [float(xs.min()), float(ys.min())],
        [float(xs.max()), float(ys.max())],
    ]


def extract_mask_polygons(
    mask,
    epsilon_ratio=0.003,
    min_area_ratio=0.0005,
    close_kernel_ratio=0.015,
    open_kernel_ratio=0.003,
    min_points=12,
    max_points=DEFAULT_MAX_POLYGON_POINTS,
    max_instances=DEFAULT_MAX_INSTANCES,
):
    """Extract cleaned external mask contours as X-AnyLabeling polygon points.

    Args:
        mask: (H, W) boolean ndarray.
        epsilon_ratio: Douglas-Peucker simplification strength relative to contour perimeter.
        min_area_ratio: discard connected components smaller than this image-area ratio.
        close_kernel_ratio: morphology close kernel relative to image size.
        open_kernel_ratio: morphology open kernel relative to image size.
        min_points: target lower bound for polygon vertices.
        max_points: target upper bound for polygon vertices.
        max_instances: max polygons to return by area; 0 means no limit.

    Returns:
        List of polygons, where each polygon is a list of [x, y] pixel points.
    """
    if not np.any(mask):
        return []

    try:
        import cv2
    except ImportError:
        polygon = extract_mask_polygon_fallback(mask)
        return [polygon] if polygon is not None else []

    refined = refine_mask(
        mask,
        close_kernel_ratio=close_kernel_ratio,
        open_kernel_ratio=open_kernel_ratio,
        min_area_ratio=min_area_ratio,
        keep_largest=False,
    )
    mask_uint8 = refined.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if max_instances and max_instances > 0:
        contours = contours[:max_instances]

    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue

        polygon = simplify_contour_adaptive(
            contour,
            epsilon_ratio=epsilon_ratio,
            min_points=min_points,
            max_points=max_points,
        )
        if len(polygon) < 3:
            polygon = cv2.convexHull(contour)
        if len(polygon) < 3:
            continue

        points = contour_to_polygon_points(polygon, mask.shape)
        if points is not None:
            polygons.append(points)

    return polygons


def extract_mask_polygon(mask, **kwargs):
    """Backward-compatible helper returning the largest polygon only."""
    polygons = extract_mask_polygons(mask, max_instances=1, **kwargs)
    return polygons[0] if polygons else None


def extract_mask_instances(
    mask,
    score_map=None,
    epsilon_ratio=0.003,
    min_area_ratio=0.0005,
    min_component_score=DEFAULT_MIN_COMPONENT_EXG_MEAN,
    close_kernel_ratio=0.015,
    open_kernel_ratio=0.003,
    min_points=12,
    max_points=DEFAULT_MAX_POLYGON_POINTS,
    max_instances=DEFAULT_MAX_INSTANCES,
):
    """Extract cleaned mask instances with a polygon and per-instance mask."""
    if not np.any(mask):
        return []

    try:
        import cv2
    except ImportError:
        polygon = extract_mask_polygon_fallback(mask)
        rectangle = mask_to_rectangle_points(mask)
        return [
            {
                "polygon": polygon,
                "rectangle": rectangle,
                "mask": mask,
                "area": int(np.sum(mask)),
            }
        ] if polygon is not None and rectangle is not None else []

    refined = refine_mask(
        mask,
        close_kernel_ratio=close_kernel_ratio,
        open_kernel_ratio=open_kernel_ratio,
        min_area_ratio=min_area_ratio,
        keep_largest=False,
    )
    mask_uint8 = refined.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if max_instances and max_instances > 0:
        contours = contours[:max_instances]

    instances = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 1, thickness=cv2.FILLED)
        component_mask = component_mask.astype(bool)
        if score_map is not None and min_component_score is not None:
            if float(np.mean(score_map[component_mask])) < min_component_score:
                continue

        polygon = simplify_contour_adaptive(
            contour,
            epsilon_ratio=epsilon_ratio,
            min_points=min_points,
            max_points=max_points,
        )
        if len(polygon) < 3:
            polygon = cv2.convexHull(contour)
        points = contour_to_polygon_points(polygon, mask.shape)
        if points is None:
            continue

        instances.append(
            {
                "polygon": points,
                "rectangle": mask_to_rectangle_points(component_mask),
                "mask": component_mask,
                "area": int(np.sum(component_mask)),
            }
        )

    return instances


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
    for instance in rec["instances"]:
        for shape_type, points in (
            ("polygon", instance["polygon"]),
            ("rectangle", instance["rectangle"]),
        ):
            shapes.append(
                {
                    "label": CLASS_NAMES[instance["class_id"]],
                    "score": instance["confidence"],
                    "points": points,
                    "group_id": None,
                    "description": instance.get("description") or instance.get("reason"),
                    "difficult": False,
                    "shape_type": shape_type,
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _analyze_image(task):
    """Analyze one image and return an annotation record, or a skip warning."""
    img_path, params = task
    try:
        rgb = load_rgb_float(img_path)
    except Exception as exc:
        return None, f"  Warning: skipped {img_path.name}: {exc}"

    indices = calculate_indices(rgb)
    mask = create_vegetation_mask(
        indices,
        params["exg_threshold"],
        rgb=rgb,
        min_saturation=params["min_saturation"],
        min_value=params["min_value"],
        max_value=params["max_value"],
    )
    extracted_instances = extract_mask_instances(
        mask,
        score_map=indices["ExG"],
        epsilon_ratio=params["polygon_epsilon"],
        min_area_ratio=params["min_area_ratio"],
        min_component_score=params["min_component_exg_mean"],
        close_kernel_ratio=params["close_kernel_ratio"],
        open_kernel_ratio=params["open_kernel_ratio"],
        min_points=params["min_polygon_points"],
        max_points=params["max_polygon_points"],
        max_instances=params["max_instances"],
    )
    instances = []
    for instance_index, extracted in enumerate(extracted_instances):
        instances.append(
            {
                "image_path": img_path,
                "image_stem": img_path.stem,
                "instance_index": instance_index,
                "polygon": extracted["polygon"],
                "rectangle": extracted["rectangle"],
                "area": extracted["area"],
                "class_id": 0,
                "confidence": 1.0,
                "description": "auto-labeled healthy; review manually",
            }
        )

    return (
        {
            "image_path": img_path,
            "stem": img_path.stem,
            "suffix": img_path.suffix,
            "image_height": rgb.shape[0],
            "image_width": rgb.shape[1],
            "instances": instances,
        },
        None,
    )


def annotate(
    input_dir,
    output_dir,
    train_ratio=0.8,
    exg_threshold=0.1,
    min_saturation=0.08,
    min_value=0.04,
    max_value=1.0,
    min_component_exg_mean=DEFAULT_MIN_COMPONENT_EXG_MEAN,
    polygon_epsilon=0.003,
    min_area_ratio=0.0005,
    close_kernel_ratio=0.015,
    open_kernel_ratio=0.003,
    min_polygon_points=12,
    max_polygon_points=DEFAULT_MAX_POLYGON_POINTS,
    max_instances=DEFAULT_MAX_INSTANCES,
    recursive=False,
    seed=42,
    workers=1,
):
    """Run the full annotation pipeline.

    Args:
        input_dir: directory of seedling images.
        output_dir: directory for X-AnyLabeling image/json pairs.
        train_ratio: fraction of images for training (rest for val).
        exg_threshold: ExG threshold for vegetation mask.
        min_saturation: minimum HSV saturation for vegetation pixels.
        min_value: minimum HSV value for vegetation pixels.
        max_value: maximum HSV value for vegetation pixels.
        min_component_exg_mean: discard components below this mean ExG score.
        polygon_epsilon: contour simplification ratio; lower keeps more detail.
        min_area_ratio: discard mask components smaller than this image-area ratio.
        close_kernel_ratio: morphology close kernel relative to image size.
        open_kernel_ratio: morphology open kernel relative to image size.
        min_polygon_points: target lower bound for polygon vertices.
        max_polygon_points: target upper bound for polygon vertices.
        max_instances: max polygons per image by area; 0 means no limit.
        recursive: whether to search input_dir recursively.
        seed: random seed for reproducible train/val split.
        workers: number of worker processes for image analysis; 1 disables multiprocessing.

    Returns:
        list of annotation records (dicts).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got: {workers}")

    # ---- 1. Discover images ----
    image_paths = discover_images(input_dir, recursive)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Found {len(image_paths)} images in {input_dir}")

    # ---- 2. Per-image: indices + instance polygons/rectangles ----
    analysis_params = {
        "exg_threshold": exg_threshold,
        "min_saturation": min_saturation,
        "min_value": min_value,
        "max_value": max_value,
        "min_component_exg_mean": min_component_exg_mean,
        "polygon_epsilon": polygon_epsilon,
        "min_area_ratio": min_area_ratio,
        "close_kernel_ratio": close_kernel_ratio,
        "open_kernel_ratio": open_kernel_ratio,
        "min_polygon_points": min_polygon_points,
        "max_polygon_points": max_polygon_points,
        "max_instances": max_instances,
    }
    tasks = [(img_path, analysis_params) for img_path in image_paths]
    if workers == 1:
        results = [_analyze_image(task) for task in tqdm(tasks, desc="Analyzing images", unit="image")]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                tqdm(
                    executor.map(_analyze_image, tasks),
                    total=len(tasks),
                    desc=f"Analyzing images ({workers} workers)",
                    unit="image",
                )
            )

    records = []
    for record, warning in results:
        if warning:
            print(warning)
        if record is not None:
            records.append(record)

    if not records:
        raise ValueError("No images could be processed")

    # ---- 3. Train/val split ----
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_train = int(len(records) * train_ratio)
    if n_train == 0 and train_ratio > 0:
        n_train = 1  # ensure at least 1 training image when split is requested
    train_indices = set(indices[:n_train])

    for i, rec in enumerate(records):
        rec["split"] = "train" if i in train_indices else "val"

    # ---- 4. Write output ----
    # dataset/{train,val}/ with image + JSON pairs, plus dataset/classes.txt

    train_count = 0
    val_count = 0
    no_polygon_count = 0

    for rec in tqdm(records, desc="Writing annotations", unit="image"):
        split = rec["split"]
        img_dst = output_dir / split / f"{rec['stem']}{rec['suffix']}"
        label_dst = output_dir / split / f"{rec['stem']}.json"

        img_dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy image
        try:
            shutil.copy2(rec["image_path"], img_dst)
        except Exception as exc:
            print(f"  Warning: failed to copy {rec['image_path'].name}: {exc}")
            continue

        # Write label
        write_xanylabeling_json(label_dst, rec, img_dst.name)
        if not rec["instances"]:
            no_polygon_count += 1

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    # ---- 5. Generate classes.txt ----
    write_classes_txt(output_dir, CLASS_NAMES)

    # ---- 6. Summary ----
    print(f"\nDone: {train_count} train + {val_count} val images -> {output_dir}")
    if no_polygon_count:
        print(f"  {no_polygon_count} images had no vegetation polygon (empty labels)")

    # Class distribution
    class_counts = {}
    for rec in records:
        for instance in rec["instances"]:
            cid = instance["class_id"]
            class_counts[cid] = class_counts.get(cid, 0) + 1

    print("\nClass distribution:")
    if class_counts:
        for cid in sorted(class_counts.keys()):
            name = CLASS_NAMES.get(cid, f"class_{cid}")
            print(f"  {cid} ({name}): {class_counts[cid]}")
    else:
        print("  no instances")

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
        help="Directory of seedling images (default: raw_data).",
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
        "--min-saturation",
        type=float,
        default=0.08,
        help="Minimum HSV saturation for vegetation pixels (default: 0.08).",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.04,
        help="Minimum HSV value for vegetation pixels (default: 0.04).",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=1.0,
        help="Maximum HSV value for vegetation pixels (default: 1.0).",
    )
    parser.add_argument(
        "--min-component-exg-mean",
        type=float,
        default=DEFAULT_MIN_COMPONENT_EXG_MEAN,
        help=f"Discard components with mean ExG below this value (default: {DEFAULT_MIN_COMPONENT_EXG_MEAN}).",
    )
    parser.add_argument(
        "--polygon-epsilon",
        type=float,
        default=0.003,
        help="Contour simplification ratio. Lower keeps more detail (default: 0.003).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0005,
        help="Discard mask components smaller than this image-area ratio (default: 0.0005).",
    )
    parser.add_argument(
        "--close-kernel-ratio",
        type=float,
        default=0.015,
        help="Morphological close kernel ratio for joining broken regions (default: 0.015).",
    )
    parser.add_argument(
        "--open-kernel-ratio",
        type=float,
        default=0.003,
        help="Morphological open kernel ratio for removing speckles (default: 0.003).",
    )
    parser.add_argument(
        "--min-polygon-points",
        type=int,
        default=12,
        help="Target lower bound for generated polygon vertices (default: 12).",
    )
    parser.add_argument(
        "--max-polygon-points",
        type=int,
        default=DEFAULT_MAX_POLYGON_POINTS,
        help=f"Target upper bound for generated polygon vertices (default: {DEFAULT_MAX_POLYGON_POINTS}).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=DEFAULT_MAX_INSTANCES,
        help=f"Maximum polygons per image by area; 0 keeps all valid instances (default: {DEFAULT_MAX_INSTANCES}).",
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
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for image analysis. Use 4-8 for large datasets (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotate(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        exg_threshold=args.exg_threshold,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
        max_value=args.max_value,
        min_component_exg_mean=args.min_component_exg_mean,
        polygon_epsilon=args.polygon_epsilon,
        min_area_ratio=args.min_area_ratio,
        close_kernel_ratio=args.close_kernel_ratio,
        open_kernel_ratio=args.open_kernel_ratio,
        min_polygon_points=args.min_polygon_points,
        max_polygon_points=args.max_polygon_points,
        max_instances=args.max_instances,
        recursive=args.recursive,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

