#!/usr/bin/env python3
"""
Auto-annotate seedling images for X-AnyLabeling editing using RGB vegetation indices.

Assumes one seedling per image (one pot, large in frame).
- Polygon: external contour of ExG vegetation mask
- Health classification: batch-relative ranking (default) or absolute thresholds

Input:  raw_datas/       鈥?directory of seedling images
Output: dataset/          鈥?X-AnyLabeling dataset (train/ and val/ image+json pairs)

Usage:
    # Batch-relative ranking (default)
    python scripts/rgb_yolo_annotate.py --input raw_datas --output dataset

    # Absolute threshold mode (fixed, reproducible across batches)
    python scripts/rgb_yolo_annotate.py --input raw_datas --output dataset \\
        --classification-mode absolute

    # Absolute mode with custom thresholds
    python scripts/rgb_yolo_annotate.py --input raw_datas --output dataset \\
        --classification-mode absolute \\
        --gli-healthy-min 0.35 --ngrdi-healthy-min 0.30

"""

import argparse
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

# Indices used for green-health scoring (higher = greener/healthier)
GREEN_FEATURES = ("ExG", "GLI", "NGRDI", "VARI")

# Pixel-mask cleanup defaults for auto-generated segmentation labels.
DEFAULT_MIN_COMPONENT_AREA = 300
DEFAULT_MIN_COMPONENT_AREA_RATIO = 0.001
DEFAULT_MIN_COMPONENT_EXG_MEAN = 0.12
DEFAULT_MAX_INSTANCES = 12
DEFAULT_MAX_POLYGON_POINTS = 96
DEFAULT_AUX_MIN_OVERLAP = 0.08

# Absolute-value thresholds for health classification.
# Normalised RGB vegetation indices (GLI, NGRDI, VARI) are preferred because
# they are less sensitive to illumination changes than raw ExG.
# Ranges derived from the literature:
#  GLI     [-1, 1]     NGRDI  [-1, 1]     VARI  [-1, 1]     ExR    [-1, 1.4]
DEFAULT_GLI_HEALTHY_MIN = 0.30
DEFAULT_NGRDI_HEALTHY_MIN = 0.25
DEFAULT_VARI_HEALTHY_MIN = 0.30
DEFAULT_ExR_HEALTHY_MAX = 0.12
DEFAULT_COVERAGE_HEALTHY_MIN = 0.03

DEFAULT_GLI_UNHEALTHY_MAX = 0.10
DEFAULT_NGRDI_UNHEALTHY_MAX = 0.15
DEFAULT_ExR_UNHEALTHY_MIN = 0.25
DEFAULT_COVERAGE_UNHEALTHY_MAX = 0.005

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


def polygon_to_mask(points, image_shape):
    """Rasterize polygon points to a boolean mask."""
    try:
        import cv2
    except ImportError:
        return None

    if len(points) < 3:
        return None

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def simplify_polygon_points(points, image_shape, epsilon_ratio=0.003, min_points=12, max_points=DEFAULT_MAX_POLYGON_POINTS):
    """Simplify raw polygon points using the same contour policy as mask polygons."""
    try:
        import cv2
    except ImportError:
        return points

    if len(points) < 3:
        return None

    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = simplify_contour_adaptive(
        contour,
        epsilon_ratio=epsilon_ratio,
        min_points=min_points,
        max_points=max_points,
    )
    return contour_to_polygon_points(simplified, image_shape)


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
        return [{"polygon": polygon, "mask": mask, "area": int(np.sum(mask))}] if polygon is not None else []

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
                "mask": component_mask,
                "area": int(np.sum(component_mask)),
            }
        )

    return instances


def load_aux_seg_model(model_path):
    """Load an optional Ultralytics segmentation model."""
    if not model_path:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Auxiliary segmentation model not found: {model_path}")

    from ultralytics import YOLO
    return YOLO(str(model_path))


def parse_bool(value):
    """Parse common command-line boolean values."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def extract_aux_model_instances(
    model,
    image_path,
    image_shape,
    score_map=None,
    min_component_score=DEFAULT_MIN_COMPONENT_EXG_MEAN,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device=None,
    min_area_ratio=0.0005,
    max_instances=DEFAULT_MAX_INSTANCES,
    polygon_epsilon=0.003,
    min_polygon_points=12,
    max_polygon_points=DEFAULT_MAX_POLYGON_POINTS,
):
    """Use a segmentation model to propose instance masks; class is assigned later by RGB stats."""
    if model is None:
        return []

    predict_kwargs = dict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    if device:
        predict_kwargs["device"] = device

    result = model.predict(**predict_kwargs)[0]
    if result.masks is None or not result.masks.xy:
        return []

    h, w = image_shape[:2]
    min_area = max(1, int(h * w * min_area_ratio))
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(result.masks.xy))

    instances = []
    for idx, raw_polygon in enumerate(result.masks.xy):
        points = [
            [
                float(min(w - 1, max(0, x))),
                float(min(h - 1, max(0, y))),
            ]
            for x, y in np.asarray(raw_polygon, dtype=np.float32)
        ]
        points = simplify_polygon_points(
            points,
            image_shape,
            epsilon_ratio=polygon_epsilon,
            min_points=min_polygon_points,
            max_points=max_polygon_points,
        )
        if points is None:
            continue

        mask = polygon_to_mask(points, image_shape)
        if mask is None:
            continue

        area = int(np.sum(mask))
        if area < min_area:
            continue
        if score_map is not None and min_component_score is not None:
            if float(np.mean(score_map[mask])) < min_component_score:
                continue

        instances.append(
            {
                "polygon": points,
                "mask": mask,
                "area": area,
                "source": "aux_model",
                "model_confidence": round(float(confs[idx]), 4) if idx < len(confs) else None,
            }
        )

    instances.sort(key=lambda item: item["area"], reverse=True)
    if max_instances and max_instances > 0:
        instances = instances[:max_instances]
    return instances


def fuse_aux_model_with_rgb_mask(
    model_instances,
    rgb_mask,
    score_map=None,
    min_overlap_ratio=DEFAULT_AUX_MIN_OVERLAP,
    min_component_score=DEFAULT_MIN_COMPONENT_EXG_MEAN,
    epsilon_ratio=0.003,
    min_area_ratio=0.0005,
    close_kernel_ratio=0.015,
    open_kernel_ratio=0.003,
    min_points=12,
    max_points=DEFAULT_MAX_POLYGON_POINTS,
    max_instances=DEFAULT_MAX_INSTANCES,
):
    """Fuse model proposals with the RGB vegetation mask by using their intersection.

    The segmentation model supplies instance separation; the RGB mask supplies
    plant-pixel validation. Final geometry is extracted from model_mask & rgb_mask,
    never from the raw model mask alone.
    """
    if not model_instances:
        return [], np.zeros(rgb_mask.shape, dtype=bool)

    fused_instances = []
    accepted_union = np.zeros(rgb_mask.shape, dtype=bool)

    for model_instance in model_instances:
        model_mask = model_instance["mask"].astype(bool)
        model_area = int(np.sum(model_mask))
        if model_area <= 0:
            continue

        intersection = model_mask & rgb_mask
        intersection_area = int(np.sum(intersection))
        overlap_ratio = intersection_area / float(model_area)
        if intersection_area <= 0 or overlap_ratio < min_overlap_ratio:
            continue

        remaining_intersection = intersection & ~accepted_union
        if not np.any(remaining_intersection):
            continue

        remaining_slots = 0
        if max_instances and max_instances > 0:
            remaining_slots = max_instances - len(fused_instances)
            if remaining_slots <= 0:
                break

        instances = extract_mask_instances(
            remaining_intersection,
            score_map=score_map,
            epsilon_ratio=epsilon_ratio,
            min_area_ratio=min_area_ratio,
            min_component_score=min_component_score,
            close_kernel_ratio=close_kernel_ratio,
            open_kernel_ratio=open_kernel_ratio,
            min_points=min_points,
            max_points=max_points,
            max_instances=remaining_slots,
        )
        for instance in instances:
            instance["source"] = "fused_model_rgb"
            instance["model_confidence"] = model_instance.get("model_confidence")
            instance["model_overlap_ratio"] = round(float(overlap_ratio), 4)
            instance["raw_model_area"] = model_area
            fused_instances.append(instance)
            accepted_union |= instance["mask"]

    return fused_instances, accepted_union


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

    Classification rules:
        healthy:  good coverage + strong green + low red
        abnormal: weak green, elevated red, low coverage, or intermediate pattern.
    """
    cov_rank = rec["coverage_rank"]
    green = rec["green_score"]
    red = rec["red_rank"]

    if cov_rank >= 0.35 and green >= 0.55 and red <= 0.75:
        class_id = 0  # healthy
        confidence = (cov_rank + green + (1.0 - red)) / 3.0
        reason = "good coverage, strong green, low red/brown signal"
    elif green <= 0.35 and red >= 0.45:
        class_id = 1  # abnormal
        confidence = (1.0 - green + red) / 2.0
        reason = "weak green indices with elevated red/brown signal"
    elif cov_rank <= 0.20:
        class_id = 1  # abnormal
        confidence = 1.0 - cov_rank
        reason = "very low vegetation coverage"
    else:
        class_id = 1  # abnormal
        confidence = 0.55
        reason = "intermediate abnormal RGB pattern; review manually"

    rec["class_id"] = class_id
    rec["confidence"] = round(float(confidence), 4)
    rec["reason"] = reason

    # Override: no vegetation pixels -> abnormal.
    if rec["stats"]["vegetation_coverage"] <= 0.0:
        rec["class_id"] = 1
        rec["confidence"] = 1.0
        rec["reason"] = "no vegetation pixels detected"


def classify_record_absolute(rec, thresholds=None):
    """Assign health class_id using absolute vegetation-index thresholds.

    Unlike ``classify_record``, this does NOT depend on batch-relative ranking.
    The same image always receives the same label when the thresholds are fixed.

    Args:
        rec: per-instance dict with ``stats`` (coverage, ExG_veg_mean, 鈥?.
        thresholds: optional dict of overrides for the default thresholds.
            Keys (all optional):
                gli_healthy_min, ngrdi_healthy_min, vari_healthy_min,
                exr_healthy_max, coverage_healthy_min,
                gli_unhealthy_max, ngrdi_unhealthy_max,
                exr_unhealthy_min, coverage_unhealthy_max

    Reads from rec["stats"]; writes class_id, confidence, reason back.
    """
    t = {
        "gli_healthy_min": DEFAULT_GLI_HEALTHY_MIN,
        "ngrdi_healthy_min": DEFAULT_NGRDI_HEALTHY_MIN,
        "vari_healthy_min": DEFAULT_VARI_HEALTHY_MIN,
        "exr_healthy_max": DEFAULT_ExR_HEALTHY_MAX,
        "coverage_healthy_min": DEFAULT_COVERAGE_HEALTHY_MIN,
        "gli_unhealthy_max": DEFAULT_GLI_UNHEALTHY_MAX,
        "ngrdi_unhealthy_max": DEFAULT_NGRDI_UNHEALTHY_MAX,
        "exr_unhealthy_min": DEFAULT_ExR_UNHEALTHY_MIN,
        "coverage_unhealthy_max": DEFAULT_COVERAGE_UNHEALTHY_MAX,
    }
    if thresholds:
        t.update(thresholds)

    stats = rec["stats"]
    coverage = stats["vegetation_coverage"]
    gli = stats.get("GLI_veg_mean", 0.0)
    ngrdi = stats.get("NGRDI_veg_mean", 0.0)
    vari = stats.get("VARI_veg_mean", 0.0)
    exr = stats.get("ExR_veg_mean", 0.0)

    # ---- no vegetation at all -> abnormal ----
    if coverage <= 0.0:
        rec["class_id"] = 1  # abnormal
        rec["confidence"] = 1.0
        rec["reason"] = "no vegetation pixels detected"
        return

    # ---- healthy: strong green *and* low red *and* sufficient coverage ----
    healthy_checks = (
        gli >= t["gli_healthy_min"],
        ngrdi >= t["ngrdi_healthy_min"],
        vari >= t["vari_healthy_min"],
        exr <= t["exr_healthy_max"],
        coverage >= t["coverage_healthy_min"],
    )
    if all(healthy_checks):
        rec["class_id"] = 0  # healthy
        n = len(healthy_checks)
        passed = sum(healthy_checks)
        rec["confidence"] = round(passed / n, 4)
        rec["reason"] = (
            f"GLI={gli:.3f}>={t['gli_healthy_min']}, "
            f"NGRDI={ngrdi:.3f}>={t['ngrdi_healthy_min']}, "
            f"VARI={vari:.3f}>={t['vari_healthy_min']}, "
            f"ExR={exr:.3f}<={t['exr_healthy_max']}, "
            f"cover={coverage:.3f}>={t['coverage_healthy_min']}"
        )
        return

    # ---- abnormal: very weak green or elevated red or barely any coverage ----
    abnormal_reasons = []
    if gli <= t["gli_unhealthy_max"] and ngrdi <= t["ngrdi_unhealthy_max"]:
        abnormal_reasons.append(
            f"GLI={gli:.3f}<={t['gli_unhealthy_max']} AND "
            f"NGRDI={ngrdi:.3f}<={t['ngrdi_unhealthy_max']}"
        )
    if exr >= t["exr_unhealthy_min"]:
        abnormal_reasons.append(
            f"ExR={exr:.3f}>={t['exr_unhealthy_min']}"
        )
    if coverage <= t["coverage_unhealthy_max"]:
        abnormal_reasons.append(
            f"cover={coverage:.3f}<={t['coverage_unhealthy_max']}"
        )

    if abnormal_reasons:
        rec["class_id"] = 1  # abnormal
        rec["confidence"] = round(min(1.0, 0.5 + 0.2 * len(abnormal_reasons)), 4)
        rec["reason"] = "; ".join(abnormal_reasons)
        return

    # ---- abnormal: everything between healthy and clearly abnormal ----
    rec["class_id"] = 1  # abnormal
    rec["confidence"] = 0.55
    rec["reason"] = (
        f"intermediate: GLI={gli:.3f}, NGRDI={ngrdi:.3f}, "
        f"VARI={vari:.3f}, ExR={exr:.3f}, cover={coverage:.3f}"
    )


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
        shapes.append(
            {
                "label": CLASS_NAMES[instance["class_id"]],
                "score": instance["confidence"],
                "points": instance["polygon"],
                "group_id": None,
                "description": instance.get("description") or instance.get("reason"),
                "difficult": False,
                "shape_type": "polygon",
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
    aux_model_path=None,
    aux_model_conf=0.25,
    aux_model_iou=0.7,
    aux_model_imgsz=640,
    aux_model_device=None,
    use_aux_model=True,
    aux_min_overlap=DEFAULT_AUX_MIN_OVERLAP,
    aux_model_fallback=True,
    recursive=False,
    seed=42,
    copy_mode="copy",
    classification_mode="rank",
    gli_healthy_min=DEFAULT_GLI_HEALTHY_MIN,
    ngrdi_healthy_min=DEFAULT_NGRDI_HEALTHY_MIN,
    vari_healthy_min=DEFAULT_VARI_HEALTHY_MIN,
    exr_healthy_max=DEFAULT_ExR_HEALTHY_MAX,
    coverage_healthy_min=DEFAULT_COVERAGE_HEALTHY_MIN,
    gli_unhealthy_max=DEFAULT_GLI_UNHEALTHY_MAX,
    ngrdi_unhealthy_max=DEFAULT_NGRDI_UNHEALTHY_MAX,
    exr_unhealthy_min=DEFAULT_ExR_UNHEALTHY_MIN,
    coverage_unhealthy_max=DEFAULT_COVERAGE_UNHEALTHY_MAX,
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
        aux_model_path: optional YOLO segmentation model used to propose instance masks.
        aux_model_conf: auxiliary model confidence threshold.
        aux_model_iou: auxiliary model IoU threshold.
        aux_model_imgsz: auxiliary model inference image size.
        aux_model_device: optional Ultralytics device string.
        use_aux_model: whether to use the auxiliary model when aux_model_path is provided.
        aux_min_overlap: minimum model-mask area ratio that must overlap the RGB mask.
        aux_model_fallback: add RGB mask instances missed by the model, or use RGB masks when fusion finds no masks.
        recursive: whether to search input_dir recursively.
        seed: random seed for reproducible train/val split.
        copy_mode: "copy" to duplicate images, "symlink" for symlinks.
        classification_mode: "rank" (batch-relative percentile) or "absolute".
        gli_healthy_min: (absolute mode) minimum GLI for healthy.
        ngrdi_healthy_min: (absolute mode) minimum NGRDI for healthy.
        vari_healthy_min: (absolute mode) minimum VARI for healthy.
        exr_healthy_max: (absolute mode) maximum ExR for healthy.
        coverage_healthy_min: (absolute mode) minimum coverage for healthy.
        gli_unhealthy_max: (absolute mode) maximum GLI for abnormal.
        ngrdi_unhealthy_max: (absolute mode) maximum NGRDI for abnormal.
        exr_unhealthy_min: (absolute mode) minimum ExR for abnormal.
        coverage_unhealthy_max: (absolute mode) maximum coverage for abnormal.

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
    aux_model = load_aux_seg_model(aux_model_path) if use_aux_model and aux_model_path else None

    # ---- 2. Per-image: indices + instance polygons + per-instance stats ----
    records = []
    instance_records = []
    for img_path in tqdm(image_paths, desc="Analyzing images", unit="image"):
        try:
            rgb = load_rgb_float(img_path)
        except Exception as exc:
            print(f"  Warning: skipped {img_path.name}: {exc}")
            continue

        indices = calculate_indices(rgb)
        mask = create_vegetation_mask(
            indices,
            exg_threshold,
            rgb=rgb,
            min_saturation=min_saturation,
            min_value=min_value,
            max_value=max_value,
        )
        extracted_instances = []
        if aux_model is not None:
            model_instances = extract_aux_model_instances(
                aux_model,
                img_path,
                rgb.shape,
                score_map=None,
                min_component_score=None,
                conf=aux_model_conf,
                iou=aux_model_iou,
                imgsz=aux_model_imgsz,
                device=aux_model_device,
                min_area_ratio=min_area_ratio,
                max_instances=max_instances,
                polygon_epsilon=polygon_epsilon,
                min_polygon_points=min_polygon_points,
                max_polygon_points=max_polygon_points,
            )
            extracted_instances, accepted_union = fuse_aux_model_with_rgb_mask(
                model_instances,
                mask,
                score_map=indices["ExG"],
                min_overlap_ratio=aux_min_overlap,
                min_component_score=min_component_exg_mean,
                epsilon_ratio=polygon_epsilon,
                min_area_ratio=min_area_ratio,
                close_kernel_ratio=close_kernel_ratio,
                open_kernel_ratio=open_kernel_ratio,
                min_points=min_polygon_points,
                max_points=max_polygon_points,
                max_instances=max_instances,
            )
            if aux_model_fallback:
                remaining_slots = 0
                if max_instances and max_instances > 0:
                    remaining_slots = max_instances - len(extracted_instances)
                    if remaining_slots <= 0:
                        remaining_slots = None
                residual_mask = mask & ~accepted_union
                if not extracted_instances:
                    residual_mask = mask
                if remaining_slots is not None:
                    residual_instances = extract_mask_instances(
                        residual_mask,
                        score_map=indices["ExG"],
                        epsilon_ratio=polygon_epsilon,
                        min_area_ratio=min_area_ratio,
                        min_component_score=min_component_exg_mean,
                        close_kernel_ratio=close_kernel_ratio,
                        open_kernel_ratio=open_kernel_ratio,
                        min_points=min_polygon_points,
                        max_points=max_polygon_points,
                        max_instances=remaining_slots,
                    )
                    for residual_instance in residual_instances:
                        residual_instance["source"] = "rgb_residual" if extracted_instances else "rgb_mask"
                    extracted_instances.extend(residual_instances)

        if not extracted_instances and (aux_model is None or aux_model_fallback):
            extracted_instances = extract_mask_instances(
                mask,
                score_map=indices["ExG"],
                epsilon_ratio=polygon_epsilon,
                min_area_ratio=min_area_ratio,
                min_component_score=min_component_exg_mean,
                close_kernel_ratio=close_kernel_ratio,
                open_kernel_ratio=open_kernel_ratio,
                min_points=min_polygon_points,
                max_points=max_polygon_points,
                max_instances=max_instances,
            )
        instances = []
        for instance_index, extracted in enumerate(extracted_instances):
            instance = {
                "image_path": img_path,
                "image_stem": img_path.stem,
                "instance_index": instance_index,
                "polygon": extracted["polygon"],
                "area": extracted["area"],
                "source": extracted.get("source", "rgb_mask"),
                "model_confidence": extracted.get("model_confidence"),
                "model_overlap_ratio": extracted.get("model_overlap_ratio"),
                "stats": compute_image_stats(indices, extracted["mask"]),
            }
            instances.append(instance)
            instance_records.append(instance)

        records.append(
            {
                "image_path": img_path,
                "stem": img_path.stem,
                "suffix": img_path.suffix,
                "image_height": rgb.shape[0],
                "image_width": rgb.shape[1],
                "instances": instances,
            }
        )

    if not records:
        raise ValueError("No images could be processed")

    # ---- 3. Per-instance classification ----
    if instance_records:
        if classification_mode == "absolute":
            abs_thresholds = {
                "gli_healthy_min": gli_healthy_min,
                "ngrdi_healthy_min": ngrdi_healthy_min,
                "vari_healthy_min": vari_healthy_min,
                "exr_healthy_max": exr_healthy_max,
                "coverage_healthy_min": coverage_healthy_min,
                "gli_unhealthy_max": gli_unhealthy_max,
                "ngrdi_unhealthy_max": ngrdi_unhealthy_max,
                "exr_unhealthy_min": exr_unhealthy_min,
                "coverage_unhealthy_max": coverage_unhealthy_max,
            }
            for instance in tqdm(instance_records, desc="Classifying instances", unit="instance"):
                classify_record_absolute(instance, thresholds=abs_thresholds)
        else:
            compute_batch_scores(instance_records)
            for instance in tqdm(instance_records, desc="Classifying instances", unit="instance"):
                classify_record(instance)

        for instance in instance_records:
            description = f"{instance['reason']}; source={instance['source']}"
            if instance.get("model_confidence") is not None:
                description += f"; model_conf={instance['model_confidence']}"
            if instance.get("model_overlap_ratio") is not None:
                description += f"; model_rgb_overlap={instance['model_overlap_ratio']}"
            instance["description"] = description

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
    no_polygon_count = 0

    for rec in tqdm(records, desc="Writing annotations", unit="image"):
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
        if not rec["instances"]:
            no_polygon_count += 1

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    # ---- 6. Generate classes.txt ----
    write_classes_txt(output_dir, CLASS_NAMES)

    # ---- 7. Summary ----
    print(f"\nDone: {train_count} train + {val_count} val images 鈫?{output_dir}")
    if no_polygon_count:
        print(f"  {no_polygon_count} images had no vegetation polygon (empty labels)")

    # Class distribution
    class_counts = {}
    for instance in instance_records:
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
        "--aux-model-path",
        default=None,
        help="Optional YOLO segmentation model path used to propose instance masks.",
    )
    parser.add_argument(
        "--aux-model-conf",
        type=float,
        default=0.25,
        help="Auxiliary segmentation model confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--aux-model-iou",
        type=float,
        default=0.7,
        help="Auxiliary segmentation model IoU threshold (default: 0.7).",
    )
    parser.add_argument(
        "--aux-model-imgsz",
        type=int,
        default=640,
        help="Auxiliary segmentation model inference size (default: 640).",
    )
    parser.add_argument(
        "--aux-model-device",
        default=None,
        help="Optional Ultralytics device string for the auxiliary model, e.g. 0, cpu, cuda.",
    )
    parser.add_argument(
        "--use-aux-model",
        type=parse_bool,
        default=True,
        help="Whether to use the auxiliary model when --aux-model-path is provided (default: true).",
    )
    parser.add_argument(
        "--aux-min-overlap",
        type=float,
        default=DEFAULT_AUX_MIN_OVERLAP,
        help=(
            "Minimum fraction of a model mask that must overlap the RGB vegetation mask "
            f"before fusion keeps it (default: {DEFAULT_AUX_MIN_OVERLAP})."
        ),
    )
    parser.add_argument(
        "--no-aux-model-fallback",
        action="store_true",
        help="Do not fall back to RGB mask extraction when the auxiliary model finds no masks.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively.",
    )
    parser.add_argument(
        "--classification-mode",
        choices=["rank", "absolute"],
        default="absolute",
        help="Health classification mode: 'rank' (batch-relative percentile, default) "
        "or 'absolute' (fixed vegetation-index thresholds).",
    )
    parser.add_argument(
        "--gli-healthy-min",
        type=float,
        default=DEFAULT_GLI_HEALTHY_MIN,
        help=f"Absolute mode: minimum GLI for healthy (default: {DEFAULT_GLI_HEALTHY_MIN}).",
    )
    parser.add_argument(
        "--ngrdi-healthy-min",
        type=float,
        default=DEFAULT_NGRDI_HEALTHY_MIN,
        help=f"Absolute mode: minimum NGRDI for healthy (default: {DEFAULT_NGRDI_HEALTHY_MIN}).",
    )
    parser.add_argument(
        "--vari-healthy-min",
        type=float,
        default=DEFAULT_VARI_HEALTHY_MIN,
        help=f"Absolute mode: minimum VARI for healthy (default: {DEFAULT_VARI_HEALTHY_MIN}).",
    )
    parser.add_argument(
        "--exr-healthy-max",
        type=float,
        default=DEFAULT_ExR_HEALTHY_MAX,
        help=f"Absolute mode: maximum ExR for healthy (default: {DEFAULT_ExR_HEALTHY_MAX}).",
    )
    parser.add_argument(
        "--coverage-healthy-min",
        type=float,
        default=DEFAULT_COVERAGE_HEALTHY_MIN,
        help=f"Absolute mode: minimum vegetation coverage for healthy (default: {DEFAULT_COVERAGE_HEALTHY_MIN}).",
    )
    parser.add_argument(
        "--gli-unhealthy-max",
        type=float,
        default=DEFAULT_GLI_UNHEALTHY_MAX,
        help=f"Absolute mode: maximum GLI for abnormal (default: {DEFAULT_GLI_UNHEALTHY_MAX}).",
    )
    parser.add_argument(
        "--ngrdi-unhealthy-max",
        type=float,
        default=DEFAULT_NGRDI_UNHEALTHY_MAX,
        help=f"Absolute mode: maximum NGRDI for abnormal (default: {DEFAULT_NGRDI_UNHEALTHY_MAX}).",
    )
    parser.add_argument(
        "--exr-unhealthy-min",
        type=float,
        default=DEFAULT_ExR_UNHEALTHY_MIN,
        help=f"Absolute mode: minimum ExR for abnormal (default: {DEFAULT_ExR_UNHEALTHY_MIN}).",
    )
    parser.add_argument(
        "--coverage-unhealthy-max",
        type=float,
        default=DEFAULT_COVERAGE_UNHEALTHY_MAX,
        help=f"Absolute mode: maximum vegetation coverage for abnormal (default: {DEFAULT_COVERAGE_UNHEALTHY_MAX}).",
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
        aux_model_path=args.aux_model_path,
        aux_model_conf=args.aux_model_conf,
        aux_model_iou=args.aux_model_iou,
        aux_model_imgsz=args.aux_model_imgsz,
        aux_model_device=args.aux_model_device,
        use_aux_model=args.use_aux_model,
        aux_min_overlap=args.aux_min_overlap,
        aux_model_fallback=not args.no_aux_model_fallback,
        recursive=args.recursive,
        seed=args.seed,
        copy_mode=args.copy_mode,
        classification_mode=args.classification_mode,
        gli_healthy_min=args.gli_healthy_min,
        ngrdi_healthy_min=args.ngrdi_healthy_min,
        vari_healthy_min=args.vari_healthy_min,
        exr_healthy_max=args.exr_healthy_max,
        coverage_healthy_min=args.coverage_healthy_min,
        gli_unhealthy_max=args.gli_unhealthy_max,
        ngrdi_unhealthy_max=args.ngrdi_unhealthy_max,
        exr_unhealthy_min=args.exr_unhealthy_min,
        coverage_unhealthy_max=args.coverage_unhealthy_max,
    )


if __name__ == "__main__":
    main()

