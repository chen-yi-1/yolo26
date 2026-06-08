import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


EPSILON = 1e-6
INDEX_NAMES = ("ExG", "ExR", "ExGR", "NGRDI", "GLI", "VARI", "CIVE")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > EPSILON,
    )


def calculate_indices(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    return {
        "ExG": exg,
        "ExR": exr,
        "ExGR": exg - exr,
        "NGRDI": safe_divide(g - r, g + r),
        "GLI": safe_divide(2 * g - r - b, 2 * g + r + b),
        "VARI": safe_divide(g - r, g + r - b),
        "CIVE": 0.441 * r - 0.811 * g + 0.385 * b,
    }


def create_vegetation_mask(indices, threshold):
    return indices["ExG"] > threshold


def summarize_values(values):
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def build_metrics_row(image_path, rgb, threshold):
    height, width = rgb.shape[:2]
    indices = calculate_indices(rgb)
    mask = create_vegetation_mask(indices, threshold)

    row = {
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "vegetation_coverage": float(np.mean(mask)),
    }

    for name in INDEX_NAMES:
        full_mean, full_std = summarize_values(indices[name])
        veg_mean, veg_std = summarize_values(indices[name][mask])
        row[f"{name}_mean"] = full_mean
        row[f"{name}_std"] = full_std
        row[f"{name}_veg_mean"] = veg_mean
        row[f"{name}_veg_std"] = veg_std

    return row, mask


def iter_image_paths(input_dir, recursive):
    pattern = "**/*" if recursive else "*"
    for path in sorted(Path(input_dir).glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_rgb_float(image_path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float32) / 255.0


def save_mask(mask, path):
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_image.save(path)


def save_overlay(rgb, mask, path, overlay_alpha):
    base = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    overlay = base.copy()
    overlay[mask] = (
        (1.0 - overlay_alpha) * overlay[mask]
        + overlay_alpha * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(path)


def csv_fieldnames():
    fields = ["image_path", "width", "height", "vegetation_coverage"]
    for name in INDEX_NAMES:
        fields.extend([f"{name}_mean", f"{name}_std", f"{name}_veg_mean", f"{name}_veg_std"])
    return fields


def write_metrics_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def process_directory(input_dir, output_dir, threshold, recursive=False, overlay_alpha=0.4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in iter_image_paths(input_dir, recursive):
        try:
            rgb = load_rgb_float(image_path)
        except Exception as exc:
            print(f"Warning: skipped unreadable image {image_path}: {exc}")
            continue

        row, mask = build_metrics_row(image_path, rgb, threshold)
        rows.append(row)

        stem = image_path.stem
        save_mask(mask, masks_dir / f"{stem}_mask.png")
        save_overlay(rgb, mask, overlays_dir / f"{stem}_overlay.png", overlay_alpha)

    write_metrics_csv(rows, output_dir / "metrics.csv")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate RGB vegetation metrics for seedling images.")
    parser.add_argument("--input", required=True, help="Input image directory.")
    parser.add_argument("--output", required=True, help="Output directory for CSV, masks, and overlays.")
    parser.add_argument("--threshold", type=float, default=0.1, help="ExG threshold for vegetation mask.")
    parser.add_argument("--recursive", action="store_true", help="Process nested image directories.")
    parser.add_argument("--overlay-alpha", type=float, default=0.4, help="Green overlay opacity from 0 to 1.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between 0 and 1")
    rows = process_directory(
        args.input,
        args.output,
        threshold=args.threshold,
        recursive=args.recursive,
        overlay_alpha=args.overlay_alpha,
    )
    print(f"Wrote {len(rows)} image rows to {Path(args.output) / 'metrics.csv'}")


if __name__ == "__main__":
    main()
