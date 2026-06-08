# RGB Vegetation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a batch RGB vegetation metrics script that exports per-image CSV metrics plus vegetation masks and overlays.

**Architecture:** Keep the script self-contained under `scripts/` so it can be run independently from YOLO training and inference. Expose small pure functions for index calculation, mask generation, row summarization, and batch processing so tests can verify behavior without GPU, model weights, or datasets.

**Tech Stack:** Python standard library, NumPy, Pillow, unittest/pytest-compatible tests.

---

## File Structure

- Create `scripts/rgb_vegetation_metrics.py`: CLI script and reusable pure functions for RGB index calculation, ExG mask generation, overlay rendering, CSV row construction, and directory processing.
- Create `tests/test_rgb_vegetation_metrics.py`: focused tests for formulas, mask coverage, and one-image batch output.

### Task 1: RGB Index Formula Tests

**Files:**
- Create: `tests/test_rgb_vegetation_metrics.py`
- Create: `scripts/rgb_vegetation_metrics.py`

- [x] **Step 1: Write the failing formula test**

Create `tests/test_rgb_vegetation_metrics.py`:

```python
import unittest

import numpy as np

from scripts.rgb_vegetation_metrics import calculate_indices


class RGBVegetationMetricsTests(unittest.TestCase):
    def test_calculate_indices_on_known_rgb_pixel(self):
        rgb = np.array([[[0.2, 0.6, 0.1]]], dtype=np.float32)

        indices = calculate_indices(rgb)

        self.assertAlmostEqual(float(indices["ExG"][0, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(indices["ExR"][0, 0]), -0.32, places=6)
        self.assertAlmostEqual(float(indices["ExGR"][0, 0]), 1.22, places=6)
        self.assertAlmostEqual(float(indices["NGRDI"][0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(indices["GLI"][0, 0]), 0.6, places=6)
        self.assertAlmostEqual(float(indices["VARI"][0, 0]), 0.57142857, places=6)
        self.assertAlmostEqual(float(indices["CIVE"][0, 0]), -0.3689, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_rgb_vegetation_metrics.py::RGBVegetationMetricsTests::test_calculate_indices_on_known_rgb_pixel -q
```

Expected: FAIL because `scripts.rgb_vegetation_metrics` does not exist yet.

- [x] **Step 3: Write minimal index implementation**

Create `scripts/rgb_vegetation_metrics.py`:

```python
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image


EPSILON = 1e-6
INDEX_NAMES = ("ExG", "ExR", "ExGR", "NGRDI", "GLI", "VARI", "CIVE")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
        "NGRDI": (g - r) / (g + r + EPSILON),
        "GLI": (2 * g - r - b) / (2 * g + r + b + EPSILON),
        "VARI": (g - r) / (g + r - b + EPSILON),
        "CIVE": 0.441 * r - 0.811 * g + 0.385 * b,
    }
```

- [x] **Step 4: Run formula test to verify it passes**

Run:

```bash
python -m pytest tests/test_rgb_vegetation_metrics.py::RGBVegetationMetricsTests::test_calculate_indices_on_known_rgb_pixel -q
```

Expected: PASS.

### Task 2: Mask Coverage And Row Summary

**Files:**
- Modify: `tests/test_rgb_vegetation_metrics.py`
- Modify: `scripts/rgb_vegetation_metrics.py`

- [x] **Step 1: Add failing mask and summary test**

Append this test method inside `RGBVegetationMetricsTests`:

```python
    def test_build_metrics_row_summarizes_full_image_and_mask(self):
        rgb = np.array(
            [
                [[0.2, 0.6, 0.1], [0.5, 0.4, 0.3]],
                [[0.1, 0.7, 0.1], [0.8, 0.2, 0.1]],
            ],
            dtype=np.float32,
        )

        row, mask = build_metrics_row("sample.jpg", rgb, threshold=0.4)

        self.assertEqual(row["image_path"], "sample.jpg")
        self.assertEqual(row["width"], 2)
        self.assertEqual(row["height"], 2)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertAlmostEqual(row["vegetation_coverage"], 0.5, places=6)
        self.assertIn("ExG_mean", row)
        self.assertIn("ExG_std", row)
        self.assertIn("ExG_veg_mean", row)
        self.assertIn("ExG_veg_std", row)
        self.assertGreater(row["ExG_veg_mean"], row["ExG_mean"])
```

Also update the import:

```python
from scripts.rgb_vegetation_metrics import build_metrics_row, calculate_indices
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_rgb_vegetation_metrics.py::RGBVegetationMetricsTests::test_build_metrics_row_summarizes_full_image_and_mask -q
```

Expected: FAIL because `build_metrics_row` is not implemented.

- [x] **Step 3: Implement mask and summary helpers**

Add these functions to `scripts/rgb_vegetation_metrics.py`:

```python
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
```

- [x] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_rgb_vegetation_metrics.py -q
```

Expected: both tests PASS.

### Task 3: Batch CLI Output

**Files:**
- Modify: `tests/test_rgb_vegetation_metrics.py`
- Modify: `scripts/rgb_vegetation_metrics.py`

- [x] **Step 1: Add failing batch output test**

Add these imports to `tests/test_rgb_vegetation_metrics.py`:

```python
import csv
import tempfile
from pathlib import Path

from PIL import Image
```

Append this test method inside `RGBVegetationMetricsTests`:

```python
    def test_process_directory_writes_csv_mask_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            image_path = input_dir / "seedling.png"
            Image.new("RGB", (2, 2), (50, 180, 30)).save(image_path)

            rows = process_directory(input_dir, output_dir, threshold=0.1, recursive=False, overlay_alpha=0.4)

            self.assertEqual(len(rows), 1)
            self.assertTrue((output_dir / "metrics.csv").exists())
            self.assertTrue((output_dir / "masks" / "seedling_mask.png").exists())
            self.assertTrue((output_dir / "overlays" / "seedling_overlay.png").exists())

            with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                csv_rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["image_path"], str(image_path))
```

Update the import:

```python
from scripts.rgb_vegetation_metrics import build_metrics_row, calculate_indices, process_directory
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_rgb_vegetation_metrics.py::RGBVegetationMetricsTests::test_process_directory_writes_csv_mask_and_overlay -q
```

Expected: FAIL because `process_directory` is not implemented.

- [x] **Step 3: Implement file discovery, image loading, mask/overlay saving, CSV writing, and CLI**

Add these functions to `scripts/rgb_vegetation_metrics.py`:

```python
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
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
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
```

- [x] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_rgb_vegetation_metrics.py -q
```

Expected: all RGB metrics tests PASS.

### Task 4: Final Verification And Commit

**Files:**
- Modify: `scripts/rgb_vegetation_metrics.py`
- Modify: `tests/test_rgb_vegetation_metrics.py`

- [x] **Step 1: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests PASS.

- [x] **Step 2: Run CLI smoke test against sample image**

Run:

```bash
python scripts/rgb_vegetation_metrics.py --input img --output runs/rgb_metrics_smoke --threshold 0.1
```

Expected: command prints that it wrote image rows to `runs/rgb_metrics_smoke/metrics.csv`, and the output directory contains `metrics.csv`, `masks/`, and `overlays/`.

- [x] **Step 3: Inspect git diff**

Run:

```bash
git diff -- scripts/rgb_vegetation_metrics.py tests/test_rgb_vegetation_metrics.py
```

Expected: diff only includes the RGB metrics script and tests.

- [x] **Step 4: Commit implementation**

Run:

```bash
git add scripts/rgb_vegetation_metrics.py tests/test_rgb_vegetation_metrics.py docs/superpowers/plans/2026-06-08-rgb-vegetation-metrics.md
git commit -m "feat: add RGB vegetation metrics script"
```

Expected: commit succeeds.
