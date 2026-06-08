import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.rgb_vegetation_metrics import build_metrics_row, calculate_indices, process_directory


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
        self.assertAlmostEqual(float(indices["CIVE"][0, 0]), -0.3599, places=6)

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


if __name__ == "__main__":
    unittest.main()
