import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.rgb_yolo_annotate import (
    annotate,
    calculate_indices,
    create_vegetation_mask,
    extract_mask_instances,
    extract_mask_polygon,
    extract_mask_polygons,
    load_rgb_float,
)


class RGBYoloAnnotateTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Index calculation
    # ------------------------------------------------------------------

    def test_calculate_indices_known_pixel(self):
        rgb = np.array([[[0.2, 0.6, 0.1]]], dtype=np.float32)
        indices = calculate_indices(rgb)

        self.assertAlmostEqual(float(indices["ExG"][0, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(indices["ExR"][0, 0]), -0.32, places=6)
        self.assertAlmostEqual(float(indices["ExGR"][0, 0]), 1.22, places=6)
        self.assertAlmostEqual(float(indices["NGRDI"][0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(indices["GLI"][0, 0]), 0.6, places=6)
        self.assertAlmostEqual(float(indices["VARI"][0, 0]), 0.57142857, places=6)
        self.assertAlmostEqual(float(indices["CIVE"][0, 0]), -0.3599, places=6)

    # ------------------------------------------------------------------
    # Mask extraction
    # ------------------------------------------------------------------

    def test_extract_mask_polygon_partial_mask(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:10] = True

        polygon = extract_mask_polygon(mask)

        self.assertIsNotNone(polygon)
        self.assertGreaterEqual(len(polygon), 3)
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        self.assertEqual(min(xs), 4.0)
        self.assertEqual(max(xs), 9.0)
        self.assertEqual(min(ys), 2.0)
        self.assertEqual(max(ys), 5.0)

    def test_extract_mask_polygons_multiple_components(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[2:10, 3:12] = True
        mask[15:25, 20:34] = True

        polygons = extract_mask_polygons(mask, close_kernel_ratio=0.0, open_kernel_ratio=0.0)

        self.assertEqual(len(polygons), 2)
        self.assertTrue(all(len(polygon) >= 3 for polygon in polygons))

    def test_extract_mask_instances_filters_low_score_components(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[2:10, 3:12] = True
        mask[15:25, 20:34] = True
        score = np.zeros((30, 40), dtype=np.float64)
        score[2:10, 3:12] = 0.08
        score[15:25, 20:34] = 0.22

        instances = extract_mask_instances(
            mask,
            score_map=score,
            min_component_score=0.13,
            close_kernel_ratio=0.0,
            open_kernel_ratio=0.0,
            max_instances=0,
        )

        self.assertEqual(len(instances), 1)
        ys, xs = np.where(instances[0]["mask"])
        self.assertGreaterEqual(xs.min(), 20)

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------

    def test_annotate_end_to_end(self):
        """Full pipeline with synthetic seedling images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_datas"
            output_dir = tmpdir / "datas"
            input_dir.mkdir()

            # Create 5 synthetic images simulating different seedling conditions
            # Healthy: bright green
            Image.new("RGB", (200, 200), (30, 180, 40)).save(input_dir / "healthy_01.jpg")
            # Dead: brown/black
            Image.new("RGB", (200, 200), (80, 40, 20)).save(input_dir / "dead_01.jpg")
            # Wilted: yellowish
            Image.new("RGB", (200, 200), (200, 180, 30)).save(input_dir / "wilted_01.jpg")
            # Overgrown: large green area
            Image.new("RGB", (200, 200), (20, 150, 35)).save(input_dir / "overgrown_01.jpg")
            # Abnormal: pale green
            Image.new("RGB", (200, 200), (140, 170, 100)).save(input_dir / "abnormal_01.jpg")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                train_ratio=0.8,
                exg_threshold=0.1,
                seed=42,
            )

            # Check record count
            self.assertEqual(len(records), 5)

            # Check default output structure: X-AnyLabeling image/json pairs.
            self.assertTrue((output_dir / "train").is_dir())
            self.assertTrue((output_dir / "val").is_dir())
            self.assertFalse((output_dir / "dataset.yaml").exists())
            self.assertFalse((output_dir / "images").exists())
            self.assertFalse((output_dir / "labels").exists())
            self.assertEqual(
                (output_dir / "classes.txt").read_text(encoding="utf-8"),
                "healthy\nabnormal\n",
            )

            # Train/val split: 0.8 脳 5 = 4 train, 1 val
            train_images = list((output_dir / "train").glob("*.jpg"))
            val_images = list((output_dir / "val").glob("*.jpg"))
            self.assertEqual(len(train_images), 4)
            self.assertEqual(len(val_images), 1)

            # Each image has a corresponding X-AnyLabeling JSON file.
            train_labels = list((output_dir / "train").glob("*.json"))
            val_labels = list((output_dir / "val").glob("*.json"))
            self.assertEqual(len(train_labels), 4)
            self.assertEqual(len(val_labels), 1)

            # Verify JSON label format.
            all_labels = train_labels + val_labels
            for lbl_path in all_labels:
                data = json.loads(lbl_path.read_text(encoding="utf-8"))
                self.assertEqual(data["imagePath"], lbl_path.with_suffix(".jpg").name)
                self.assertIn("shapes", data)
                self.assertTrue(all(shape["label"] == "healthy" for shape in data["shapes"]))

    def test_annotate_all_val_when_train_ratio_zero(self):
        """Edge case: train_ratio=0 puts everything in val."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_datas"
            output_dir = tmpdir / "datas"
            input_dir.mkdir()
            Image.new("RGB", (100, 100), (30, 180, 40)).save(input_dir / "img.jpg")

            records = annotate(input_dir=input_dir, output_dir=output_dir, train_ratio=0.0, seed=42)

            train_images = list((output_dir / "train").glob("*.jpg"))
            val_images = list((output_dir / "val").glob("*.jpg"))
            self.assertEqual(len(train_images), 0)
            self.assertEqual(len(val_images), 1)

    def test_annotate_with_multiple_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_datas"
            output_dir = tmpdir / "datas"
            input_dir.mkdir()

            for idx in range(3):
                Image.new("RGB", (80, 80), (30, 180, 40)).save(input_dir / f"plant_{idx}.png")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                train_ratio=1.0,
                exg_threshold=0.1,
                seed=42,
                workers=2,
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(len(list((output_dir / "train").glob("*.json"))), 3)

    def test_annotate_writes_multiple_instance_polygons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_datas"
            output_dir = tmpdir / "datas"
            input_dir.mkdir()

            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image[:] = (60, 35, 20)
            image[10:35, 10:40] = (30, 180, 40)
            image[55:90, 70:110] = (30, 180, 40)
            Image.fromarray(image, "RGB").save(input_dir / "multi.jpg")

            annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                train_ratio=0.0,
                exg_threshold=0.1,
                close_kernel_ratio=0.0,
                open_kernel_ratio=0.0,
                min_area_ratio=0.001,
                max_instances=0,
                seed=42,
            )

            data = json.loads((output_dir / "val" / "multi.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["shapes"]), 4)
            self.assertEqual(
                [shape["shape_type"] for shape in data["shapes"]],
                ["polygon", "rectangle", "polygon", "rectangle"],
            )
            self.assertTrue(all(shape["label"] == "healthy" for shape in data["shapes"]))

    def test_annotate_xanylabeling_output(self):
        """X-AnyLabeling mode writes image/json pairs under train and val."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_datas"
            output_dir = tmpdir / "dataset"
            input_dir.mkdir()

            Image.new("RGB", (100, 80), (30, 180, 40)).save(input_dir / "healthy_01.jpg")
            Image.new("RGB", (100, 80), (80, 40, 20)).save(input_dir / "dead_01.jpg")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                train_ratio=0.5,
                exg_threshold=0.1,
                seed=1,
            )

            self.assertEqual(len(records), 2)
            self.assertTrue((output_dir / "train").is_dir())
            self.assertTrue((output_dir / "val").is_dir())
            self.assertFalse((output_dir / "images").exists())
            self.assertFalse((output_dir / "labels").exists())
            self.assertFalse((output_dir / "dataset.yaml").exists())
            self.assertTrue((output_dir / "classes.txt").exists())

            json_paths = sorted(output_dir.glob("*/*.json"))
            image_paths = sorted(output_dir.glob("*/*.jpg"))
            self.assertEqual(len(json_paths), 2)
            self.assertEqual(len(image_paths), 2)

            data = json.loads(json_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(data["version"], "4.0.0-beta.7")
            self.assertEqual(data["imageHeight"], 80)
            self.assertEqual(data["imageWidth"], 100)
            self.assertEqual(data["imagePath"], json_paths[0].with_suffix(".jpg").name)
            self.assertEqual(len(data["shapes"]), 2)
            polygon_shape = data["shapes"][0]
            rectangle_shape = data["shapes"][1]
            self.assertEqual(polygon_shape["label"], "healthy")
            self.assertEqual(polygon_shape["shape_type"], "polygon")
            self.assertGreaterEqual(len(polygon_shape["points"]), 3)
            self.assertEqual(rectangle_shape["label"], "healthy")
            self.assertEqual(rectangle_shape["shape_type"], "rectangle")
            self.assertEqual(len(rectangle_shape["points"]), 2)

    def test_annotate_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                annotate(input_dir=Path(tmpdir) / "nonexistent", output_dir=Path(tmpdir) / "out")


if __name__ == "__main__":
    unittest.main()

