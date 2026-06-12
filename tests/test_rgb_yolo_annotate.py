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
            input_dir = tmpdir / "raw_data"
            output_dir = tmpdir / "datas"
            class_dir = input_dir / "healthy"
            class_dir.mkdir(parents=True)

            # Create 5 synthetic images simulating different seedling conditions
            # Healthy: bright green
            Image.new("RGB", (200, 200), (30, 180, 40)).save(class_dir / "healthy_01.jpg")
            # Dead: brown/black
            Image.new("RGB", (200, 200), (80, 40, 20)).save(class_dir / "dead_01.jpg")
            # Wilted: yellowish
            Image.new("RGB", (200, 200), (200, 180, 30)).save(class_dir / "wilted_01.jpg")
            # Overgrown: large green area
            Image.new("RGB", (200, 200), (20, 150, 35)).save(class_dir / "overgrown_01.jpg")
            # Abnormal: pale green
            Image.new("RGB", (200, 200), (140, 170, 100)).save(class_dir / "abnormal_01.jpg")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                exg_threshold=0.1,
            )

            # Check record count
            self.assertEqual(len(records), 5)

            # Check default output structure: X-AnyLabeling image/json pairs mirror input folders.
            self.assertTrue((output_dir / "healthy").is_dir())
            self.assertFalse((output_dir / "train").exists())
            self.assertFalse((output_dir / "val").exists())
            self.assertFalse((output_dir / "dataset.yaml").exists())
            self.assertFalse((output_dir / "images").exists())
            self.assertFalse((output_dir / "labels").exists())
            self.assertEqual(
                (output_dir / "classes.txt").read_text(encoding="utf-8"),
                "healthy\n",
            )

            output_images = list((output_dir / "healthy").glob("*.jpg"))
            self.assertEqual(len(output_images), 5)

            # Each image has a corresponding X-AnyLabeling JSON file.
            output_labels = list((output_dir / "healthy").glob("*.json"))
            self.assertEqual(len(output_labels), 5)

            # Verify JSON label format.
            for lbl_path in output_labels:
                data = json.loads(lbl_path.read_text(encoding="utf-8"))
                self.assertEqual(data["imagePath"], lbl_path.with_suffix(".jpg").name)
                self.assertIn("shapes", data)
                self.assertTrue(all(shape["label"] == "healthy" for shape in data["shapes"]))

    def test_annotate_uses_input_folder_names_as_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_data"
            output_dir = tmpdir / "datas"
            (input_dir / "abnormal").mkdir(parents=True)
            (input_dir / "healthy").mkdir()

            Image.new("RGB", (100, 100), (30, 180, 40)).save(input_dir / "abnormal" / "abnormal_01.jpg")
            Image.new("RGB", (100, 100), (30, 180, 40)).save(input_dir / "healthy" / "healthy_01.jpg")

            annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                exg_threshold=0.1,
            )

            self.assertEqual(
                (output_dir / "classes.txt").read_text(encoding="utf-8"),
                "abnormal\nhealthy\n",
            )

            abnormal_data = json.loads((output_dir / "abnormal" / "abnormal_01.json").read_text(encoding="utf-8"))
            healthy_data = json.loads((output_dir / "healthy" / "healthy_01.json").read_text(encoding="utf-8"))
            self.assertTrue(all(shape["label"] == "abnormal" for shape in abnormal_data["shapes"]))
            self.assertTrue(all(shape["label"] == "healthy" for shape in healthy_data["shapes"]))

    def test_annotate_writes_mirrored_output_without_train_val_split(self):
        """Output mirrors input folders instead of splitting into train/val."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_data"
            output_dir = tmpdir / "datas"
            class_dir = input_dir / "healthy"
            class_dir.mkdir(parents=True)
            Image.new("RGB", (100, 100), (30, 180, 40)).save(class_dir / "img.jpg")

            records = annotate(input_dir=input_dir, output_dir=output_dir)

            self.assertFalse((output_dir / "train").exists())
            self.assertFalse((output_dir / "val").exists())
            output_images = list((output_dir / "healthy").glob("*.jpg"))
            self.assertEqual(len(output_images), 1)

    def test_annotate_with_multiple_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_data"
            output_dir = tmpdir / "datas"
            class_dir = input_dir / "healthy"
            class_dir.mkdir(parents=True)

            for idx in range(3):
                Image.new("RGB", (80, 80), (30, 180, 40)).save(class_dir / f"plant_{idx}.png")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                exg_threshold=0.1,
                workers=2,
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(len(list((output_dir / "healthy").glob("*.json"))), 3)

    def test_annotate_writes_multiple_instance_polygons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_data"
            output_dir = tmpdir / "datas"
            class_dir = input_dir / "healthy"
            class_dir.mkdir(parents=True)

            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image[:] = (60, 35, 20)
            image[10:35, 10:40] = (30, 180, 40)
            image[55:90, 70:110] = (30, 180, 40)
            Image.fromarray(image, "RGB").save(class_dir / "multi.jpg")

            annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                exg_threshold=0.1,
                close_kernel_ratio=0.0,
                open_kernel_ratio=0.0,
                min_area_ratio=0.001,
                max_instances=0,
            )

            data = json.loads((output_dir / "healthy" / "multi.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["shapes"]), 4)
            self.assertEqual(
                [shape["shape_type"] for shape in data["shapes"]],
                ["polygon", "rectangle", "polygon", "rectangle"],
            )
            self.assertTrue(all(shape["label"] == "healthy" for shape in data["shapes"]))

    def test_annotate_xanylabeling_output(self):
        """X-AnyLabeling mode writes image/json pairs under mirrored input directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_data"
            output_dir = tmpdir / "dataset"
            class_dir = input_dir / "healthy"
            class_dir.mkdir(parents=True)

            Image.new("RGB", (100, 80), (30, 180, 40)).save(class_dir / "healthy_01.jpg")
            Image.new("RGB", (100, 80), (80, 40, 20)).save(class_dir / "dead_01.jpg")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                exg_threshold=0.1,
            )

            self.assertEqual(len(records), 2)
            self.assertTrue((output_dir / "healthy").is_dir())
            self.assertFalse((output_dir / "train").exists())
            self.assertFalse((output_dir / "val").exists())
            self.assertFalse((output_dir / "images").exists())
            self.assertFalse((output_dir / "labels").exists())
            self.assertFalse((output_dir / "dataset.yaml").exists())
            self.assertTrue((output_dir / "classes.txt").exists())

            json_paths = sorted(output_dir.glob("*/*.json"))
            image_paths = sorted(output_dir.glob("*/*.jpg"))
            self.assertEqual(len(json_paths), 2)
            self.assertEqual(len(image_paths), 2)

            healthy_json = output_dir / "healthy" / "healthy_01.json"
            data = json.loads(healthy_json.read_text(encoding="utf-8"))
            self.assertEqual(data["version"], "4.0.0-beta.7")
            self.assertEqual(data["imageHeight"], 80)
            self.assertEqual(data["imageWidth"], 100)
            self.assertEqual(data["imagePath"], healthy_json.with_suffix(".jpg").name)
            self.assertEqual(len(data["shapes"]), 2)
            polygon_shape = data["shapes"][0]
            rectangle_shape = data["shapes"][1]
            self.assertEqual(polygon_shape["label"], "healthy")
            self.assertEqual(polygon_shape["shape_type"], "polygon")
            self.assertGreaterEqual(len(polygon_shape["points"]), 3)
            self.assertEqual(rectangle_shape["label"], "healthy")
            self.assertEqual(rectangle_shape["shape_type"], "rectangle")
            self.assertEqual(len(rectangle_shape["points"]), 4)

    def test_annotate_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                annotate(input_dir=Path(tmpdir) / "nonexistent", output_dir=Path(tmpdir) / "out")

    def test_annotate_without_class_folders_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_data"
            input_dir.mkdir()
            Image.new("RGB", (100, 100), (30, 180, 40)).save(input_dir / "img.jpg")

            with self.assertRaisesRegex(ValueError, "No class folders found"):
                annotate(input_dir=input_dir, output_dir=tmpdir / "out")


if __name__ == "__main__":
    unittest.main()

