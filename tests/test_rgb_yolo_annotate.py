import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.rgb_yolo_annotate import (
    CLASS_NAMES,
    annotate,
    calculate_indices,
    classify_record,
    compute_batch_scores,
    compute_image_stats,
    create_vegetation_mask,
    convert_yolo_dataset_to_xanylabeling,
    extract_yolo_bbox,
    load_rgb_float,
    percentile_ranks,
)


class RGBYoloAnnotateTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Index calculation (same math as rgb_vegetation_metrics)
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
    # Bbox extraction
    # ------------------------------------------------------------------

    def test_extract_bbox_full_mask(self):
        mask = np.ones((100, 200), dtype=bool)
        cx, cy, w, h = extract_yolo_bbox(mask)
        # Full mask: x1=0, x2=199, y1=0, y2=99
        # cx = (0+199+1)/2/200 = 0.5, cy = (0+99+1)/2/100 = 0.5
        self.assertAlmostEqual(cx, 0.5, places=3)
        self.assertAlmostEqual(cy, 0.5, places=3)
        self.assertAlmostEqual(w, 1.0, places=3)
        self.assertAlmostEqual(h, 1.0, places=3)

    def test_extract_bbox_partial_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:60, 40:120] = True  # rows 20-59, cols 40-119
        cx, cy, w, h = extract_yolo_bbox(mask)
        # x1=40, x2=119 → cx=(40+119+1)/2/200=0.4, w=(119-40+1)/200=0.4
        self.assertAlmostEqual(cx, 0.4, places=3)
        self.assertAlmostEqual(cy, 0.4, places=3)
        self.assertAlmostEqual(w, 0.4, places=3)
        self.assertAlmostEqual(h, 0.4, places=3)

    def test_extract_bbox_empty_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        self.assertIsNone(extract_yolo_bbox(mask))

    def test_extract_bbox_single_pixel(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[25, 25] = True
        cx, cy, w, h = extract_yolo_bbox(mask)
        # x1=x2=25, cx=(25+25+1)/2/50=0.51, w=1/50=0.02
        self.assertAlmostEqual(cx, 0.51, places=3)
        self.assertAlmostEqual(cy, 0.51, places=3)
        self.assertAlmostEqual(w, 0.02, places=3)
        self.assertAlmostEqual(h, 0.02, places=3)

    # ------------------------------------------------------------------
    # Per-image statistics
    # ------------------------------------------------------------------

    def test_compute_image_stats_coverage(self):
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        rgb[..., 1] = 0.8  # green
        indices = calculate_indices(rgb)
        mask = create_vegetation_mask(indices, threshold=0.1)

        stats = compute_image_stats(indices, mask)

        # Everything should be vegetation since G is high
        self.assertAlmostEqual(stats["vegetation_coverage"], 1.0, places=3)
        self.assertIn("ExG_veg_mean", stats)
        self.assertIn("ExR_veg_std", stats)

    def test_compute_image_stats_no_vegetation(self):
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.1  # dark, low green
        indices = calculate_indices(rgb)
        mask = create_vegetation_mask(indices, threshold=0.9)

        stats = compute_image_stats(indices, mask)
        self.assertEqual(stats["vegetation_coverage"], 0.0)
        # All veg means should be 0 since mask is empty
        self.assertEqual(stats["ExG_veg_mean"], 0.0)

    # ------------------------------------------------------------------
    # Percentile ranks
    # ------------------------------------------------------------------

    def test_percentile_ranks_uniform(self):
        vals = np.array([1.0, 1.0, 1.0])
        ranks = percentile_ranks(vals)
        # Ties get averaged rank
        self.assertTrue(np.allclose(ranks, 0.5))

    def test_percentile_ranks_sorted(self):
        vals = np.array([0.0, 0.5, 1.0])
        ranks = percentile_ranks(vals)
        self.assertAlmostEqual(ranks[0], 0.0)
        self.assertAlmostEqual(ranks[1], 0.5)
        self.assertAlmostEqual(ranks[2], 1.0)

    def test_percentile_ranks_single_value(self):
        vals = np.array([42.0])
        ranks = percentile_ranks(vals)
        self.assertEqual(ranks[0], 1.0)

    def test_percentile_ranks_empty(self):
        vals = np.array([], dtype=np.float64)
        ranks = percentile_ranks(vals)
        self.assertEqual(len(ranks), 0)

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def _make_record(self, coverage, exg_mean, gli_mean, ngrdi_mean, vari_mean, exr_mean):
        return {
            "stats": {
                "vegetation_coverage": coverage,
                "ExG_veg_mean": exg_mean,
                "GLI_veg_mean": gli_mean,
                "NGRDI_veg_mean": ngrdi_mean,
                "VARI_veg_mean": vari_mean,
                "ExR_veg_mean": exr_mean,
            }
        }

    def test_classify_separates_extremes(self):
        """Clear extremes (dead vs healthy) get different class_ids."""
        records = [
            self._make_record(-0.05, -0.1, -0.05, -0.05, -0.1, 0.3)
            for _ in range(8)
        ] + [
            self._make_record(0.6, 0.4, 0.3, 0.3, 0.25, -0.2)
            for _ in range(8)
        ]
        compute_batch_scores(records)
        for rec in records:
            classify_record(rec)

        classes = [r["class_id"] for r in records]
        # First 8 (worst) and last 8 (best) should be different
        worst_class = classes[0]
        best_class = classes[-1]
        self.assertNotEqual(worst_class, best_class,
                            f"Extremes should get different classes, got {worst_class} for both")
        # All class_ids should be valid
        for cid in classes:
            self.assertIn(cid, range(3),
                          f"Invalid class_id {cid}; {dict(enumerate(classes))}")

    def test_classify_middle_is_subhealthy(self):
        """Middle-of-batch records → subhealthy."""
        records = [
            self._make_record(-0.05, -0.2, -0.1, -0.1, -0.15, 0.4)
            for _ in range(3)
        ] + [
            self._make_record(0.2, 0.05, 0.03, 0.03, 0.02, 0.05)
            for _ in range(4)
        ] + [
            self._make_record(0.5, 0.35, 0.25, 0.25, 0.20, -0.15)
            for _ in range(3)
        ]
        compute_batch_scores(records)
        for rec in records:
            classify_record(rec)

        classes = [r["class_id"] for r in records]
        # Middle group should be subhealthy
        middle_classes = classes[3:7]
        self.assertTrue(
            all(c == 1 for c in middle_classes),
            f"Middle group should all be subhealthy (1), got {middle_classes}",
        )

    def test_classify_zero_coverage_is_dead(self):
        """No vegetation pixels → dead_rotten regardless of other scores."""
        rec = self._make_record(coverage=0.0, exg_mean=0.0, gli_mean=0.0, ngrdi_mean=0.0, vari_mean=0.0, exr_mean=0.0)
        rec["green_score"] = 0.9
        rec["coverage_rank"] = 0.9
        rec["red_rank"] = 0.1
        classify_record(rec)
        self.assertEqual(rec["class_id"], 2)  # unhealthy

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
            # Subhealthy: pale green
            Image.new("RGB", (200, 200), (140, 170, 100)).save(input_dir / "subhealthy_01.jpg")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                train_ratio=0.8,
                exg_threshold=0.1,
                seed=42,
                copy_mode="copy",
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
                "healthy\nsubhealthy\nunhealthy\n",
            )

            # Train/val split: 0.8 × 5 = 4 train, 1 val
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
                copy_mode="copy",
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
            self.assertEqual(len(data["shapes"]), 1)
            shape = data["shapes"][0]
            self.assertIn(shape["label"], CLASS_NAMES.values())
            self.assertEqual(shape["shape_type"], "rectangle")
            self.assertEqual(len(shape["points"]), 4)

    def test_convert_yolo_dataset_to_xanylabeling(self):
        """Existing YOLO dataset is converted to split image/json pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yolo_dir = tmpdir / "dataset"
            output_dir = tmpdir / "dataset_xany"
            (yolo_dir / "images" / "train").mkdir(parents=True)
            (yolo_dir / "labels" / "train").mkdir(parents=True)
            (yolo_dir / "images" / "val").mkdir(parents=True)
            (yolo_dir / "labels" / "val").mkdir(parents=True)

            Image.new("RGB", (100, 80), (30, 180, 40)).save(yolo_dir / "images" / "train" / "plant.jpg")
            (yolo_dir / "labels" / "train" / "plant.txt").write_text(
                "1 0.500000 0.500000 0.400000 0.250000\n",
                encoding="utf-8",
            )
            Image.new("RGB", (50, 40), (80, 40, 20)).save(yolo_dir / "images" / "val" / "empty.jpg")
            (yolo_dir / "labels" / "val" / "empty.txt").write_text("", encoding="utf-8")
            (yolo_dir / "classes.txt").write_text("healthy\nsubhealthy\nunhealthy\n", encoding="utf-8")

            records = convert_yolo_dataset_to_xanylabeling(yolo_dir, output_dir)

            self.assertEqual(len(records), 2)
            self.assertTrue((output_dir / "train" / "plant.jpg").exists())
            self.assertTrue((output_dir / "train" / "plant.json").exists())
            self.assertTrue((output_dir / "val" / "empty.jpg").exists())
            self.assertTrue((output_dir / "val" / "empty.json").exists())
            self.assertEqual(
                (output_dir / "classes.txt").read_text(encoding="utf-8"),
                "healthy\nsubhealthy\nunhealthy\n",
            )

            plant = json.loads((output_dir / "train" / "plant.json").read_text(encoding="utf-8"))
            self.assertEqual(plant["imagePath"], "plant.jpg")
            self.assertEqual(plant["imageWidth"], 100)
            self.assertEqual(plant["imageHeight"], 80)
            self.assertEqual(plant["shapes"][0]["label"], "subhealthy")
            self.assertEqual(
                plant["shapes"][0]["points"],
                [[30.0, 30.0], [70.0, 30.0], [70.0, 50.0], [30.0, 50.0]],
            )

            empty = json.loads((output_dir / "val" / "empty.json").read_text(encoding="utf-8"))
            self.assertEqual(empty["shapes"], [])

    def test_annotate_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                annotate(input_dir=Path(tmpdir) / "nonexistent", output_dir=Path(tmpdir) / "out")


if __name__ == "__main__":
    unittest.main()
