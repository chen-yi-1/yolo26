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
    extract_mask_instances,
    extract_mask_polygon,
    extract_mask_polygons,
    fuse_aux_model_with_rgb_mask,
    load_rgb_float,
    percentile_ranks,
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

    def test_aux_model_fusion_uses_intersection_not_raw_model_mask(self):
        rgb_mask = np.zeros((30, 40), dtype=bool)
        rgb_mask[8:18, 12:25] = True
        model_mask = np.zeros((30, 40), dtype=bool)
        model_mask[2:25, 4:35] = True
        score = np.zeros((30, 40), dtype=np.float64)
        score[rgb_mask] = 0.25

        instances, accepted_union = fuse_aux_model_with_rgb_mask(
            [{"mask": model_mask, "model_confidence": 0.9}],
            rgb_mask,
            score_map=score,
            min_overlap_ratio=0.01,
            min_component_score=0.12,
            close_kernel_ratio=0.0,
            open_kernel_ratio=0.0,
            min_area_ratio=0.001,
            max_instances=0,
        )

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["source"], "fused_model_rgb")
        self.assertEqual(int(np.sum(instances[0]["mask"])), int(np.sum(rgb_mask)))
        self.assertLess(int(np.sum(instances[0]["mask"])), int(np.sum(model_mask)))
        self.assertEqual(int(np.sum(accepted_union)), int(np.sum(rgb_mask)))

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
            self.assertIn(cid, range(2),
                          f"Invalid class_id {cid}; {dict(enumerate(classes))}")

    def test_classify_middle_is_abnormal(self):
        """Middle-of-batch records are abnormal."""
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
        # Middle group should be abnormal.
        middle_classes = classes[3:7]
        self.assertTrue(
            all(c == 1 for c in middle_classes),
            f"Middle group should all be abnormal (1), got {middle_classes}",
        )

    def test_classify_zero_coverage_is_dead(self):
        """No vegetation pixels 鈫?dead_rotten regardless of other scores."""
        rec = self._make_record(coverage=0.0, exg_mean=0.0, gli_mean=0.0, ngrdi_mean=0.0, vari_mean=0.0, exr_mean=0.0)
        rec["green_score"] = 0.9
        rec["coverage_rank"] = 0.9
        rec["red_rank"] = 0.1
        classify_record(rec)
        self.assertEqual(rec["class_id"], 1)  # abnormal

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
            self.assertEqual(len(data["shapes"]), 2)
            self.assertTrue(all(shape["shape_type"] == "polygon" for shape in data["shapes"]))
            self.assertTrue(all(shape["label"] in CLASS_NAMES.values() for shape in data["shapes"]))

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
            self.assertEqual(shape["shape_type"], "polygon")
            self.assertGreaterEqual(len(shape["points"]), 3)

    def test_annotate_missing_input_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                annotate(input_dir=Path(tmpdir) / "nonexistent", output_dir=Path(tmpdir) / "out")

    def test_annotate_can_disable_aux_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "raw_datas"
            output_dir = tmpdir / "dataset"
            input_dir.mkdir()

            Image.new("RGB", (100, 80), (30, 180, 40)).save(input_dir / "healthy_01.jpg")

            records = annotate(
                input_dir=input_dir,
                output_dir=output_dir,
                aux_model_path=tmpdir / "missing-seg.pt",
                use_aux_model=False,
                train_ratio=0.0,
                seed=1,
            )

            self.assertEqual(len(records), 1)
            self.assertTrue((output_dir / "classes.txt").exists())


if __name__ == "__main__":
    unittest.main()

