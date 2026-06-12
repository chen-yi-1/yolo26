import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from scripts.prepare_yolo_dataset import (
    prepare_yolo_dataset,
    validate_detection_label,
    validate_segmentation_label,
)


class PrepareYoloDatasetTests(unittest.TestCase):
    def test_prepare_yolo_dataset_from_labels_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "dataset"
            output = root / "datasets"
            yaml_path = output / "datasets.yaml"

            (source / "labels").mkdir(parents=True)
            (source / "healthy").mkdir()
            (source / "abnormal").mkdir()
            (source / "classes.txt").write_text("healthy\nabnormal\n", encoding="utf-8")

            Image.new("RGB", (20, 20), (30, 180, 40)).save(source / "healthy" / "plant_train.jpg")
            Image.new("RGB", (20, 20), (30, 180, 40)).save(source / "abnormal" / "plant_val.jpg")
            (source / "healthy" / "plant_train.json").write_text('{"imageWidth": 20, "imageHeight": 20, "shapes": []}', encoding="utf-8")
            (source / "abnormal" / "plant_val.json").write_text('{"imageWidth": 20, "imageHeight": 20, "shapes": []}', encoding="utf-8")

            (source / "labels" / "plant_train.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9\n",
                encoding="utf-8",
            )
            (source / "labels" / "plant_val.txt").write_text(
                "1 0.2 0.2 0.8 0.2 0.8 0.8\n",
                encoding="utf-8",
            )

            result = prepare_yolo_dataset(source, output, yaml_path, train_ratio=1.0, seed=7)

            self.assertEqual(result["train"], 2)
            self.assertEqual(result["val"], 0)
            self.assertTrue((output / "images" / "train" / "plant_train.jpg").exists())
            self.assertTrue((output / "labels" / "train" / "plant_train.txt").exists())
            self.assertTrue((output / "images" / "train" / "plant_val.jpg").exists())
            self.assertTrue((output / "labels" / "train" / "plant_val.txt").exists())
            self.assertFalse((output / "images" / "train" / "plant_train.json").exists())
            self.assertTrue((source / "healthy" / "plant_train.json").exists())
            self.assertTrue((source / "abnormal" / "plant_val.json").exists())

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            self.assertEqual(data["path"], str(output.resolve()).replace("\\", "/"))
            self.assertEqual(data["train"], "images/train")
            self.assertEqual(data["val"], "images/val")
            self.assertEqual(data["nc"], 2)
            self.assertEqual(data["names"], {0: "healthy", 1: "abnormal"})

    def test_validate_segmentation_label_rejects_bbox_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / "bad.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_segmentation_label(label_path, class_count=2)

    def test_prepare_detection_dataset_from_labels_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "dataset"
            output = root / "datasets"
            yaml_path = output / "datasets.yaml"

            (source / "labels").mkdir(parents=True)
            (source / "healthy").mkdir()
            (source / "abnormal").mkdir()
            (source / "classes.txt").write_text("healthy\nabnormal\n", encoding="utf-8")

            Image.new("RGB", (100, 80), (30, 180, 40)).save(source / "healthy" / "plant_train.jpg")
            Image.new("RGB", (100, 80), (30, 180, 40)).save(source / "abnormal" / "plant_val.jpg")
            (source / "healthy" / "plant_train.json").write_text('{"imageWidth": 100, "imageHeight": 80, "shapes": []}', encoding="utf-8")
            (source / "abnormal" / "plant_val.json").write_text('{"imageWidth": 100, "imageHeight": 80, "shapes": []}', encoding="utf-8")

            (source / "labels" / "plant_train.txt").write_text(
                "0 0.300000 0.500000 0.400000 0.500000\n",
                encoding="utf-8",
            )
            (source / "labels" / "plant_val.txt").write_text(
                "1 0.250000 0.500000 0.200000 0.300000\n",
                encoding="utf-8",
            )

            result = prepare_yolo_dataset(source, output, yaml_path, task="detect", train_ratio=1.0)

            self.assertEqual(result["task"], "detect")
            self.assertEqual(result["train"], 2)
            label_text = (output / "labels" / "train" / "plant_train.txt").read_text(encoding="utf-8")
            self.assertEqual(label_text, "0 0.300000 0.500000 0.400000 0.500000\n")
            validate_detection_label(output / "labels" / "train" / "plant_train.txt", class_count=2)
            self.assertFalse((output / "images" / "train" / "plant_train.json").exists())
            self.assertTrue((source / "healthy" / "plant_train.json").exists())

    def test_prepare_dataset_samples_then_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "dataset"
            output = root / "datasets"
            yaml_path = output / "datasets.yaml"

            (source / "labels").mkdir(parents=True)
            (source / "healthy").mkdir()
            (source / "abnormal").mkdir()
            (source / "classes.txt").write_text("healthy\nabnormal\n", encoding="utf-8")

            for index in range(4):
                image_name = f"plant_{index}.jpg"
                class_dir = source / ("healthy" if index % 2 == 0 else "abnormal")
                Image.new("RGB", (100, 80), (30, 180, 40)).save(class_dir / image_name)
                (source / "labels" / f"plant_{index}.txt").write_text(
                    "0 0.1 0.1 0.9 0.1 0.9 0.9\n",
                    encoding="utf-8",
                )

            result = prepare_yolo_dataset(
                source,
                output,
                yaml_path,
                task="segment",
                sample_percent=50,
                train_ratio=0.8,
                seed=7,
            )

            self.assertEqual(result["train"], 1)
            self.assertEqual(result["val"], 1)
            self.assertEqual(len(list((output / "images" / "train").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output / "images" / "val").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output / "labels" / "train").glob("*.txt"))), 1)
            self.assertEqual(len(list((output / "labels" / "val").glob("*.txt"))), 1)

    def test_prepare_yolo_dataset_rejects_missing_labels_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "dataset"
            output = root / "datasets"
            source.mkdir(parents=True)
            (source / "healthy").mkdir()
            (source / "classes.txt").write_text("healthy\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "labels"):
                prepare_yolo_dataset(source, output, train_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
