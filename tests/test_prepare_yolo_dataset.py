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
    def test_prepare_yolo_dataset_from_edited_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "dataset"
            output = root / "datasets"
            yaml_path = output / "datasets.yaml"

            (source / "train").mkdir(parents=True)
            (source / "val").mkdir()
            (source / "labels").mkdir()
            (source / "classes.txt").write_text("healthy\nabnormal\n", encoding="utf-8")

            Image.new("RGB", (20, 20), (30, 180, 40)).save(source / "train" / "plant_train.JPG")
            Image.new("RGB", (20, 20), (30, 180, 40)).save(source / "val" / "plant_val.JPG")
            (source / "labels" / "plant_train.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9\n",
                encoding="utf-8",
            )
            (source / "labels" / "plant_val.txt").write_text(
                "1 0.2 0.2 0.8 0.2 0.8 0.8\n",
                encoding="utf-8",
            )

            result = prepare_yolo_dataset(source, output, yaml_path)

            self.assertEqual(result["train"], 1)
            self.assertEqual(result["val"], 1)
            self.assertTrue((output / "images" / "train" / "plant_train.JPG").exists())
            self.assertTrue((output / "labels" / "train" / "plant_train.txt").exists())
            self.assertTrue((output / "images" / "val" / "plant_val.JPG").exists())
            self.assertTrue((output / "labels" / "val" / "plant_val.txt").exists())

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

    def test_prepare_detection_dataset_from_xanylabeling_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "dataset"
            output = root / "datasets"
            yaml_path = output / "datasets.yaml"

            (source / "train").mkdir(parents=True)
            (source / "val").mkdir()
            (source / "classes.txt").write_text("healthy\nabnormal\n", encoding="utf-8")

            Image.new("RGB", (100, 80), (30, 180, 40)).save(source / "train" / "plant_train.jpg")
            Image.new("RGB", (100, 80), (30, 180, 40)).save(source / "val" / "plant_val.jpg")

            label_data = {
                "imageWidth": 100,
                "imageHeight": 80,
                "shapes": [
                    {
                        "label": "healthy",
                        "shape_type": "polygon",
                        "points": [[10, 20], [40, 20], [40, 60]],
                    },
                    {
                        "label": "healthy",
                        "shape_type": "rectangle",
                        "points": [[10, 20], [50, 20], [50, 60], [10, 60]],
                    },
                ],
            }
            (source / "train" / "plant_train.json").write_text(
                json.dumps(label_data),
                encoding="utf-8",
            )
            (source / "val" / "plant_val.json").write_text(
                json.dumps(label_data),
                encoding="utf-8",
            )

            result = prepare_yolo_dataset(source, output, yaml_path, task="detect")

            self.assertEqual(result["task"], "detect")
            self.assertEqual(result["train"], 1)
            label_text = (output / "labels" / "train" / "plant_train.txt").read_text(encoding="utf-8")
            self.assertEqual(label_text, "0 0.300000 0.500000 0.400000 0.500000\n")
            validate_detection_label(output / "labels" / "train" / "plant_train.txt", class_count=2)


if __name__ == "__main__":
    unittest.main()
