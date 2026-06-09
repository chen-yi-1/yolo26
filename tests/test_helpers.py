import unittest
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw, ImageFont


class HelperTests(unittest.TestCase):
    def test_measure_text_uses_pillow_textbbox(self):
        from utils.utils import measure_text

        image = Image.new("RGB", (100, 50))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        width, height = measure_text(draw, "healthy 0.90", font)

        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_torch_load_patch_is_restored(self):
        from train import torch_load_weights_only_false

        original_load = torch.load
        with torch_load_weights_only_false():
            self.assertIsNot(torch.load, original_load)

        self.assertIs(torch.load, original_load)

    def test_fresh_phase_names_are_split(self):
        from train import phase_train_names

        freeze_name, unfreeze_name = phase_train_names("train_20260608_010203", False)

        self.assertEqual(freeze_name, "train_20260608_010203_freeze")
        self.assertEqual(unfreeze_name, "train_20260608_010203_unfreeze")

    def test_resume_phase_names_reuse_checkpoint_run(self):
        from train import phase_train_names

        freeze_name, unfreeze_name = phase_train_names("train_20260608_010203_unfreeze", True)

        self.assertEqual(freeze_name, "train_20260608_010203_unfreeze")
        self.assertEqual(unfreeze_name, "train_20260608_010203_unfreeze")

    def test_phase2_epochs_are_remaining_not_total(self):
        from train import phase2_epochs

        self.assertEqual(phase2_epochs(0, 50, 100, True), 50)
        self.assertEqual(phase2_epochs(20, 50, 100, True), 50)
        self.assertEqual(phase2_epochs(60, 50, 100, True), 40)
        self.assertEqual(phase2_epochs(20, 50, 100, False), 80)

    def test_torch_load_patch_forces_weights_only_false(self):
        from train import torch_load_weights_only_false

        def fake_load(*args, **kwargs):
            return kwargs

        with patch.object(torch, "load", fake_load):
            with torch_load_weights_only_false():
                kwargs = torch.load("checkpoint.pt", weights_only=True)

        self.assertIs(kwargs["weights_only"], False)

    def test_default_map_model_path_falls_back(self):
        from get_map import default_model_path

        with patch("glob.glob", return_value=[]):
            self.assertEqual(default_model_path(), "model_data/yolo26x-seg.pt")

    def test_default_map_model_path_uses_latest_unfreeze_best(self):
        from get_map import default_model_path

        paths = [
            "runs/segment/logs/a_unfreeze/weights/best.pt",
            "runs/segment/logs/b_unfreeze/weights/best.pt",
        ]
        mtimes = {
            paths[0]: 1,
            paths[1]: 2,
        }
        with patch("glob.glob", return_value=paths), patch("os.path.getmtime", side_effect=mtimes.get):
            self.assertEqual(default_model_path(), paths[1])


if __name__ == "__main__":
    unittest.main()
