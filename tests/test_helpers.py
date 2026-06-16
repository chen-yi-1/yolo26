import os
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

    def test_resume_phase_names_keep_freeze_and_split_unfreeze(self):
        from train import phase_train_names

        freeze_name, unfreeze_name = phase_train_names("train_20260608_010203_unfreeze", True)

        self.assertEqual(freeze_name, "train_20260608_010203_unfreeze")
        self.assertEqual(unfreeze_name, "train_20260608_010203_unfreeze")

    def test_resume_phase_names_advance_from_freeze_to_unfreeze(self):
        from train import phase_train_names

        freeze_name, unfreeze_name = phase_train_names("train_20260608_010203_freeze", True)

        self.assertEqual(freeze_name, "train_20260608_010203_freeze")
        self.assertEqual(unfreeze_name, "train_20260608_010203_unfreeze")

    def test_resume_checkpoint_prefers_phase_last_in_latest_run(self):
        from train import find_latest_resume_checkpoint

        dirs = ["old_unfreeze", "new_freeze"]
        files = {
            os.path.normpath("runs/segment/logs/old_unfreeze/weights/last.pt"),
            os.path.normpath("runs/segment/logs/new_freeze/weights/last.pt"),
            os.path.normpath("runs/segment/logs/new_freeze/weights/phase_last.pt"),
        }
        mtimes = {
            os.path.normpath("runs/segment/logs/old_unfreeze/weights/last.pt"): 10,
            os.path.normpath("runs/segment/logs/new_freeze/weights/last.pt"): 20,
            os.path.normpath("runs/segment/logs/new_freeze/weights/phase_last.pt"): 19,
        }

        with patch("os.path.isdir", return_value=True), \
             patch("os.listdir", return_value=dirs), \
             patch("os.path.isfile", side_effect=lambda path: os.path.normpath(path) in files), \
             patch("os.path.getmtime", side_effect=lambda path: mtimes[os.path.normpath(path)]):
            self.assertEqual(
                os.path.normpath(find_latest_resume_checkpoint("runs/segment/logs")),
                os.path.normpath("runs/segment/logs/new_freeze/weights/phase_last.pt"),
            )

    def test_phase2_from_completed_freeze_loads_weights_without_resume(self):
        from train import phase2_checkpoint

        checkpoint, resume = phase2_checkpoint(
            freeze_save_dir=None,
            model_path="runs/segment/logs/train_20260608_010203_freeze/weights/phase_last.pt",
            is_resuming=True,
            train_name="train_20260608_010203_freeze",
            init_epoch=50,
            freeze_epoch=50,
            freeze_train=True,
        )

        self.assertEqual(
            checkpoint,
            "runs/segment/logs/train_20260608_010203_freeze/weights/phase_last.pt",
        )
        self.assertFalse(resume)

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
            self.assertEqual(default_model_path(), "model_data/yolo26n-seg.pt")

    def test_default_map_model_path_falls_back_for_detect(self):
        from get_map import default_model_path

        with patch("glob.glob", return_value=[]):
            self.assertEqual(default_model_path("detect"), "model_data/yolo26n.pt")

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
