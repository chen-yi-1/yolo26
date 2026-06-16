import os
import unittest
from unittest.mock import patch


class HelperTests(unittest.TestCase):
    def test_default_map_model_path_falls_back_for_segment(self):
        from get_map import default_model_path

        with patch("glob.glob", return_value=[]):
            self.assertEqual(default_model_path("segment"), "model_data/yolo26n-seg.pt")

    def test_default_map_model_path_falls_back_for_detect(self):
        from get_map import default_model_path

        with patch("glob.glob", return_value=[]):
            self.assertEqual(default_model_path("detect"), "model_data/yolo26n.pt")

    def test_default_map_model_path_uses_latest_official_best(self):
        from get_map import default_model_path

        official_pattern = os.path.join("runs", "segment", "*", "weights", "best.pt")
        old_pattern = os.path.join("runs", "segment", "logs", "*_unfreeze", "weights", "best.pt")
        paths = [
            os.path.join("runs", "segment", "train", "weights", "best.pt"),
            os.path.join("runs", "segment", "train2", "weights", "best.pt"),
        ]
        mtimes = {
            paths[0]: 1,
            paths[1]: 2,
        }

        def glob_side_effect(pattern):
            if pattern == official_pattern:
                return paths
            if pattern == old_pattern:
                return []
            self.fail(f"Unexpected glob pattern: {pattern}")

        with patch("glob.glob", side_effect=glob_side_effect), \
             patch("os.path.getmtime", side_effect=lambda path: mtimes[path]):
            self.assertEqual(default_model_path("segment"), paths[1])


if __name__ == "__main__":
    unittest.main()
