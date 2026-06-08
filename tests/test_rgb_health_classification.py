import csv
import tempfile
import unittest
from pathlib import Path

from scripts.classify_rgb_seedling_health import classify_metrics_csv


class RGBHealthClassificationTests(unittest.TestCase):
    def test_classify_metrics_csv_writes_directly_usable_statuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_csv = tmpdir / "metrics.csv"
            output_csv = tmpdir / "health.csv"

            rows = [
                self._row("healthy.jpg", 0.55, 0.28, 0.18, 0.16, 0.20, 0.05),
                self._row("yellow.jpg", 0.28, 0.14, 0.06, 0.04, 0.04, 0.17),
                self._row("dead.jpg", 0.03, 0.08, 0.02, 0.01, 0.01, 0.20),
                self._row("overgrown.jpg", 0.88, 0.24, 0.14, 0.11, 0.14, 0.08),
            ]
            with input_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            classified = classify_metrics_csv(input_csv, output_csv)

            self.assertEqual(len(classified), 4)
            statuses = {row["image_path"]: row["health_status"] for row in classified}
            self.assertEqual(statuses["healthy.jpg"], "healthy")
            self.assertEqual(statuses["yellow.jpg"], "wilted_yellowing")
            self.assertEqual(statuses["dead.jpg"], "dead_rotten")
            self.assertEqual(statuses["overgrown.jpg"], "overgrown")
            self.assertTrue(output_csv.exists())
            self.assertIn("confidence", classified[0])
            self.assertIn("reason", classified[0])

    def _row(self, image_path, coverage, exg, gli, ngrdi, vari, exr):
        return {
            "image_path": image_path,
            "width": "2",
            "height": "2",
            "vegetation_coverage": str(coverage),
            "ExG_veg_mean": str(exg),
            "GLI_veg_mean": str(gli),
            "NGRDI_veg_mean": str(ngrdi),
            "VARI_veg_mean": str(vari),
            "ExR_veg_mean": str(exr),
        }


if __name__ == "__main__":
    unittest.main()
