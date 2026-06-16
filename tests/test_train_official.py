import unittest
from unittest.mock import patch

import train


class OfficialTrainTests(unittest.TestCase):
    def test_build_train_kwargs_omits_none_project_to_preserve_ultralytics_paths(self):
        kwargs = train.build_train_kwargs(
            data_yaml="datasets/datasets.yaml",
            task="segment",
            imgsz=640,
            epochs=100,
            batch=16,
            device=0,
            workers=4,
            project=None,
            name="train",
            exist_ok=False,
            pretrained=True,
            resume=False,
            optimizer="auto",
            lr0=0.01,
            patience=100,
            save_period=10,
            amp=True,
            freeze=None,
            cache=False,
            plots=True,
            val=True,
            verbose=True,
        )

        self.assertEqual(kwargs["data"], "datasets/datasets.yaml")
        self.assertEqual(kwargs["task"], "segment")
        self.assertEqual(kwargs["name"], "train")
        self.assertNotIn("project", kwargs)
        self.assertNotIn("freeze", kwargs)

    def test_run_training_calls_ultralytics_train(self):
        with patch("train.require_existing_file"), patch("train.YOLO") as yolo_cls:
            train.run_training()

        yolo_cls.assert_called_once_with(train.model_path)
        kwargs = yolo_cls.return_value.train.call_args.kwargs
        self.assertEqual(kwargs["data"], train.data_yaml)
        self.assertEqual(kwargs["task"], train.task)
        self.assertEqual(kwargs["imgsz"], train.input_shape[0])


if __name__ == "__main__":
    unittest.main()
