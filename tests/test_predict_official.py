import unittest
from unittest.mock import patch

import predict


class OfficialPredictTests(unittest.TestCase):
    def test_common_predict_kwargs_omits_none_project_to_preserve_ultralytics_paths(self):
        kwargs = predict.common_predict_kwargs(save=True)

        self.assertEqual(kwargs["imgsz"], predict.input_shape[0])
        self.assertEqual(kwargs["conf"], predict.confidence)
        self.assertTrue(kwargs["save"])
        self.assertNotIn("project", kwargs)

    def test_run_dir_predict_calls_ultralytics_predict(self):
        with patch("predict.YOLO") as yolo_cls:
            predict.run_mode("dir_predict")

        yolo_cls.assert_called_once_with(predict.model_path)
        yolo_cls.return_value.predict.assert_called_once()
        kwargs = yolo_cls.return_value.predict.call_args.kwargs
        self.assertEqual(kwargs["source"], predict.dir_origin_path)
        self.assertTrue(kwargs["save"])
        self.assertEqual(kwargs["imgsz"], predict.input_shape[0])

    def test_run_export_onnx_calls_ultralytics_export(self):
        with patch("predict.YOLO") as yolo_cls:
            predict.run_mode("export_onnx")

        yolo_cls.assert_called_once_with(predict.model_path)
        yolo_cls.return_value.export.assert_called_once()
        kwargs = yolo_cls.return_value.export.call_args.kwargs
        self.assertEqual(kwargs["format"], "onnx")
        self.assertEqual(kwargs["imgsz"], predict.input_shape[0])
        self.assertEqual(kwargs["simplify"], predict.simplify)


if __name__ == "__main__":
    unittest.main()
