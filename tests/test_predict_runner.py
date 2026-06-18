from utils.predict_runner import build_common_predict_kwargs, build_export_kwargs


def test_build_common_predict_kwargs_filters_none_project():
    kwargs = build_common_predict_kwargs(
        input_shape=[640, 640],
        confidence=0.5,
        iou=0.7,
        device="cpu",
        save=True,
        show=False,
        save_txt=False,
        save_conf=False,
        name="predict",
        exist_ok=False,
        verbose=True,
        project=None,
    )

    assert kwargs == {
        "imgsz": 640,
        "conf": 0.5,
        "iou": 0.7,
        "device": "cpu",
        "save": True,
        "show": False,
        "save_txt": False,
        "save_conf": False,
        "name": "predict",
        "exist_ok": False,
        "verbose": True,
    }


def test_build_export_kwargs_uses_official_onnx_format():
    kwargs = build_export_kwargs(
        input_shape=[640, 640],
        simplify=True,
        dynamic=False,
        opset=None,
        device=0,
    )

    assert kwargs == {
        "format": "onnx",
        "imgsz": 640,
        "simplify": True,
        "dynamic": False,
        "opset": None,
        "device": 0,
    }
