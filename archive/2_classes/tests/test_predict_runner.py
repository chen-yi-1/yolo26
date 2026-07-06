from utils.predict_runner import (
    build_common_predict_kwargs,
    build_export_kwargs,
    display_prediction_windows,
    run_interactive_predict,
)


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


def test_run_interactive_predict_holds_window_and_disables_ultralytics_show():
    class FakeResult:
        pass

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [FakeResult()]

    displayed = []

    def predict_kwargs_factory(save):
        return {"save": save, "show": True}

    def display_func(results, window_size=None):
        displayed.extend((results, window_size))

    model = FakeModel()
    result = run_interactive_predict(
        model,
        predict_source="image.jpg",
        save=True,
        predict_kwargs_factory=predict_kwargs_factory,
        hold_show=True,
        display_window_size=(960, 720),
        display_func=display_func,
    )

    assert len(displayed) == 2
    assert displayed[0] == result
    assert displayed[1] == (960, 720)
    assert isinstance(result[0], FakeResult)
    assert model.kwargs == {"source": "image.jpg", "save": True, "show": False}


def test_display_prediction_windows_uses_fixed_window_size(monkeypatch):
    class FakeResult:
        def plot(self):
            return "plotted-image"

    class FakeCv2:
        WINDOW_NORMAL = 0

        def __init__(self):
            self.calls = []

        def namedWindow(self, title, flag):
            self.calls.append(("namedWindow", title, flag))

        def resizeWindow(self, title, width, height):
            self.calls.append(("resizeWindow", title, width, height))

        def imshow(self, title, image):
            self.calls.append(("imshow", title, image))

        def waitKey(self, delay):
            self.calls.append(("waitKey", delay))

        def destroyAllWindows(self):
            self.calls.append(("destroyAllWindows",))

    fake_cv2 = FakeCv2()
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    display_prediction_windows([FakeResult()], window_size=(960, 720))

    assert fake_cv2.calls == [
        ("namedWindow", "YOLO26 prediction", fake_cv2.WINDOW_NORMAL),
        ("resizeWindow", "YOLO26 prediction", 960, 720),
        ("imshow", "YOLO26 prediction", "plotted-image"),
        ("waitKey", 0),
        ("destroyAllWindows",),
    ]
