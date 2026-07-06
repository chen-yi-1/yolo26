# YOLO26 Official Ultralytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace local YOLO26 training/inference internals with official Ultralytics train, val, predict, and export APIs while preserving edit-in-file entry points.

**Architecture:** `train.py`, `get_map.py`, and `predict.py` become thin wrappers around `ultralytics.YOLO`. The wrappers validate local input files and dispatch arguments, but they do not rebuild Ultralytics path handling, dataloaders, loss, NMS, plotting, or export behavior. Obsolete local training/inference modules and tests are removed.

**Tech Stack:** Python, Ultralytics 8.4.x, PyTorch through Ultralytics, pytest/unittest.

---

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `train.py` | Rewrite | File-configured official `YOLO.train()` entry point |
| `get_map.py` | Modify | File-configured official `YOLO.val()` entry point with official-run checkpoint lookup |
| `predict.py` | Rewrite | File-configured official `YOLO.predict()` and `YOLO.export()` entry point |
| `yolo.py` | Delete | Remove local inference wrapper and custom drawing/export |
| `nets/yolo_training.py` | Delete | Remove custom loss, EMA, scheduler |
| `utils/utils_fit.py` | Delete | Remove custom epoch loop |
| `utils/callbacks.py` | Delete | Remove local training loss/mAP callbacks |
| `utils/utils_bbox.py` | Delete | Remove local decode/NMS |
| `utils/utils_map.py` | Delete | Remove local mAP implementation |
| `utils/utils.py` | Delete | Remove utilities only used by deleted local wrappers |
| `tests/test_helpers.py` | Modify | Remove custom-training helper tests; keep/update official validation tests |
| `tests/test_utils_bbox.py` | Delete | Remove tests for deleted local decoder |
| `tests/test_train_official.py` | Create | Verify `train.py` dispatches official training arguments |
| `tests/test_predict_official.py` | Create | Verify `predict.py` dispatches official predict/export calls |
| `README.md` | Modify | Document official workflow |
| `CLAUDE.md` | Modify | Update project instructions to official architecture |

## Task 1: Rewrite `train.py`

**Files:**
- Modify: `train.py`
- Test: `tests/test_train_official.py`

- [ ] **Step 1: Write failing tests for official training dispatch**

Create `tests/test_train_official.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q tests\test_train_official.py
```

Expected: FAIL because `build_train_kwargs()` and `run_training()` do not exist.

- [ ] **Step 3: Replace `train.py` with official wrapper**

Replace `train.py` with:

```python
#-------------------------------------#
#       使用 Ultralytics 官方训练
#-------------------------------------#
from pathlib import Path

from ultralytics import YOLO


def require_existing_file(path, label):
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def build_train_kwargs(
    data_yaml,
    task,
    imgsz,
    epochs,
    batch,
    device,
    workers,
    project,
    name,
    exist_ok,
    pretrained,
    resume,
    optimizer,
    lr0,
    patience,
    save_period,
    amp,
    freeze,
    cache,
    plots,
    val,
    verbose,
):
    kwargs = {
        "data": data_yaml,
        "task": task,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "workers": workers,
        "name": name,
        "exist_ok": exist_ok,
        "pretrained": pretrained,
        "resume": resume,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
        "save_period": save_period,
        "amp": amp,
        "cache": cache,
        "plots": plots,
        "val": val,
        "verbose": verbose,
    }
    if project is not None:
        kwargs["project"] = project
    if freeze is not None:
        kwargs["freeze"] = freeze
    return {key: value for key, value in kwargs.items() if value is not None}


def run_training():
    require_existing_file(data_yaml, "Dataset YAML")
    if any(sep in model_path for sep in ("/", "\\")):
        require_existing_file(model_path, "Model")

    model = YOLO(model_path)
    train_kwargs = build_train_kwargs(
        data_yaml=data_yaml,
        task=task,
        imgsz=input_shape[0],
        epochs=epochs,
        batch=batch,
        device=device,
        workers=num_workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        resume=resume,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        save_period=save_period,
        amp=amp,
        freeze=freeze,
        cache=cache,
        plots=plots,
        val=val,
        verbose=verbose,
    )
    return model.train(**train_kwargs)


if __name__ == "__main__":
    #---------------------------------#
    #   task: detect / segment
    #---------------------------------#
    task = "segment"
    #---------------------------------#
    #   模型与数据配置
    #   model_path 可为本地 .pt 路径或官方模型名。
    #   data_yaml 传给 Ultralytics，不在本地重写其内部路径逻辑。
    #---------------------------------#
    model_path = "model_data/yolo26n-seg.pt"
    data_yaml = "datasets/datasets.yaml"
    input_shape = [640, 640]
    #---------------------------------#
    #   官方训练参数
    #---------------------------------#
    epochs = 100
    batch = 16
    device = 0
    num_workers = 4
    optimizer = "auto"
    lr0 = 0.01
    patience = 100
    save_period = 10
    amp = True
    cache = False
    plots = True
    val = True
    verbose = True
    #---------------------------------#
    #   官方保存路径参数
    #   project=None 时使用 Ultralytics 默认 runs/<task>/<name>。
    #   不要写 project="runs"，否则本版本会生成 runs/<task>/runs/<name>。
    #---------------------------------#
    project = None
    name = "train"
    exist_ok = False
    #---------------------------------#
    #   resume=True 时，model_path 应指向官方 last.pt。
    #   freeze 可为 None、整数或层索引列表，由 Ultralytics 处理。
    #---------------------------------#
    pretrained = True
    resume = False
    freeze = None

    run_training()
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest -q tests\test_train_official.py
```

Expected: PASS.

## Task 2: Rewrite `predict.py` Around Official Prediction/Export

**Files:**
- Modify: `predict.py`
- Test: `tests/test_predict_official.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_predict_official.py`:

```python
import unittest
from unittest.mock import patch

import predict


class OfficialPredictTests(unittest.TestCase):
    def test_common_predict_kwargs_omits_none_project(self):
        kwargs = predict.common_predict_kwargs(save=True)

        self.assertEqual(kwargs["imgsz"], predict.input_shape[0])
        self.assertEqual(kwargs["conf"], predict.confidence)
        self.assertNotIn("project", kwargs)

    def test_dir_predict_calls_official_predict(self):
        with patch("predict.YOLO") as yolo_cls:
            predict.run_mode("dir_predict")

        yolo_cls.assert_called_once_with(predict.model_path)
        kwargs = yolo_cls.return_value.predict.call_args.kwargs
        self.assertEqual(kwargs["source"], predict.dir_origin_path)
        self.assertTrue(kwargs["save"])

    def test_export_onnx_calls_official_export(self):
        with patch("predict.YOLO") as yolo_cls:
            predict.run_mode("export_onnx")

        kwargs = yolo_cls.return_value.export.call_args.kwargs
        self.assertEqual(kwargs["format"], "onnx")
        self.assertEqual(kwargs["imgsz"], predict.input_shape[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest -q tests\test_predict_official.py
```

Expected: FAIL because `common_predict_kwargs()` and `run_mode()` do not exist.

- [ ] **Step 3: Replace `predict.py` with official wrapper**

Replace `predict.py` with:

```python
#-----------------------------------------------------------------------#
#   predict.py — 官方 Ultralytics 推理与导出入口
#   modes: predict / video / fps / dir_predict / export_onnx
#-----------------------------------------------------------------------#
import time

from ultralytics import YOLO


def common_predict_kwargs(save):
    kwargs = {
        "imgsz": input_shape[0],
        "conf": confidence,
        "iou": nms_iou,
        "device": device,
        "save": save,
        "show": show,
        "stream": stream,
        "verbose": verbose,
        "name": name,
        "exist_ok": exist_ok,
    }
    if project is not None:
        kwargs["project"] = project
    return {key: value for key, value in kwargs.items() if value is not None}


def run_predict_loop(model):
    while True:
        image_path = input("Input image filename:")
        if not image_path:
            break
        model.predict(source=image_path, **common_predict_kwargs(save=save))


def run_fps(model):
    kwargs = common_predict_kwargs(save=False)
    kwargs["verbose"] = False
    model.predict(source=fps_image_path, **kwargs)
    start = time.time()
    for _ in range(test_interval):
        model.predict(source=fps_image_path, **kwargs)
    tact_time = (time.time() - start) / test_interval
    print(str(tact_time) + " seconds, " + str(1 / tact_time) + "FPS, @batch_size 1")
    return tact_time


def run_export(model):
    kwargs = {
        "format": "onnx",
        "imgsz": input_shape[0],
        "simplify": simplify,
        "device": device,
    }
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    if opset is not None:
        kwargs["opset"] = opset
    return model.export(**{key: value for key, value in kwargs.items() if value is not None})


def run_mode(selected_mode=None):
    selected_mode = selected_mode or mode
    model = YOLO(model_path)

    if selected_mode == "predict":
        return run_predict_loop(model)
    if selected_mode == "video":
        return model.predict(source=video_path, **common_predict_kwargs(save=save))
    if selected_mode == "fps":
        return run_fps(model)
    if selected_mode == "dir_predict":
        return model.predict(source=dir_origin_path, **common_predict_kwargs(save=True))
    if selected_mode == "export_onnx":
        return run_export(model)
    raise AssertionError("Please specify mode: 'predict', 'video', 'fps', 'dir_predict', 'export_onnx'.")


#---------------------------------#
#   模型与数据配置
#---------------------------------#
model_path = "model_data/yolo26n-seg.pt"
data_yaml = "datasets/datasets.yaml"
input_shape = [640, 640]
#---------------------------------#
#   推理通用参数
#---------------------------------#
confidence = 0.25
nms_iou = 0.7
device = 0
save = True
show = False
stream = False
verbose = True
#---------------------------------#
#   官方保存路径参数
#   project=None 时使用 Ultralytics 默认 runs/<task>/<name>。
#---------------------------------#
project = None
name = "predict"
exist_ok = False
#---------------------------------#
#   模式配置
#---------------------------------#
mode = "predict"
video_path = 0
test_interval = 100
fps_image_path = "img/street.jpg"
dir_origin_path = "img/"
simplify = True
dynamic = None
opset = None


if __name__ == "__main__":
    run_mode()
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest -q tests\test_predict_official.py
```

Expected: PASS.

## Task 3: Update `get_map.py` For Official Run Paths

**Files:**
- Modify: `get_map.py`
- Test: `tests/test_helpers.py`

- [ ] **Step 1: Replace old default path tests**

In `tests/test_helpers.py`, remove tests for `train.phase_*` helpers and keep official validation path tests:

```python
import os
import unittest
from unittest.mock import patch


class HelperTests(unittest.TestCase):
    def test_default_map_model_path_falls_back(self):
        from get_map import default_model_path

        with patch("glob.glob", return_value=[]):
            self.assertEqual(default_model_path(), "model_data/yolo26n-seg.pt")

    def test_default_map_model_path_falls_back_for_detect(self):
        from get_map import default_model_path

        with patch("glob.glob", return_value=[]):
            self.assertEqual(default_model_path("detect"), "model_data/yolo26n.pt")

    def test_default_map_model_path_uses_latest_official_best(self):
        from get_map import default_model_path

        paths = [
            os.path.normpath("runs/segment/train/weights/best.pt"),
            os.path.normpath("runs/segment/train2/weights/best.pt"),
        ]
        mtimes = {paths[0]: 1, paths[1]: 2}

        with patch("glob.glob", return_value=paths), \
             patch("os.path.getmtime", side_effect=lambda path: mtimes[os.path.normpath(path)]):
            self.assertEqual(os.path.normpath(default_model_path()), paths[1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify latest path test fails**

Run:

```bash
python -m pytest -q tests\test_helpers.py
```

Expected: FAIL because `default_model_path()` still searches `logs/*_unfreeze`.

- [ ] **Step 3: Modify `get_map.py`**

Change `default_model_path()` to:

```python
def default_model_path(task="segment"):
    pattern = os.path.join("runs", task, "*", "weights", "best.pt")
    candidates = glob.glob(pattern)
    if candidates:
        return max(candidates, key=os.path.getmtime)
    if task == "detect":
        return "model_data/yolo26n.pt"
    return "model_data/yolo26n-seg.pt"
```

Keep `.val()` as the official implementation. Keep `data_yaml` as the exact string passed to Ultralytics.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest -q tests\test_helpers.py
```

Expected: PASS.

## Task 4: Delete Obsolete Local Modules

**Files:**
- Delete: `yolo.py`
- Delete: `nets/yolo_training.py`
- Delete: `utils/utils_fit.py`
- Delete: `utils/callbacks.py`
- Delete: `utils/utils_bbox.py`
- Delete: `utils/utils_map.py`
- Delete: `utils/utils.py`
- Delete: `tests/test_utils_bbox.py`

- [ ] **Step 1: Delete files**

Use `apply_patch` delete hunks for each listed file.

- [ ] **Step 2: Search for stale imports**

Run:

```bash
rg -n "from yolo import|utils_bbox|utils_fit|utils_map|utils.callbacks|yolo_training|utils.utils" .
```

Expected: no matches in tracked source/tests except historical docs under `docs/superpowers`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest -q tests\test_train_official.py tests\test_predict_official.py tests\test_helpers.py
```

Expected: PASS.

## Task 5: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README**

Rewrite the training, prediction, validation, tests, and key files sections so they say:

```markdown
## Training

Edit the configuration block in `train.py`, then run:

```bash
python train.py
```

`train.py` is a thin wrapper around official `ultralytics.YOLO.train()`.
It passes `model_path`, `data_yaml`, `task`, `imgsz`, `epochs`, `batch`,
`device`, `name`, and related options directly to Ultralytics.

Do not set `project="runs"` unless you intentionally want an extra nested
folder. With `project=None`, Ultralytics writes to `runs/<task>/<name>`.
```

Also remove references to local loss, local EMA, local dataloaders, local
decode/NMS, local mAP callback, and `utils/utils_fit.py`.

- [ ] **Step 2: Update CLAUDE.md**

Replace custom-loop architecture notes with:

```markdown
## Architecture

`train.py`, `get_map.py`, and `predict.py` are thin file-configured wrappers
around official Ultralytics APIs:

- `YOLO.train()` for training
- `YOLO.val()` for validation/mAP
- `YOLO.predict()` for inference
- `YOLO.export()` for ONNX export

Do not reintroduce custom dataloaders, loss, EMA, decode, NMS, or mAP
implementations unless the user explicitly asks for a non-official training
architecture.
```

- [ ] **Step 3: Search docs for stale custom-loop claims**

Run:

```bash
rg -n "custom training|自定义训练|utils_fit|yolo_training|utils_bbox|EvalCallback|DecodeBox|ModelEMA|Loss\\(" README.md CLAUDE.md
```

Expected: no stale claims remain.

## Task 6: Full Verification

**Files:**
- All modified files

- [ ] **Step 1: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m compileall -q train.py get_map.py predict.py scripts tests
```

Expected: exit code 0.

- [ ] **Step 3: Run import checks**

Run:

```bash
python -c "import train, get_map, predict; print('official wrappers import OK')"
```

Expected:

```text
official wrappers import OK
```

- [ ] **Step 4: Final stale-code search**

Run:

```bash
rg -n "from yolo import|utils_bbox|utils_fit|utils_map|utils.callbacks|yolo_training|DecodeBox|EvalCallback|ModelEMA" . --glob "*.py" --glob "*.md"
```

Expected: no matches outside old design/plan docs under `docs/superpowers`.
