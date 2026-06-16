# YOLO26 Official Ultralytics Refactor Design

## Goal

Refactor YOLO26 to use official Ultralytics training, validation, prediction,
and export paths instead of local training and inference implementations, while
keeping the current "edit the top config block, then run `python train.py`"
workflow.

## Scope

This refactor replaces local duplicated YOLO behavior with official Ultralytics
APIs:

- `train.py` calls `ultralytics.YOLO.train()`.
- `get_map.py` calls `ultralytics.YOLO.val()`.
- `predict.py` calls `ultralytics.YOLO.predict()` or `ultralytics.YOLO.export()`.
- Local decode, NMS, loss, EMA, scheduler, dataloader loop, and mAP callback
  code is removed when no longer referenced.

The dataset preparation scripts stay in the repository because they convert the
user's edited source folders into the official Ultralytics dataset layout.

## Architecture

### Training

`train.py` remains the primary training entry point. It contains a single
configuration block near the top with paths and training options, then builds a
small `train_kwargs` dictionary and passes it to `YOLO(model_path).train()`.

Training outputs use official Ultralytics run directories:

```text
runs/<task>/<name>/weights/best.pt
runs/<task>/<name>/weights/last.pt
```

Resume is delegated to Ultralytics by setting `resume=True` and pointing
`model_path` at an existing `last.pt` checkpoint when needed.

### Validation

`get_map.py` stays as a simple file-configured validation entry point. It loads
`YOLO(model_path)` and calls `.val(data=..., imgsz=..., conf=..., iou=...)`.

The default checkpoint lookup changes from the old custom `logs/*_unfreeze`
layout to official run layouts:

```text
runs/<task>/*/weights/best.pt
```

### Prediction And Export

`predict.py` becomes the main official inference entry point. It keeps the
existing mode names where practical:

- `predict`: one image with `YOLO.predict()`
- `dir_predict`: folder input with `YOLO.predict()`
- `video`: video/webcam source with `YOLO.predict()`
- `fps`: lightweight timing around repeated official predictions
- `export_onnx`: `YOLO.export(format="onnx")`

The old `yolo.py` wrapper is removed because it duplicates official prediction,
result parsing, NMS, export, and drawing behavior. Official Ultralytics result
rendering replaces custom PIL drawing and local YOLO26 output decoding.

### Deleted Local Training And Inference Code

Delete these files if they become unreferenced:

- `nets/yolo_training.py`
- `utils/utils_fit.py`
- `utils/callbacks.py`
- `utils/utils_bbox.py`
- `utils/utils_map.py`
- `yolo.py`

Keep these shared/project files:

- `scripts/prepare_yolo_dataset.py`
- `scripts/rgb_yolo_annotate.py`
- `utils/utils.py`, only if still used by scripts or retained entry points
- `get_map.py`
- `train.py`
- `predict.py`

## Data Flow

1. User prepares data with `scripts/prepare_yolo_dataset.py`.
2. `datasets/datasets.yaml` points to `images/train`, `images/val`,
   `labels/train`, and `labels/val`.
3. `train.py` passes that YAML directly to official Ultralytics training.
4. Ultralytics handles model class-count adaptation, augmentation, dataloading,
   optimizer, loss, EMA, validation, metrics, and checkpoint saving.
5. `get_map.py` validates saved checkpoints through official validation.
6. `predict.py` runs official prediction and export.

## Error Handling

The local entry points should fail early for missing model files, missing data
YAML files, and unsupported modes. They should not catch and hide Ultralytics
training, validation, or prediction errors because those messages are the most
accurate source of failure details.

## Tests

Tests should verify local wrapper behavior, not re-test Ultralytics internals:

- `train.py` builds expected train kwargs and calls `YOLO.train()`.
- `get_map.py` finds the latest official `best.pt` checkpoint.
- `predict.py` dispatches each mode to the official method.
- Dataset preparation tests remain unchanged.
- Old tests for custom training helpers are removed.

## Documentation

`README.md` and `CLAUDE.md` should describe the official Ultralytics workflow
and remove references to the custom training loop, local loss, local mAP
callback, and local output decoder.
