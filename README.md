# YOLO26

YOLO26 instance segmentation and detection project built around Ultralytics model
components, with a custom training loop for YOLO26 dual-head outputs.

The current training path is not a pure `ultralytics.YOLO.train()` wrapper.
`train.py` uses Ultralytics dataset loading, then trains with the local
`Loss`, `ModelEMA`, optimizer setup, scheduler, and epoch loop.

## Environment

```bash
pip install -r requirements.txt
```

Model weights and fonts are expected under `model_data/`, for example:

```text
model_data/yolo26x.pt
model_data/yolo26x-seg.pt
model_data/simhei.ttf
```

## Dataset

The prepared dataset follows the Ultralytics YOLO layout:

```text
datasets/
  images/
    train/
    val/
  labels/
    train/
    val/
  datasets.yaml
```

Example `datasets/datasets.yaml`:

```yaml
path: C:/Users/EDY/Desktop/yolo26/datasets
train: images/train
val: images/val
nc: 2
names:
  0: abnormal
  1: healthy
```

For segmentation labels, rows are normalized polygons:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

For detection labels, rows are normalized boxes:

```text
class_id x_center y_center width height
```

Prepare `datasets/` from an edited source directory:

```bash
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task segment
```

For detection labels:

```bash
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task detect
```

The script validates labels, copies matching images, writes
`datasets/images/{train,val}`, `datasets/labels/{train,val}`, and creates
`datasets/datasets.yaml`. Use `--sample-count` to sample a fixed count per
class.

## Training

Edit the configuration block in `train.py`, then run:

```bash
python train.py
```

Current training behavior:

- Loads the model with `ultralytics.YOLO(model_path).model`.
- Replaces YOLO26 classification heads when `datasets.yaml` has a different
  class count from the pretrained checkpoint.
- Builds train and validation datasets with
  `ultralytics.data.build.build_yolo_dataset`.
- Converts Ultralytics batch dictionaries in `utils/utils_fit.py`.
- Uses local YOLO26 loss in `nets/yolo_training.py`.
- Uses local EMA, optimizer parameter grouping, LR scheduler, and checkpoint
  saving.
- Supports two-stage freeze/unfreeze training. Freezing is done by layer name
  prefixes `model.0.` through `model.9.` because Ultralytics models do not
  expose a `.backbone` attribute here.

Important defaults are near the top of `train.py`:

```python
Cuda = False
classes_path = "datasets/datasets.yaml"
model_path = "model_data/yolo26x.pt"
input_shape = [640, 640]
Freeze_Train = True
optimizer_type = "auto"
```

`optimizer_type="auto"` chooses SGD for longer runs and AdamW for shorter runs.
Adam-family optimizers receive `betas`, not `momentum`, to match PyTorch
optimizer signatures.

Checkpoints are saved under `logs/loss_<timestamp>/`.

## Prediction

Edit defaults in `yolo.py`, especially:

```python
"model_path": "model_data/yolo26x.pt"
"classes_path": "datasets/datasets.yaml"
```

Then run:

```bash
python predict.py
```

Supported `predict.py` modes:

- `predict`
- `video`
- `fps`
- `dir_predict`
- `heatmap`
- `export_onnx`

The custom `YOLO` wrapper in `yolo.py` loads the Ultralytics model, decodes
YOLO26 outputs through `utils/utils_bbox.py`, and draws boxes and Chinese class
labels with `model_data/simhei.ttf`.

## Validation

`get_map.py` runs official Ultralytics validation:

```bash
python get_map.py
```

By default it tries to use the latest
`runs/<task>/logs/*_unfreeze/weights/best.pt`; if none exists it falls back to
`model_data/yolo26n-seg.pt` for segmentation or `model_data/yolo26n.pt` for
detection.

Edit `data_yaml`, `input_shape`, `confidence`, `nms_iou`, and `split` in
`get_map.py` as needed.

## Tests

```bash
python -m pytest -q
python -m compileall -q train.py utils\utils_fit.py get_map.py scripts
```

## Key Files

| File | Purpose |
| --- | --- |
| `train.py` | Custom YOLO26 training entry point |
| `utils/utils_fit.py` | One-epoch train/validation loop |
| `nets/yolo_training.py` | YOLO26 loss, assigner, EMA, LR helpers |
| `yolo.py` | Inference wrapper, heatmap, ONNX export |
| `predict.py` | Interactive and batch prediction modes |
| `get_map.py` | Ultralytics validation entry point |
| `scripts/prepare_yolo_dataset.py` | Dataset conversion and validation |
| `utils/utils.py` | Shared image, class, seed, and display helpers |
