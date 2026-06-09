# YOLO26 Instance Segmentation

YOLO26 instance segmentation project using the official `ultralytics.YOLO` training,
prediction, validation, and export pipeline.

## Environment

```bash
pip install -r requirements.txt
```

Model weights and fonts are expected under `model_data/`, for example:

```text
model_data/yolo26n-seg.pt
model_data/yolo26x-seg.pt
model_data/simhei.ttf
```

## Dataset Format

This project uses the official Ultralytics YOLO dataset layout.

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

Each label file should use normalized YOLO segmentation polygons:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Example `datasets/datasets.yaml`:

```yaml
path: /home/zhuye/yolo26/datasets
train: images/train
val: images/val
nc: 2
names:
  0: healthy
  1: abnormal
```

After editing labels under `dataset/labels`, prepare the Ultralytics layout:

```bash
python scripts/prepare_yolo_dataset.py --source dataset --output datasets
```

This writes `datasets/images/{train,val}`, `datasets/labels/{train,val}`, and
`datasets/datasets.yaml`.

## Train

Edit the config section in `train.py`, then run:

```bash
python train.py
```

`train.py` refreshes `datasets/` from `dataset/` before training, so manual
edits in `dataset/labels` are picked up automatically.

Training uses a two-phase freeze/unfreeze strategy with YOLO26 `-seg` weights:

- Freeze phase writes to `runs/segment/logs/<train_name>_freeze/`
- Unfreeze phase writes to `runs/segment/logs/<train_name>_unfreeze/`

For resume training, set `Init_Epoch > 0`. The script discovers the latest
`last.pt` under `runs/segment/logs/` and resumes into that checkpoint's existing
run directory.

## Predict

Edit `yolo.py` defaults if needed:

```python
"model_path": "model_data/yolo26x-seg.pt"
"classes_path": "datasets/datasets.yaml"
```

Prediction overlays instance masks, mask contours, boxes, class names, and confidence scores.

Run:

```bash
python predict.py
```

Supported modes in `predict.py`:

- `predict`
- `video`
- `fps`
- `dir_predict`
- `heatmap`
- `export_onnx`

## Validate mAP

`get_map.py` now calls the official Ultralytics validation path against
`datasets/datasets.yaml`.

Edit `model_path`, `data_yaml`, and `split` in `get_map.py`, then run:

```bash
python get_map.py
```

Use `split="val"` for the validation set, or add a `test:` entry to
`datasets/datasets.yaml` and set `split="test"`.

## Model Summary

```bash
python summary.py
```

Edit `phi` in `summary.py` to select `n/s/m/l/x`.
