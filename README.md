# YOLO26 Instance Segmentation / Object Detection

YOLO26 project using the official `ultralytics.YOLO` training, prediction,
validation, and export pipeline. Set `TASK` in `config.py` to choose:

- `segment`: instance segmentation with polygon labels
- `detect`: object detection with rectangle/bbox labels

## Environment

```bash
pip install -r requirements.txt
```

Model weights and fonts are expected under `model_data/`, for example:

```text
model_data/yolo26n-seg.pt
model_data/yolo26n.pt
model_data/simhei.ttf
```

## Configuration

Training, validation, and inference defaults are centralized in:

```text
config.py
```

For object detection, change:

```python
TASK = "detect"
```

For instance segmentation, use:

```python
TASK = "segment"
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

For segmentation, each label file uses normalized YOLO polygons:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

For detection, each label file uses normalized YOLO boxes:

```text
class_id x_center y_center width height
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

To prepare detection labels from X-AnyLabeling rectangle shapes:

```bash
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task detect
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

Training uses a two-phase freeze/unfreeze strategy. The run directory follows
`TASK`:

- Freeze phase writes to `runs/segment/logs/<train_name>_freeze/`
- Unfreeze phase writes to `runs/segment/logs/<train_name>_unfreeze/`
- Detection writes to `runs/detect/logs/...`

For resume training, set `Init_Epoch > 0`. The script discovers the latest
`last.pt` under `runs/segment/logs/` and resumes into that checkpoint's existing
run directory.

## Predict

Edit inference defaults in `config.py` under `INFER`.

Prediction overlays boxes, class names, and confidence scores. Segmentation
models also overlay instance masks and mask contours.

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

