# YOLO26 Object Detection

YOLO26 object detection project using the official `ultralytics.YOLO` training,
prediction, validation, and export pipeline.

## Environment

```bash
pip install -r requirements.txt
```

Model weights and fonts are expected under `model_data/`, for example:

```text
model_data/yolo26n.pt
model_data/yolo26x.pt
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
dataset.yaml
```

Each label file uses normalized YOLO boxes:

```text
class_id x_center y_center width height
```

Example `dataset.yaml`:

```yaml
path: /home/zhuye/yolo26/datasets
train: images/train
val: images/val
nc: 2
names:
  0: healthy
  1: unhealthy
```

Prepare data directly in the Ultralytics format above.

## Train

Edit the config section in `train.py`, then run:

```bash
python train.py
```

Training uses a two-phase freeze/unfreeze strategy:

- Freeze phase writes to `runs/detect/logs/<train_name>_freeze/`
- Unfreeze phase writes to `runs/detect/logs/<train_name>_unfreeze/`

For resume training, set `Init_Epoch > 0`. The script discovers the latest
`last.pt` under `runs/detect/logs/` and resumes into that checkpoint's existing
run directory.

## Predict

Edit `yolo.py` defaults if needed:

```python
"model_path": "model_data/yolo26x.pt"
"classes_path": "dataset.yaml"
```

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
`dataset.yaml`.

Edit `model_path`, `data_yaml`, and `split` in `get_map.py`, then run:

```bash
python get_map.py
```

Use `split="val"` for the validation set, or add a `test:` entry to
`dataset.yaml` and set `split="test"`.

## Model Summary

```bash
python summary.py
```

Edit `phi` in `summary.py` to select `n/s/m/l/x`.
