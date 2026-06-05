# YOLO26 Official Training Adaptation — Design Spec

## Goal

Replace the custom training loop with ultralytics official `model.train()` API, while preserving the project's inference, evaluation, and visualization features.

## Decisions

| Topic | Decision |
|---|---|
| Data format | YOLO format (`.txt` per image + `dataset.yaml`) |
| Training | `ultralytics.YOLO.train()` — handles data loading, augmentation, loss, EMA, optimizer, validation, checkpointing |
| Inference | `ultralytics.YOLO` as backend, preserve all modes (predict/video/fps/dir_predict/heatmap/export_onnx) |
| Evaluation | Adapt get_map.py to use ultralytics model, keep VOC mAP logic |
| Freeze-thaw | Use ultralytics native `freeze` parameter |

## What Gets Removed

- `nets/` — entire directory (YoloBody wrapper, custom Loss/TaskAlignedAssigner/EMA/optimizer)
- `utils/dataloader.py` — custom YoloDataset with mosaic/mixup (ultralytics provides this)
- `utils/utils_fit.py` — custom training loop (ultralytics DetectionTrainer handles this)
- `utils/utils_bbox.py` — DecodeBox class (ultralytics provides prediction)

## What Gets Rewritten

- `train.py` — minimal config + `model.train()` call
- `yolo.py` — YOLO class backed by `ultralytics.YOLO`
- `predict.py` — adapt to new YOLO class, preserve all modes
- `get_map.py` — adapt to ultralytics model inference
- `summary.py` — use ultralytics model info API

## What Gets Kept (as-is or lightly modified)

- `utils/utils.py` — utility functions (cvtColor, get_classes, seed_everything, preprocess_input, show_config)
- `utils/utils_map.py` — VOC/COCO mAP computation logic
- `utils/callbacks.py` — tensorboard logging, may become redundant with ultralytics built-in
- `voc_annotation.py` — VOC→YOLO conversion utility
- `model_data/simhei.ttf` — Chinese font for visualization
- `model_data/coco_classes.txt` — class names reference

## Files After Migration

```
train.py              # Minimal: config dict + YOLO().train()
yolo.py               # YOLO class wrapping ultralytics
predict.py            # Multi-mode inference (unchanged interface)
get_map.py            # VOC mAP evaluation
summary.py            # FLOPs/params analysis
voc_annotation.py     # VOC XML → YOLO txt converter
utils/
  utils.py            # General utilities (kept)
  utils_map.py        # mAP computation (kept)
  callbacks.py        # Optional: custom logging (kept)
model_data/
  simhei.ttf          # Chinese font (kept)
  coco_classes.txt    # Class names (kept)
```

## What `train.py` Becomes

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26x.pt")
    model.train(
        data="dataset.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        optimizer="auto",     # Adam for small datasets
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0,
        warmup_epochs=3,
        cos_lr=True,
        freeze=10,            # freeze backbone first 10 epochs
        box=7.5, cls=0.5, dfl=1.5,
        amp=True,
        seed=11,
        ...
    )
```
