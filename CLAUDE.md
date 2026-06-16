# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

YOLO26 is a local training and inference project for YOLO26 detection and
instance segmentation models. The project uses Ultralytics model and dataset
components, but the current training entry point is a custom loop, not a thin
`ultralytics.YOLO.train()` wrapper.

YOLO26 traits used by this codebase:

- NMS-free style end-to-end head with `one2one` and `one2many` branches.
- DFL-free box regression for YOLO26 checkpoints with `reg_max=1`.
- Training loss reads the `one2many` branch.
- Inference decodes the `one2one` branch.
- Classification heads are replaced in `train.py` when the dataset class count
  differs from the pretrained checkpoint.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare a YOLO dataset from an edited source directory
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task segment
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task detect

# Train. Edit the config block in train.py first.
python train.py

# Predict. Edit yolo.py _defaults first.
python predict.py

# Validate mAP through official Ultralytics validation
python get_map.py

# Run tests
python -m pytest -q
python -m compileall -q train.py utils\utils_fit.py get_map.py scripts
```

## Dataset Layout

Prepared datasets use the official Ultralytics layout:

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

`datasets/datasets.yaml` should contain:

```yaml
path: C:/Users/EDY/Desktop/yolo26/datasets
train: images/train
val: images/val
nc: 2
names:
  0: abnormal
  1: healthy
```

`scripts/prepare_yolo_dataset.py` supports `--task segment` for polygon labels
and `--task detect` for bbox labels. It can also sample with `--sample-count`.

## Architecture

### Training (`train.py`)

Current training architecture:

- Loads model weights through `ultralytics.YOLO(model_path).model`.
- Reads class names and `nc` from `datasets/datasets.yaml` via
  `utils.get_classes`.
- If `num_classes != model.nc`, replaces classification heads under both
  `head.one2many["cls_head"]` and `head.one2one["cls_head"]`.
- Builds datasets with `ultralytics.data.build.build_yolo_dataset`.
- Uses PyTorch `DataLoader` with the dataset-provided `collate_fn`.
- Uses `nets.yolo_training.Loss`, `ModelEMA`, `get_lr_scheduler`, and
  `set_optimizer_lr`.
- Runs one epoch at a time through `utils.utils_fit.fit_one_epoch`.
- Saves `.pth` checkpoints under `logs/loss_<timestamp>/`.

Two-stage freeze/unfreeze training is still present. Ultralytics models in this
project do not expose `.backbone`, so freezing and unfreezing are implemented by
matching parameter names with prefixes `model.0.` through `model.9.`.

`optimizer_type="auto"` resolves to SGD for longer runs and AdamW for shorter
runs. Adam-family optimizers must receive `betas`, not `momentum`.

Resume helper functions are defined at module scope for tests and future resume
work:

- `torch_load_weights_only_false()`
- `phase_train_names()`
- `find_latest_resume_checkpoint()`
- `phase2_checkpoint()`
- `phase2_epochs()`

The main training loop currently still saves local `.pth` state dict files; it
does not resume Ultralytics `.pt` trainer checkpoints end-to-end.

### Training Batch Format (`utils/utils_fit.py`)

Ultralytics datasets yield batch dictionaries. `fit_one_epoch` converts them to
the local loss format:

```text
batch["img"] -> float tensor in [0, 1]
batch["batch_idx"], batch["cls"], batch["bboxes"] -> [N, 6]
```

The local loss expects `[batch_idx, cls, x, y, w, h]`.

### Loss (`nets/yolo_training.py`)

`Loss` expects YOLO26 model outputs shaped as a dictionary with an `one2many`
branch:

- `outputs["one2many"]["boxes"]`
- `outputs["one2many"]["scores"]`
- `outputs["one2many"]["feats"]`

It uses task-aligned assignment, CIoU box loss, BCE class loss, and no DFL for
YOLO26 `reg_max=1` models.

### Inference (`yolo.py`)

`YOLO` wraps `ultralytics.YOLO(...).model` for local inference. Methods:

- `detect_image(image, crop, count)`
- `get_FPS(image, test_interval)`
- `detect_heatmap(image, save_path)`
- `convert_to_onnx(simplify, path)`

Inference uses `utils.utils_bbox.DecodeBox`, Chinese font rendering through
`model_data/simhei.ttf`, and modes configured in `predict.py`.

### Evaluation (`get_map.py`)

`get_map.py` uses official `ultralytics.YOLO.val()`.

`default_model_path()` prefers the latest
`runs/<task>/logs/*_unfreeze/weights/best.pt`. If none exists, it falls back to
`model_data/yolo26n-seg.pt` for segmentation or `model_data/yolo26n.pt` for
detection.

## Important Project Notes

- There is currently no `config.py`; edit config blocks in `train.py`,
  `yolo.py`, `predict.py`, and `get_map.py`.
- Keep changes surgical. This repository is mid-refactor and tests may describe
  intended helper contracts even when the main loop is still custom.
- Do not replace the custom training loop with `YOLO.train()` unless the user
  explicitly asks for that architecture change.
- `model_data/`, `logs/`, `datasets/`, and local run outputs are environment
  artifacts and may be ignored by git.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

### 1. Think Before Coding

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear, stop and name what is unclear.

### 2. Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- No speculative configurability.
- If a small fix is enough, prefer the small fix.

### 3. Surgical Changes

When editing existing code:

- Touch only what is needed.
- Do not refactor unrelated code.
- Match existing style.
- Remove only imports, variables, or functions made unused by your own changes.

### 4. Goal-Driven Execution

Transform tasks into verifiable goals:

```text
1. Change X -> verify with Y
2. Change Z -> verify with tests
```

For bugs, reproduce or isolate the failure before fixing, then verify with the
smallest relevant command and the full test suite when practical.
