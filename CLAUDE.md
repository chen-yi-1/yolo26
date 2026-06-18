# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

YOLO26 is a local training, prediction, export, and validation project for
YOLO26 detection and instance segmentation models. The project now uses the
official Ultralytics wrappers as the runtime architecture:

- `train.py` calls `ultralytics.YOLO.train()`.
- `predict.py` calls `ultralytics.YOLO.predict()` and `YOLO.export()`.
- `get_map.py` calls `ultralytics.YOLO.val()`.

Keep this official-wrapper architecture in place. Do not reintroduce custom
dataloaders, loss, EMA, decode, NMS, or mAP code unless the user explicitly asks
for a non-official architecture.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare a YOLO dataset from an edited source directory
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task segment
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task detect

# Train. Edit the config block in train.py first.
python train.py

# Predict or export. Edit the config block in predict.py first.
python predict.py

# Validate mAP through official Ultralytics validation
python get_map.py

# Run tests
python -m pytest -q
python -m compileall -q train.py get_map.py predict.py scripts utils tests
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

- Config values live near the top of `train.py`.
- `require_existing_file()` checks local dataset and model paths.
- `build_train_kwargs()` maps local config names to official Ultralytics train
  arguments.
- `run_training()` loads `YOLO(model_path)` and returns
  `model.train(**train_kwargs)`.
- Ultralytics owns dataset construction, augmentation, optimization,
  checkpointing, validation, freeze handling, resume behavior, and metrics.

Important path behavior:

- `project = None` preserves the official `runs/<task>/<name>` layout.
- Do not set `project = "runs"` as a default. In this installed Ultralytics
  version, relative project values become
  `runs/<task>/<project>/<name>`.

For resume, set `resume = True` and point `model_path` at an official
Ultralytics `last.pt` checkpoint.

### Prediction And Export (`predict.py`)

Current prediction architecture:

- Config values live near the top of `predict.py`.
- `common_predict_kwargs()` builds arguments for `YOLO.predict()`.
- `export_kwargs()` builds arguments for `YOLO.export(format="onnx")`.
- `run_mode()` dispatches `predict`, `video`, `fps`, `dir_predict`, and
  `export_onnx`.
- Ultralytics owns preprocessing, postprocessing, rendering, artifact paths, and
  export internals.

Keep `project = None` by default for the standard run layout.

### Evaluation (`get_map.py`)

`get_map.py` uses official `ultralytics.YOLO.val()`.

`default_model_path()` prefers the latest
`runs/<task>/*/weights/best.pt`. If none exists, it falls back to
`model_data/yolo26n-seg.pt` for segmentation or `model_data/yolo26n.pt` for
detection.

## Important Project Notes

- There is currently no `config.py`; edit config blocks in `train.py`,
  `predict.py`, and `get_map.py`.
- Keep changes surgical. This repository has been refactored toward official
  Ultralytics entry points, and tests focus on wrapper dispatch and helper
  contracts.
- Deleted local modules should stay deleted unless the user requests a
  non-official architecture.
- `model_data/`, `logs/`, `datasets/`, `runs/`, and local run outputs are
  environment artifacts and may be ignored by git.

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
