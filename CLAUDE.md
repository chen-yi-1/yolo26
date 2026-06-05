# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO26 object detection — uses the official `ultralytics.YOLO` training pipeline with a thin configuration wrapper. YOLO format datasets, freeze-thaw training strategy, custom inference with Chinese font support and multiple prediction modes.

YOLO26 key traits:
- **NMS-Free**: `end2end=True`, dual head (one2one for inference, one2many for training)
- **DFL-Free**: `reg_max=1`, direct bbox regression without distribution
- Backbone: Conv + C3k2 + SPPF + C2PSA (attention), DWConv in classification head
- Scales: n/s/m/l/x (x: 58.8M params, 208.5 GFLOPs)

## Commands

```bash
# Install dependencies
pip install torch torchvision ultralytics

# Prepare dataset (VOC → YOLO format conversion)
# 1. Place images in VOCdevkit/VOC2007/JPEGImages/
# 2. Place XML annotations in VOCdevkit/VOC2007/Annotations/
# 3. Edit voc_annotation.py: set classes_path
python voc_annotation.py       # generates 2007_train.txt, 2007_val.txt
# 4. Create dataset.yaml pointing to the generated txt files

# Train (edit train.py config section first)
python train.py

# Predict (edit yolo.py _defaults to set model_path and classes_path)
python predict.py              # interactive: input image path to detect
# Modes in predict.py: predict / video / fps / dir_predict / heatmap / export_onnx

# Evaluate mAP (VOC-format ground truth required)
python get_map.py              # map_mode 0=full VOC, 4=COCO via pycocotools

# View model FLOPs & params
python summary.py              # edit phi first
```

## Architecture

### Training (`train.py`)
Thin wrapper around `ultralytics.YOLO.train()`. Two-phase freeze-thaw:
- **Phase 1** (freeze): `freeze=10` freezes backbone layers, larger batch size, mosaic stays on
- **Phase 2** (unfreeze): `freeze=None` unfreezes all, smaller batch, mosaic closes for last N epochs

All training params (optimizer, LR, augmentation, loss gains) are passed directly to ultralytics. The old custom training loop, Loss, EMA, optimizer, and DataLoader are deleted — ultralytics handles everything internally.

Config section at the top of `if __name__ == "__main__":` uses the same parameter names as the old codebase (`Freeze_Epoch`, `Freeze_Train`, `Freeze_batch_size`, etc.) for familiarity.

**Init_Epoch and resume:** `Init_Epoch > 0` triggers resume mode:
- Auto-discovers the latest `last.pt` under `runs/detect/{save_dir}/`, validates it has `epoch`/`optimizer` state
- Extracts `train_name` from the checkpoint path so results write back to the same directory
- `_train_with_resume()` monkey-patches `torch.load` to force `weights_only=False` (required for PyTorch 2.6+ compatibility with ultralytics)
- When checkpoint epoch ≠ `Init_Epoch`, warns but respects `Init_Epoch` for phase/skip logic; actual resume epoch is determined by the checkpoint
- Phase 1 also supports resume when `Init_Epoch > 0` and `Init_Epoch < Freeze_Epoch`
- `_add_per_epoch_plotting()` callback updates `results.png` after each epoch so curves are visible even if training is interrupted

### Inference (`yolo.py`)
`YOLO` class wraps `ultralytics.YOLO`. Methods:
- `detect_image(image, crop, count)` — PIL Image → detect → draw boxes with Chinese labels
- `get_FPS(image, test_interval)` — FPS benchmark
- `detect_heatmap(image, save_path)` — class-activation heatmap from `one2one` branch feats
- `get_map_txt(image_id, image, class_names, out_path)` — dump detections for mAP
- `convert_to_onnx(simplify, path)` — ONNX export

Supports letterbox resize, confidence/NMS filtering, and Chinese font rendering (simhei.ttf).

### Evaluation (`get_map.py`)
Computes VOC mAP (mode 0-3) or COCO mAP via pycocotools (mode 4). Uses `YOLO.get_map_txt()` with low confidence threshold to capture all possible detections, then matches against VOC XML ground truth. Actual mAP computation logic lives in `utils/utils_map.py`.

### Utility files

| File | Purpose |
|---|---|
| `utils/utils.py` | `cvtColor`, `get_classes`, `seed_everything`, `preprocess_input`, `resize_image`, `show_config` |
| `utils/utils_map.py` | VOC mAP computation + COCO mAP via pycocotools |
| `voc_annotation.py` | VOC XML annotations → YOLO-format txt files (train/val splits) |

### Gitignored directories
`model_data/`, `VOCdevkit/`, `logs/`, `datasets/` — not tracked. Classes txt, model weights (`yolo26x.pt`), and fonts go in `model_data/`.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
