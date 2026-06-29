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

# ── Rust ONNX inference (rust-inference/ sub-crate, no Python required) ──
cd rust-inference
cargo build --release              # Compile binary -> target/release/yolo26-rust-inference.exe
cargo run --release -- --help      # Print all CLI flags
cargo run --release                # Start Axum Web UI on http://0.0.0.0:3001 (default mode)
cargo run --release -- --source "C:\images\test.jpg"   # Single image CLI inference
cargo run --release --features "nvidia" -- --device cuda:0 --half --source test.jpg
cd ..

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

### Rust ONNX Inference (`rust-inference/`)

The repository also ships a standalone Rust inference crate under
`rust-inference/`. This is a separate build target (not a Python binding) and
is intentionally kept independent so it can be compiled and distributed
without a Python runtime.

Current Rust architecture:

- `rust-inference/Cargo.toml` pins dependencies and exposes Cargo features
  for GPU backends: `nvidia` (= cuda + tensorrt), `cuda`, `tensorrt`,
  `coreml`, `openvino`, `directml`, `rocm`, `xnnpack`, plus `video`
  (requires FFmpeg 7+).
- `rust-inference/src/main.rs` owns clap argument parsing and dispatches:
  - CLI mode (`--source <path>`) → `process_source()` → single image or
    directory batch, with annotated writes to `runs/<task>/predict/` to
    match the Ultralytics output layout used by the Python stack.
  - Web mode (no `--source`, the default) → `run_web_mode()` which hands
    off to `web.rs`.
- `rust-inference/src/web.rs` runs an Axum/tokio server on `0.0.0.0:3001`
  exposing `GET /` (single-file dark-themed HTML UI with drag-drop +
  Ctrl+V paste), `GET /api/health`, and `POST /api/detect` (accepts
  base64 image body, returns annotated base64 JPEG + detection JSON).
- Saved outputs mirror the Python side: `runs/<task>/predict/`.

Important conventions / contracts for the Rust side:

- Default model path is relative: `models/best_12.onnx` (matches the Python
  side's `models\best_12.pt` sibling convention after running
  `predict.py` with `mode = "export_onnx"`). Known Ultralytics model names
  (e.g. `yolo26n.onnx`) auto-download.
- Build artifacts live in `rust-inference/target/` which is covered by the
  repo-root `.gitignore` `target/` rule (also matches PyBuilder `target/`).
- Uploaded Web images are persisted in `web_uploads/` (same directory as
  the Python FastAPI server); already gitignored at repo root.
- Keep touching `rust-inference/` surgical. Treat it as its own crate with
  its own entry points. Do not try to unify its implementation with the
  Python wrappers — they solve different stages (training/iteration vs.
  deployment/distribution).

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

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

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

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Code Management Practices

**Version Control and Documentation Standards**

- Always commit changes before making significant code modifications or refactoring to maintain a clear history and enable rollback when using Git.
- Document all changes in Chinese, including code comments, commit messages, and log files.

