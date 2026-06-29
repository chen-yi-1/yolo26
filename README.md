# YOLO26

YOLO26 instance segmentation and detection project built around the official
Ultralytics workflow. The entry points in this repository configure and call
Ultralytics wrappers directly:

- `train.py` calls `ultralytics.YOLO.train()`.
- `predict.py` calls `ultralytics.YOLO.predict()` and `YOLO.export()`.
- `get_map.py` calls `ultralytics.YOLO.val()`.

## Environment

```bash
pip install -r requirements.txt
```

Model weights are expected under `model_data/`, for example:

```text
model_data/yolo26n.pt
model_data/yolo26n-seg.pt
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

For imbalanced abnormal/healthy training data, oversample the minority class in
the training split only:

```bash
python scripts/prepare_yolo_dataset.py --source dataset --output datasets --task detect --oversample-class abnormal --oversample-target-ratio 1.0
```

`--oversample-target-ratio 1.0` duplicates `abnormal` training samples until
they match the largest other class count. Validation data is not oversampled.

## Training

Edit the configuration block in `train.py`, then run:

```bash
python train.py
```

During training, `train.py` reads the run directory `results.csv` after each
epoch update and overwrites `training_summary.png` next to it for a quick visual
check of train loss, validation loss, validation metrics, and learning rate.

Current training behavior:

- Validates that `data_yaml` exists and that local model paths exist.
- Loads the model with `ultralytics.YOLO(model_path)`.
- Builds keyword arguments with `build_train_kwargs()`.
- Calls `model.train(**train_kwargs)`.
- Lets Ultralytics own dataset loading, optimization, checkpointing,
  validation, resume handling, freezing, metrics, and run artifacts.

Important defaults are near the top of `train.py`:

```python
task = "segment"
model_path = "model_data/yolo26n-seg.pt"
data_yaml = "datasets/datasets.yaml"
input_shape = [640, 640]
epochs = 100
batch = 16
optimizer = "auto"
project = None
name = "train"
resume = False
freeze = None
```

Path note: keep `project = None` unless you deliberately want a nested custom
location. In this installed Ultralytics version, `project=None` writes to the
standard `runs/<task>/<name>` layout. A relative `project` value is nested as
`runs/<task>/<project>/<name>`.

For resume, set `resume = True` and point `model_path` at an official
Ultralytics `last.pt` checkpoint.

## Prediction And Export

Edit the configuration block in `predict.py`, especially:

```python
mode = "predict"
model_path = "model_data/yolo26n-seg.pt"
input_shape = [640, 640]
confidence = 0.5
iou = 0.7
project = None
name = "predict"
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
- `export_onnx`

Prediction and export are delegated to Ultralytics. Saved outputs follow the
same path behavior as training: `project=None` keeps the standard
`runs/<task>/<name>` layout.

## Rust ONNX Inference (No Python)

The project ships a second inference stack inside `rust-inference/`, built on
the official `ultralytics-inference` crate. It runs entirely on ONNX Runtime
and requires **no Python environment** once compiled.

Key differences vs. the Python side:

| Dimension | Python (`predict.py`) | Rust (`rust-inference/`) |
|-----------|-----------------------|--------------------------|
| Runtime   | PyTorch + Ultralytics (Python) | ONNX Runtime (pure Rust) |
| Model     | `.pt` weights | `.onnx` models |
| Web port  | 8081 (FastAPI/uvicorn) | 3001 (Axum/tokio) |
| Use case  | Training loop, iteration, export | Deployment, zero-dependency binaries |

### Environment

```powershell
# Install Rust toolchain (Windows)
winget install Rustlang.Rustup
rustup default stable
rustc --version   # >= 1.89.0
```

(Optional) Install FFmpeg 7+ for video/webcam/RTSP support:

```powershell
winget install "FFmpeg (Shared)"
```

### Prepare an ONNX Model

Option 1 — Export from the local PyTorch weights via the Python pipeline
(preferred, converts `model_data/*.pt` or your trained `models/best_*.pt`):

```bash
# Edit predict.py -> mode = "export_onnx" and model_path to your .pt file, e.g.
#   mode       = "export_onnx"
#   model_path = "model_data/yolo26n-seg.pt"
python predict.py
# -> writes the ONNX file next to the source .pt
```

Copy or symlink the exported `.onnx` to `models/best_12.onnx` (the Rust
default) or pass `--model path/to/file.onnx` explicitly.

Option 2 — Auto-download at runtime. Pass a known model name, e.g.
`--model yolo26n.onnx`, and the binary fetches it from Ultralytics on first
run.

### Run

```powershell
cd rust-inference

# Compile (once)
cargo build --release

# CLI help
cargo run --release -- --help

# CLI — single image (defaults to models/best_12.onnx, conf=0.3)
cargo run --release -- --source ..\img\street.jpg

# CLI — directory batch
cargo run --release -- --source "C:\images" --conf 0.5

# Web mode (default when no --source) — opens http://0.0.0.0:3001
cargo run --release
```

GPU acceleration is enabled at compile time through Cargo features. See
`rust-inference/README.md` for the full feature matrix (CUDA / TensorRT /
CoreML / OpenVINO / DirectML / XNNPACK), task modes (detect / segment /
pose / obb / classify / semantic), and the complete parameter reference.

## Validation

`get_map.py` runs official Ultralytics validation:

```bash
python get_map.py
```

By default it tries to use the latest
`runs/<task>/*/weights/best.pt`; if none exists it falls back to
`model_data/yolo26n-seg.pt` for segmentation or `model_data/yolo26n.pt` for
detection.

Edit `data_yaml`, `input_shape`, `confidence`, `nms_iou`, and `split` in
`get_map.py` as needed. The validation call itself is `YOLO(model_path).val()`.

## Tests

```bash
python -m pytest -q
python -m compileall -q train.py get_map.py predict.py scripts utils tests
```

## Key Files

| File | Purpose |
| --- | --- |
| `train.py` | Official Ultralytics training wrapper |
| `predict.py` | Official prediction, video, FPS, directory prediction, and ONNX export wrapper |
| `get_map.py` | Official Ultralytics validation entry point |
| `utils/training_plots.py` | Training `results.csv` plotting and epoch callback helpers |
| `utils/predict_runner.py` | Prediction/export/FPS/Web helper functions used by `predict.py` |
| `scripts/prepare_yolo_dataset.py` | Dataset conversion and validation |
| `tests/` | Focused tests for wrapper dispatch and helper behavior |
| `rust-inference/Cargo.toml` | Rust crate manifest — dependencies and GPU feature flags |
| `rust-inference/src/main.rs` | Rust CLI entry point — argument parsing + dispatch to CLI/Web |
| `rust-inference/src/web.rs` | Rust Axum Web server + browser UI (dark theme, drag & drop, paste) |
| `rust-inference/README.md` | Rust-side deep dive: GPU feature matrix, all task modes, full CLI flags |
| `docs/superpowers/` | Design documents and plans from the original rust-inference repo |
