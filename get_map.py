import glob
import os

import torch
from ultralytics import YOLO

from project_config import DATA, TASK, VAL, get_model_path, get_run_task


def default_model_path(task="segment"):
    run_task = get_run_task(task)
    fallback = get_model_path(task)
    candidates = sorted(
        glob.glob(f"runs/{run_task}/logs/*_unfreeze/weights/best.pt"),
        key=os.path.getmtime,
    )
    if candidates:
        return candidates[-1]
    return fallback


if __name__ == "__main__":
    # Official Ultralytics dataset format:
    # datasets/datasets.yaml -> path, train/val/test image directories, nc, names.
    task = TASK
    model_path = default_model_path(task)
    data_yaml = DATA["yaml"]
    input_shape = VAL["input_shape"]
    confidence = VAL["confidence"]
    nms_iou = VAL["nms_iou"]
    split = VAL["split"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=input_shape[0],
        conf=confidence,
        iou=nms_iou,
        split=split,
        device=device,
        plots=True,
    )

    print("Validation metrics:")
    for key, value in metrics.results_dict.items():
        print(f"{key}: {value:.6f}")
    if hasattr(metrics, "seg"):
        print(f"mask mAP50-95: {metrics.seg.map:.6f}")
        print(f"mask mAP50: {metrics.seg.map50:.6f}")
    if hasattr(metrics, "box"):
        print(f"box mAP50-95: {metrics.box.map:.6f}")
        print(f"box mAP50: {metrics.box.map50:.6f}")
