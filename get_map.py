import glob
import os

import torch
from ultralytics import YOLO


def default_model_path():
    candidates = sorted(
        glob.glob("runs/segment/logs/*_unfreeze/weights/best.pt"),
        key=os.path.getmtime,
    )
    if candidates:
        return candidates[-1]
    return "model_data/yolo26n-seg.pt"


if __name__ == "__main__":
    # Official Ultralytics dataset format:
    # datasets/datasets.yaml -> path, train/val/test image directories, nc, names.
    model_path = default_model_path()
    data_yaml = os.path.join("datasets", "datasets.yaml")
    input_shape = [640, 640]
    confidence = 0.001
    nms_iou = 0.7
    split = "val"
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
