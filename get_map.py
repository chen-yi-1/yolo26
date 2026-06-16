import glob
import os

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    #------------------------------------------------------#
    #   model_path      指向训练好的权值文件
    #   data_yaml       数据集配置文件路径
    #   input_shape     输入的shape大小
    #   confidence      置信度阈值
    #   nms_iou         NMS IOU阈值
    #   split           验证集或测试集 (val / test)
    #------------------------------------------------------#
    model_path  = 'model_data/yolo26x.pt'
    data_yaml   = 'datasets/datasets.yaml'
    input_shape = [640, 640]
    confidence  = 0.001
    nms_iou     = 0.7
    split       = "val"
    device      = "cuda" if torch.cuda.is_available() else "cpu"

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
