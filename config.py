"""Central project configuration for training, validation, and inference."""

import os


TASK = "detect"  # "segment" or "detect"

DATA = {
    "source": "dataset",
    "output": "datasets",
    "yaml": os.path.join("datasets", "datasets.yaml"),
}

MODEL_PATHS = {
    "segment": "model_data/yolo26n-seg.pt",
    "detect": "model_data/yolo26n.pt",
}

TRAIN = {
    "cuda": False,
    "seed": 11,
    "fp16": True,
    "input_shape": [640, 640],
    "init_epoch": 0,
    "freeze_epoch": 5,
    "unfreeze_epoch": 10,
    "freeze_train": True,
    "freeze_batch_size": 32,
    "unfreeze_batch_size": 16,
    "init_lr": 1e-3,
    "optimizer_type": "auto",
    "momentum": 0.937,
    "weight_decay": 0,
    "lr_decay_type": "cos",
    "mosaic": True,
    "mosaic_prob": 0.5,
    "mixup": True,
    "mixup_prob": 0.5,
    "special_aug_ratio": 0.7,
    "save_period": 10,
    "save_dir": "logs",
    "eval_flag": True,
    "num_workers": 4,
}

VAL = {
    "input_shape": [640, 640],
    "confidence": 0.001,
    "nms_iou": 0.7,
    "split": "val",
}

INFER = {
    "model_path": None,  # None means use MODEL_PATHS[TASK].
    "classes_path": DATA["yaml"],
    "input_shape": [640, 640],
    "confidence": 0.5,
    "nms_iou": 0.3,
    "mask_alpha": 0.35,
    "letterbox_image": True,
    "cuda": True,
}

def get_run_task(task=TASK):
    return "detect" if task == "detect" else "segment"


def get_model_path(task=TASK):
    return MODEL_PATHS[task]


def get_infer_model_path(task=TASK):
    return INFER["model_path"] or get_model_path(task)
