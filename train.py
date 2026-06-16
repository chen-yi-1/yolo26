#-------------------------------------#
#       使用 Ultralytics 官方训练
#-------------------------------------#
from pathlib import Path

from ultralytics import YOLO


#---------------------------------#
#   task: detect / segment
#---------------------------------#
task = "segment"
#---------------------------------#
#   模型与数据配置
#   model_path 可为本地 .pt 路径或官方模型名。
#   data_yaml 传给 Ultralytics，不在本地重写其内部路径逻辑。
#---------------------------------#
model_path = "model_data/yolo26n-seg.pt"
data_yaml = "datasets/datasets.yaml"
input_shape = [640, 640]
#---------------------------------#
#   官方训练参数
#---------------------------------#
epochs = 100
batch = 16
device = 0
num_workers = 4
optimizer = "auto"
lr0 = 0.01
patience = 100
save_period = 10
amp = True
cache = False
plots = True
val = True
verbose = True
#---------------------------------#
#   官方保存路径参数
#   project=None 时使用 Ultralytics 默认 runs/<task>/<name>。
#   不要写 project="runs"，否则本版本会生成 runs/<task>/runs/<name>。
#---------------------------------#
project = None
name = "train"
exist_ok = False
#---------------------------------#
#   resume=True 时，model_path 应指向官方 last.pt。
#   freeze 可为 None、整数或层索引列表，由 Ultralytics 处理。
#---------------------------------#
pretrained = True
resume = False
freeze = None


def require_existing_file(path, label):
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def build_train_kwargs(
    data_yaml,
    task,
    imgsz,
    epochs,
    batch,
    device,
    workers,
    project,
    name,
    exist_ok,
    pretrained,
    resume,
    optimizer,
    lr0,
    patience,
    save_period,
    amp,
    freeze,
    cache,
    plots,
    val,
    verbose,
):
    kwargs = {
        "data": data_yaml,
        "task": task,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "workers": workers,
        "name": name,
        "exist_ok": exist_ok,
        "pretrained": pretrained,
        "resume": resume,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
        "save_period": save_period,
        "amp": amp,
        "cache": cache,
        "plots": plots,
        "val": val,
        "verbose": verbose,
    }
    if project is not None:
        kwargs["project"] = project
    if freeze is not None:
        kwargs["freeze"] = freeze
    return {key: value for key, value in kwargs.items() if value is not None}


def run_training():
    require_existing_file(data_yaml, "Dataset YAML")
    if any(sep in model_path for sep in ("/", "\\")):
        require_existing_file(model_path, "Model")

    model = YOLO(model_path)
    train_kwargs = build_train_kwargs(
        data_yaml=data_yaml,
        task=task,
        imgsz=input_shape[0],
        epochs=epochs,
        batch=batch,
        device=device,
        workers=num_workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        resume=resume,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        save_period=save_period,
        amp=amp,
        freeze=freeze,
        cache=cache,
        plots=plots,
        val=val,
        verbose=verbose,
    )
    return model.train(**train_kwargs)


if __name__ == "__main__":
    run_training()
