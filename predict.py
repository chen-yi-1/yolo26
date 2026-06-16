#-------------------------------------#
#       使用 Ultralytics 官方预测
#-------------------------------------#
import time

from ultralytics import YOLO


#---------------------------------#
#   模式: predict / video / fps / dir_predict / export_onnx
#---------------------------------#
mode = "predict"
#---------------------------------#
#   模型与输入配置
#   model_path 可为本地 .pt 路径或官方模型名。
#---------------------------------#
model_path = "model_data/yolo26n-seg.pt"
input_shape = [640, 640]
confidence = 0.5
iou = 0.7
device = 0
verbose = True
#---------------------------------#
#   官方保存路径参数
#   project=None 时使用 Ultralytics 默认 runs/<task>/<name>。
#   不要写 project="runs"，否则本版本会生成 runs/<task>/runs/<name>。
#---------------------------------#
project = None
name = "predict"
exist_ok = False
save = True
show = False
save_txt = False
save_conf = False
#---------------------------------#
#   predict 模式。source 为空时进入交互输入。
#---------------------------------#
predict_source = ""
#---------------------------------#
#   video 模式。video_path=0 表示摄像头。
#   video_save_path 保留给用户编辑习惯；官方保存位置由 project/name 控制。
#---------------------------------#
video_path = 0
video_save_path = ""
#---------------------------------#
#   fps 模式
#---------------------------------#
test_interval = 100
fps_image_path = "img/street.jpg"
#---------------------------------#
#   dir_predict 模式
#---------------------------------#
dir_origin_path = "img/"
#---------------------------------#
#   export_onnx 模式
#---------------------------------#
simplify = True
dynamic = False
opset = None


def common_predict_kwargs(save):
    kwargs = {
        "imgsz": input_shape[0],
        "conf": confidence,
        "iou": iou,
        "device": device,
        "save": save,
        "show": show,
        "save_txt": save_txt,
        "save_conf": save_conf,
        "name": name,
        "exist_ok": exist_ok,
        "verbose": verbose,
    }
    if project is not None:
        kwargs["project"] = project
    return {key: value for key, value in kwargs.items() if value is not None}


def export_kwargs():
    return {
        "format": "onnx",
        "imgsz": input_shape[0],
        "simplify": simplify,
        "dynamic": dynamic,
        "opset": opset,
        "device": device,
    }


def run_predict(model):
    if predict_source:
        return model.predict(source=predict_source, **common_predict_kwargs(save=save))

    result = None
    while True:
        source = input("Input image filename:")
        if not source:
            break
        result = model.predict(source=source, **common_predict_kwargs(save=save))
    return result


def run_fps(model):
    start = time.perf_counter()
    for _ in range(test_interval):
        model.predict(source=fps_image_path, **common_predict_kwargs(save=False))
    tact_time = (time.perf_counter() - start) / test_interval
    print(str(tact_time) + " seconds, " + str(1 / tact_time) + "FPS, @batch_size 1")
    return tact_time


def run_mode(selected_mode=mode):
    model = YOLO(model_path)

    if selected_mode == "predict":
        return run_predict(model)
    if selected_mode == "video":
        return model.predict(source=video_path, **common_predict_kwargs(save=save))
    if selected_mode == "fps":
        return run_fps(model)
    if selected_mode == "dir_predict":
        return model.predict(source=dir_origin_path, **common_predict_kwargs(save=True))
    if selected_mode == "export_onnx":
        return model.export(**export_kwargs())

    raise AssertionError(
        "Please specify the correct mode: 'predict', 'video', 'fps', 'export_onnx', 'dir_predict'."
    )


if __name__ == "__main__":
    run_mode(mode)
