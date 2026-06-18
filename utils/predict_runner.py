import time


def build_common_predict_kwargs(
    input_shape,
    confidence,
    iou,
    device,
    save,
    show,
    save_txt,
    save_conf,
    name,
    exist_ok,
    verbose,
    project=None,
):
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


def build_export_kwargs(input_shape, simplify, dynamic, opset, device):
    return {
        "format": "onnx",
        "imgsz": input_shape[0],
        "simplify": simplify,
        "dynamic": dynamic,
        "opset": opset,
        "device": device,
    }


def run_interactive_predict(model, predict_source, save, predict_kwargs_factory, input_func=input):
    if predict_source:
        return model.predict(source=predict_source, **predict_kwargs_factory(save=save))

    result = None
    while True:
        source = input_func("Input image filename:")
        if not source:
            break
        result = model.predict(source=source, **predict_kwargs_factory(save=save))
    return result


def run_fps_test(model, test_interval, fps_image_path, predict_kwargs_factory):
    start = time.perf_counter()
    for _ in range(test_interval):
        model.predict(source=fps_image_path, **predict_kwargs_factory(save=False))
    tact_time = (time.perf_counter() - start) / test_interval
    print(str(tact_time) + " seconds, " + str(1 / tact_time) + "FPS, @batch_size 1")
    return tact_time


def run_web_server(model, predict_kwargs_factory, web_host, web_port):
    """Start a browser upload server for image prediction."""
    import base64
    import uuid
    from pathlib import Path

    import cv2
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="YOLO26 Web 预测")

    html_page = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>YOLO26 图片预测</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; text-align: center; background: #f5f5f5; }
h1 { color: #222; font-size: 24px; }
.card { background: #fff; border-radius: 12px; padding: 30px; box-shadow: 0 2px 12px rgba(0,0,0,.08); margin: 20px 0; }
.upload-row { display: flex; align-items: center; justify-content: center; gap: 12px; flex-wrap: wrap; }
input[type=file] { padding: 8px; }
button { padding: 10px 28px; background: #4a90d9; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
button:hover { background: #357abd; }
.images { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 20px; }
.images div { flex: 1; min-width: 280px; }
.images img { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
.label { font-weight: 600; margin: 10px 0; color: #555; }
#loading { display: none; margin: 20px; color: #888; }
#error { color: #c00; margin: 10px; }
</style>
</head>
<body>
<h1>YOLO26 图片预测</h1>
<div class="card">
<div class="upload-row">
<input type="file" id="imageInput" accept="image/*">
<button onclick="predict()">开始预测</button>
</div>
</div>
<div id="loading">正在预测中...</div>
<div id="error"></div>
<div class="images" id="results" style="display:none">
<div><div class="label">原始图片</div><img id="originalImg"></div>
<div><div class="label">预测结果</div><img id="predictedImg"></div>
</div>
<script>
async function predict() {
  const f = document.getElementById('imageInput').files[0];
  if (!f) { alert('请先选择一张图片'); return; }
  document.getElementById('loading').style.display = '';
  document.getElementById('error').textContent = '';
  document.getElementById('results').style.display = 'none';
  const fd = new FormData(); fd.append('file', f);
  try {
    const r = await fetch('/predict', { method:'POST', body:fd });
    const d = await r.json();
    document.getElementById('originalImg').src = 'data:image/jpeg;base64,' + d.original;
    document.getElementById('predictedImg').src = 'data:image/jpeg;base64,' + d.predicted;
    document.getElementById('results').style.display = 'flex';
  } catch(e) {
    document.getElementById('error').textContent = '预测失败：' + e.message;
  }
  document.getElementById('loading').style.display = 'none';
}
</script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return html_page

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        contents = await file.read()

        upload_dir = Path("web_uploads")
        upload_dir.mkdir(exist_ok=True)
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        filename = file.filename if file.filename else f"{uuid.uuid4().hex}{suffix}"
        save_path = upload_dir / filename
        save_path.write_bytes(contents)

        results = model.predict(source=str(save_path), **predict_kwargs_factory(save=False))
        plot_arr = results[0].plot()
        _, buffer = cv2.imencode(".jpg", plot_arr)
        predicted_b64 = base64.b64encode(buffer).decode("utf-8")
        original_b64 = base64.b64encode(contents).decode("utf-8")
        return {"original": original_b64, "predicted": predicted_b64}

    print(f"[Info] Web 服务器已启动：http://{web_host}:{web_port}")
    print("[Info] 请在浏览器中打开以上地址上传图片进行预测")
    uvicorn.run(app, host=web_host, port=web_port, log_level="info")
