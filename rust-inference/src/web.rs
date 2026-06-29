//! Web 服务模块 — HTML 页面 & API handler

/// 应用共享状态
#[derive(Clone)]
pub struct AppState {
    pub model: std::sync::Arc<std::sync::Mutex<ultralytics_inference::YOLOModel>>,
    pub model_name: String,
    pub device: String,
}

/// 前端页面 HTML（单页，零外部依赖）
pub const INDEX_HTML: &str = r###"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO26 推理可视化</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; min-height: 100vh;
  }
  .container { max-width: 960px; margin: 0 auto; padding: 24px; }
  header { text-align: center; margin-bottom: 24px; }
  header h1 { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }
  header .info { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  .upload-zone {
    border: 2px dashed #475569; border-radius: 12px;
    padding: 48px 24px; text-align: center; cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: #38bdf8; background: rgba(56,189,248,0.05);
  }
  .upload-zone .icon { font-size: 40px; margin-bottom: 12px; }
  .upload-zone p { color: #94a3b8; font-size: 14px; }
  .upload-zone p.hint { font-size: 12px; margin-top: 8px; color: #64748b; }
  .result { margin-top: 24px; display: none; }
  .result.show { display: block; }
  .result img { width: 100%; border-radius: 8px; border: 1px solid #334155; }
  .result .stats {
    text-align: center; margin-top: 12px; font-size: 14px; color: #94a3b8;
  }
  .result .stats strong { color: #38bdf8; }
  .loading {
    display: none; text-align: center; margin-top: 24px;
  }
  .loading.show { display: block; }
  .spinner {
    width: 36px; height: 36px; border: 3px solid #334155;
    border-top-color: #38bdf8; border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: #f87171; text-align: center; margin-top: 12px; display: none; }
  .error.show { display: block; }
  input[type=file] { display: none; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>YOLO26 推理可视化</h1>
    <div class="info" id="modelInfo">加载中...</div>
  </header>
  <div class="upload-zone" id="uploadZone">
    <div class="icon">📁</div>
    <p>点击选择图片 / 拖拽图片到此处</p>
    <p class="hint">支持 Ctrl+V 粘贴截图</p>
  </div>
  <input type="file" id="fileInput" accept="image/*">
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>正在推理...</p>
  </div>
  <div class="error" id="error"></div>
  <div class="result" id="result">
    <img id="resultImage" alt="检测结果">
    <div class="stats" id="stats"></div>
  </div>
</div>
<script>
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const loading = document.getElementById('loading');
  const error = document.getElementById('error');
  const result = document.getElementById('result');
  const resultImage = document.getElementById('resultImage');
  const stats = document.getElementById('stats');
  const modelInfo = document.getElementById('modelInfo');

  // 页面加载时获取模型信息
  fetch('/api/health').then(r => r.json()).then(d => {
    modelInfo.textContent = '模型: ' + d.model + ' | 设备: ' + d.device;
  }).catch(() => { modelInfo.textContent = '无法连接服务'; });

  // 点击上传区域
  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) upload(fileInput.files[0]);
  });

  // 拖拽上传
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) upload(file);
  });

  // Ctrl+V 粘贴
  document.addEventListener('paste', e => {
    const items = e.clipboardData.items;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        upload(item.getAsFile());
        break;
      }
    }
  });

  function upload(file) {
    error.classList.remove('show');
    result.classList.remove('show');
    loading.classList.add('show');

    const reader = new FileReader();
    reader.onload = function() {
      const base64 = reader.result.split(',')[1];
      fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain', 'X-Filename': encodeURIComponent(file.name) },
        body: base64
      })
      .then(function(resp) {
        if (!resp.ok) { return resp.text().then(function(t) { throw new Error(t); }); }
        return resp.json();
      })
      .then(function(data) {
        resultImage.src = data.annotated_image_base64;
        stats.innerHTML = data.filename + ' | 检测到 <strong>' + data.count + '</strong> 个目标 | 推理耗时: <strong>' + data.speed.inference + 'ms</strong>';
        result.classList.add('show');
      })
      .catch(function(e) {
        error.textContent = '推理失败: ' + e.message;
        error.classList.add('show');
      })
      .finally(function() {
        loading.classList.remove('show');
        fileInput.value = '';
      });
    };
    reader.readAsDataURL(file);
  }
</script>
</body>
</html>"###;

use anyhow::Result;
use axum::{
    extract::State,
    extract::DefaultBodyLimit,
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use base64::Engine;
use serde::Serialize;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use ultralytics_inference::{annotate::annotate_image, YOLOModel};

// ============================================================================
// 数据类型
// ============================================================================

/// 单个检测目标
#[derive(Serialize)]
struct DetectionItem {
    class: String,
    confidence: f32,
    bbox: [f32; 4],
}

/// 推理耗时
#[derive(Serialize)]
struct SpeedInfo {
    preprocess: f32,
    inference: f32,
    postprocess: f32,
}

/// POST /api/detect 响应
#[derive(Serialize)]
struct DetectResponse {
    annotated_image_base64: String,
    detections: Vec<DetectionItem>,
    count: usize,
    speed: SpeedInfo,
    filename: String,
}

/// GET /api/health 响应
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    device: String,
}

// ============================================================================
// API handlers
// ============================================================================

/// GET / — 返回前端页面
async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// GET /api/health — 健康检查
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        model: state.model_name.clone(),
        device: state.device.clone(),
    })
}

/// POST /api/detect — 上传图片并推理（接收 base64 文本）
async fn detect_handler(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: String,
) -> Result<Json<DetectResponse>, (StatusCode, String)> {
    // 提取文件名
    let filename = headers
        .get("x-filename")
        .and_then(|v| v.to_str().ok())
        .map(|f| urlencoding::decode(f).unwrap_or(std::borrow::Cow::Borrowed(f)).into_owned())
        .unwrap_or_else(|| "upload".to_string());

    // base64 解码
    let image_data = base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("base64 解码失败: {}", e)))?;

    // 保存上传的图片到 web_uploads/
    let save_dir = std::path::Path::new("web_uploads");
    std::fs::create_dir_all(save_dir).ok();
    let save_path = save_dir.join(&filename);
    std::fs::write(&save_path, &image_data).ok();

    // 解码图片
    let img = image::load_from_memory(&image_data).map_err(|e| {
        (StatusCode::BAD_REQUEST, format!("无法解码图片: {}", e))
    })?;

    // 推理（加锁串行执行）
    let (annotated, speed, detections, count) = {
        // 使用 unwrap_or_else 恢复污染锁，避免 Mutex 污染导致服务器永久不可用
        let mut model = state.model.lock().unwrap_or_else(|poison| {
            eprintln!("⚠ 模型锁被污染，尝试恢复...");
            poison.into_inner()
        });
        let results = model
            .predict_image(&img, filename.clone())
            .map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {}", e))
            })?;

        // 检查推理结果是否为空
        if results.is_empty() {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "推理返回了空结果".into(),
            ));
        }

        let annotated = annotate_image(&img, &results[0], None);
        let speed = SpeedInfo {
            preprocess: {
                let v = results[0].speed.preprocess;
                if v.is_none() {
                    eprintln!("⚠ preprocess 计时数据缺失，回退为 0.0");
                }
                v.unwrap_or(0.0) as f32
            },
            inference: {
                let v = results[0].speed.inference;
                if v.is_none() {
                    eprintln!("⚠ inference 计时数据缺失，回退为 0.0");
                }
                v.unwrap_or(0.0) as f32
            },
            postprocess: {
                let v = results[0].speed.postprocess;
                if v.is_none() {
                    eprintln!("⚠ postprocess 计时数据缺失，回退为 0.0");
                }
                v.unwrap_or(0.0) as f32
            },
        };

        let mut dets = Vec::new();
        if let Some(ref boxes) = results[0].boxes {
            let xyxy = boxes.xyxy();
            for i in 0..boxes.len() {
                let cls_id = boxes.cls()[i] as usize;
                let name = results[0]
                    .names
                    .get(&cls_id)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let conf = boxes.conf()[i];
                let row = xyxy.row(i);
                dets.push(DetectionItem {
                    class: name,
                    confidence: conf,
                    bbox: [row[0], row[1], row[2], row[3]],
                });
            }
        }
        let count = dets.len();
        (annotated, speed, dets, count)
    };

    // 编码标注图为 JPEG base64
    let mut buf = Cursor::new(Vec::new());
    annotated.write_to(&mut buf, image::ImageFormat::Jpeg).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("图片编码失败: {}", e))
    })?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
    let annotated_image_base64 = format!("data:image/jpeg;base64,{}", b64);

    Ok(Json(DetectResponse {
        annotated_image_base64,
        detections,
        count,
        speed,
        filename,
    }))
}

// ============================================================================
// 服务器启动
// ============================================================================

/// 启动 Web 推理服务（阻塞当前线程）
pub fn run_web_server(
    model: Arc<Mutex<YOLOModel>>,
    model_name: String,
    device: String,
    bind: String,
) -> Result<()> {
    let state = AppState {
        model,
        model_name: model_name.clone(),
        device: device.clone(),
    };

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/health", get(health_handler))
        .route("/api/detect", post(detect_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        // 注意：permissive() 允许任意来源的跨域请求，仅适合本地/内网调试。
        // 若将此服务部署到公网，请改为 CorsLayer::new() + allow_origin(具体域名)
        // 以避免 CSRF 类攻击和未授权的跨站调用。
        .layer(CorsLayer::permissive())
        .with_state(state);

    println!("🌐 YOLO26 Web 推理服务已启动");
    println!("   地址: http://{}", bind);
    println!("   模型: {}", model_name);
    println!("   设备: {}", device);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                tokio::signal::ctrl_c().await.ok();
                eprintln!("\n🛑 收到终止信号，正在优雅关闭服务...");
            })
            .await?;
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}
