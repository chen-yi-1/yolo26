mod web;

use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

// ============================================================================
// 命令行参数定义（仅 Web 服务模式）
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "yolo26-inference",
    version,
    about = "YOLO26 ONNX 推理 Web 服务 — 纯 Rust 实现",
    long_about = "基于 ultralytics-inference crate 的 YOLO 推理 Web 服务。\n\
                  启动 Axum 服务器提供 HTML 拖拽界面与 /api/detect JSON 接口。"
)]
struct Args {
    /// ONNX 模型路径（或已知模型名如 yolo26n.onnx，会自动下载）
    #[arg(short, long, default_value = "models/best_12.onnx")]
    model: String,

    /// 置信度阈值 (0.0 ~ 1.0)
    #[arg(long, default_value = "0.3")]
    conf: f32,

    /// IoU 阈值 (0.0 ~ 1.0)
    #[arg(long, default_value = "0.7")]
    iou: f32,

    /// 最大检测数
    #[arg(long, default_value = "300")]
    max_det: usize,

    /// 推理图像尺寸（宽 高），设为 0 则使用模型默认值
    #[arg(long, value_names = ["WIDTH", "HEIGHT"], num_args = 2)]
    imgsz: Option<Vec<u32>>,

    /// 执行设备 (cpu, cuda:0, tensorrt:0, coreml, openvino, directml:0, xnnpack)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// 使用 FP16 半精度推理
    #[arg(long)]
    half: bool,

    /// Web 服务端口（默认 3001）
    #[arg(long, default_value = "3001")]
    port: u16,

    /// Web 服务绑定地址
    #[arg(long, default_value = "0.0.0.0")]
    bind: String,

    /// 标签生成模式：为数据集目录下的图片生成 YOLO 格式 txt 标签（仅框，类别由文件夹决定）
    #[arg(long, value_name = "DATASET_ROOT")]
    labels: Option<PathBuf>,
}

// ============================================================================
// 主入口（唯一模式：Web 服务）
// ============================================================================

fn main() -> Result<()> {
    let args = Args::parse();

    if !Path::new(&args.model).exists() && !is_known_model(&args.model) {
        println!(
            "⚠ 模型文件 '{}' 不存在，将尝试自动下载...",
            args.model
        );
    }

    let device = parse_device(&args.device)?;

    if let Some(dataset_root) = &args.labels {
        run_label_mode(&args, device, dataset_root)
    } else {
        run_web_mode(&args, device)
    }
}

// ============================================================================
// 标签生成模式入口
// ============================================================================

/// 读取 classes.txt，返回 {class_name: index}
fn read_class_map(classes_file: &Path) -> Result<HashMap<String, usize>> {
    let content = std::fs::read_to_string(classes_file)
        .with_context(|| format!("无法读取类别文件: {}", classes_file.display()))?;
    let mut map = HashMap::new();
    for (idx, line) in content.lines().enumerate() {
        let name = line.trim();
        if !name.is_empty() {
            map.insert(name.to_string(), idx);
        }
    }
    Ok(map)
}

fn run_label_mode(args: &Args, device: Device, dataset_root: &Path) -> Result<()> {
    let class_map = read_class_map(&dataset_root.join("classes.txt"))?;
    println!("类别映射 ({} 类):", class_map.len());
    let mut class_vec: Vec<_> = class_map.iter().collect();
    class_vec.sort_by_key(|(_, &idx)| idx);
    for (name, idx) in &class_vec {
        println!("  {}: {}", idx, name);
    }

    let mut config = InferenceConfig::new()
        .with_confidence(args.conf)
        .with_iou(args.iou)
        .with_max_det(args.max_det)
        .with_device(device)
        .with_half(args.half);
    if let Some(ref sz) = args.imgsz {
        if sz.len() >= 2 {
            config = config.with_imgsz(sz[0] as usize, sz[1] as usize);
        }
    }

    println!("\n🔧 加载模型: {}", args.model);
    let mut model = YOLOModel::load_with_config(&args.model, config)
        .with_context(|| format!("无法加载模型: {}", args.model))?;

    let img_exts = |p: &Path| -> bool {
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp" | "webp" | "tif" | "tiff"))
            .unwrap_or(false)
    };

    let mut total = 0usize;
    let mut labeled = 0usize;
    let mut no_det = 0usize;
    let mut errors = 0usize;

    let mut entries: Vec<_> = std::fs::read_dir(dataset_root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let folder = entry.file_name();
        let folder_str = folder.to_string_lossy();
        let class_id = match class_map.get(folder_str.as_ref()) {
            Some(&id) => id,
            None => continue,
        };

        let mut images: Vec<_> = std::fs::read_dir(entry.path())?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && img_exts(p))
            .collect();
        images.sort();

        if images.is_empty() {
            continue;
        }
        println!("\n[{}] class_id={}, {} 张", folder_str, class_id, images.len());

        for img_path in &images {
            total += 1;
            let results = match model.predict(img_path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  [跳过] 推理失败 {}: {}", img_path.display(), e);
                    errors += 1;
                    continue;
                }
            };

            let result = &results[0];
            let boxes = match result.boxes.as_ref().filter(|b| b.len() > 0) {
                Some(b) => b,
                None => {
                    // 无检测：写空文件
                    let txt_path = img_path.with_extension("txt");
                    std::fs::write(txt_path, b"")?;
                    no_det += 1;
                    continue;
                }
            };

            // xyxyn 返回归一化 [x1,y1,x2,y2]；转为 YOLO xywh
            let xyxy = boxes.xyxyn();
            let mut lines = Vec::with_capacity(boxes.len());
            for i in 0..xyxy.nrows() {
                let x1 = xyxy[[i, 0]];
                let y1 = xyxy[[i, 1]];
                let x2 = xyxy[[i, 2]];
                let y2 = xyxy[[i, 3]];
                let cx = ((x1 + x2) / 2.0).clamp(0.0, 1.0);
                let cy = ((y1 + y2) / 2.0).clamp(0.0, 1.0);
                let w = (x2 - x1).clamp(0.0, 1.0);
                let h = (y2 - y1).clamp(0.0, 1.0);
                lines.push(format!("{} {:.6} {:.6} {:.6} {:.6}", class_id, cx, cy, w, h));
            }

            let txt_path = img_path.with_extension("txt");
            let mut content = lines.join("\n");
            if !content.is_empty() {
                content.push('\n');
            }
            std::fs::write(txt_path, content.as_bytes())?;
            labeled += 1;
        }
    }

    println!(
        "\n完成。处理 {} 张，有框 {} 张，无检测 {} 张，错误 {} 张。",
        total, labeled, no_det, errors
    );
    Ok(())
}

// ============================================================================
// Web 服务模式入口
// ============================================================================

fn run_web_mode(args: &Args, device: Device) -> Result<()> {
    use std::sync::{Arc, Mutex};

    let mut config = InferenceConfig::new()
        .with_confidence(args.conf)
        .with_iou(args.iou)
        .with_max_det(args.max_det)
        .with_device(device)
        .with_half(args.half);

    if let Some(ref sz) = args.imgsz {
        if sz.len() >= 2 {
            config = config.with_imgsz(sz[0] as usize, sz[1] as usize);
        }
    }

    println!("🔧 加载模型: {}", args.model);
    println!("   配置: conf={}, iou={}, device={}, half={}",
        args.conf, args.iou, args.device, args.half);

    let model = YOLOModel::load_with_config(&args.model, config)
        .with_context(|| format!("无法加载模型: {}", args.model))?;

    let bind_addr = format!("{}:{}", args.bind, args.port);
    web::run_web_server(
        Arc::new(Mutex::new(model)),
        args.model.clone(),
        args.device.clone(),
        bind_addr,
    )
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 解析设备字符串为 Device 枚举
fn parse_device(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "xnnpack" => Ok(Device::Xnnpack),
        "coreml" => Ok(Device::CoreMl),
        "openvino" => Ok(Device::OpenVino),
        _ if s.starts_with("cuda:") => {
            let id: usize = s
                .strip_prefix("cuda:")
                .unwrap()
                .parse()
                .context("cuda 设备 ID 无效")?;
            Ok(Device::Cuda(id))
        }
        _ if s.starts_with("tensorrt:") => {
            let id: usize = s
                .strip_prefix("tensorrt:")
                .unwrap()
                .parse()
                .context("tensorrt 设备 ID 无效")?;
            Ok(Device::TensorRt(id))
        }
        _ if s.starts_with("directml:") => {
            let id: usize = s
                .strip_prefix("directml:")
                .unwrap()
                .parse()
                .context("directml 设备 ID 无效")?;
            Ok(Device::DirectMl(id))
        }
        _ if s.starts_with("rocm:") => {
            let id: usize = s
                .strip_prefix("rocm:")
                .unwrap()
                .parse()
                .context("rocm 设备 ID 无效")?;
            Ok(Device::Rocm(id))
        }
        other => anyhow::bail!(
            "不支持的设备: '{}'。支持: cpu, cuda:N, tensorrt:N, coreml, openvino, directml:N, rocm:N, xnnpack",
            other
        ),
    }
}

/// 判断是否为已知的模型名（会自动从 Ultralytics 下载）
fn is_known_model(name: &str) -> bool {
    let known = [
        "yolo26n.onnx", "yolo26s.onnx", "yolo26m.onnx", "yolo26l.onnx", "yolo26x.onnx",
        "yolo26n-seg.onnx", "yolo26s-seg.onnx", "yolo26m-seg.onnx", "yolo26l-seg.onnx", "yolo26x-seg.onnx",
        "yolo26n-pose.onnx", "yolo26s-pose.onnx", "yolo26m-pose.onnx", "yolo26l-pose.onnx", "yolo26x-pose.onnx",
        "yolo26n-obb.onnx", "yolo26s-obb.onnx", "yolo26m-obb.onnx", "yolo26l-obb.onnx", "yolo26x-obb.onnx",
        "yolo26n-cls.onnx", "yolo26s-cls.onnx", "yolo26m-cls.onnx", "yolo26l-cls.onnx", "yolo26x-cls.onnx",
        "yolo11n.onnx", "yolo11s.onnx", "yolo11m.onnx", "yolo11l.onnx", "yolo11x.onnx",
        "yolov8n.onnx", "yolov8s.onnx", "yolov8m.onnx", "yolov8l.onnx", "yolov8x.onnx",
    ];
    known.contains(&name)
}
