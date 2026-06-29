mod web;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::Path;
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
    run_web_mode(&args, device)
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
