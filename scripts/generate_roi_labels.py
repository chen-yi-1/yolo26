"""
从预训练的植物 ROI 检测模型生成数据集标签。

只用模型预测的 bounding box 几何信息；类别号不使用模型的预测类别，
而是根据图片所在文件夹的品种名，按 dataset/classes.txt 中的顺序写入。

输出 YOLO 格式：每张图片一个同名 .txt，每行一个框：
    <class_id> <x_center> <y_center> <width> <height>
坐标归一化到 [0, 1]。
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def load_class_map(dataset_root: Path) -> dict[str, int]:
    """读取 classes.txt，返回 {class_name: index} 映射。"""
    classes_file = dataset_root / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"未找到类别文件: {classes_file}")

    names = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)

    return {name: idx for idx, name in enumerate(names)}


def xyxy_to_yolo(box, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """将像素坐标 (x1, y1, x2, y2) 转为归一化的 (cx, cy, w, h)。"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (
        max(0.0, min(1.0, cx / img_w)),
        max(0.0, min(1.0, cy / img_h)),
        max(0.0, min(1.0, w / img_w)),
        max(0.0, min(1.0, h / img_h)),
    )


def generate_labels(
    model_path: str,
    dataset_root: Path,
    conf: float = 0.25,
    imgsz: int | None = None,
    device: str = "cpu",
    classes: list[str] | None = None,
) -> None:
    class_map = load_class_map(dataset_root)
    print(f"类别映射 ({len(class_map)} 类): {class_map}")

    # 校验请求的品种都在映射中
    if classes:
        unknown = [c for c in classes if c not in class_map]
        if unknown:
            raise ValueError(f"以下品种不在 classes.txt 中: {unknown}")

    model = YOLO(model_path)
    print(f"模型加载完成: {model_path}  (task={model.task}, names={model.names})")

    # 收集待处理的 (品种文件夹, 图片路径) 列表
    tasks: list[tuple[str, list[Path]]] = []
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name not in class_map:
            continue
        if classes and entry.name not in classes:
            continue

        images = sorted(
            p
            for p in entry.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        )
        if images:
            tasks.append((entry.name, images))

    if not tasks:
        print("未找到任何待处理图片。")
        return

    total_images = sum(len(imgs) for _, imgs in tasks)
    print(f"共 {len(tasks)} 个品种，{total_images} 张图片待处理。\n")

    # 推理参数
    predict_kwargs: dict = {"conf": conf, "device": device, "verbose": False}
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    stats = {"processed": 0, "labeled": 0, "no_detection": 0, "errors": 0}

    for variety, images in tasks:
        class_id = class_map[variety]
        print(f"[{variety}] class_id={class_id}, {len(images)} 张")

        # 逐张推理，单张损坏（truncated 等）不中断整个流程
        for img_path in images:
            stats["processed"] += 1
            try:
                results = model.predict([str(img_path)], **predict_kwargs)
            except Exception as e:
                print(f"  [跳过] 推理失败 {img_path.name}: {e}")
                stats["errors"] += 1
                continue

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                stats["no_detection"] += 1
                # 无检测：写入空文件，便于后续区分"已处理但无目标"
                txt_path = img_path.with_suffix(".txt")
                txt_path.write_text("", encoding="utf-8")
                continue

            img_h, img_w = result.orig_shape  # orig_shape = (H, W)
            lines: list[str] = []
            for box in boxes:
                cx, cy, w, h = xyxy_to_yolo(box.xyxy[0].tolist(), img_w, img_h)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            txt_path = img_path.with_suffix(".txt")
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["labeled"] += 1

    print(
        f"\n完成。处理 {stats['processed']} 张，"
        f"有框 {stats['labeled']} 张，"
        f"无检测 {stats['no_detection']} 张，"
        f"错误 {stats['errors']} 张。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="用预训练 ROI 检测模型生成数据集标签（仅框，类别由文件夹决定）")
    parser.add_argument("--model", default=r"C:\Users\EDY\Desktop\yolo26\archive\2_classes\models\best_12.pt", help="检测模型路径 (.pt)")
    parser.add_argument("--dataset", default=r"C:\Users\EDY\Desktop\yolo26\dataset", help="数据集根目录（含 classes.txt 与品种子文件夹）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--imgsz", type=int, default=None, help="推理尺寸（默认用模型自带）")
    parser.add_argument("--device", default="cpu", help="推理设备：cpu / cuda:0 等（默认 cpu）")
    parser.add_argument("--classes", nargs="*", default=None, help="仅处理指定品种（默认全部）")
    args = parser.parse_args()

    generate_labels(
        model_path=args.model,
        dataset_root=Path(args.dataset),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        classes=args.classes,
    )


if __name__ == "__main__":
    main()
