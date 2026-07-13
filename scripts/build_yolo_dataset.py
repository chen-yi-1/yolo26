"""
将 dataset/ 下的品种图片 + ROI 标签转换为 Ultralytics 标准数据集布局。

输入（源）：
    dataset/<class_name>/*.jpg（品种子目录下的图片）
    dataset/labels/*.txt（YOLO 格式标签，全部集中在此目录，文件名与图片 stem 对应）
    dataset/classes.txt（品种名 -> 类别序号）

输出（目标）：
    datasets/images/train/<class_name>_<idx>.jpg
    datasets/images/val/<class_name>_<idx>.jpg
    datasets/labels/train/<class_name>_<idx>.txt
    datasets/labels/val/<class_name>_<idx>.txt
    datasets/datasets.yaml

说明：
    - 标签中每行的首列 class_id 会被忽略，统一替换为 classes.txt 中该品种对应的序号
    - 标签文件不存在或内容为空的图片会被跳过
    - 划分比例默认 80/20（train/val），同一品种的所有图片随机打乱后按比例切分
    - 复制而非移动，源 dataset/ 保持不变
"""

import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_class_map(classes_file: Path) -> list[str]:
    """读取 classes.txt，返回按序号排序的品种名列表。"""
    names: list[str] = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def collect_images(class_dir: Path) -> list[Path]:
    """收集文件夹下所有图片文件。"""
    imgs = []
    for f in sorted(class_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            imgs.append(f)
    return imgs


def split_images(images: list[Path], val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    """随机切分 train/val。"""
    imgs = images[:]
    rng.shuffle(imgs)
    n_val = max(0, round(len(imgs) * val_ratio))
    return imgs[n_val:], imgs[:n_val]


def safe_name(name: str) -> str:
    """把品种名转为安全的文件名片段（替换空格和特殊字符）。"""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def copy_image_with_ext(img_path: Path, dst_img: Path) -> None:
    """原样复制图片，保持原始文件名和扩展名。"""
    shutil.copy2(str(img_path), str(dst_img))


def build_dataset(
    src_root: Path,
    dst_root: Path,
    val_ratio: float,
    seed: int,
    conf_filter: float | None,
) -> None:
    class_names = read_class_map(src_root / "classes.txt")
    print(f"类别数: {len(class_names)}")
    print(f"类别: {class_names}")

    labels_dir = src_root / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"源目录缺少 labels 文件夹: {labels_dir}")

    # 创建目标目录
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dst_root / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    total_train = 0
    total_val = 0
    total_skip_no_label = 0
    total_skip_collision = 0
    used_train_names: set[str] = set()
    used_val_names: set[str] = set()

    for class_idx, class_name in enumerate(class_names):
        class_dir = src_root / class_name
        if not class_dir.is_dir():
            print(f"[警告] 跳过不存在的品种文件夹: {class_name}")
            continue

        images = collect_images(class_dir)
        if not images:
            print(f"[警告] 品种 {class_name} 下无图片")
            continue

        # 预过滤：只保留存在且非空标签的图片
        valid_images: list[Path] = []
        for img_path in images:
            txt = labels_dir / f"{img_path.stem}.txt"
            if txt.is_file() and txt.stat().st_size > 0:
                valid_images.append(img_path)
            else:
                total_skip_no_label += 1

        if not valid_images:
            print(f"[警告] 品种 {class_name} 下无有效标签图片（共{len(images)}张被跳过）")
            continue

        train_imgs, val_imgs = split_images(valid_images, val_ratio, rng)
        print(
            f"  [{class_name}] 原图={len(images)}, 有效={len(valid_images)}, "
            f"train={len(train_imgs)}, val={len(val_imgs)}"
        )

        # 复制 train
        for img_path in train_imgs:
            txt_content = load_and_prepare_label(labels_dir, img_path, class_idx, conf_filter)
            if txt_content is None:
                total_skip_no_label += 1
                continue

            orig_name = img_path.name
            if orig_name in used_train_names:
                print(f"  [警告] train中文件名重复，跳过: {orig_name} ({class_name})")
                total_skip_collision += 1
                continue
            used_train_names.add(orig_name)
            orig_stem = img_path.stem
            orig_suffix = img_path.suffix

            dst_img = dst_root / "images" / "train" / orig_name
            copy_image_with_ext(img_path, dst_img)
            dst_txt = dst_root / "labels" / "train" / f"{orig_stem}.txt"
            dst_txt.write_text(txt_content, encoding="utf-8")
            total_train += 1

        # 复制 val
        for img_path in val_imgs:
            txt_content = load_and_prepare_label(labels_dir, img_path, class_idx, conf_filter)
            if txt_content is None:
                total_skip_no_label += 1
                continue

            orig_name = img_path.name
            if orig_name in used_val_names:
                print(f"  [警告] val中文件名重复，跳过: {orig_name} ({class_name})")
                total_skip_collision += 1
                continue
            used_val_names.add(orig_name)
            orig_stem = img_path.stem
            orig_suffix = img_path.suffix

            dst_img = dst_root / "images" / "val" / orig_name
            copy_image_with_ext(img_path, dst_img)
            dst_txt = dst_root / "labels" / "val" / f"{orig_stem}.txt"
            dst_txt.write_text(txt_content, encoding="utf-8")
            total_val += 1

    collision_msg = f", 跳过（重名冲突）={total_skip_collision}" if total_skip_collision else ""
    print(f"\n总计 train={total_train}, val={total_val}, 跳过（无有效标签）={total_skip_no_label}{collision_msg}")

    # 写 datasets.yaml
    write_datasets_yaml(dst_root, class_names)


def rewrite_label_class_id(txt_content: str, class_idx: int) -> str:
    """替换标签每行首列的 class_id 为指定的品种序号，保留后续 ROI 坐标列。
    同时兼容 5 列（class cx cy w h）和 6 列（带 conf）以及多边形多列格式。"""
    if not txt_content.strip():
        return txt_content
    lines = txt_content.strip().split("\n")
    rewritten = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        parts[0] = str(class_idx)
        rewritten.append(" ".join(parts))
    return "\n".join(rewritten) + "\n" if rewritten else ""


def load_and_prepare_label(
    labels_dir: Path,
    img_path: Path,
    class_idx: int,
    conf_filter: float | None,
) -> str | None:
    """从统一 labels 目录加载标签，执行置信度过滤 + class_id 重写。
    标签不存在或内容为空则返回 None。"""
    src_txt = labels_dir / f"{img_path.stem}.txt"
    if not src_txt.is_file():
        return None
    txt_content = src_txt.read_text(encoding="utf-8")
    if not txt_content.strip():
        return None
    if conf_filter is not None:
        txt_content = filter_by_conf(txt_content, conf_filter)
        if not txt_content.strip():
            return None
    txt_content = rewrite_label_class_id(txt_content, class_idx)
    if not txt_content.strip():
        return None
    return txt_content


def filter_by_conf(txt_content: str, conf_threshold: float) -> str:
    """按置信度阈值过滤标签行。YOLO 格式：class cx cy w h（无置信度列），
    若模型输出带置信度列（class cx cy w h conf），则过滤。"""
    if not txt_content.strip():
        return txt_content
    lines = txt_content.strip().split("\n")
    kept = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 6:
            # class cx cy w h conf
            try:
                conf = float(parts[5])
            except ValueError:
                kept.append(line)
                continue
            if conf >= conf_threshold:
                # 去掉置信度列，保持标准 5 列格式
                kept.append(" ".join(parts[:5]))
        else:
            # 5 列或更少，不过滤
            kept.append(line)
    return "\n".join(kept) + "\n" if kept else ""


def write_datasets_yaml(dst_root: Path, class_names: list[str]) -> None:
    """生成 datasets.yaml，path 使用绝对路径（Windows 反斜杠转为正斜杠）。"""
    abs_path = str(dst_root.resolve()).replace("\\", "/")
    lines = [
        f"path: {abs_path}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        # 用引号包裹含特殊字符的名字
        safe = name.replace('"', '\\"')
        lines.append(f'  {idx}: "{safe}"')

    yaml_path = dst_root / "datasets.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已生成 {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 dataset/ 转为 Ultralytics 标准数据集布局 datasets/")
    parser.add_argument("--src", default="dataset", help="源 dataset 根目录")
    parser.add_argument("--dst", default="datasets", help="目标数据集根目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42，保证可复现）")
    parser.add_argument("--conf", type=float, default=None, help="标签置信度阈值（仅当标签含 conf 列时生效，默认不过滤）")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not (src_root / "classes.txt").exists():
        raise SystemExit(f"源目录缺少 classes.txt: {src_root / 'classes.txt'}")

    if any(dst_root.rglob("*")):
        print(f"[警告] 目标目录 {dst_root} 非空，继续将覆盖同名文件。")

    build_dataset(
        src_root=src_root,
        dst_root=dst_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        conf_filter=args.conf,
    )


if __name__ == "__main__":
    main()
