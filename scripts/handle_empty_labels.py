"""
处理空标签文件：
1. 找出 dataset/labels/ 中所有空的 .txt 标签文件
2. 删除对应的 .json 文件（在类别文件夹下）
3. 将对应的图片复制到 dataset/tmp/ 下，保持文件名唯一
"""

import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASET_ROOT = Path(r"C:\Users\EDY\Desktop\yolo26\dataset")
LABELS_DIR = DATASET_ROOT / "labels"
TMP_DIR = DATASET_ROOT / "tmp"


def find_empty_labels() -> list[str]:
    """返回空标签的 stem 列表（不含 .txt 后缀）"""
    stems = []
    for txt_path in sorted(LABELS_DIR.glob("*.txt")):
        try:
            content = txt_path.read_text(encoding="utf-8")
        except Exception:
            content = ""
        if not content.strip():
            stems.append(txt_path.stem)
    return stems


def resolve_class_and_paths(stem: str) -> tuple[Path | None, Path | None]:
    """根据 stem（如 Kale_01）找到对应的 json 文件和图片文件。
    返回 (json_path, img_path)，找不到为 None。
    """
    # 遍历所有类别目录（排除 labels 和 tmp）
    class_dirs = [d for d in DATASET_ROOT.iterdir() if d.is_dir() and d.name not in ("labels", "tmp")]
    json_path = None
    img_path = None
    for class_dir in class_dirs:
        # 找 json
        candidate_json = class_dir / f"{stem}.json"
        if candidate_json.is_file():
            json_path = candidate_json
        # 找图片
        for ext in IMG_EXTS:
            candidate_img = class_dir / f"{stem}{ext}"
            if candidate_img.is_file():
                img_path = candidate_img
                break
        if json_path is not None and img_path is not None:
            break
    return json_path, img_path


def get_unique_dst(dst_dir: Path, filename: str) -> Path:
    """生成唯一目标路径，避免重名覆盖。若已存在则追加 _1, _2 ..."""
    dst = dst_dir / filename
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    empty_stems = find_empty_labels()
    print(f"[INFO] 发现空标签文件: {len(empty_stems)} 个")

    deleted_json = 0
    missing_json = 0
    copied_imgs = 0
    missing_imgs = 0

    for stem in empty_stems:
        json_path, img_path = resolve_class_and_paths(stem)

        # 1. 删除 json
        if json_path is not None:
            try:
                json_path.unlink()
                deleted_json += 1
            except Exception as e:
                print(f"[ERROR] 删除 JSON 失败 {json_path}: {e}")
        else:
            missing_json += 1

        # 2. 复制图片到 tmp
        if img_path is not None:
            dst = get_unique_dst(TMP_DIR, img_path.name)
            try:
                shutil.copy2(img_path, dst)
                copied_imgs += 1
            except Exception as e:
                print(f"[ERROR] 复制图片失败 {img_path} -> {dst}: {e}")
        else:
            missing_imgs += 1

    print(f"\n[DONE] 处理完成")
    print(f"  删除 JSON 数量: {deleted_json}")
    print(f"  找不到 JSON:   {missing_json}")
    print(f"  复制图片数量:   {copied_imgs}")
    print(f"  找不到图片:     {missing_imgs}")
    print(f"  图片输出目录:   {TMP_DIR}")


if __name__ == "__main__":
    main()
