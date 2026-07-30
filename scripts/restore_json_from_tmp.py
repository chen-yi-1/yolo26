"""
将 tmp 中对应标签已填充的 JSON 文件移回原类别文件夹。

逻辑：
1. 遍历 dataset/tmp/*.json
2. 检查 dataset/labels/<stem>.txt 是否非空（即用户已打标签）
3. 若是，将 JSON 移回对应的类别文件夹（dataset/<Class>/<stem>.json）
"""

import shutil
from pathlib import Path

DATASET_ROOT = Path(r"dataset")
LABELS_DIR = DATASET_ROOT / "labels"
TMP_DIR = DATASET_ROOT / "tmp"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_label_filled(stem: str) -> bool:
    """检查 labels 中对应 stem 的 txt 是否非空。"""
    txt_path = LABELS_DIR / f"{stem}.txt"
    if not txt_path.is_file():
        return False
    try:
        content = txt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(content.strip())


def resolve_class_dir_for_stem(stem: str) -> Path | None:
    """根据 stem 找到对应的类别文件夹。
    优先方法：扫描所有类别目录，找到包含同名图片的目录。
    """
    class_dirs = [
        d
        for d in DATASET_ROOT.iterdir()
        if d.is_dir() and d.name not in ("labels", "tmp")
    ]
    for class_dir in class_dirs:
        for ext in IMG_EXTS:
            if (class_dir / f"{stem}{ext}").is_file():
                return class_dir
        # 也检查 JSON 是否已经在类别文件夹（极端情况）
        if (class_dir / f"{stem}.json").is_file():
            return class_dir
    return None


def main() -> None:
    json_files = sorted(TMP_DIR.glob("*.json"))
    print(f"[INFO] tmp 中共有 JSON 文件: {len(json_files)} 个")

    moved = 0
    skipped_label_empty = 0
    skipped_no_class = 0
    already_exists = 0

    for json_file in json_files:
        stem = json_file.stem

        # 1. 检查标签是否已填充
        if not is_label_filled(stem):
            skipped_label_empty += 1
            continue

        # 2. 找到对应的类别文件夹
        class_dir = resolve_class_dir_for_stem(stem)
        if class_dir is None:
            print(f"[WARN] 找不到类别文件夹（标签已填充）: {stem}")
            skipped_no_class += 1
            continue

        # 3. 移动 JSON（目标已存在则覆盖）
        dst = class_dir / json_file.name
        overwritten = False
        if dst.exists():
            try:
                dst.unlink()
                overwritten = True
                already_exists += 1
            except (OSError, FileNotFoundError) as e:
                print(f"[ERROR] 无法删除已有文件 {dst}: {e}")
                continue

        try:
            shutil.move(str(json_file), str(dst))
            moved += 1
            tag = "[OVERWRITE]" if overwritten else "[OK]"
            print(f"  {tag} {stem}.json -> {class_dir.name}/")
        except (shutil.Error, OSError) as e:
            print(f"[ERROR] 移动失败 {json_file} -> {dst}: {e}")

    print("\n[DONE] 处理完成")
    print(f"  成功移动 JSON:       {moved}")
    print(f"  其中覆盖已有文件:    {already_exists}")
    print(f"  标签仍为空（跳过）:  {skipped_label_empty}")
    print(f"  找不到类别文件夹:    {skipped_no_class}")
    print(f"  tmp 中剩余 JSON:     {len(list(TMP_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
