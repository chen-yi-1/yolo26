"""
重命名 dataset 目录下的植物图片和对应 JSON 文件。
命名格式: 类别_编号.ext (例如 Basil_01.png, Basil_01.json)

用法:
    python scripts/rename_dataset.py --dry-run   # 预览模式，不实际改名
    python scripts/rename_dataset.py              # 执行改名
"""

import argparse
import json
from pathlib import Path

DATASET_DIR = Path(r"C:\Users\EDY\Desktop\yolo26\dataset")

# 支持的图片扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_image_groups(category_dir: Path):
    """收集目录下所有图片及其对应 JSON，按图片分组。"""
    groups = []
    for f in sorted(category_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS:
            json_path = f.with_suffix(".json")
            groups.append((f, json_path if json_path.exists() else None))
    return groups


def update_json_image_path(json_path: Path, new_image_name: str):
    """更新 JSON 文件中的 imagePath 字段。"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "imagePath" in data:
            data["imagePath"] = new_image_name
            json_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return True
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [警告] 更新 JSON 失败 {json_path.name}: {e}")
    return False


def rename_category(category_dir: Path, dry_run: bool):
    """重命名单个类别目录下的所有文件。"""
    category_name = category_dir.name
    groups = collect_image_groups(category_dir)

    if not groups:
        print(f"  [{category_name}] 无图片文件，跳过")
        return 0, 0

    renamed_count = 0
    json_updated_count = 0

    for idx, (img_path, json_path) in enumerate(groups, start=1):
        new_base = f"{category_name}_{idx:02d}"
        new_img_name = f"{new_base}{img_path.suffix}"
        new_img_path = img_path.parent / new_img_name

        # 防止自身重名冲突
        if new_img_path.resolve() == img_path.resolve():
            pass  # 同名，跳过
        elif new_img_path.exists():
            print(f"  [冲突] 目标已存在: {new_img_name}")
            continue

        if dry_run:
            print(f"  {img_path.name} -> {new_img_name}")
        else:
            img_path.rename(new_img_path)
            renamed_count += 1

        # 处理对应 JSON
        if json_path and json_path.exists():
            new_json_name = f"{new_base}.json"
            new_json_path = json_path.parent / new_json_name

            if new_json_path.resolve() != json_path.resolve() and new_json_path.exists():
                print(f"  [冲突] 目标 JSON 已存在: {new_json_name}")
                continue

            if dry_run:
                print(f"  {json_path.name} -> {new_json_name}")
            else:
                json_path.rename(new_json_path)
                renamed_count += 1
                if update_json_image_path(new_json_path, new_img_name):
                    json_updated_count += 1

    return renamed_count, json_updated_count


def main():
    parser = argparse.ArgumentParser(description="重命名 dataset 图片和 JSON 文件")
    parser.add_argument(
        "--dry-run", action="store_true", help="预览模式，不实际改名"
    )
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        print(f"数据集目录不存在: {DATASET_DIR}")
        return

    total_renamed = 0
    total_json_updated = 0

    # 按目录名排序，处理所有子目录
    categories = sorted(
        [d for d in DATASET_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name.lower(),
    )

    for cat_dir in categories:
        renamed, json_updated = rename_category(cat_dir, args.dry_run)
        total_renamed += renamed
        total_json_updated += json_updated

    print("\n" + "=" * 50)
    if args.dry_run:
        print("预览模式完成，未实际改名")
    else:
        print(f"改名完成: {total_renamed} 个文件")
        print(f"JSON imagePath 更新: {total_json_updated} 个")


if __name__ == "__main__":
    main()
