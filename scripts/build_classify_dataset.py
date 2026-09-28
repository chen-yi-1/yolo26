"""
将 luyav1 实验记录 + 盆位 ROI 裁剪图转换为 Ultralytics YOLO classify 官方数据集布局。

输入（源）：
    luyav1/.tmp/实验记录.csv      每行一个盆位一条记录，标签取 expert_plant_state 列
    luyav1/dataset/captures/<capture_id>/white_on.png 等 5 个拍摄帧
    luyav1/config/machine.json    按 layer_id + pot_slot 定义的多边形盆位 ROI

输出（目标，classify 官方格式）：
    <out>/train/<expert_plant_state>/<capture_id>_L<layer>_P<slot>.png
    <out>/val/<expert_plant_state>/<capture_id>_L<layer>_P<slot>.png

说明：
    - 裁剪范围 = ROI 多边形的外接矩形，与 luyav1/utils/roi.py 的
      polygon_bounds（x2/y2 取 max+1）保持一致
    - 默认只用 white_on 可见光帧，可用 --frame 指定其他帧
    - train/val 按盆（layer_id + pot_slot）分组划分：同一盆多天拍摄的
      高度相似图片只会整体进入 train 或 val，避免验证集信息泄漏
    - 复制式生成，不修改源数据；重复运行会覆盖同名输出
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

VALID_STATES = {
    "seed_germination",
    "light_initiation",
    "rapid_growth",
    "harvest",
}


def load_pot_rois(machine_path: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """读取 machine.json，返回 (layer_id, pot_slot) -> 多边形顶点列表。"""
    raw = json.loads(machine_path.read_text(encoding="utf-8"))
    rois: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for layer_id, layer in raw["layers"].items():
        for slot, polygon in layer["pot_rois"].items():
            rois[(str(layer_id), str(slot))] = [(int(x), int(y)) for x, y in polygon]
    return rois


def polygon_bounds(polygon: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """外接矩形 (x1, y1, x2, y2)，x2/y2 为 max+1，与 luyav1/utils/roi.py 一致。"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取实验记录 CSV（兼容 utf-8 BOM），返回非空行列表。"""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            row = {
                (key or "").strip(): (value or "").strip()
                for key, value in row.items()
            }
            if row.get("capture_id"):
                rows.append(row)
    return rows


def imread_safe(path: Path) -> np.ndarray | None:
    """Windows 下兼容中文路径的图片读取。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_safe(path: Path, image: np.ndarray) -> bool:
    """Windows 下兼容中文路径的图片写入。"""
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def split_pot_groups(
    pot_groups: list[tuple[str, str]], val_ratio: float, seed: int
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """按盆分组随机划分 train/val，同一盆的所有记录只进一侧。"""
    groups = pot_groups[:]
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_ratio)) if groups else 0
    return set(groups[n_val:]), set(groups[:n_val])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 YOLO26 classify 官方格式数据集（expert_plant_state 分类）"
    )
    parser.add_argument(
        "--csv",
        default=r"C:\Users\EDY\Desktop\workspace\luyav1\.tmp\实验记录.csv",
        help="实验记录 CSV 路径",
    )
    parser.add_argument(
        "--captures",
        default=r"C:\Users\EDY\Desktop\workspace\luyav1\dataset\captures",
        help="captures 数据根目录",
    )
    parser.add_argument(
        "--machine",
        default=r"C:\Users\EDY\Desktop\workspace\luyav1\config\machine.json",
        help="machine.json 盆位 ROI 配置",
    )
    parser.add_argument(
        "--out",
        default=r"C:\Users\EDY\Desktop\workspace\yolo26\datasets\plant_state_classify",
        help="输出数据集根目录",
    )
    parser.add_argument(
        "--frame",
        default="white_on",
        help="裁剪使用的拍摄帧名（默认 white_on，可选 red_on/red_ambient/nir_on/nir_ambient）",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="验证集盆数比例（默认 0.2）"
    )
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子（默认 42）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    captures_dir = Path(args.captures)
    machine_path = Path(args.machine)
    out_root = Path(args.out)
    frame_exts = (".png", ".jpg", ".jpeg")

    rois = load_pot_rois(machine_path)
    rows = read_rows(csv_path)
    print(f"CSV 记录数: {len(rows)}")

    # 过滤出有效记录，并按盆 (layer_id, pot_slot) 分组
    items: list[tuple[dict[str, str], Path]] = []
    pot_groups: set[tuple[str, str]] = set()
    skipped_state = 0
    skipped_missing = 0
    for row in rows:
        state = row.get("expert_plant_state", "")
        if state not in VALID_STATES:
            skipped_state += 1
            continue
        layer_id = row.get("layer_id", "")
        pot_slot = row.get("pot_slot", "")
        roi = rois.get((layer_id, pot_slot))
        capture_dir = captures_dir / row["capture_id"]
        frame_path = next(
            (capture_dir / f"{args.frame}{ext}" for ext in frame_exts
             if (capture_dir / f"{args.frame}{ext}").is_file()),
            None,
        )
        if roi is None or frame_path is None:
            skipped_missing += 1
            print(
                f"[警告] 跳过 capture={row['capture_id']} layer={layer_id} "
                f"pot={pot_slot}（ROI 缺失或 {args.frame} 帧不存在）"
            )
            continue
        items.append((row, frame_path))
        pot_groups.add((layer_id, pot_slot))

    print(
        f"有效记录: {len(items)}（无效标签 {skipped_state} 条，"
        f"缺图/缺ROI {skipped_missing} 条），盆组数: {len(pot_groups)}"
    )
    if not items:
        raise SystemExit("没有可用的有效记录")

    train_pots, val_pots = split_pot_groups(
        sorted(pot_groups), args.val_ratio, args.seed
    )
    print(
        f"按盆划分: train {len(train_pots)} 盆 {sorted(train_pots)} | "
        f"val {len(val_pots)} 盆 {sorted(val_pots)}"
    )

    for sub in ("train", "val"):
        for state in sorted(VALID_STATES):
            (out_root / sub / state).mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }
    bounds_cache = {
        key: polygon_bounds(polygon) for key, polygon in rois.items()
    }
    failed = 0
    # CSV 中同一 capture 的记录连续出现，仅缓存当前 capture 的整图以控制内存
    cached_id: str | None = None
    image: np.ndarray | None = None
    for row, frame_path in items:
        state = row["expert_plant_state"]
        layer_id = row["layer_id"]
        pot_slot = row["pot_slot"]
        subset = "val" if (layer_id, pot_slot) in val_pots else "train"
        capture_id = row["capture_id"]
        if capture_id != cached_id:
            cached_id = capture_id
            image = imread_safe(frame_path)
        if image is None:
            failed += 1
            print(f"[警告] 图片读取失败: {frame_path}")
            continue
        x1, y1, x2, y2 = bounds_cache[(layer_id, pot_slot)]
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            failed += 1
            print(f"[警告] 裁剪为空: {frame_path} L{layer_id} P{pot_slot}")
            continue
        dst = out_root / subset / state / f"{capture_id}_L{layer_id}_P{pot_slot}.png"
        if not imwrite_safe(dst, crop):
            failed += 1
            print(f"[警告] 写入失败: {dst}")
            continue
        counts[subset][state] += 1

    total_train = sum(counts["train"].values())
    total_val = sum(counts["val"].values())
    print(f"\n完成: train={total_train}, val={total_val}, 失败={failed}")
    for state in sorted(VALID_STATES):
        t, v = counts["train"][state], counts["val"][state]
        warn = "  [警告] val 中无该类别样本" if t > 0 and v == 0 else ""
        print(f"  {state}: train={t}, val={v}{warn}")
    print(f"\n输出目录: {out_root}")
    print(f"训练命令示例: yolo classify train data={out_root} model=yolo26n-cls.pt epochs=100 imgsz=224")


if __name__ == "__main__":
    main()
