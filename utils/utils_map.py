import glob
import json
import math
import os
import shutil
import operator
from functools import reduce

import cv2
import numpy as np
from PIL import Image


def log_average_miss_rate(prec, rec, num_images):
    """计算 log-average miss rate (lamr) for object detection evaluation."""
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    mr_9 = np.zeros(9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1]
        if j.size > 0:
            mr_9[i] = mr_tmp[j[-1]]
        else:
            mr_9[i] = 1.0

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, 1 - mr_9))))
    return lamr, mr, fppi


def compute_ap(rec, prec, use_07_metric=False):
    """计算 AP (Average Precision)."""
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out'):
    """计算 mAP (Pascal VOC 标准).

    Returns:
        mAP 值
    """
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')

    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    gt_counter_per_class = {}
    images_gt = glob.glob(GT_PATH + "/*.txt")
    if not images_gt:
        print(f"Error: No ground-truth files found in {GT_PATH}")
        return 0

    # 统计每个类别的 ground truth 数量
    for txt_file in images_gt:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            class_name = line.strip().split(' ')[0]
            gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1

    # 读取检测结果
    dr_files_list = glob.glob(DR_PATH + "/*.txt")
    if not dr_files_list:
        print(f"Error: No detection-results files found in {DR_PATH}")
        return 0

    # 为每个类别处理
    sum_AP = 0.0
    count_real_classes = 0

    for class_name, npos in gt_counter_per_class.items():
        nd = 0
        dr_data = []
        for dr_file in dr_files_list:
            with open(dr_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if parts[0] == class_name:
                    score = float(parts[1])
                    if score >= score_threhold:
                        nd += 1

        if nd == 0:
            continue

        # 创建 detection-results 临时文件
        temp_dr_file = os.path.join(TEMP_FILES_PATH, f"{class_name}_dr.txt")
        with open(temp_dr_file, 'w') as f:
            for dr_file in dr_files_list:
                image_id = os.path.basename(dr_file).replace('.txt', '')
                with open(dr_file, 'r') as dr_f:
                    lines = dr_f.readlines()
                for line in lines:
                    parts = line.strip().split(' ')
                    if parts[0] == class_name and float(parts[1]) >= score_threhold:
                        f.write(f"{image_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}\n")

        # 按置信度排序
        with open(temp_dr_file, 'r') as f:
            lines_dr = f.readlines()
        lines_dr = sorted(lines_dr, key=lambda x: float(x.strip().split(' ')[1]), reverse=True)

        tp = np.zeros(len(lines_dr))
        fp = np.zeros(len(lines_dr))
        gt_checked = {}

        for idx, line in enumerate(lines_dr):
            parts = line.strip().split(' ')
            image_id = parts[0]
            bb = [float(x) for x in parts[2:]]

            gt_file = os.path.join(GT_PATH, f"{image_id}.txt")
            if gt_file not in gt_checked:
                gt_checked[gt_file] = []
                with open(gt_file, 'r') as f:
                    for gt_line in f.readlines():
                        gt_parts = gt_line.strip().split(' ')
                        if gt_parts[0] == class_name:
                            gt_checked[gt_file].append([float(x) for x in gt_parts[1:]])

            ovmax = -1
            gt_match = -1
            for j, gt_bb in enumerate(gt_checked[gt_file]):
                bi = [max(bb[0], gt_bb[0]), max(bb[1], gt_bb[1]),
                      min(bb[2], gt_bb[2]), min(bb[3], gt_bb[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                         (gt_bb[2] - gt_bb[0] + 1) * (gt_bb[3] - gt_bb[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = j

            if ovmax >= MINOVERLAP:
                gt_checked[gt_file][gt_match] = [-1, -1, -1, -1, -1]  # mark as matched
                tp[idx] = 1
            else:
                fp[idx] = 1

        # 计算 AP
        cumsum = 0
        for idx in range(len(tp)):
            tp[idx] += cumsum
            cumsum += tp[idx]
        cumsum = 0
        for idx in range(len(fp)):
            fp[idx] += cumsum
            cumsum += fp[idx]

        rec = tp / float(npos) if npos > 0 else np.zeros(len(tp))
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = compute_ap(rec, prec, False)
        sum_AP += ap
        count_real_classes += 1

    shutil.rmtree(TEMP_FILES_PATH)
    mAP = sum_AP / count_real_classes if count_real_classes > 0 else 0
    return mAP


def get_coco_map(class_names, path):
    """使用 pycocotools 计算 COCO mAP (需要先安装 pycocotools).

    Returns:
        (mAP50, mAP50-95)
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')

    # Build COCO format annotations
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    for i, name in enumerate(class_names):
        coco_gt['categories'].append({"id": i, "name": name, "supercategory": "none"})

    gt_files = glob.glob(GT_PATH + "/*.txt")
    image_id = 0
    ann_id = 0
    for gt_file in gt_files:
        image_id += 1
        img_name = os.path.basename(gt_file).replace('.txt', '.jpg')
        coco_gt['images'].append({"id": image_id, "file_name": img_name})
        with open(gt_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                class_name = parts[0]
                cat_id = class_names.index(class_name)
                bbox = [int(float(parts[1])), int(float(parts[2])),
                        int(float(parts[3])) - int(float(parts[1])),
                        int(float(parts[4])) - int(float(parts[2]))]
                ann_id += 1
                coco_gt['annotations'].append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": cat_id, "bbox": bbox,
                    "area": bbox[2] * bbox[3], "iscrowd": 0
                })

    # Build COCO format detections
    coco_dt = []
    dr_files = glob.glob(DR_PATH + "/*.txt")
    for dr_file in dr_files:
        img_name = os.path.basename(dr_file).replace('.txt', '.jpg')
        img_id = None
        for img in coco_gt['images']:
            if img['file_name'] == img_name:
                img_id = img['id']
                break
        if img_id is None:
            continue
        with open(dr_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                class_name = parts[0]
                score = float(parts[1])
                bbox = [float(parts[2]), float(parts[3]),
                        float(parts[4]) - float(parts[2]),
                        float(parts[5]) - float(parts[3])]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                cat_id = class_names.index(class_name)
                coco_dt.append({
                    "image_id": img_id, "category_id": cat_id,
                    "bbox": bbox, "score": score
                })

    if not coco_dt:
        return 0, 0

    gt_path = os.path.join(path, 'gt.json')
    dt_path = os.path.join(path, 'dt.json')
    with open(gt_path, 'w') as f:
        json.dump(coco_gt, f)
    with open(dt_path, 'w') as f:
        json.dump(coco_dt, f)

    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    os.remove(gt_path)
    os.remove(dt_path)

    return cocoEval.stats[1], cocoEval.stats[0]
