# YOLO26 自定义训练循环重构 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 废弃 ultralytics.YOLO.train()，实现 yolov8-pytorch 风格的自定义训练循环，简化路径管理。

**Architecture:** 使用 ultralytics 的 DetectionModel 作为模型定义 + ultralytics 的 DataLoader，自己实现 Loss/EMA/LR/train loop/inference。配置集中在 train.py 顶部，纯相对路径。Loss 针对 yolo26 (end2end + reg_max=1) 简化：只含 CIoU box loss + BCE cls loss，无 DFL。

**Model API (verified):**
- `model.nc = 80` (num classes)
- `model.stride = tensor([8., 16., 32.])` (3 detection layers)
- `model.train()` → `model(x)` returns `{'one2many': {'boxes': [B,4,8400], 'scores': [B,80,8400], 'feats': [3 tensors]}, 'one2one': {...}}`

**Tech Stack:** PyTorch, ultralytics (model + dataloader only), numpy, opencv-python, matplotlib, Pillow

---

## File Map

| File | Action | Lines | Responsibility |
|------|--------|-------|----------------|
| `nets/__init__.py` | Create | 0 | Package init |
| `nets/yolo_training.py` | Create | ~300 | Loss, TaskAlignedAssigner, ModelEMA, LR scheduler |
| `utils/utils_bbox.py` | Create | ~200 | make_anchors, dist2bbox, DecodeBox |
| `utils/utils_map.py` | Create | ~100 | get_map, get_coco_map |
| `utils/callbacks.py` | Create | ~200 | LossHistory, EvalCallback |
| `utils/utils_fit.py` | Create | ~120 | fit_one_epoch |
| `train.py` | Rewrite | ~350 | Config + two-phase training loop |
| `yolo.py` | Rewrite | ~250 | Inference (no ultralytics.YOLO) |
| `get_map.py` | Modify | ~20 | Remove config.py dependency |
| `config.py` | Delete | — | Configs moved to train.py / yolo.py |
| `utils/utils.py` | No change | — | Keep existing |

---

### Task 1: Create `utils/utils_bbox.py`

**Files:**
- Create: `utils/utils_bbox.py`

- [ ] **Step 1: Write the file**

```python
import numpy as np
import torch
import torch.nn as nn


def make_anchors(feats, strides, grid_cell_offset=1):
    """从特征图生成锚点和步长张量。

    Args:
        feats: 特征图列表, 每个 shape [B, C, H, W]
        strides: 步长列表
        grid_cell_offset: 网格偏移

    Returns:
        anchor_points: [num_anchors, 2] 每个锚点的 (x, y) 坐标
        stride_tensor: [num_anchors, 1] 每个锚点的步长
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将 anchor 距离解码为 bbox (基于左/上/右/下距离)。
    
    yolo26 的 reg_max=1, distance 即直接回归的 bbox 偏移。
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class DecodeBox:
    """推理时将模型输出解码为边界框。

    用于 yolo.py 推理流程中从模型原始输出到最终检测结果。
    """
    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def decode_box(self, outputs):
        """从模型输出解码边界框。

        Args:
            outputs: 模型推理输出, yolo26 为 dict with 'one2one' key

        Returns:
            list of tensors, 每个 shape [1, num_anchors, 4 + 1 + num_classes]
        """
        if isinstance(outputs, dict):
            # yolo26 end2end 推理: 使用 one2one 分支
            out = outputs['one2one']
            boxes = out['boxes']  # [1, 4, num_anchors]
            scores = out['scores']  # [1, num_classes, num_anchors]
            # 转置为 [1, num_anchors, 4] 和 [1, num_anchors, num_classes]
            boxes = boxes.permute(0, 2, 1)
            scores = scores.permute(0, 2, 1)
            # 拼接: [1, num_anchors, 4 + num_classes]
            output = torch.cat([boxes, scores], dim=-1)
            # 转换为 [batch, num_anchors, 4 + 1 + num_classes] 格式
            # 加入 objectness (全 1, yolo26 无 objectness)
            obj = torch.ones((output.shape[0], output.shape[1], 1), device=output.device)
            output = torch.cat([output[..., :4], obj, output[..., 4:]], dim=-1)
            return [output]
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape,
                            letterbox_image, conf_thres=0.5, nms_thres=0.4):
        """NMS 后处理, 参考 yolov8-pytorch 实现。

        Returns:
            list of tensors, 每个 [num_detections, 7] (x1,y1,x2,y2,conf,cls,...)
        """
        # 将预测框缩放到原图大小
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # 置信度过滤
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue

            # 合并置信度: obj_conf * class_conf
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # NMS 按类别
            unique_labels = detections[:, -1].cpu().unique()
            if unique_labels.shape[0] == 0:
                continue

            boxes, scores = detections[:, :4], detections[:, 4] * detections[:, 5]
            iou = self._box_iou(boxes, boxes)
            
            keep = torch.ones(len(boxes), dtype=torch.bool)
            for cls in unique_labels:
                cls_mask = (detections[:, -1] == cls)
                if cls_mask.sum() == 0:
                    continue
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                cls_keep = torch.ones(len(cls_boxes), dtype=torch.bool)
                _, sorted_idx = cls_scores.sort(descending=True)
                for idx in sorted_idx:
                    if not cls_keep[idx]:
                        continue
                    ious = self._box_iou(cls_boxes[idx:idx+1], cls_boxes)
                    cls_keep[ious[0] > nms_thres] = False
                    cls_keep[idx] = True
                keep[cls_mask] = cls_keep

            detections = detections[keep]
            if not detections.size(0):
                continue

            # 缩放回原图
            if letterbox_image:
                scale = min(input_shape[1] / image_shape[1], input_shape[0] / image_shape[0])
                nw = int(image_shape[1] * scale)
                nh = int(image_shape[0] * scale)
                dx = (input_shape[1] - nw) // 2
                dy = (input_shape[0] - nh) // 2
                detections[:, [0, 2]] = (detections[:, [0, 2]] - dx) / scale
                detections[:, [1, 3]] = (detections[:, [1, 3]] - dy) / scale
            else:
                detections[:, [0, 2]] = detections[:, [0, 2]] * image_shape[1] / input_shape[1]
                detections[:, [1, 3]] = detections[:, [1, 3]] * image_shape[0] / input_shape[0]

            # Clip to image bounds
            detections[:, 0] = torch.clamp(detections[:, 0], min=0, max=image_shape[1])
            detections[:, 1] = torch.clamp(detections[:, 1], min=0, max=image_shape[0])
            detections[:, 2] = torch.clamp(detections[:, 2], min=0, max=image_shape[1])
            detections[:, 3] = torch.clamp(detections[:, 3], min=0, max=image_shape[0])

            output[i] = detections.cpu().numpy()

        return output

    @staticmethod
    def _box_iou(box1, box2):
        """计算 box IoU, 用于 NMS。"""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
        inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
        inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
        inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (area1.unsqueeze(1) + area2 - inter_area + 1e-16)
        return iou
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, '.'); from utils.utils_bbox import make_anchors, dist2bbox, DecodeBox; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/utils_bbox.py
git commit -m "feat: add utils_bbox (make_anchors, dist2bbox, DecodeBox)"
```

---

### Task 2: Create `utils/utils_map.py`

**Files:**
- Create: `utils/utils_map.py`

- [ ] **Step 1: Write the file**

```python
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
    """计算 mAP (Pascal VOC 标准)。

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

            if ovmax >= MINOVERLAP and gt_match not in [g for g, _ in enumerate(gt_checked[gt_file]) if False]:
                if gt_match not in [x[0] for x in enumerate(gt_checked[gt_file])]:
                    tp[idx] = 1
                    gt_checked[gt_file][gt_match] = [-1, -1, -1, -1, -1]
                else:
                    fp[idx] = 1
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
    """使用 pycocotools 计算 COCO mAP (需要先安装 pycocotools)。

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
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, '.'); from utils.utils_map import get_map, get_coco_map; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/utils_map.py
git commit -m "feat: add utils_map (get_map, get_coco_map)"
```

---

### Task 3: Create `nets/yolo_training.py`

**Files:**
- Create: `nets/__init__.py`
- Create: `nets/yolo_training.py`

- [ ] **Step 1: Create `nets/__init__.py`**

```python
# nets package
```

- [ ] **Step 2: Create `nets/yolo_training.py`**

```python
import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_bbox import make_anchors


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """选择在真实框内的锚点中心。

    Args:
        xy_centers (Tensor): shape(num_anchors, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4) xyxy format
    Return:
        (Tensor): shape(b, n_boxes, num_anchors)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                            dim=2).view(bs, n_boxes, n_anchors, -1)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """如果锚点被分配给多个真实框, 选择 IoU 最高的那个。"""
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner, 参考 yolov8-pytorch。"""

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.roll_out_thr = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        self.roll_out = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts,
            topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.long().squeeze(-1)
        bbox_scores = pd_scores[ind[0], :, ind[1]]
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1),
                            xywh=False, CIoU=True).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs[~topk_mask] = 0
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64,
                                 device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """计算 bbox IoU (支持 CIoU/DIoU/GIoU)。"""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def xywh2xyxy(x):
    """Convert xywh to xyxy format."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class BboxLoss(nn.Module):
    """yolo26 Box Loss: CIoU only (reg_max=1, no DFL)."""

    def __init__(self, reg_max=1):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = reg_max > 1  # yolo26: False

    def forward(self, pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        loss_dfl = torch.tensor(0.0).to(pred_bboxes.device)
        return loss_iou, loss_dfl


class Loss:
    """yolo26 自定义 Loss。

    针对 yolo26 特点:
    - reg_max=1 (无 DFL): 只计算 CIoU box loss + BCE cls loss
    - dual-head: 训练使用 one2many 分支
    - 模型已输出解码好的 boxes 和 scores
    """
    def __init__(self, model):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # tensor([8., 16., 32.])
        self.nc = model.nc
        self.reg_max = getattr(model, 'reg_max', 1)
        self.no = self.nc + self.reg_max * 4  # nc + 4

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, roll_out_thr=64)
        self.bbox_loss = BboxLoss(self.reg_max - 1 if self.reg_max > 1 else 1)

    def preprocess(self, targets, batch_size, scale_tensor):
        """预处理标签: padding + 缩放到原图大小。"""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts_max = counts.max() if counts.numel() > 0 else 0
            out = torch.zeros(batch_size, counts_max, 5, device=targets.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def __call__(self, outputs, batch):
        device = outputs['one2many']['scores'].device
        loss = torch.zeros(3, device=device)

        # yolo26 模型已输出 boxes 和 scores
        # boxes:  [B, 4, num_anchors]
        # scores: [B, nc, num_anchors]
        pred_bboxes = outputs['one2many']['boxes'].permute(0, 2, 1)     # [B, num_anch, 4]
        pred_scores = outputs['one2many']['scores'].permute(0, 2, 1)    # [B, num_anch, nc]
        feats = outputs['one2many']['feats']

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 1.0)

        # 标签预处理
        targets = torch.cat((batch[:, 0].view(-1, 1),
                             batch[:, 1].view(-1, 1),
                             batch[:, 2:]), 1)
        targets = self.preprocess(targets.to(device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Task-Aligned Assigner
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # BCE 分类损失
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # CIoU 边界框损失 (无 DFL)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_bboxes, target_bboxes,
                                              target_scores, target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain (always 0 for yolo26)
        return loss.sum()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Exponential Moving Average (EMA), 参考 yolov8-pytorch。"""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters,
                     warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    """获取学习率调度函数。"""

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters,
                          warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) /
                               (total_iters - warmup_total_iters - no_aug_iter)))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        return lr * decay_rate ** n

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters,
                       warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """设置优化器学习率。"""
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, '.'); from nets.yolo_training import Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add nets/__init__.py nets/yolo_training.py
git commit -m "feat: add yolo_training (Loss, TaskAlignedAssigner, EMA, LR)"
```

---

### Task 4: Create `utils/callbacks.py`

**Files:**
- Create: `utils/callbacks.py`

- [ ] **Step 1: Write the file**

```python
import datetime
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox
from utils.utils_map import get_coco_map, get_map


class LossHistory:
    """训练损失记录和可视化。"""

    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss) + "\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss) + "\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback:
    """训练过程中定期在验证集上计算 mAP。"""

    def __init__(self, net, input_shape, class_names, num_classes, val_lines,
                 log_dir, cuda, map_out_path=".temp_map_out", max_boxes=100,
                 confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        super().__init__()

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        self.maps = [0]
        self.epoches = [0]

        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write("0\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w",
                 encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]),
                                  self.letterbox_image)
        image_data = np.expand_dims(np.transpose(
            preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        import torch
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou)

            if results[0] is None:
                f.close()
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            f.write(f"{predicted_class} {score[:6]} {str(int(left))} "
                    f"{str(int(top))} {str(int(right))} {str(int(bottom))}\n")
        f.close()

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            import torch
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path, exist_ok=True)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))

            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                image = Image.open(line[0])
                gt_boxes = np.array([np.array(list(map(int, box.split(','))))
                                     for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"),
                          "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except Exception:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)

            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map) + "\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, '.'); from utils.callbacks import LossHistory, EvalCallback; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/callbacks.py
git commit -m "feat: add callbacks (LossHistory, EvalCallback)"
```

---

### Task 5: Create `utils/utils_fit.py`

**Files:**
- Create: `utils/utils_fit.py`

- [ ] **Step 1: Write the file**

```python
import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history,
                  eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda,
                  fp16, scaler, save_period, save_dir, local_rank=0):
    """训练和验证一个 epoch, 并保存 checkpoint。

    参考 yolov8-pytorch/utils/utils_fit.py。
    """
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',
                    postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, batch_targets = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                batch_targets = batch_targets.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, batch_targets)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, batch_targets)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',
                    postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, batch_targets = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                batch_targets = batch_targets.cuda(local_rank)
            optimizer.zero_grad()
            outputs = model_train_eval(images)
            loss_value = yolo_loss(outputs, batch_targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print(f'Epoch:{epoch + 1}/{Epoch}')
        print(f'Total Loss: {loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # 保存 checkpoint
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict,
                       os.path.join(save_dir,
                                    f"ep{epoch + 1:03d}-loss{loss / epoch_step:.3f}"
                                    f"-val_loss{val_loss / epoch_step_val:.3f}.pth"))

        if len(loss_history.val_loss) <= 1 or \
           (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import sys; sys.path.insert(0, '.'); from utils.utils_fit import fit_one_epoch; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/utils_fit.py
git commit -m "feat: add utils_fit (fit_one_epoch)"
```

---

### Task 6: Rewrite `train.py`

**Files:**
- Modify: `train.py` (complete rewrite)

- [ ] **Step 1: Write the new `train.py`**

```python
#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import LOGGER

from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr)
from utils.callbacks import EvalCallback, LossHistory
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)


'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为YOLO格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.txt格式，每张图片对应一个同名txt，文件中会有需要检测的目标信息。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'datasets/datasets.yaml'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolo26x.pt'
    #------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从0开始训练：
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch尽量不小于300。optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           YOLO26 小数据集推荐 100 epochs
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #------------------------------------------------------------------#
    eval_flag           = True
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers         = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    device = torch.device('cuda' if Cuda and torch.cuda.is_available() else 'cpu')

    #------------------------------------------------------#
    #   获取classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model = YOLO(model_path).model
    model.nc = num_classes  # override num_classes for custom dataset

    if Init_Epoch > 0:
        # 断点续训: 用户手动设置 model_path 指向 last_epoch_weights.pth
        print(f'Load weights for resume: {model_path}')
    else:
        print(f'Load pretrained weights: {model_path}')

    model_train = model.train()

    #----------------------#
    #   获得损失函数
    #----------------------#
    yolo_loss = Loss(model)

    #----------------------#
    #   记录Loss
    #----------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if Cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #----------------------------#
    #   权值平滑
    #----------------------------#
    ema = ModelEMA(model_train)

    #---------------------------#
    #   读取数据集
    #---------------------------#
    # 使用 ultralytics 的 DataLoader 加载标准 YOLO 格式数据集
    import yaml
    with open(classes_path, encoding='utf-8') as f:
        ydata = yaml.safe_load(f)

    dataset_path = ydata.get('path', '')
    train_path = os.path.join(dataset_path, ydata['train'], '') if dataset_path else ydata['train']
    val_path = os.path.join(dataset_path, ydata['val'], '') if dataset_path else ydata['val']

    # 构建 ultralytics YOLO dataset
    train_dataset = build_yolo_dataset(
        cfg=model.yaml,
        img_path=os.path.join(train_path, 'images'),
        batch=Freeze_batch_size,
        data=classes_path,
        rect=False,
        stride=32,
        augment=True,
        prefix='train: ',
    )

    val_dataset = build_yolo_dataset(
        cfg=model.yaml,
        img_path=os.path.join(val_path, 'images'),
        batch=Unfreeze_batch_size,
        data=classes_path,
        rect=True,
        stride=32,
        augment=False,
        prefix='val: ',
    )

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    show_config(
        classes_path=classes_path, model_path=model_path, input_shape=input_shape,
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
        Freeze_Train=Freeze_Train,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
        momentum=momentum, lr_decay_type=lr_decay_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers,
        num_train=num_train, num_val=num_val
    )

    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
            optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，"
              "共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                  num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，"
              "建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    UnFreeze_flag = False

    #------------------------------------#
    #   冻结一定部分训练
    #------------------------------------#
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    #-------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    #-------------------------------------------------------------------#
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    #---------------------------------------#
    #   根据optimizer_type选择优化器
    #---------------------------------------#
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    #---------------------------------------#
    #   获得学习率下降的公式
    #---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    #---------------------------------------#
    #   判断每一个世代的长度
    #---------------------------------------#
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    if ema:
        ema.updates = epoch_step * Init_Epoch

    #---------------------------------------#
    #   构建数据集加载器。
    #---------------------------------------#
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=True, drop_last=True,
                     collate_fn=getattr(train_dataset, 'collate_fn', None),
                     worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True, drop_last=True,
                         collate_fn=getattr(val_dataset, 'collate_fn', None),
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    #---------------------------------------#
    #   构建训练用的 val_lines 和 eval callback
    #---------------------------------------#
    # 从 val_dataset 获取图片路径列表用于评估
    val_lines = []
    if eval_flag:
        # 构建简单的 val_lines 格式: "image_path x1,y1,x2,y2,cls_id"
        val_img_dir = os.path.join(val_path, 'images')
        val_lbl_dir = os.path.join(val_path, 'labels')
        if os.path.isdir(val_img_dir) and os.path.isdir(val_lbl_dir):
            import glob
            for img_path in sorted(glob.glob(os.path.join(val_img_dir, '*.[jJ][pP][gG]'))):
                lbl_path = os.path.join(val_lbl_dir,
                                        os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                boxes_str = []
                if os.path.isfile(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(float(parts[0]))
                                # YOLO 格式: cls x_center y_center w h (normalized)
                                # 转为: x1,y1,x2,y2,cls
                                if len(parts) >= 5:
                                    x, y, w, h = [float(p) for p in parts[1:5]]
                                    # 转为像素坐标 (假设图片 640x640)
                                    x1 = int((x - w / 2) * input_shape[1])
                                    y1 = int((y - h / 2) * input_shape[0])
                                    x2 = int((x + w / 2) * input_shape[1])
                                    y2 = int((y + h / 2) * input_shape[0])
                                    boxes_str.append(f"{x1},{y1},{x2},{y2},{cls_id}")
                val_lines.append(f"{img_path} {' '.join(boxes_str)}")

    eval_callback = EvalCallback(
        model, input_shape, class_names, num_classes, val_lines, log_dir,
        Cuda, eval_flag=eval_flag, period=save_period)

    #---------------------------------------#
    #   开始模型训练
    #---------------------------------------#
    from utils.utils_fit import fit_one_epoch

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        #---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        #---------------------------------------#
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            #-------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            #-------------------------------------------------------------------#
            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if ema:
                ema.updates = epoch_step * epoch

            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True, drop_last=True,
                             collate_fn=getattr(train_dataset, 'collate_fn', None),
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 collate_fn=getattr(val_dataset, 'collate_fn', None),
                                 worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, ema, yolo_loss, loss_history,
                      eval_callback, optimizer, epoch, epoch_step,
                      epoch_step_val, gen, gen_val, UnFreeze_Epoch,
                      Cuda, fp16, scaler, save_period, log_dir)

    loss_history.writer.close()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('train.py', doraise=True); print('Syntax OK')"`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "refactor: rewrite train.py with custom training loop"
```

---

### Task 7: Rewrite `yolo.py` for Inference

**Files:**
- Modify: `yolo.py` (complete rewrite)

- [ ] **Step 1: Write the new `yolo.py`**

```python
import colorsys
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO as UltralyticsYOLO

from utils.utils import (cvtColor, get_classes, measure_text, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolo26x.pt',
        "classes_path"      : 'datasets/datasets.yaml',
        "input_shape"       : [640, 640],
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "mask_alpha"        : 0.35,
        "letterbox_image"   : True,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        self.class_names, self.num_classes = get_classes(self.classes_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                               self.colors))

        self.generate()
        show_config(**self._defaults)

    def generate(self):
        """加载模型。"""
        self.net = UltralyticsYOLO(self.model_path).model
        self.net.eval()
        if self.cuda and torch.cuda.is_available():
            self.net = self.net.cuda()
        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        print(f'{self.model_path} model loaded, classes: {self.num_classes}')

    # --------------------------------------------------- #
    #   Detect image
    # --------------------------------------------------- #
    def detect_image(self, image, crop=False, count=False):
        image = cvtColor(image)

        if not self.letterbox_image:
            orig_w, orig_h = image.size
            infer_image = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        else:
            infer_image = image

        image_data = resize_image(infer_image, (self.input_shape[1], self.input_shape[0]),
                                  self.letterbox_image)
        image_data = np.expand_dims(np.transpose(
            preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            image_shape = np.array(np.shape(image)[0:2])
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou)

        if results[0] is None:
            return image

        boxes_xyxy = results[0][:, :4]
        confs = results[0][:, 4]
        labels = results[0][:, 5].astype(np.int32)

        font = ImageFont.truetype(
            font='model_data/simhei.ttf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        if count:
            print("top_label:", labels)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(labels == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i in range(len(boxes_xyxy)):
                left, top, right, bottom = boxes_xyxy[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, f"crop_{i}.png"),
                                quality=95, subsampling=0)
                print(f"save crop_{i}.png to {dir_save_path}")

        draw = ImageDraw.Draw(image)
        for i, c in enumerate(labels):
            c = int(c)
            predicted_class = self.class_names[int(c)]
            box = boxes_xyxy[i]
            score = confs[i]

            left, top, right, bottom = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            label_size = measure_text(draw, label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle([left, top, right, bottom], outline=self.colors[c],
                           width=thickness)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

        return image

    # --------------------------------------------------- #
    #   FPS test
    # --------------------------------------------------- #
    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]),
                                  self.letterbox_image)
        image_data = np.expand_dims(np.transpose(
            preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                self.net(images)
        t2 = time.time()
        return (t2 - t1) / test_interval

    # --------------------------------------------------- #
    #   Heatmap visualization
    # --------------------------------------------------- #
    def detect_heatmap(self, image, heatmap_save_path):
        import matplotlib.pyplot as plt

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]),
                                  self.letterbox_image)
        image_data = np.expand_dims(np.transpose(
            preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        self.net.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()
                self.net = self.net.cuda()
            out = self.net(images)

        if isinstance(out, dict) and 'one2one' in out:
            feats = out['one2one']['feats']
            outputs = [fm.split((fm.size(1) - self.num_classes, self.num_classes), 1)[1]
                       for fm in feats]
        else:
            outputs = [out] if not isinstance(out, (list, tuple)) else list(out)

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]),
                                      [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., :]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    # --------------------------------------------------- #
    #   ONNX export
    # --------------------------------------------------- #
    def convert_to_onnx(self, simplify, model_path):
        import shutil
        m = UltralyticsYOLO(self.model_path)
        exported = m.export(format="onnx", imgsz=self.input_shape[0], simplify=simplify)
        if exported and model_path and str(exported) != model_path:
            shutil.move(str(exported), model_path)
        print(f'Onnx model exported to {model_path}')
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('yolo.py', doraise=True); print('Syntax OK')"`

- [ ] **Step 3: Commit**

```bash
git add yolo.py
git commit -m "refactor: rewrite yolo.py for custom model inference"
```

---

### Task 8: Adapt `get_map.py`

**Files:**
- Modify: `get_map.py`

- [ ] **Step 1: Modify `get_map.py`**

Replace the entire content of `get_map.py`:

```python
import glob
import os

import torch
from ultralytics import YOLO

if __name__ == "__main__":
    #------------------------------------------------------#
    #   model_path      指向训练好的权值文件
    #   data_yaml       数据集配置文件路径
    #   input_shape     输入的shape大小
    #   confidence      置信度阈值
    #   nms_iou         NMS IOU阈值
    #   split           验证集或测试集 (val / test)
    #------------------------------------------------------#
    model_path  = 'model_data/yolo26x.pt'
    data_yaml   = 'datasets/datasets.yaml'
    input_shape = [640, 640]
    confidence  = 0.001
    nms_iou     = 0.7
    split       = "val"
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=input_shape[0],
        conf=confidence,
        iou=nms_iou,
        split=split,
        device=device,
        plots=True,
    )

    print("Validation metrics:")
    for key, value in metrics.results_dict.items():
        print(f"{key}: {value:.6f}")
    if hasattr(metrics, "seg"):
        print(f"mask mAP50-95: {metrics.seg.map:.6f}")
        print(f"mask mAP50: {metrics.seg.map50:.6f}")
    if hasattr(metrics, "box"):
        print(f"box mAP50-95: {metrics.box.map:.6f}")
        print(f"box mAP50: {metrics.box.map50:.6f}")
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('get_map.py', doraise=True); print('Syntax OK')"`

- [ ] **Step 3: Commit**

```bash
git add get_map.py
git commit -m "refactor: adapt get_map.py, remove config.py dependency"
```

---

### Task 9: Delete `config.py` and Clean Up

**Files:**
- Delete: `config.py`

- [ ] **Step 1: Delete config.py**

Run: `git rm config.py`

- [ ] **Step 2: Check for remaining config references**

Run: `git grep -n "from config\|import config"` and fix any remaining references.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: delete config.py, configs moved to train.py and yolo.py"
```

---

### Task 10: Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Verify all modules import correctly**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from nets.yolo_training import Loss, ModelEMA, get_lr_scheduler, set_optimizer_lr
from utils.utils_bbox import make_anchors, dist2bbox, DecodeBox
from utils.utils_map import get_map, get_coco_map
from utils.callbacks import LossHistory, EvalCallback
from utils.utils_fit import fit_one_epoch
print('All imports OK')
"
```

- [ ] **Step 2: Verify train.py syntax and structure**

Run:
```bash
python -c "import py_compile; py_compile.compile('train.py', doraise=True); print('train.py OK')"
python -c "import py_compile; py_compile.compile('yolo.py', doraise=True); print('yolo.py OK')"
python -c "import py_compile; py_compile.compile('get_map.py', doraise=True); print('get_map.py OK')"
python -c "import py_compile; py_compile.compile('predict.py', doraise=True); print('predict.py OK')"
```

- [ ] **Step 3: Verify Loss instantiation with actual model**

Run:
```bash
python -c "
from ultralytics import YOLO
from nets.yolo_training import Loss

model = YOLO('yolo26n.pt').model
model.nc = 80
loss_fn = Loss(model)
print('Loss instantiated OK')
print('nc:', loss_fn.nc)
print('stride:', loss_fn.stride)
print('reg_max:', loss_fn.reg_max)
"
```

- [ ] **Step 4: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "chore: verify and fix integration issues"
```
