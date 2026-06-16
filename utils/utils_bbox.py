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
