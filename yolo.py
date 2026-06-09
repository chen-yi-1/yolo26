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


class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolo26x-seg.pt',
        "classes_path"      : os.path.join('datasets', 'datasets.yaml'),
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
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        show_config(**self._defaults)

    def generate(self):
        self.model = UltralyticsYOLO(self.model_path)
        print(f'{self.model_path} model loaded, classes: {self.num_classes}')

    # --------------------------------------------------- #
    #   Detect image
    # --------------------------------------------------- #
    def detect_image(self, image, crop=False, count=False):
        image = cvtColor(image)

        # When letterbox_image=False, use direct stretch-resize before inference
        if not self.letterbox_image:
            orig_w, orig_h = image.size
            infer_image = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        else:
            infer_image = image

        results = self.model.predict(
            infer_image,
            imgsz=self.input_shape[0],
            conf=self.confidence,
            iou=self.nms_iou,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return image

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(np.int32)
        masks_xy = result.masks.xy if result.masks is not None else []

        # Scale boxes back to original coordinates for direct-resize mode
        if not self.letterbox_image:
            boxes_xyxy[:, [0, 2]] *= orig_w / self.input_shape[1]
            boxes_xyxy[:, [1, 3]] *= orig_h / self.input_shape[0]
            scaled_masks_xy = []
            for polygon in masks_xy:
                polygon = np.asarray(polygon, dtype=np.float32).copy()
                polygon[:, 0] *= orig_w / self.input_shape[1]
                polygon[:, 1] *= orig_h / self.input_shape[0]
                scaled_masks_xy.append(polygon)
            masks_xy = scaled_masks_xy

        font = ImageFont.truetype(
            font='model_data/simhei.ttf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        )
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
                crop_image.save(os.path.join(dir_save_path, f"crop_{i}.png"), quality=95, subsampling=0)
                print(f"save crop_{i}.png to {dir_save_path}")

        if masks_xy:
            mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_layer)
            for i, polygon in enumerate(masks_xy):
                if polygon is None or len(polygon) < 3:
                    continue
                c = int(labels[i])
                color = self.colors[c]
                points = [
                    (
                        int(np.clip(x, 0, image.size[0] - 1)),
                        int(np.clip(y, 0, image.size[1] - 1)),
                    )
                    for x, y in polygon
                ]
                fill = (*color, int(255 * self.mask_alpha))
                mask_draw.polygon(points, fill=fill)
                mask_draw.line(points + [points[0]], fill=(*color, 255), width=max(1, thickness))
            image = Image.alpha_composite(image.convert("RGBA"), mask_layer).convert("RGB")

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

            draw.rectangle([left, top, right, bottom], outline=self.colors[c], width=thickness)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

        return image

    # --------------------------------------------------- #
    #   FPS test
    # --------------------------------------------------- #
    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        t1 = time.time()
        for _ in range(test_interval):
            self.model.predict(
                image,
                imgsz=self.input_shape[0],
                conf=self.confidence,
                iou=self.nms_iou,
                verbose=False,
            )
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
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        self.model.model.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()
                self.model.model = self.model.model.cuda()
            # YOLO26 with end2end=True returns (preds, {'one2many': {...}, 'one2one': {...}})
            # one2one['feats'] contains P3/P4/P5 feature maps from the detect head
            out = self.model.model(images)

        # Extract classification feature maps from inference (one2one) branch
        if isinstance(out, (list, tuple)) and len(out) >= 2 and isinstance(out[1], dict):
            feats = out[1]['one2one']['feats']   # 3 feature maps: [B, C, H, W]
            outputs = [fm.split((fm.size(1) - self.num_classes, self.num_classes), 1)[1] for fm in feats]
        else:
            # Fallback: use raw output (older ultralytics or different model)
            outputs = [out] if not isinstance(out, (list, tuple)) else list(out)

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
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
        exported = self.model.export(format="onnx", imgsz=self.input_shape[0], simplify=simplify)
        # Move to desired path if different from default export path
        if exported and model_path and str(exported) != model_path:
            shutil.move(str(exported), model_path)
        print(f'Onnx model exported to {model_path}')
