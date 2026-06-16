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
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/yolo26x.pt',
        "classes_path"      : 'datasets/datasets.yaml',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
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
