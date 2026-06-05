# YOLO26 Official Training Adaptation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace custom training loop with `ultralytics.YOLO.train()`, rewrite inference to use ultralytics backend, preserve all prediction modes and VOC evaluation.

**Architecture:** `train.py` becomes a thin config → `model.train()` call. `yolo.py` YOLO class wraps `ultralytics.YOLO` for inference. `predict.py` preserves all 6 modes. `get_map.py` keeps VOC/COCO mAP computation. All custom model/loss/dataloader/training-loop code under `nets/` and `utils/` is deleted.

**Tech Stack:** ultralytics 8.4.x, PyTorch, numpy, PIL, OpenCV

---

## File State After Migration

| File | Action | Purpose |
|---|---|---|
| `train.py` | Rewrite | Config → `model.train()` |
| `yolo.py` | Rewrite | YOLO class wrapping ultralytics |
| `predict.py` | Adapt | 6 prediction modes |
| `get_map.py` | Adapt | VOC/COCO mAP eval |
| `summary.py` | Rewrite | FLOPs/params display |
| `voc_annotation.py` | Keep | VOC→YOLO converter |
| `utils/utils.py` | Keep | General utilities |
| `utils/utils_map.py` | Keep | mAP computation |
| `nets/__init__.py` | Delete | |
| `nets/yolo.py` | Delete | Replaced by ultralytics |
| `nets/yolo_training.py` | Delete | Replaced by ultralytics |
| `utils/dataloader.py` | Delete | Replaced by ultralytics |
| `utils/utils_fit.py` | Delete | Replaced by ultralytics |
| `utils/utils_bbox.py` | Delete | Replaced by ultralytics |
| `utils/callbacks.py` | Delete | Replaced by ultralytics built-in |
| `utils/__init__.py` | Recreate | Update exports |

---

### Task 1: Delete replaced modules

**Files:**
- Delete: `nets/__init__.py`
- Delete: `nets/yolo.py`
- Delete: `nets/yolo_training.py`
- Delete: `utils/dataloader.py`
- Delete: `utils/utils_fit.py`
- Delete: `utils/utils_bbox.py`
- Delete: `utils/callbacks.py`
- Delete: `utils_coco/coco_annotation.py`
- Delete: `utils_coco/get_map_coco.py`

- [ ] **Step 1: Remove files**

```bash
rm -rf nets/ utils/dataloader.py utils/utils_fit.py utils/utils_bbox.py utils/callbacks.py utils_coco/
```

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "remove: delete custom model, loss, dataloader, training loop modules"
```

---

### Task 2: Rewrite train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Rewrite train.py**

```python
"""
YOLO26 training — uses ultralytics official training pipeline.
Freeze-thaw strategy via ultralytics `freeze` parameter.
"""
import datetime
import os

from ultralytics import YOLO

if __name__ == "__main__":
    # -------------------------------- #
    #   Training Configuration
    # -------------------------------- #
    # Model & Data
    model_path  = 'yolo26x.pt'          # pretrained weights (download from ultralytics)
    data_yaml   = 'dataset.yaml'        # YOLO-format dataset config
    imgsz       = 640

    # Training epochs
    epochs      = 100

    # Batch sizes (freeze phase uses freeze_batch, then switches to un-freeze batch)
    freeze_batch    = 32
    unfreeze_batch  = 16

    # Freeze-thaw: backbone frozen for first N epochs
    freeze_epochs   = 50                  # backbone frozen for epochs 0..freeze_epochs

    # Optimizer (ultralytics 'auto' uses Adam for YOLO26)
    lr0         = 1e-3
    lrf         = 0.01                    # final lr = lr0 * lrf
    momentum    = 0.937
    weight_decay= 0                       # Adam default: no weight decay
    warmup_epochs = 3.0
    cos_lr      = True

    # Augmentation
    mosaic      = 1.0                     # mosaic probability
    mixup       = 0.5                     # mixup probability (only on mosaic images)
    close_mosaic= 15                      # disable mosaic for last N epochs
    hsv_h       = 0.015
    hsv_s       = 0.7
    hsv_v       = 0.4
    degrees     = 0.0
    translate   = 0.1
    scale       = 0.5
    shear       = 0.0
    perspective = 0.0
    flipud      = 0.0
    fliplr      = 0.5

    # Loss gains (matching YOLO26 defaults)
    box_gain    = 7.5
    cls_gain    = 0.5
    dfl_gain    = 1.5

    # System
    device      = 'cuda'
    workers     = 4
    amp         = True
    seed        = 11

    # Logging
    project     = 'logs'
    name        = f"yolo26_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_period = 10
    val         = True
    plots       = True

    # -------------------------------- #
    #   Phase 1: Freeze backbone
    # -------------------------------- #
    if freeze_epochs > 0:
        print(f"\n[Phase 1] Freezing backbone for {freeze_epochs} epochs (batch={freeze_batch})")
        model = YOLO(model_path)
        model.train(
            data=data_yaml,
            epochs=freeze_epochs,
            batch=freeze_batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            cos_lr=cos_lr,
            mosaic=mosaic,
            mixup=mixup,
            close_mosaic=0,              # don't close mosaic in freeze phase
            box=box_gain,
            cls=cls_gain,
            dfl=dfl_gain,
            amp=amp,
            seed=seed,
            project=project,
            name=name,
            save_period=save_period,
            val=val,
            plots=plots,
            freeze=10,                   # freeze first 10 layers (backbone)
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
        )

    # -------------------------------- #
    #   Phase 2: Unfreeze and train
    # -------------------------------- #
    remaining = epochs - freeze_epochs
    if remaining > 0:
        print(f"\n[Phase 2] Unfreezing all layers for {remaining} epochs (batch={unfreeze_batch})")
        model = YOLO(os.path.join(project, name, 'weights', 'last.pt'))
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=unfreeze_batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=0,             # no warmup in phase 2
            cos_lr=cos_lr,
            mosaic=mosaic,
            mixup=mixup,
            close_mosaic=close_mosaic,
            box=box_gain,
            cls=cls_gain,
            dfl=dfl_gain,
            amp=amp,
            seed=seed,
            project=project,
            name=name,
            save_period=save_period,
            val=val,
            plots=plots,
            freeze=None,                 # unfreeze all
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            resume=True,
        )

    print("\nTraining complete. Best weights saved in:", os.path.join(project, name, 'weights', 'best.pt'))
```

- [ ] **Step 2: Commit**

```bash
git add train.py && git commit -m "rewrite: train.py uses ultralytics YOLO.train() with freeze-thaw"
```

---

### Task 3: Rewrite yolo.py (YOLO inference class)

**Files:**
- Modify: `yolo.py`

- [ ] **Step 1: Rewrite yolo.py**

The YOLO class wraps ultralytics.YOLO and preserves the existing inference interface (detect_image, get_FPS, detect_heatmap, get_map_txt, convert_to_onnx).

```python
import colorsys
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO as UltralyticsYOLO

from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)


class YOLO(object):
    _defaults = {
        "model_path"        : 'reference/yolo26x.pt',
        "classes_path"      : 'model_data/voc_classes.txt',
        "input_shape"       : [640, 640],
        "phi"               : 'x',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
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
        device = 'cuda' if self.cuda and torch.cuda.is_available() else 'cpu'
        self.model = UltralyticsYOLO(self.model_path)
        self.model.to(device)
        print(f'{self.model_path} model loaded on {device}, classes: {self.num_classes}')

    # --------------------------------------------------- #
    #   Detect image
    # --------------------------------------------------- #
    def detect_image(self, image, crop=False, count=False):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)

        results = self.model.predict(
            image,
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

        top_label = labels
        top_conf = confs
        top_boxes = boxes_xyxy

        font = ImageFont.truetype(
            font='model_data/simhei.ttf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        )
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, c in list(enumerate(top_boxes)):
                left, top, right, bottom = top_boxes[i]
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

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            left, top, right, bottom = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    # --------------------------------------------------- #
    #   FPS test
    # --------------------------------------------------- #
    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        t1 = time.time()
        for _ in range(test_interval):
            self.model.predict(
                image_data,
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

        # Get raw feature maps for heatmap — use ultralytics internal forward
        self.model.model.eval()
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda and torch.cuda.is_available():
                images = images.cuda()
            # Run predict to get internal feature maps
            feat_maps = self.model.model(images)

        # Extract classification scores from detect head outputs
        # feat_maps is a tuple of 3 feature tensors from detect layer
        # Each is [B, nc+reg_max*4, H, W]
        if isinstance(feat_maps, (list, tuple)):
            outputs = [fm.split((fm.size(1) - self.num_classes, self.num_classes), 1)[1] for fm in feat_maps]
        else:
            outputs = [feat_maps.split((feat_maps.size(1) - self.num_classes, self.num_classes), 1)[1]]

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
        import onnx
        self.model.export(format="onnx", imgsz=self.input_shape[0], simplify=simplify)
        print(f'Onnx model exported')

    # --------------------------------------------------- #
    #   Get mAP detection results
    # --------------------------------------------------- #
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)

        results = self.model.predict(
            image,
            imgsz=self.input_shape[0],
            conf=0.001,
            iou=0.5,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            f.close()
            return

        top_boxes = result.boxes.xyxy.cpu().numpy()
        top_conf = result.boxes.conf.cpu().numpy()
        top_label = result.boxes.cls.cpu().numpy().astype(np.int32)

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            left, top, right, bottom = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6],
                str(int(left)), str(int(top)),
                str(int(right)), str(int(bottom))
            ))
        f.close()
```

- [ ] **Step 2: Commit**

```bash
git add yolo.py && git commit -m "rewrite: yolo.py wraps ultralytics.YOLO for inference"
```

---

### Task 4: Adapt predict.py

**Files:**
- Modify: `predict.py`

- [ ] **Step 1: Update predict.py**

The predict.py stays almost identical — just update default model_path:

```python
"""
predict.py — multi-mode prediction: predict / video / fps / dir_predict / heatmap / export_onnx
"""
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # ------------------------------------------------------------------------ #
    #   mode: 'predict' | 'video' | 'fps' | 'dir_predict' | 'heatmap' | 'export_onnx'
    # ------------------------------------------------------------------------ #
    mode = "predict"

    # crop & count only for mode='predict'
    crop            = False
    count           = False

    # video mode settings
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    # fps mode settings
    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    # dir_predict mode settings
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    # heatmap mode settings
    heatmap_save_path = "model_data/heatmap_vision.png"

    # export_onnx mode settings
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        """
        1. Save result: r_image.save("img.jpg")
        2. Get bbox coords: read top, left, bottom, right in yolo.detect_image
        3. Crop objects: use crop=True
        4. Add text: modify yolo.detect_image drawing section
        """
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Cannot read camera/video. Check camera connection or video path.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print(f"fps= {fps:.2f}")
            frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)
            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(f"{tact_time} seconds, {1/tact_time}FPS, @batch_size 1")

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
```

- [ ] **Step 2: Commit**

```bash
git add predict.py && git commit -m "adapt: predict.py works with new ultralytics-backed YOLO class"
```

---

### Task 5: Adapt get_map.py

**Files:**
- Modify: `get_map.py`

- [ ] **Step 1: Update get_map.py**

```python
import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    """
    Recall and Precision values depend on the confidence threshold.
    By default, confidence=0.5 is used as the threshold.
    """
    map_mode        = 0
    classes_path    = 'model_data/voc_classes.txt'
    MINOVERLAP      = 0.5
    confidence      = 0.001
    nms_iou         = 0.5
    score_threhold  = 0.5
    map_vis         = False
    VOCdevkit_path  = 'VOCdevkit'
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
```

- [ ] **Step 2: Commit**

```bash
git add get_map.py && git commit -m "adapt: get_map.py works with new ultralytics-backed YOLO class"
```

---

### Task 6: Rewrite summary.py

**Files:**
- Modify: `summary.py`

- [ ] **Step 1: Rewrite summary.py**

```python
"""
Model summary — FLOPs & parameter count using ultralytics model.
"""
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    input_shape = [640, 640]
    phi         = 'x'

    model_path = f'yolo26{phi}.pt'
    model = YOLO(model_path)

    # Print model info
    print(f"\nYOLO26{phi} summary:")
    print(model.info())

    # Detailed FLOPs via thop
    from thop import clever_format, profile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model.model.to(device)
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m, (dummy_input,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print(f'Total GFLOPS: {flops}')
    print(f'Total params: {params}')
```

- [ ] **Step 2: Commit**

```bash
git add summary.py && git commit -m "rewrite: summary.py uses ultralytics model info API"
```

---

### Task 7: Update utils/__init__.py

**Files:**
- Modify: `utils/__init__.py`

- [ ] **Step 1: Rewrite utils/__init__.py**

```python
from .utils import (
    cvtColor,
    resize_image,
    get_classes,
    get_lr,
    seed_everything,
    worker_init_fn,
    preprocess_input,
    show_config,
)
from .utils_map import get_map, get_coco_map
```

- [ ] **Step 2: Commit**

```bash
git add utils/__init__.py && git commit -m "update: utils/__init__.py exports for new module structure"
```

---

### Task 8: Final cleanup and verification

- [ ] **Step 1: Remove downloaded test model**

```bash
rm -f yolo26n.pt
```

- [ ] **Step 2: Verify all imports resolve**

```bash
python3 -c "
from utils.utils import cvtColor, get_classes, seed_everything, preprocess_input, resize_image, show_config
from utils.utils_map import get_map, get_coco_map
from yolo import YOLO
print('All imports OK')
"
```

- [ ] **Step 3: Verify train.py syntax**

```bash
python3 -c "compile(open('train.py').read(), 'train.py', 'exec'); print('train.py syntax OK')"
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "cleanup: remove temp files, verify all modules"
```
