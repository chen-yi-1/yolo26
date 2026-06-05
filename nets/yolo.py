"""
YOLO26 model body — thin wrapper around ultralytics DetectionModel.

Key YOLO26 differences from YOLOv8:
  - end2end=True: dual head (one2one for NMS-Free inference, one2many for training)
  - reg_max=1:    DFL-Free, direct bbox regression (no distribution)
  - C3k2 blocks:  CSP with kernel-size-2 bottlenecks
  - C2PSA:        cross-stage partial + self-attention in backbone
"""

import torch
import torch.nn as nn

from utils.utils_bbox import make_anchors


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d layers."""
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels,
                          kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding, dilation=conv.dilation,
                          groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


# YOLO26 architecture config (from ultralytics yolo26x.yaml)
YOLO26_CFG = {
    'end2end': True,
    'reg_max': 1,
    'scales': {
        'n': [0.5, 0.25, 1024],
        's': [0.5, 0.5, 1024],
        'm': [0.5, 1.0, 512],
        'l': [1.0, 1.0, 512],
        'x': [1.0, 1.5, 512],
    },
    'backbone': [
        [-1, 1, 'Conv', [64, 3, 2]],           # 0
        [-1, 1, 'Conv', [128, 3, 2]],           # 1
        [-1, 2, 'C3k2', [256, False, 0.25]],    # 2
        [-1, 1, 'Conv', [256, 3, 2]],            # 3
        [-1, 2, 'C3k2', [512, False, 0.25]],    # 4
        [-1, 1, 'Conv', [512, 3, 2]],            # 5
        [-1, 2, 'C3k2', [512, True]],            # 6
        [-1, 1, 'Conv', [1024, 3, 2]],           # 7
        [-1, 2, 'C3k2', [1024, True]],           # 8
        [-1, 1, 'SPPF', [1024, 5, 3, True]],     # 9
        [-1, 2, 'C2PSA', [1024]],                # 10
    ],
    'head': [
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],            # 11
        [[-1, 6], 1, 'Concat', [1]],                                # 12
        [-1, 2, 'C3k2', [512, True]],                               # 13
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],            # 14
        [[-1, 4], 1, 'Concat', [1]],                                # 15
        [-1, 2, 'C3k2', [256, True]],                               # 16
        [-1, 1, 'Conv', [256, 3, 2]],                               # 17
        [[-1, 13], 1, 'Concat', [1]],                               # 18
        [-1, 2, 'C3k2', [512, True]],                               # 19
        [-1, 1, 'Conv', [512, 3, 2]],                               # 20
        [[-1, 10], 1, 'Concat', [1]],                               # 21
        [-1, 1, 'C3k2', [1024, True, 0.5, True]],                  # 22
        [[16, 19, 22], 1, 'Detect', ['nc']],                        # 23
    ],
}

YOLO26_SCALES = YOLO26_CFG['scales']


class YoloBody(nn.Module):
    """YOLO26 model — ultralytics DetectionModel under the hood."""

    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        from ultralytics.nn.tasks import DetectionModel

        if phi not in YOLO26_SCALES:
            raise ValueError(f"YOLO26 phi '{phi}' not in {list(YOLO26_SCALES.keys())}")

        cfg = {
            'nc': num_classes, 'end2end': YOLO26_CFG['end2end'],
            'reg_max': YOLO26_CFG['reg_max'], 'scales': YOLO26_CFG['scales'],
            'backbone': YOLO26_CFG['backbone'], 'head': YOLO26_CFG['head'],
            'scale': phi,
        }

        self.model = DetectionModel(cfg, ch=3, nc=num_classes)
        self.num_classes = num_classes
        self.phi = phi
        self.input_shape = input_shape

        # --- Metadata for native pipeline ---
        detect = self.model.model[-1]
        self.nl = detect.nl
        self.reg_max = detect.reg_max
        self.no = detect.no
        self.feat_indices = [16, 19, 22]  # P3, P4, P5 layer indices
        self.stride = detect.stride
        self.ch = [detect.cv2[i][0].conv.in_channels for i in range(self.nl)]

        # Shape tracking for anchor grid caching
        self._shape = None
        self._anchors = None
        self._strides = None

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if hasattr(m, 'conv') and hasattr(m, 'bn'):
                if isinstance(m.conv, nn.Conv2d) and isinstance(m.bn, nn.BatchNorm2d):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
        return self

    def _run_layers(self, x):
        """Run input through all layers, tracking intermediate outputs.
        Mirrors ultralytics DetectionModel._forward_once."""
        y = []
        for m in self.model.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    # -1 in list means current x (previous layer's output)
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return y

    def forward(self, x):
        """
        Training forward pass.
        Returns (dbox, cls, feat_list, anchors, strides) for the native Loss.
        """
        y = self._run_layers(x)

        # Extract P3, P4, P5 feature maps from the Detect head inputs
        feats = [y[i] for i in self.feat_indices]  # [B,C,H,W] × 3

        # Build anchor grid (cached by shape)
        shape = feats[0].shape
        if self._shape != shape:
            self._anchors, self._strides = (
                z.transpose(0, 1) for z in make_anchors(feats, self.stride, 0.5)
            )
            self._shape = shape

        # Apply Detect head's cv2 (box) and cv3 (cls) to each scale
        # Matching the original YoloBody pattern: torch.cat((cv2(feat), cv3(feat)), 1)
        detect = self.model.model[-1]
        x_head = []
        for i in range(self.nl):
            x_head.append(torch.cat((detect.cv2[i](feats[i]), detect.cv3[i](feats[i])), 1))

        # Concatenate across spatial dimensions
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x_head], 2)

        # Split: reg_max*4 channels for box, num_classes channels for cls
        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)

        # DFL-Free: box is already direct regression (reg_max=1), no DFL softmax needed
        # box shape: [B, 4, N]
        self.anchors = self._anchors
        self.strides = self._strides
        return box, cls, x_head, self._anchors.to(box.device), self._strides.to(box.device)

    def load_pretrained(self, model_path, device='cpu'):
        """
        Load pretrained weights from yolo26x.pt (ultralytics checkpoint).

        Maps checkpoint keys (model.X...) to our wrapper keys (model.model.X...).
        Handles 80→2 class remapping automatically.
        """
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'model' in checkpoint:
            src_state = checkpoint['model'].state_dict()
        else:
            src_state = checkpoint

        # Map checkpoint keys to our model's state dict
        # Checkpoint: model.0.conv.weight → Our model: model.model.0.conv.weight
        dst_state = self.state_dict()

        # Build a key lookup: strip leading 'model.' from checkpoint keys
        # Checkpoint has "model.X..." → our model has "model.model.X..."
        matched, skipped = {}, []

        for ckpt_key, v in src_state.items():
            if 'num_batches_tracked' in ckpt_key:
                continue
            # Map: model.X... → model.model.X...
            our_key = 'model.' + ckpt_key
            if our_key in dst_state:
                if dst_state[our_key].shape == v.shape:
                    matched[our_key] = v
                else:
                    skipped.append(f"{ckpt_key}: src{v.shape} → dst{dst_state[our_key].shape}")
            else:
                skipped.append(f"{ckpt_key}: not in target")

        self.load_state_dict(matched, strict=False)

        print(f"Loaded {len(matched)}/{len(matched) + len(skipped)} weight tensors")
        if skipped:
            cls_skipped = [s for s in skipped if 'cv3' in s and 'bias' in s]
            print(f"\nSkipped {len(skipped)} keys ({len(cls_skipped)} class-head: 80→{self.num_classes})")
        return len(matched), skipped
