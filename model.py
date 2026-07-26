"""
YOLACT-style real-time instance segmentation model, following the architecture
used in Caballas et al. (IECBES 2020) for laparoscopic gallbladder detection.

Architecture (Bolya et al., "YOLACT: Real-time Instance Segmentation"):
    image -> ResNet backbone -> FPN (P3-P7)
    P3 -> ProtoNet -> k prototype masks (shared across whole image)
    P3-P7 -> Prediction Head -> per-anchor: class scores, box regression, mask coefficients
    final mask = sigmoid( prototypes @ mask_coefficients ) cropped to predicted box

This is a from-scratch, readable reimplementation (not the original repo) sized
for a single-class (or few-class) surgical segmentation task like gallbladder
detection. It's built to be trained on COCO-format polygon/mask annotations,
which is exactly the label format used in the paper (Table I: "Labels
(polygon masks)").
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------- #
# Backbone + FPN
# --------------------------------------------------------------------------- #
class ResNetFPNBackbone(nn.Module):
    """ResNet-50 backbone (ImageNet-pretrained) + Feature Pyramid Network.

    Produces feature maps P3, P4, P5, P6, P7 at strides 8, 16, 32, 64, 128,
    matching the original YOLACT paper's pyramid.
    """

    def __init__(self, pretrained: bool = True, fpn_channels: int = 256):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)

        # Stem + stages. We need C3, C4, C5 (strides 8, 16, 32).
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # -> C2 (stride 4), not used directly
        self.layer2 = resnet.layer2  # -> C3 (stride 8),  512 channels
        self.layer3 = resnet.layer3  # -> C4 (stride 16), 1024 channels
        self.layer4 = resnet.layer4  # -> C5 (stride 32), 2048 channels

        c3_ch, c4_ch, c5_ch = 512, 1024, 2048

        # Lateral 1x1 convs to unify channel dims for top-down FPN pathway
        self.lat_c5 = nn.Conv2d(c5_ch, fpn_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4_ch, fpn_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3_ch, fpn_channels, kernel_size=1)

        # 3x3 smoothing convs after top-down merge
        self.smooth_p5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # Extra downsample layers to get P6, P7 (as in RetinaNet/YOLACT)
        self.p6 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        return [p3, p4, p5, p6, p7]  # strides 8, 16, 32, 64, 128


# --------------------------------------------------------------------------- #
# ProtoNet — generates k prototype masks from the highest-resolution FPN level
# --------------------------------------------------------------------------- #
class ProtoNet(nn.Module):
    def __init__(self, in_channels: int = 256, proto_channels: int = 256, num_prototypes: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, proto_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(proto_channels, proto_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(proto_channels, proto_channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(proto_channels, num_prototypes, kernel_size=1)

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        x = self.net(p3)
        x = self.upsample(x)  # upsample before final conv, per YOLACT
        x = F.relu(self.final_conv(x))
        return x  # (B, k, H/4, W/4)


# --------------------------------------------------------------------------- #
# Prediction head — shared across pyramid levels, predicts per-anchor
# class logits, box regression deltas, and mask coefficients
# --------------------------------------------------------------------------- #
class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3, num_prototypes: int = 32):
        super().__init__()
        self.num_classes = num_classes  # includes background
        self.num_anchors = num_anchors
        self.num_prototypes = num_prototypes

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.class_head = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.mask_head = nn.Conv2d(in_channels, num_anchors * num_prototypes, 3, padding=1)

    def forward(self, feat: torch.Tensor):
        x = self.shared(feat)
        B, _, H, W = x.shape

        cls = self.class_head(x).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
        box = self.box_head(x).permute(0, 2, 3, 1).reshape(B, -1, 4)
        mask_coef = torch.tanh(self.mask_head(x)).permute(0, 2, 3, 1).reshape(B, -1, self.num_prototypes)

        return cls, box, mask_coef


# --------------------------------------------------------------------------- #
# Anchor generation
# --------------------------------------------------------------------------- #
def generate_anchors(feature_shapes: List[Tuple[int, int]], strides: List[int],
                      scales=(1.0, 1.26, 1.59), aspect_ratios=(1.0, 0.5, 2.0),
                      base_size: int = 24, device="cpu") -> torch.Tensor:
    """Generates anchor boxes (cx, cy, w, h) for every pyramid level, matching
    the 3-scale x 3-ratio scheme from the original YOLACT paper."""
    anchors = []
    for (h, w), stride in zip(feature_shapes, strides):
        for i in range(h):
            for j in range(w):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                for scale in scales:
                    area = (base_size * stride / 8) ** 2 * scale ** 2
                    for ratio in aspect_ratios:
                        aw = math.sqrt(area * ratio)
                        ah = math.sqrt(area / ratio)
                        anchors.append([cx, cy, aw, ah])
    return torch.tensor(anchors, dtype=torch.float32, device=device)


NUM_ANCHORS_PER_CELL = 9  # 3 scales x 3 aspect ratios


# --------------------------------------------------------------------------- #
# Full YOLACT model
# --------------------------------------------------------------------------- #
class YOLACT(nn.Module):
    """
    num_classes: number of FOREGROUND classes (e.g. 1 for gallbladder-only,
                 as in the paper). Background is added automatically.
    """

    def __init__(self, num_classes: int = 1, num_prototypes: int = 32, pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes + 1  # + background
        self.num_prototypes = num_prototypes

        self.backbone = ResNetFPNBackbone(pretrained=pretrained_backbone)
        self.protonet = ProtoNet(num_prototypes=num_prototypes)

        # Single shared prediction head applied at every pyramid level (P3-P7)
        self.head = PredictionHead(
            in_channels=256,
            num_classes=self.num_classes,
            num_anchors=NUM_ANCHORS_PER_CELL,
            num_prototypes=num_prototypes,
        )
        self.strides = [8, 16, 32, 64, 128]

    def forward(self, x: torch.Tensor):
        img_h, img_w = x.shape[-2:]
        feats = self.backbone(x)          # [P3..P7]
        protos = self.protonet(feats[0])  # from P3 only

        all_cls, all_box, all_maskcoef, shapes = [], [], [], []
        for feat in feats:
            cls, box, mc = self.head(feat)
            all_cls.append(cls)
            all_box.append(box)
            all_maskcoef.append(mc)
            shapes.append(feat.shape[-2:])

        cls = torch.cat(all_cls, dim=1)
        box = torch.cat(all_box, dim=1)
        mask_coef = torch.cat(all_maskcoef, dim=1)

        anchors = generate_anchors(shapes, self.strides, device=x.device)

        return {
            "class_logits": cls,        # (B, num_anchors_total, num_classes)
            "box_deltas": box,          # (B, num_anchors_total, 4)
            "mask_coef": mask_coef,     # (B, num_anchors_total, k)
            "prototypes": protos,       # (B, k, H/4, W/4)
            "anchors": anchors,         # (num_anchors_total, 4) cx,cy,w,h
            "img_size": (img_h, img_w),
        }


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode (dx,dy,dw,dh) regression deltas relative to anchors -> xyxy boxes."""
    cx = anchors[:, 0] + deltas[..., 0] * anchors[:, 2]
    cy = anchors[:, 1] + deltas[..., 1] * anchors[:, 3]
    w = anchors[:, 2] * torch.exp(deltas[..., 2])
    h = anchors[:, 3] * torch.exp(deltas[..., 3])
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == "__main__":
    # Quick smoke test — verifies shapes line up end-to-end.
    model = YOLACT(num_classes=1, pretrained_backbone=False)
    dummy = torch.randn(1, 3, 550, 550)  # 550px, same as the paper's training res
    out = model(dummy)
    print("class_logits:", out["class_logits"].shape)
    print("box_deltas:  ", out["box_deltas"].shape)
    print("mask_coef:   ", out["mask_coef"].shape)
    print("prototypes:  ", out["prototypes"].shape)
    print("anchors:     ", out["anchors"].shape)
