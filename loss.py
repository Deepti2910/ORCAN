"""
YOLACT loss: classification (cross-entropy over anchors matched via IoU),
box regression (smooth L1), and mask loss (per-instance pixel-wise BCE on
prototype-assembled masks, cropped to GT box — this is the "Lincomb" mask
loss from the paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import decode_boxes


def box_iou(anchors_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """IoU between every anchor and every GT box. Returns (num_anchors, num_gt)."""
    area1 = (anchors_xyxy[:, 2] - anchors_xyxy[:, 0]) * (anchors_xyxy[:, 3] - anchors_xyxy[:, 1])
    area2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])

    lt = torch.max(anchors_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    rb = torch.min(anchors_xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def encode_boxes(anchors_cxcywh: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """Inverse of decode_boxes: GT xyxy -> (dx, dy, dw, dh) targets relative to anchors."""
    gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2
    gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
    gt_w = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=1e-4)
    gt_h = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=1e-4)

    dx = (gt_cx - anchors_cxcywh[:, 0]) / anchors_cxcywh[:, 2]
    dy = (gt_cy - anchors_cxcywh[:, 1]) / anchors_cxcywh[:, 3]
    dw = torch.log(gt_w / anchors_cxcywh[:, 2])
    dh = torch.log(gt_h / anchors_cxcywh[:, 3])
    return torch.stack([dx, dy, dw, dh], dim=-1)


class YOLACTLoss(nn.Module):
    def __init__(self, num_classes: int, pos_iou_thresh: float = 0.5, neg_iou_thresh: float = 0.4,
                 mask_loss_weight: float = 6.125, box_loss_weight: float = 1.5, ohem_ratio: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.mask_loss_weight = mask_loss_weight
        self.box_loss_weight = box_loss_weight
        self.ohem_ratio = ohem_ratio

    def forward(self, preds: dict, targets: list):
        """
        preds: output dict from YOLACT.forward()
        targets: list (len = batch) of dicts with keys:
            'boxes'  -> (N, 4) xyxy, in the SAME pixel scale as img_size
            'labels' -> (N,) long, 1..num_fg_classes  (0 is reserved for background)
            'masks'  -> (N, H, W) binary float tensor, full-image-resolution GT masks
        """
        anchors = preds["anchors"]                      # (A, 4) cx,cy,w,h
        anchors_xyxy = cxcywh_to_xyxy(anchors)
        prototypes = preds["prototypes"]                 # (B, k, Hp, Wp)
        img_h, img_w = preds["img_size"]

        total_cls_loss, total_box_loss, total_mask_loss = 0.0, 0.0, 0.0
        n_pos_total = 0

        B = prototypes.shape[0]
        for b in range(B):
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]
            gt_masks = targets[b]["masks"]

            cls_logits = preds["class_logits"][b]  # (A, C)
            box_deltas = preds["box_deltas"][b]     # (A, 4)
            mask_coef = preds["mask_coef"][b]       # (A, k)

            if gt_boxes.numel() == 0:
                # No objects: everything is background.
                cls_target = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
                total_cls_loss += F.cross_entropy(cls_logits, cls_target)
                continue

            iou = box_iou(anchors_xyxy, gt_boxes)          # (A, N)
            best_gt_iou, best_gt_idx = iou.max(dim=1)       # per-anchor best GT

            cls_target = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
            pos_mask = best_gt_iou >= self.pos_iou_thresh
            neutral_mask = (best_gt_iou >= self.neg_iou_thresh) & (~pos_mask)
            cls_target[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

            # --- classification loss with online hard-negative mining (OHEM) ---
            per_anchor_cls_loss = F.cross_entropy(cls_logits, cls_target, reduction="none")
            per_anchor_cls_loss = per_anchor_cls_loss.masked_fill(neutral_mask, 0.0)

            n_pos = int(pos_mask.sum().item())
            n_pos_total += max(n_pos, 1)
            n_neg = min(self.ohem_ratio * max(n_pos, 1), int((~pos_mask & ~neutral_mask).sum().item()))
            neg_losses = per_anchor_cls_loss.clone()
            neg_losses[pos_mask] = -1  # exclude positives from negative mining
            hard_neg_idx = torch.topk(neg_losses, k=n_neg).indices if n_neg > 0 else torch.tensor([], dtype=torch.long)

            cls_loss = per_anchor_cls_loss[pos_mask].sum() + per_anchor_cls_loss[hard_neg_idx].sum()
            cls_loss = cls_loss / max(n_pos, 1)
            total_cls_loss += cls_loss

            if n_pos == 0:
                continue

            # --- box regression loss (smooth L1), positives only ---
            pos_anchors = anchors[pos_mask]
            pos_gt_boxes = gt_boxes[best_gt_idx[pos_mask]]
            box_targets = encode_boxes(pos_anchors, pos_gt_boxes)
            total_box_loss += F.smooth_l1_loss(box_deltas[pos_mask], box_targets) * self.box_loss_weight

            # --- mask loss: assemble masks = sigmoid(protos . coef), crop to GT box, BCE ---
            pos_mask_coef = mask_coef[pos_mask]                     # (P, k)
            protos_b = prototypes[b]                                # (k, Hp, Wp)
            Hp, Wp = protos_b.shape[-2:]
            assembled = torch.einsum("pk,khw->phw", pos_mask_coef, protos_b)  # (P, Hp, Wp)
            assembled = torch.sigmoid(assembled)

            pos_gt_idx = best_gt_idx[pos_mask]
            gt_masks_matched = gt_masks[pos_gt_idx]                 # (P, H, W) full-res
            gt_masks_small = F.interpolate(
                gt_masks_matched.unsqueeze(1), size=(Hp, Wp), mode="bilinear", align_corners=False
            ).squeeze(1)

            # crop loss to each instance's GT box (as in the paper, to avoid
            # penalizing prototype activations far outside the object)
            mask_bce = F.binary_cross_entropy(assembled, gt_masks_small, reduction="none")
            scale_x, scale_y = Wp / img_w, Hp / img_h
            crop_loss = 0.0
            for i in range(pos_gt_boxes.shape[0]):
                x1, y1, x2, y2 = pos_gt_boxes[i]
                px1, py1 = max(int(x1 * scale_x), 0), max(int(y1 * scale_y), 0)
                px2, py2 = min(int(x2 * scale_x) + 1, Wp), min(int(y2 * scale_y) + 1, Hp)
                if px2 <= px1 or py2 <= py1:
                    continue
                region = mask_bce[i, py1:py2, px1:px2]
                crop_loss += region.mean()
            total_mask_loss += (crop_loss / max(pos_gt_boxes.shape[0], 1)) * self.mask_loss_weight

        total_cls_loss = total_cls_loss / B
        total_box_loss = total_box_loss / B
        total_mask_loss = total_mask_loss / B

        return {
            "loss": total_cls_loss + total_box_loss + total_mask_loss,
            "cls_loss": total_cls_loss,
            "box_loss": total_box_loss,
            "mask_loss": total_mask_loss,
        }
