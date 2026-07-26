"""Inference: decode predictions, apply Fast NMS, assemble + crop final masks."""

import torch
import torch.nn.functional as F

from loss import cxcywh_to_xyxy
from model import decode_boxes


def fast_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5,
             top_k: int = 200) -> torch.Tensor:
    """YOLACT's Fast NMS: matrix-based, non-sequential (much faster than
    standard NMS, at the cost of being slightly more permissive)."""
    scores, idx = scores.sort(descending=True)
    idx = idx[:top_k]
    scores = scores[:top_k]
    boxes = boxes[idx]

    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    lt = torch.max(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.min(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    iou = inter / (areas[:, None] + areas[None, :] - inter).clamp(min=1e-6)

    iou = torch.triu(iou, diagonal=1)  # only compare each box to higher-scoring ones
    keep = iou.max(dim=0).values <= iou_threshold
    return idx[keep]


@torch.no_grad()
def postprocess(preds: dict, score_thresh: float = 0.3, nms_thresh: float = 0.5,
                 top_k: int = 200, mask_thresh: float = 0.5):
    """
    Returns, per image in the batch, a dict with:
        boxes  (M, 4) xyxy in image-pixel coords
        scores (M,)
        labels (M,)
        masks  (M, H, W) binary, full image resolution
    """
    img_h, img_w = preds["img_size"]
    anchors = preds["anchors"]
    B = preds["class_logits"].shape[0]

    results = []
    for b in range(B):
        cls_probs = F.softmax(preds["class_logits"][b], dim=-1)  # (A, C)
        scores, labels = cls_probs[:, 1:].max(dim=-1)  # exclude background (index 0)
        labels = labels + 1

        keep = scores > score_thresh
        if keep.sum() == 0:
            results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                             "labels": torch.zeros(0, dtype=torch.long),
                             "masks": torch.zeros(0, img_h, img_w)})
            continue

        boxes = decode_boxes(anchors, preds["box_deltas"][b])[keep]
        scores_k = scores[keep]
        labels_k = labels[keep]
        mask_coef_k = preds["mask_coef"][b][keep]

        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)

        nms_idx = fast_nms(boxes, scores_k, nms_thresh, top_k)
        boxes, scores_k, labels_k, mask_coef_k = (
            boxes[nms_idx], scores_k[nms_idx], labels_k[nms_idx], mask_coef_k[nms_idx]
        )

        protos = preds["prototypes"][b]  # (k, Hp, Wp)
        assembled = torch.sigmoid(torch.einsum("mk,khw->mhw", mask_coef_k, protos))
        assembled = F.interpolate(assembled.unsqueeze(0), size=(img_h, img_w),
                                   mode="bilinear", align_corners=False).squeeze(0)

        # crop each mask to its predicted box (standard YOLACT postprocessing)
        final_masks = torch.zeros_like(assembled)
        for i, (x1, y1, x2, y2) in enumerate(boxes.int()):
            x1, y1 = max(x1.item(), 0), max(y1.item(), 0)
            x2, y2 = min(x2.item(), img_w), min(y2.item(), img_h)
            final_masks[i, y1:y2, x1:x2] = assembled[i, y1:y2, x1:x2]
        final_masks = (final_masks > mask_thresh).float()

        results.append({"boxes": boxes, "scores": scores_k, "labels": labels_k, "masks": final_masks})

    return results
