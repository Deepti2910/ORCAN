"""
COCO-format dataset loader for gallbladder (or other organ) instance
segmentation. This matches the label format described in the paper
(Table I: images + polygon-mask labels per class).

Expected directory layout (standard COCO instance-segmentation format —
this is what CVAT, Roboflow, and Label Studio all export to):

    dataset/
        train/
            images/
                img_0001.jpg
                ...
            annotations.json      <- COCO instances format
        val/
            images/
                ...
            annotations.json

annotations.json format (per COCO spec):
    {
      "images": [{"id": 1, "file_name": "img_0001.jpg", "height": 1080, "width": 1920}, ...],
      "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                        "bbox": [x,y,w,h], "segmentation": [[x1,y1,x2,y2,...]]}, ...],
      "categories": [{"id": 1, "name": "gallbladder"}]
    }

If your masks are pixel PNGs instead of polygons (e.g. CholecSeg8k, see
README), use `masks_to_coco.py`-style conversion first, or adapt
`_load_mask` below to read from your mask PNGs instead of polygons.
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask_utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CocoInstanceSegDataset(Dataset):
    def __init__(self, images_dir: str, annotation_file: str, img_size: int = 550, train: bool = True):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size
        self.train = train

        # Map COCO category_id -> contiguous 1..N label (0 reserved for background)
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self):
        return len(self.img_ids)

    def _load_mask(self, ann: dict, h: int, w: int) -> np.ndarray:
        seg = ann["segmentation"]
        if isinstance(seg, list):  # polygon
            rles = coco_mask_utils.frPyObjects(seg, h, w)
            rle = coco_mask_utils.merge(rles)
        else:  # already RLE
            rle = seg
        return coco_mask_utils.decode(rle).astype(np.float32)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            masks.append(self._load_mask(ann, orig_h, orig_w))

        # Resize image + boxes + masks to fixed square training resolution.
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        sx, sy = self.img_size / orig_w, self.img_size / orig_h

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            labels = torch.tensor(labels, dtype=torch.long)
            masks_resized = np.stack([
                np.array(Image.fromarray(m).resize((self.img_size, self.img_size), Image.NEAREST))
                for m in masks
            ])
            masks = torch.from_numpy(masks_resized).float()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            masks = torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)

        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return img_tensor, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
