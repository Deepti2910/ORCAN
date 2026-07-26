# YOLACT-style Instance Segmentation for Laparoscopic Organ Detection

A PyTorch reimplementation of the architecture used in Caballas et al., *"Development of a Visual Guidance System for Laparoscopic Surgical Palpation using Computer Vision"* (IECBES 2020) — a ResNet-50 + FPN backbone, ProtoNet, and YOLACT-style prediction head for real-time instance segmentation of organs (e.g. the gallbladder) in laparoscopic video, trainable end-to-end on COCO-format instance masks.

## Overview

Palpation — using touch to detect tissue anomalies like tumors or cysts — isn't possible in laparoscopic surgery since surgeons can't directly access the patient's viscera by hand. This project implements the visual guidance component of a motion-based laparoscopic palpation system: a real-time instance segmentation model that localizes a target organ in the endoscope feed so that palpation guidance can be overlaid on it.

## Architecture

- **Backbone:** ResNet-50 (ImageNet-pretrained) + Feature Pyramid Network, producing feature maps at strides 8, 16, 32, 64, and 128 (P3–P7)
- **ProtoNet:** generates 32 shared prototype masks from the highest-resolution feature map (P3)
- **Prediction head:** shared across all pyramid levels, predicting per-anchor class scores, box regression deltas, and mask coefficients
- **Mask assembly:** final instance masks are produced by linearly combining prototypes with per-instance mask coefficients, then cropping to the predicted bounding box ("Lincomb" mask formulation)
- **NMS:** Fast NMS (matrix-based, non-sequential) for efficient real-time inference

## Data preparation

1. Download CholecSeg8k from Kaggle: https://www.kaggle.com/datasets/newslab/cholecseg8k
2. Convert the pixel-mask annotations to COCO instance-segmentation format (polygons + bounding boxes per organ/tool class), matching the schema described below.
3. Split into `train/` and `val/` image + annotation sets.

## Installation

```bash
pip install torch torchvision pycocotools pillow numpy
```

## Usage

### Sanity check
```bash
python model.py
```
Runs a forward pass on a dummy input and prints output tensor shapes.

### Training
```bash
python train.py \
    --train-images dataset/train/images --train-ann dataset/train/annotations.json \
    --val-images   dataset/val/images   --val-ann   dataset/val/annotations.json \
    --num-classes 1 --epochs 50 --batch-size 4 --img-size 550 --lr 1e-3
```

`--num-classes` is the number of foreground classes (1 for gallbladder-only, as in the original paper; increase if segmenting additional organs or tools).

## Demonstration Video
Kindly consider this demonstration video explaining the results that can be obtained from this model.
https://www.loom.com/share/b0fc2706e68d4b0aac9c922a89f4efc2

## Reference

K. Caballas, H. J. M. Bolingot, N. J. C. Libatique, and G. L. Tangonan, "Development of a Visual Guidance System for Laparoscopic Surgical Palpation using Computer Vision," 2020 IEEE EMBS Conference on Biomedical Engineering and Sciences (IECBES), 2021, doi: 10.1109/IECBES48179.2021.9398796.
