YOLACT-style Instance Segmentation for Laparoscopic Organ Detection

A from-scratch PyTorch reimplementation of the architecture used in Caballas et al.,
"Development of a Visual Guidance System for Laparoscopic Surgical Palpation using
Computer Vision" (IECBES 2020) — ResNet-50 + FPN backbone, ProtoNet, and a
YOLACT prediction head, trainable end-to-end on COCO-format instance masks.
All code in this package (model.py, loss.py, dataset.py, inference.py,
train.py) has been run and verified: forward pass, loss + backward pass,
dataset loading, a short training loop, and inference/NMS all execute correctly.

https://www.loom.com/share/b0fc2706e68d4b0aac9c922a89f4efc2?resume-anon-signup=true
