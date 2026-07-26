"""
Training script.

Usage:
    python train.py \
        --train-images dataset/train/images --train-ann dataset/train/annotations.json \
        --val-images   dataset/val/images   --val-ann   dataset/val/annotations.json \
        --num-classes 1 --epochs 50 --batch-size 4 --lr 1e-3

Matches the paper's training setup where reasonable (550px input, batch size
scaled to your GPU memory — the paper used batch size 4 on a 6GB GTX 2060).
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from dataset import CocoInstanceSegDataset, collate_fn
from loss import YOLACTLoss
from model import YOLACT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--val-images", default=None)
    p.add_argument("--val-ann", default=None)
    p.add_argument("--num-classes", type=int, required=True, help="number of FOREGROUND classes")
    p.add_argument("--img-size", type=int, default=550)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-out", default="yolact_checkpoint.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    train_ds = CocoInstanceSegDataset(args.train_images, args.train_ann, img_size=args.img_size, train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    val_loader = None
    if args.val_images and args.val_ann:
        val_ds = CocoInstanceSegDataset(args.val_images, args.val_ann, img_size=args.img_size, train=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)

    model = YOLACT(num_classes=args.num_classes, pretrained_backbone=True).to(device)
    loss_fn = YOLACTLoss(num_classes=args.num_classes + 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)], gamma=0.1
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "mask_loss": 0.0}

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)
            losses = loss_fn(preds, targets)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()

            if i % 20 == 0:
                print(f"epoch {epoch} iter {i}/{len(train_loader)} "
                      f"loss={losses['loss'].item():.3f} "
                      f"(cls={losses['cls_loss'].item():.3f} "
                      f"box={losses['box_loss'].item():.3f} "
                      f"mask={losses['mask_loss'].item():.3f})")

        scheduler.step()
        n = len(train_loader)
        print(f"== epoch {epoch} done in {time.time()-t0:.1f}s | "
              f"avg loss={running['loss']/n:.3f} ==")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    preds = model(images)
                    losses = loss_fn(preds, targets)
                    val_loss += losses["loss"].item()
            val_loss /= max(len(val_loader), 1)
            print(f"   val loss={val_loss:.3f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.checkpoint_out)
                print(f"   saved new best checkpoint -> {args.checkpoint_out}")
        else:
            torch.save(model.state_dict(), args.checkpoint_out)

    print("Training complete.")


if __name__ == "__main__":
    main()
