import argparse
import copy
import csv
import math
import os
import time

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import COCOSubset, collate_fn
from data.augment import COCOSubsetAug
from models.yolo import YOLOv8Scratch
from utils.loss import YOLOLoss

cv2.setNumThreads(0)
NUM_CLASSES = 10


class ModelEMA:
    """Exponential moving average of the weights. Evaluating the EMA copy is
    smoother and usually scores better than the raw weights."""
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))  # ramps up from 0
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def build_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)  # no weight decay on BN / biases
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.937, 0.999))


def lr_at(it, total, warmup, base_lr, final_frac=0.01):
    """Linear warmup, then cosine decay down to 1% of the base learning rate."""
    if it < warmup:
        return base_lr * (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return base_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def validate(model, criterion, loader, device, amp, max_batches=0):
    model.eval()
    sums, n = torch.zeros(3, device=device), 0
    for i, (imgs, targets) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            preds = model(imgs.to(device, non_blocking=True))
        preds = [(b.float(), c.float()) for b, c in preds]
        _, parts = criterion(preds, targets.to(device))
        sums += parts
        n += 1
    return (sums / max(n, 1)).tolist()  # [box, cls, dfl], already weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/projects/cv-portfolio/datasets/coco_subset"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nominal-batch", type=int, default=32, help="effective batch via gradient accumulation")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--base-ch", type=int, default=32, help="model width; 64 = the 43.8M-param model")
    ap.add_argument("--warmup-epochs", type=float, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision (risky on GTX 16xx)")
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: only run N batches per epoch")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--resume", default="")
    ap.add_argument("--aug", choices=["basic", "strong"], default="basic")
    ap.add_argument("--close-mosaic", type=int, default=10, help="last N epochs without mosaic")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    if args.aug == "strong":
        train_ds = COCOSubsetAug(args.data, "train", img_size=args.imgsz)
    else:
        train_ds = COCOSubset(args.data, "train", img_size=args.imgsz, augment=True)
    val_ds = COCOSubset(args.data, "val", img_size=args.imgsz)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True, collate_fn=collate_fn)

    model = YOLOv8Scratch(num_classes=NUM_CLASSES, base_ch=args.base_ch).to(device)
    model.head.init_bias(img_size=args.imgsz)   # class bias must match the real grid size
    criterion = YOLOLoss(num_classes=NUM_CLASSES).to(device)
    optimizer = build_optimizer(model, args.lr, args.wd)
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    nb = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
    accum = max(1, round(args.nominal_batch / args.batch))
    total_its = args.epochs * nb
    warmup_its = int(args.warmup_epochs * nb)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,} | train {len(train_ds)} val {len(val_ds)} | "
          f"batch {args.batch} x accum {accum} | imgsz {args.imgsz} | amp {args.amp}")

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.ema.load_state_dict(ck["ema"])
        ema.updates = ck["ema_updates"]
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "tr_box", "tr_cls", "tr_dfl", "val_box", "val_cls", "val_dfl", "lr"])

    for epoch in range(start_epoch, args.epochs):
        if args.aug == "strong":
            train_ds.mosaic_p = 0.0 if epoch >= args.epochs - args.close_mosaic else 1.0
        model.train()
        t0 = time.time()
        running, seen = torch.zeros(3, device=device), 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, total=nb, desc=f"epoch {epoch + 1}/{args.epochs}")
        for i, (imgs, targets) in enumerate(pbar):
            if i >= nb:
                break
            lr = lr_at(epoch * nb + i, total_its, warmup_its, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                preds = model(imgs)
            preds = [(b.float(), c.float()) for b, c in preds]   # loss always runs in fp32
            loss, parts = criterion(preds, targets)

            if not torch.isfinite(loss):
                print("\nnon-finite loss, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            # YOLOLoss multiplies by batch size (made for SGD). AdamW ignores gradient
            # scale, so divide it back out to keep gradient clipping meaningful.
            scaler.scale(loss / imgs.shape[0] / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == nb:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            running += parts
            seen += 1
            avg = (running / seen).tolist()
            pbar.set_postfix(box=f"{avg[0]:.2f}", cls=f"{avg[1]:.2f}", dfl=f"{avg[2]:.2f}")

        tr = (running / max(seen, 1)).tolist()
        va = validate(ema.ema, criterion, val_loader, device, args.amp, args.max_batches)
        val_total = sum(va)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"epoch {epoch + 1}: train {sum(tr):.3f} | val {val_total:.3f} "
              f"(box {va[0]:.2f} cls {va[1]:.2f} dfl {va[2]:.2f}) | "
              f"{time.time() - t0:.0f}s | peak GPU {peak:.2f} GB")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, *tr, *va, lr])

        ck = {"epoch": epoch, "model": model.state_dict(), "ema": ema.ema.state_dict(),
              "ema_updates": ema.updates, "optimizer": optimizer.state_dict(),
              "best": min(best_val, val_total), "args": vars(args)}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if val_total < best_val:
            best_val = val_total
            torch.save({"ema": ema.ema.state_dict(), "args": vars(args), "epoch": epoch},
                       os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
