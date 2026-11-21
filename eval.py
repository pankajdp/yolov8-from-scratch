import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

from data.dataset import COCOSubset, CLASSES, collate_fn
from models.yolo import YOLOv8Scratch
from utils.loss import make_anchors, dist2bbox


@torch.no_grad()
def decode(preds, strides, reg_max=16):
    """Raw head outputs -> boxes (xyxy, pixels of the letterboxed input) + class probabilities."""
    device = preds[0][0].device
    feat_shapes = [b.shape[2:] for b, _ in preds]
    box = torch.cat([b.flatten(2) for b, _ in preds], 2).permute(0, 2, 1)   # [B, N, 64]
    cls = torch.cat([c.flatten(2) for _, c in preds], 2).permute(0, 2, 1)   # [B, N, nc]
    anchors, stride_t = make_anchors(feat_shapes, strides, device)
    B, N, _ = box.shape
    proj = torch.arange(reg_max, dtype=torch.float32, device=device)
    dist = box.reshape(B, N, 4, reg_max).softmax(-1) @ proj                 # [B, N, 4] grid units
    boxes = dist2bbox(dist, anchors / stride_t) * stride_t                  # pixels
    return boxes, cls.sigmoid()


def postprocess(boxes, scores, conf, iou, pre_nms=3000, max_det=300):
    """Per image: keep confident (box, class) pairs, then NMS within each class."""
    results = []
    for b, s in zip(boxes, scores):
        nc = s.shape[1]
        flat = s.flatten()
        idx = (flat > conf).nonzero().squeeze(1)
        if idx.numel() > pre_nms:
            idx = idx[flat[idx].topk(pre_nms).indices]
        if idx.numel() == 0:
            results.append((b.new_zeros((0, 4)), b.new_zeros(0), idx))
            continue
        bx, sc, ci = b[idx // nc], flat[idx], idx % nc
        keep = batched_nms(bx, sc, ci, iou)[:max_det]
        results.append((bx[keep], sc[keep], ci[keep]))
    return results


def letterbox_meta(path, size):
    """Same arithmetic as the dataset's letterbox, so we can undo it."""
    w0, h0 = Image.open(path).size
    r = size / max(h0, w0)
    nh, nw = round(h0 * r), round(w0 * r)
    return w0, h0, r, (size - nw) // 2, (size - nh) // 2


def build_gt(ds):
    """Ground truth in COCO format (original pixel coordinates). Category ids are class+1."""
    images, anns, aid = [], [], 1
    for i, name in enumerate(ds.names):
        w0, h0 = Image.open(os.path.join(ds.img_dir, name + ".jpg")).size
        images.append({"id": i, "width": w0, "height": h0})
        p = os.path.join(ds.lbl_dir, name + ".txt")
        if not os.path.exists(p):
            continue
        for c, cx, cy, w, h in np.loadtxt(p, ndmin=2, dtype=np.float32):
            bw, bh = float(w * w0), float(h * h0)
            anns.append({"id": aid, "image_id": i, "category_id": int(c) + 1,
                         "bbox": [float((cx - w / 2) * w0), float((cy - h / 2) * h0), bw, bh],
                         "area": bw * bh, "iscrowd": 0})
            aid += 1
    gt = COCO()
    gt.dataset = {"info": {}, "images": images, "annotations": anns,
                  "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}
    gt.createIndex()
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--data", default=os.path.expanduser("~/projects/cv-portfolio/datasets/coco_subset"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--sanity-gt", action="store_true",
                    help="score the ground truth against itself; should give AP 1.0")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.weights, map_location=device, weights_only=False) if args.weights else None
    imgsz = ck["args"]["imgsz"] if ck else 416
    val_ds = COCOSubset(args.data, "val", img_size=imgsz)
    gt = build_gt(val_ds)
    dets = []

    if args.sanity_gt:
        for a in gt.dataset["annotations"]:
            dets.append({"image_id": a["image_id"], "category_id": a["category_id"],
                         "bbox": a["bbox"], "score": 1.0})
    else:
        model = YOLOv8Scratch(num_classes=len(CLASSES), base_ch=ck["args"]["base_ch"]).to(device)
        model.load_state_dict(ck["ema"])
        model.eval()
        loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_fn)
        t_model, n_img, k = 0.0, 0, 0
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                torch.cuda.synchronize(); t0 = time.time()
                preds = model(imgs)
                torch.cuda.synchronize(); t_model += time.time() - t0
                boxes, scores = decode(preds, model.head.strides)
                for bx, sc, ci in postprocess(boxes, scores, args.conf, args.iou):
                    name = val_ds.names[k]
                    w0, h0, r, left, top = letterbox_meta(os.path.join(val_ds.img_dir, name + ".jpg"), imgsz)
                    bx = bx.cpu().numpy()
                    bx[:, [0, 2]] = ((bx[:, [0, 2]] - left) / r).clip(0, w0)
                    bx[:, [1, 3]] = ((bx[:, [1, 3]] - top) / r).clip(0, h0)
                    for (x1, y1, x2, y2), s, c in zip(bx.tolist(), sc.tolist(), ci.tolist()):
                        if x2 - x1 > 0 and y2 - y1 > 0:
                            dets.append({"image_id": k, "category_id": c + 1,
                                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
                    k += 1
                n_img += len(imgs)
        print(f"model forward: {1000 * t_model / n_img:.1f} ms/img (batch {args.batch}, fp32, {torch.cuda.get_device_name(0)})")

    print(f"{len(val_ds)} val images, {len(gt.dataset['annotations'])} GT boxes, {len(dets)} detections")
    if not dets:
        print("no detections above the confidence threshold: AP = 0")
        return

    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    prec = ev.eval["precision"]   # [IoU, recall, class, area, maxDets]
    per_class = {}
    print("\nper-class AP@[.5:.95]:")
    for ki, name in enumerate(CLASSES):
        p = prec[:, :, ki, 0, 2]
        ap_k = float(p[p > -1].mean()) if (p > -1).any() else 0.0
        per_class[name] = ap_k
        print(f"  {name:15s} {ap_k:.3f}")

    if args.weights:
        out = os.path.join(os.path.dirname(args.weights), "metrics.json")
        json.dump({"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2],
                   "per_class_AP": per_class, "epoch": ck.get("epoch")}, open(out, "w"), indent=2)
        print("saved", out)


if __name__ == "__main__":
    main()
