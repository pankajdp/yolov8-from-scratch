import argparse
import json
import os

import numpy as np
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from data.dataset import COCOSubset, CLASSES
from eval import build_gt

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default=os.path.expanduser("~/projects/cv-portfolio/datasets/coco_subset"))
ap.add_argument("--imgsz", type=int, default=416)
args = ap.parse_args()

val_ds = COCOSubset(args.data, "val", img_size=args.imgsz)
gt = build_gt(val_ds)                      # same ground truth as eval.py, image_id = index in val_ds.names
model = YOLO(args.weights)

paths = [os.path.join(val_ds.img_dir, n + ".jpg") for n in val_ds.names]
dets = []
for start in range(0, len(paths), 32):
    results = model.predict(paths[start:start + 32], imgsz=args.imgsz, conf=0.001, iou=0.65,
                            max_det=300, verbose=False)
    for k, r in enumerate(results, start=start):
        b = r.boxes                         # already in original-image pixels
        for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
            dets.append({"image_id": k, "category_id": int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})

print(f"{len(val_ds)} val images, {len(gt.dataset['annotations'])} GT boxes, {len(dets)} detections")
ev = COCOeval(gt, gt.loadRes(dets), "bbox")
ev.evaluate(); ev.accumulate(); ev.summarize()

prec = ev.eval["precision"]
per_class = {}
print("\nper-class AP@[.5:.95]:")
for ki, name in enumerate(CLASSES):
    p = prec[:, :, ki, 0, 2]
    per_class[name] = float(p[p > -1].mean()) if (p > -1).any() else 0.0
    print(f"  {name:15s} {per_class[name]:.3f}")

out = os.path.join(os.path.dirname(args.weights), "metrics_coco.json")
json.dump({"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "per_class_AP": per_class},
          open(out, "w"), indent=2)
print("saved", out)
