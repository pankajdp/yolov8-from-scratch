"""
Draw the README charts from this project's own training logs and evaluation results.

Run from the repository root:
    python3 make_figures.py
Reads:
    runs/exp1/log.csv, runs/exp2_aug/log.csv                     (written by train.py)
    runs/exp1/metrics.json, runs/exp2_aug/metrics.json            (written by eval.py)
    runs/detect/runs/ultra/v8s_pretrained/weights/metrics_coco.json  (written by eval_ultra.py)
Writes:
    assets/loss_curves.png, assets/results.png, assets/per_class.png
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "assets"
os.makedirs(OUT, exist_ok=True)
GREY, BLUE, ORANGE = "#9ca3af", "#2563eb", "#d97706"

RUNS = [
    ("Scratch, flip only", "runs/exp1", GREY),
    ("Scratch, strong aug", "runs/exp2_aug", BLUE),
]
METRICS = [
    ("Scratch, flip only", ["runs/exp1/metrics_baseline_ep26.json", "runs/exp1/metrics_best_ep26.json",
                            "runs/exp1/metrics.json"], GREY),
    ("Scratch, strong aug", ["runs/exp2_aug/metrics_aug_best.json", "runs/exp2_aug/metrics.json"], BLUE),
    ("YOLOv8s, COCO-pretrained", ["runs/detect/runs/ultra/v8s_pretrained/weights/metrics_coco.json",
                                  "runs/ultra/v8s_pretrained/weights/metrics_coco.json"], ORANGE),
]


def read_log(path):
    """Return {epoch: (train_total, val_total)}; later rows win (a resumed run may repeat an epoch)."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            tr = float(r["tr_box"]) + float(r["tr_cls"]) + float(r["tr_dfl"])
            va = float(r["val_box"]) + float(r["val_cls"]) + float(r["val_dfl"])
            rows[int(float(r["epoch"]))] = (tr, va)
    return dict(sorted(rows.items()))


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# 1. Loss curves --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
for name, folder, color in RUNS:
    path = os.path.join(folder, "log.csv")
    if not os.path.exists(path):
        print("skipping (not found):", path)
        continue
    log = read_log(path)
    ep = list(log)
    tr = [log[e][0] for e in ep]
    va = [log[e][1] for e in ep]
    ax.plot(ep, tr, "--", color=color, lw=1.2, alpha=0.8, label=f"{name}: train")
    ax.plot(ep, va, "-", color=color, lw=2.2, label=f"{name}: validation")
    best = min(log, key=lambda e: log[e][1])
    ax.plot(best, log[best][1], "o", color=color)
    ax.annotate(f"best val {log[best][1]:.2f} (epoch {best})", (best, log[best][1]),
                (best + 2, log[best][1] + 0.6), fontsize=8, color=color)
ax.set_xlabel("epoch")
ax.set_ylabel("loss = box + cls + dfl")
ax.set_title("Training vs validation loss: flip-only overfits, strong augmentation does not")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_curves.png"), dpi=160)
plt.close(fig)

# 2. Overall results ----------------------------------------------------------
loaded = []
for name, paths, color in METRICS:
    p = first_existing(paths)
    if p is None:
        print("skipping (no metrics file):", name)
        continue
    print(f"{name:26s} <- {p}")
    loaded.append((name, json.load(open(p)), color))

keys = [("AP", "mAP@[.5:.95]"), ("AP50", "mAP@.5"), ("AP75", "mAP@.75")]
x = np.arange(len(keys))
w = 0.8 / max(len(loaded), 1)
fig, ax = plt.subplots(figsize=(9, 4))
for i, (name, m, color) in enumerate(loaded):
    vals = [m[k] for k, _ in keys]
    bars = ax.bar(x + (i - (len(loaded) - 1) / 2) * w, vals, w, color=color, label=name)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in keys])
ax.set_ylim(0, 0.7)
ax.set_title("945 validation images, same pycocotools evaluation, 416×416 input")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "results.png"), dpi=160)
plt.close(fig)

# 3. Per-class AP -------------------------------------------------------------
classes = list(loaded[0][1]["per_class_AP"]) if loaded else []
y = np.arange(len(classes))
h = 0.8 / max(len(loaded), 1)
fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, m, color) in enumerate(loaded):
    vals = [m["per_class_AP"][c] for c in classes]
    ax.barh(y + (i - (len(loaded) - 1) / 2) * h, vals, h, color=color, label=name)
ax.set_yticks(y)
ax.set_yticklabels(classes)
ax.invert_yaxis()
ax.set_xlabel("AP@[.5:.95]")
ax.set_title("Per-class AP")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_class.png"), dpi=160)
plt.close(fig)

print("saved assets/loss_curves.png, assets/results.png, assets/per_class.png")
