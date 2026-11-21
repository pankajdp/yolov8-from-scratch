"""
Build a small COCO subset in YOLO label format.
Only downloads the images that contain the chosen classes.

Output layout:
  <out>/images/{train,val}/<image_id>.jpg
  <out>/labels/{train,val}/<image_id>.txt   one line per object: cls cx cy w h (all 0-1)
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from pycocotools.coco import COCO
from tqdm import tqdm

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus",
           "truck", "traffic light", "stop sign", "dog", "cat"]


def download(url, path):
    if os.path.exists(path):
        return True  # already there -- lets you re-run safely after an interruption
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return False
    with open(path, "wb") as f:
        f.write(r.content)
    return True


def pick_images(coco, cat_ids, max_images):
    """Take roughly the same number of images per class, so 'person' doesn't swamp the rest."""
    per_class = max_images // len(cat_ids)
    chosen = []
    seen = set()
    for c in cat_ids:
        taken = 0
        for img_id in sorted(coco.getImgIds(catIds=[c])):
            if taken == per_class:
                break
            if img_id not in seen:
                seen.add(img_id)
                chosen.append(img_id)
                taken += 1
    return chosen


def yolo_lines(coco, img_id, cat_ids, id_map):
    """COCO boxes are [x_left, y_top, w, h] in pixels -> YOLO 'cls cx cy w h' normalized."""
    info = coco.loadImgs(img_id)[0]
    W, H = info["width"], info["height"]
    lines = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)):
        x, y, w, h = ann["bbox"]
        if w < 1 or h < 1:
            continue  # skip degenerate boxes
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        lines.append(f"{id_map[ann['category_id']]} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def build_split(coco_root, out_root, split, max_images, workers):
    coco = COCO(os.path.join(coco_root, "annotations", f"instances_{split}2017.json"))
    cat_ids = coco.getCatIds(catNms=CLASSES)
    # COCO's category ids are not 0..9 and not in our order -> remap to our CLASSES index
    id_map = {c["id"]: CLASSES.index(c["name"]) for c in coco.loadCats(cat_ids)}

    img_dir = os.path.join(out_root, "images", split)
    lbl_dir = os.path.join(out_root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    jobs = []
    for img_id in pick_images(coco, cat_ids, max_images):
        lines = yolo_lines(coco, img_id, cat_ids, id_map)
        if lines:  # images whose only boxes were crowd/degenerate are skipped
            jobs.append((img_id, coco.loadImgs(img_id)[0]["coco_url"], lines))

    def work(job):
        img_id, url, lines = job
        if not download(url, os.path.join(img_dir, f"{img_id:012d}.jpg")):
            return False
        with open(os.path.join(lbl_dir, f"{img_id:012d}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        return True

    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(work, jobs), total=len(jobs), desc=f"{split} images"))
    print(f"{split}: {sum(results)} images saved, {results.count(False)} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default=os.path.expanduser("~/projects/cv-portfolio/datasets/coco"))
    parser.add_argument("--out", default=os.path.expanduser("~/projects/cv-portfolio/datasets/coco_subset"))
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    build_split(args.coco_root, args.out, "val", args.max_val, args.workers)
    build_split(args.coco_root, args.out, "train", args.max_train, args.workers)
    print("classes:", CLASSES)
