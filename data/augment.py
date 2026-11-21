import os
import random

import cv2
import numpy as np
import torch

from data.dataset import COCOSubset, CLASSES

cv2.setNumThreads(0)


def hsv_jitter(img, hgain=0.015, sgain=0.5, vgain=0.3):
    """Random hue / saturation / brightness shift. img: RGB uint8."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    out = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)


class COCOSubsetAug(COCOSubset):
    """
    Same outputs as COCOSubset, but with probability mosaic_p a sample is a 4-image mosaic
    with random scale/shift and colour jitter. Otherwise it falls back to the basic pipeline.
    """
    def __init__(self, root, split="train", img_size=416, mosaic_p=1.0, scale=0.5, translate=0.1):
        super().__init__(root, split, img_size, augment=True)
        self.mosaic_p, self.scale, self.translate = mosaic_p, scale, translate

    def load_resized(self, i):
        """Image resized so its longer side == img_size; labels as (cls, x1, y1, x2, y2) in pixels."""
        name = self.names[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name + ".jpg")), cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        nw, nh = round(w0 * r), round(h0 * r)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        p = os.path.join(self.lbl_dir, name + ".txt")
        lb = np.loadtxt(p, ndmin=2, dtype=np.float32).reshape(-1, 5) if os.path.exists(p) \
            else np.zeros((0, 5), np.float32)
        out = np.zeros((len(lb), 5), np.float32)
        if len(lb):
            out[:, 0] = lb[:, 0]
            out[:, 1] = (lb[:, 1] - lb[:, 3] / 2) * nw
            out[:, 2] = (lb[:, 2] - lb[:, 4] / 2) * nh
            out[:, 3] = (lb[:, 1] + lb[:, 3] / 2) * nw
            out[:, 4] = (lb[:, 2] + lb[:, 4] / 2) * nh
        return img, out

    def load_mosaic(self, i):
        """Paste 4 images around a random centre point on a 2S x 2S canvas."""
        S = self.img_size
        xc, yc = [int(random.uniform(S * 0.5, S * 1.5)) for _ in range(2)]
        idxs = [i] + random.choices(range(len(self)), k=3)
        canvas = np.full((2 * S, 2 * S, 3), 114, np.uint8)
        all_labels = []
        for k, idx in enumerate(idxs):
            img, lb = self.load_resized(idx)
            h, w = img.shape[:2]
            if k == 0:    # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * S), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * S, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:         # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * S), min(2 * S, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(lb):
                lb[:, [1, 3]] += x1a - x1b
                lb[:, [2, 4]] += y1a - y1b
                all_labels.append(lb)
        labels = np.concatenate(all_labels) if all_labels else np.zeros((0, 5), np.float32)
        labels[:, 1:] = labels[:, 1:].clip(0, 2 * S)
        return canvas, labels

    def random_affine(self, img, labels):
        """Random zoom + shift of the 2S canvas down to the S x S network input."""
        S = self.img_size
        H, W = img.shape[:2]
        C = np.eye(3); C[0, 2], C[1, 2] = -W / 2, -H / 2          # move canvas centre to origin
        s = random.uniform(1 - self.scale, 1 + self.scale)
        Sc = np.eye(3); Sc[0, 0] = Sc[1, 1] = s                     # zoom
        T = np.eye(3)                                               # place near the output centre
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * S
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * S
        M = T @ Sc @ C
        img = cv2.warpAffine(img, M[:2], (S, S), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        n = len(labels)
        if n:
            pts = np.ones((n * 4, 3), np.float32)
            pts[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # the 4 box corners
            pts = (pts @ M.T)[:, :2].reshape(n, 8)
            xs, ys = pts[:, [0, 2, 4, 6]], pts[:, [1, 3, 5, 7]]
            new = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], 1).clip(0, S)
            area0 = (labels[:, 3] - labels[:, 1]) * (labels[:, 4] - labels[:, 2]) * s * s
            w, h = new[:, 2] - new[:, 0], new[:, 3] - new[:, 1]
            keep = ((w > 3) & (h > 3) & (w * h > 0.1 * area0)          # drop slivers and mostly-cropped boxes
                    & (np.maximum(w / (h + 1e-9), h / (w + 1e-9)) < 20))
            labels = np.concatenate([labels[keep, :1], new[keep]], 1).astype(np.float32)
        return img, labels

    def __getitem__(self, i):
        if random.random() >= self.mosaic_p:
            return super().__getitem__(i)      # basic letterbox + flip
        S = self.img_size
        img, labels = self.load_mosaic(i)
        img, labels = self.random_affine(img, labels)
        img = hsv_jitter(img)
        if random.random() < 0.5:              # horizontal flip
            img = img[:, ::-1]
            if len(labels):
                labels[:, [1, 3]] = S - labels[:, [3, 1]]
        out = np.zeros((len(labels), 5), np.float32)   # xyxy pixels -> normalized cx cy w h
        if len(labels):
            out[:, 0] = labels[:, 0]
            out[:, 1] = (labels[:, 1] + labels[:, 3]) / 2 / S
            out[:, 2] = (labels[:, 2] + labels[:, 4]) / 2 / S
            out[:, 3] = (labels[:, 3] - labels[:, 1]) / S
            out[:, 4] = (labels[:, 4] - labels[:, 2]) / S

        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        return img, torch.from_numpy(out)


if __name__ == "__main__":
    root = os.path.expanduser("~/projects/cv-portfolio/datasets/coco_subset")
    ds = COCOSubsetAug(root, "train")
    tiles = []
    for k in range(4):
        img, lb = ds[k]
        im = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
        S = im.shape[0]
        for c, cx, cy, w, h in lb.tolist():
            x1, y1 = int((cx - w / 2) * S), int((cy - h / 2) * S)
            x2, y2 = int((cx + w / 2) * S), int((cy + h / 2) * S)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, CLASSES[int(c)], (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles.append(im)
    cv2.imwrite("aug_check.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print("saved aug_check.jpg: open it and check the green boxes sit on the objects")
