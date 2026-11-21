import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASSES = ["person", "bicycle", "car", "motorcycle", "bus",
           "truck", "traffic light", "stop sign", "dog", "cat"]


def letterbox(img, size, pad_value=114):
    """Resize so the longer side == size (keeps aspect ratio), then pad to a size x size square."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, left, top


class COCOSubset(Dataset):
    """
    Reads the YOLO-format subset made by download_coco_subset.py.
    Each item: image [3, S, S] float 0-1, labels [n, 5] = (cls, cx, cy, w, h) normalized 0-1.
    """
    def __init__(self, root, split="train", img_size=416, augment=False):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.img_size = img_size
        self.augment = augment
        self.names = sorted(f[:-4] for f in os.listdir(self.img_dir) if f.endswith(".jpg"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(os.path.join(self.img_dir, name + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR, models expect RGB
        h0, w0 = img.shape[:2]
        img, r, left, top = letterbox(img, self.img_size)

        lbl_path = os.path.join(self.lbl_dir, name + ".txt")
        if os.path.exists(lbl_path):
            labels = np.loadtxt(lbl_path, ndmin=2, dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        # Labels were normalized to the ORIGINAL image; move them to the letterboxed square
        S = self.img_size
        labels[:, 1] = (labels[:, 1] * w0 * r + left) / S
        labels[:, 2] = (labels[:, 2] * h0 * r + top) / S
        labels[:, 3] = labels[:, 3] * w0 * r / S
        labels[:, 4] = labels[:, 4] * h0 * r / S

        if self.augment and np.random.rand() < 0.5:  # horizontal flip
            img = img[:, ::-1]
            labels[:, 1] = 1 - labels[:, 1]

        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        return img, torch.from_numpy(labels)


def collate_fn(batch):
    """
    Images have different numbers of objects, so they can't be stacked directly.
    Stack the images, and join all labels into one [K, 6] tensor with the image index
    in front: (batch_idx, cls, cx, cy, w, h) -- exactly what YOLOLoss expects.
    """
    imgs, labels = zip(*batch)
    targets = [torch.cat([torch.full((len(l), 1), float(i)), l], dim=1) for i, l in enumerate(labels)]
    return torch.stack(imgs), torch.cat(targets)


if __name__ == "__main__":
    root = os.path.expanduser("~/projects/cv-portfolio/datasets/coco_subset")
    train_ds = COCOSubset(root, "train", augment=True)
    val_ds = COCOSubset(root, "val")
    print(f"train images: {len(train_ds)}, val images: {len(val_ds)}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    imgs, targets = next(iter(loader))
    print("batch images:", imgs.shape)    # expect [4, 3, 416, 416]
    print("batch targets:", targets.shape)  # expect [K, 6]
    print("first rows:\n", targets[:3])

    # Draw the boxes of the first image so you can check them by eye
    img = (imgs[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    S = img.shape[0]
    for _, cls, cx, cy, w, h in targets[targets[:, 0] == 0].tolist():
        x1, y1 = int((cx - w / 2) * S), int((cy - h / 2) * S)
        x2, y2 = int((cx + w / 2) * S), int((cy + h / 2) * S)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, CLASSES[int(cls)], (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("sample_check.jpg", img)
    print("saved sample_check.jpg -- open it and check the boxes sit on the objects")

