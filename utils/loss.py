import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def bbox_iou(box1, box2, ciou=True, eps=1e-7):
    """
    Compute IoU (or CIoU) between two sets of boxes.
    box1, box2: [..., 4] in (x1, y1, x2, y2) format.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if not ciou:
        return iou

    # Enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance
    center_dist = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    # Aspect ratio term
    v = (4 / (3.14159265 ** 2)) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    ciou_val = iou - (center_dist / c2 + v * alpha)
    return ciou_val


def dist2bbox(distance, anchor_points, xywh=False):
    """Convert (left, top, right, bottom) distances to (x1,y1,x2,y2) boxes."""
    lt, rb = distance.chunk(2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim=-1)
    return torch.cat([x1y1, x2y2], dim=-1)


def make_anchors(feat_shapes, strides, device):
    """
    Build anchor center points for each scale.
    feat_shapes: list of (H, W) per scale.
    Returns: anchor_points [N, 2], stride_tensor [N, 1]
    """
    anchor_points, stride_list = [], []
    for (h, w), stride in zip(feat_shapes, strides):
        sx = torch.arange(w, device=device, dtype=torch.float32) + 0.5
        sy = torch.arange(h, device=device, dtype=torch.float32) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack([sx, sy], dim=-1).reshape(-1, 2) * stride)
        stride_list.append(torch.full((h * w, 1), stride, device=device, dtype=torch.float32))
    return torch.cat(anchor_points), torch.cat(stride_list)


class TaskAlignedAssigner(nn.Module):
    """
    Assigns ground-truth objects to predictions based on an alignment score:
    classification confidence (for the right class) x IoU with the GT box.
    Picks top-k predictions per GT as positive samples.
    """
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha  # weight on classification score
        self.beta = beta    # weight on IoU (exponent)
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, gt_mask):
        """
        pred_scores: [B, N, num_classes]  (sigmoid probabilities)
        pred_bboxes: [B, N, 4]            (x1,y1,x2,y2, decoded)
        anchor_points: [N, 2]
        gt_labels: [B, M]                 (class indices, padded)
        gt_bboxes: [B, M, 4]              (x1,y1,x2,y2, padded)
        gt_mask: [B, M]                   (1 = real GT, 0 = padding)

        Returns:
          target_labels [B, N], target_bboxes [B, N, 4],
          target_scores [B, N, num_classes], fg_mask [B, N]
        """
        B, N, _ = pred_scores.shape
        M = gt_bboxes.shape[1]

        if M == 0:
            device = pred_scores.device
            return (torch.zeros(B, N, dtype=torch.long, device=device),
                    torch.zeros(B, N, 4, device=device),
                    torch.zeros(B, N, self.num_classes, device=device),
                    torch.zeros(B, N, dtype=torch.bool, device=device))

        # 1. Is each anchor point inside each GT box? [B, M, N]
        gt_x1y1 = gt_bboxes[..., :2].unsqueeze(2)     # [B, M, 1, 2]
        gt_x2y2 = gt_bboxes[..., 2:].unsqueeze(2)     # [B, M, 1, 2]
        ap = anchor_points.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        in_box = ((ap >= gt_x1y1) & (ap <= gt_x2y2)).all(dim=-1)  # [B, M, N]

        # 2. Classification score for the GT's class, per anchor. [B, M, N]
        gt_labels_long = gt_labels.long().clamp(0, self.num_classes - 1)
        cls_scores = pred_scores.gather(
            2, gt_labels_long.unsqueeze(2).expand(-1, -1, N).transpose(1, 2)
        ).transpose(1, 2)  # -> [B, M, N]

        # 3. IoU between each prediction's box and each GT box. [B, M, N]
        pb = pred_bboxes.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, N, 4]
        gb = gt_bboxes.unsqueeze(2).expand(-1, -1, N, -1)    # [B, M, N, 4]
        ious = bbox_iou(pb, gb, ciou=False).clamp(0)          # [B, M, N]

        # 4. Combined alignment score
        align_score = (cls_scores.clamp(0) ** self.alpha) * (ious ** self.beta)
        align_score = align_score * in_box                 # zero out anchors outside the GT box
        align_score = align_score * gt_mask.unsqueeze(-1)  # zero out padded GTs

        # 5. Pick top-k anchors per GT by alignment score
        topk = min(self.topk, N)
        topk_scores, topk_idx = torch.topk(align_score, topk, dim=-1)  # [B, M, topk]
        pos_mask = torch.zeros_like(align_score, dtype=torch.bool)
        pos_mask.scatter_(-1, topk_idx, topk_scores > self.eps)       # [B, M, N]

        # 6. If an anchor is claimed by several GTs, keep the claimant with the highest IoU
        overlap_count = pos_mask.sum(1)                                  # [B, N]
        if (overlap_count > 1).any():
            multi = (overlap_count > 1).unsqueeze(1).expand(-1, M, -1)   # [B, M, N]
            best_iou_gt = (ious * pos_mask).argmax(1)                    # [B, N]
            best_mask = F.one_hot(best_iou_gt, M).permute(0, 2, 1).bool()
            pos_mask = torch.where(multi, best_mask, pos_mask)

        fg_mask = pos_mask.any(1)                                        # [B, N]
        best_gt_idx = pos_mask.float().argmax(1).clamp(0, M - 1)         # [B, N]

        # 7. Gather final targets per anchor
        target_labels = gt_labels_long.gather(1, best_gt_idx)            # [B, N]
        target_bboxes = gt_bboxes.gather(
            1, best_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
        )                                                                # [B, N, 4]

        # 8. Soft class targets: scale each positive by its normalized alignment,
        #    so well-localized predictions are trained toward ~their IoU, not 1.0
        pos_align = align_score * pos_mask                               # [B, M, N]
        max_align = pos_align.amax(-1, keepdim=True)                     # [B, M, 1]
        max_iou = (ious * pos_mask).amax(-1, keepdim=True)               # [B, M, 1]
        norm_align = (pos_align * max_iou / (max_align + self.eps)).amax(1)  # [B, N]

        target_scores = F.one_hot(target_labels, self.num_classes).float()
        target_scores = target_scores * (fg_mask.float() * norm_align).unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask


def bbox2dist(anchor_points, bbox, reg_max):
    """Inverse of dist2bbox: (x1,y1,x2,y2) -> (left, top, right, bottom) distances,
    clamped so the target always falls inside the DFL bin range [0, reg_max-1)."""
    x1y1, x2y2 = bbox.chunk(2, dim=-1)
    dist = torch.cat([anchor_points - x1y1, x2y2 - anchor_points], dim=-1)
    return dist.clamp(0, reg_max - 1 - 0.01)



class YOLOLoss(nn.Module):
    """
    Full YOLOv8 training loss = box (CIoU) + classification (BCE) + DFL.
    Takes the raw head outputs and ground-truth targets, returns one scalar.
    """
    def __init__(self, num_classes=80, reg_max=16, strides=(8, 16, 32),
                 box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, topk=10):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.box_gain = box_gain    # same default weights as official YOLOv8
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=num_classes)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # [0, 1, ..., 15] -- used to turn a bin distribution into one expected distance
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))

    def decode_dist(self, box_raw):
        """[B, N, 4*reg_max] logits -> [B, N, 4] expected distances (grid units)."""
        B, N, _ = box_raw.shape
        probs = box_raw.reshape(B, N, 4, self.reg_max).softmax(-1)
        return probs @ self.proj

    def dfl_loss(self, pred_logits, target):
        """
        pred_logits: [P, 4, reg_max], target: [P, 4] continuous distances.
        Cross-entropy against the two bins on either side of the target,
        weighted by how close the target is to each.
        """
        tl = target.long()           # left bin
        tr = tl + 1                  # right bin
        wl = tr.float() - target     # weight for left bin
        wr = 1 - wl                  # weight for right bin
        logits = pred_logits.reshape(-1, self.reg_max)
        loss = (F.cross_entropy(logits, tl.reshape(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(logits, tr.reshape(-1), reduction="none").view(tl.shape) * wr)
        return loss.mean(-1, keepdim=True)  # [P, 1]

    def preprocess_targets(self, targets, B, img_w, img_h, device):
        """
        targets: [K, 6] = (batch_idx, cls, cx, cy, w, h), box normalized 0-1.
        Returns padded per-image tensors the assigner expects:
          gt_labels [B, M], gt_bboxes [B, M, 4] (x1,y1,x2,y2 pixels), gt_mask [B, M]
        """
        if targets.shape[0] == 0:
            return (torch.zeros(B, 0, dtype=torch.long, device=device),
                    torch.zeros(B, 0, 4, device=device),
                    torch.zeros(B, 0, device=device))

        targets = targets.to(device)
        batch_idx = targets[:, 0].long()
        M = int(batch_idx.bincount(minlength=B).max())  # most objects in any one image

        scale = torch.tensor([img_w, img_h, img_w, img_h], device=device, dtype=torch.float32)
        xywh = targets[:, 2:6] * scale
        xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2,
                          xywh[:, :2] + xywh[:, 2:] / 2], dim=-1)

        gt_labels = torch.zeros(B, M, dtype=torch.long, device=device)
        gt_bboxes = torch.zeros(B, M, 4, device=device)
        gt_mask = torch.zeros(B, M, device=device)
        for b in range(B):
            sel = batch_idx == b
            n = int(sel.sum())
            gt_labels[b, :n] = targets[sel, 1].long()
            gt_bboxes[b, :n] = xyxy[sel]
            gt_mask[b, :n] = 1
        return gt_labels, gt_bboxes, gt_mask

    def forward(self, preds, targets):
        """
        preds: list of (box_pred [B, 4*reg_max, H, W], cls_pred [B, nc, H, W]) per scale
        targets: [K, 6] as described in preprocess_targets
        Returns: (total_loss scaled by batch size, detached [box, cls, dfl] for logging)
        """
        device = preds[0][0].device
        B = preds[0][0].shape[0]
        feat_shapes = [box.shape[2:] for box, _ in preds]

        # Flatten all 3 scales into one long list of N predictions (8400 at 640x640)
        box_raw = torch.cat([box.flatten(2) for box, _ in preds], dim=2).permute(0, 2, 1)  # [B, N, 64]
        cls_raw = torch.cat([cls.flatten(2) for _, cls in preds], dim=2).permute(0, 2, 1)  # [B, N, nc]

        anchor_points, stride_tensor = make_anchors(feat_shapes, self.strides, device)  # pixels
        anchor_grid = anchor_points / stride_tensor                                     # grid units
        img_h = feat_shapes[0][0] * self.strides[0]
        img_w = feat_shapes[0][1] * self.strides[0]

        gt_labels, gt_bboxes, gt_mask = self.preprocess_targets(targets, B, img_w, img_h, device)

        # Decode predicted boxes (grid units), then ask the assigner who's responsible for what
        pred_dist = self.decode_dist(box_raw)
        pred_bboxes = dist2bbox(pred_dist, anchor_grid)  # [B, N, 4] grid units

        _, target_bboxes, target_scores, fg_mask = self.assigner(
            cls_raw.detach().sigmoid(),
            pred_bboxes.detach() * stride_tensor,   # assigner works in pixels
            anchor_points,
            gt_labels, gt_bboxes, gt_mask,
        )
        target_scores_sum = target_scores.sum().clamp(min=1)

        # 1. Classification loss -- over ALL predictions (positives and negatives)
        loss_cls = self.bce(cls_raw, target_scores.to(cls_raw.dtype)).sum() / target_scores_sum

        # 2 & 3. Box + DFL loss -- only over positive (assigned) predictions
        loss_box = torch.zeros((), device=device)
        loss_dfl = torch.zeros((), device=device)
        if fg_mask.any():
            target_bboxes = target_bboxes / stride_tensor           # pixels -> grid units
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)   # [P, 1]

            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], ciou=True)
            loss_box = ((1.0 - iou).unsqueeze(-1) * weight).sum() / target_scores_sum

            target_ltrb = bbox2dist(anchor_grid, target_bboxes, self.reg_max)  # [B, N, 4]
            pred_logits = box_raw[fg_mask].view(-1, 4, self.reg_max)
            loss_dfl = (self.dfl_loss(pred_logits, target_ltrb[fg_mask]) * weight).sum() / target_scores_sum

        loss_box = loss_box * self.box_gain
        loss_cls = loss_cls * self.cls_gain
        loss_dfl = loss_dfl * self.dfl_gain
        total = (loss_box + loss_cls + loss_dfl) * B
        return total, torch.stack([loss_box, loss_cls, loss_dfl]).detach()


if __name__ == "__main__":
    from models.yolo import YOLOv8Scratch

    torch.manual_seed(0)
    model = YOLOv8Scratch(num_classes=80)
    criterion = YOLOLoss(num_classes=80)

    imgs = torch.randn(2, 3, 640, 640)
    # 2 objects in image 0, 1 object in image 1: (batch_idx, cls, cx, cy, w, h)
    targets = torch.tensor([
        [0,  3, 0.50, 0.50, 0.30, 0.40],
        [0, 17, 0.20, 0.30, 0.10, 0.10],
        [1,  0, 0.70, 0.60, 0.25, 0.50],
    ])

    preds = model(imgs)
    loss, (lbox, lcls, ldfl) = criterion(preds, targets)
    loss.backward()

    print(f"total loss: {loss.item():.3f}")
    print(f"  box: {lbox:.3f}  cls: {lcls:.3f}  dfl: {ldfl:.3f}")
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"gradients reach every parameter: {grad_ok}")
