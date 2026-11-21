import math
import torch
import torch.nn as nn
from models.backbone import ConvBlock


class DetectHead(nn.Module):
    """
    YOLOv8-style anchor-free decoupled detection head.
    For each of the 3 feature maps (P3, P4, P5), predicts:
      - box regression (4 values per location, using distribution focal loss style)
      - class scores (num_classes values per location)
    Applied identically (shared design, separate weights) to each scale.
    """
    def __init__(self, num_classes=80, ch=(256, 512, 1024), reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max  # DFL bins for box regression
        self.num_outputs_box = 4 * reg_max  # box branch output size
        self.strides = [8, 16, 32]  # P3, P4, P5 downsampling factors

        # Separate branches per scale, each with its own small conv stack
        self.box_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()

        for in_ch in ch:
            hidden = max(in_ch // 4, 64)
            self.box_branches.append(nn.Sequential(
                ConvBlock(in_ch, hidden, k=3),
                ConvBlock(hidden, hidden, k=3),
                nn.Conv2d(hidden, self.num_outputs_box, 1)
            ))
            self.cls_branches.append(nn.Sequential(
                ConvBlock(in_ch, hidden, k=3),
                ConvBlock(hidden, hidden, k=3),
                nn.Conv2d(hidden, num_classes, 1)
            ))

        self.init_bias()  # NEW

    def init_bias(self, img_size=640):  # NEW
        """
        Start the class outputs at a tiny probability instead of 50%.
        Assumes ~5 objects per image spread over all grid cells and classes,
        so the starting loss reflects that most locations are background.
        """
        for box_branch, cls_branch, s in zip(self.box_branches, self.cls_branches, self.strides):
            box_branch[-1].bias.data.fill_(1.0)
            cls_branch[-1].bias.data.fill_(math.log(5 / self.num_classes / (img_size / s) ** 2))

    def forward(self, feats):
        """
        feats: tuple of (p3_out, p4_out, p5_out) from the neck
        Returns: list of (box_pred, cls_pred) per scale, raw (not yet decoded)
        """
        outputs = []
        for i, feat in enumerate(feats):
            box_pred = self.box_branches[i](feat)   # [B, 4*reg_max, H, W]
            cls_pred = self.cls_branches[i](feat)    # [B, num_classes, H, W]
            outputs.append((box_pred, cls_pred))
        return outputs


if __name__ == "__main__":
    from models.backbone import CSPDarknet
    from models.neck import PANet

    backbone = CSPDarknet()
    neck = PANet()
    head = DetectHead(num_classes=80)

    dummy = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = backbone(dummy)
    p3_out, p4_out, p5_out = neck(p3, p4, p5)
    outputs = head((p3_out, p4_out, p5_out))

    for i, (box_pred, cls_pred) in enumerate(outputs):
        stride = head.strides[i]
        print(f"Scale stride={stride}: box_pred {box_pred.shape}, cls_pred {cls_pred.shape}")
    print("starting class probability (P3):", torch.sigmoid(outputs[0][1]).mean().item())  # NEW
