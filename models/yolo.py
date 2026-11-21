import torch
import torch.nn as nn
from models.backbone import CSPDarknet
from models.neck import PANet
from models.head import DetectHead


class YOLOv8Scratch(nn.Module):
    """
    Full YOLOv8-style model: backbone -> neck -> head.
    Takes an image, returns raw multi-scale predictions.
    """
    def __init__(self, num_classes=80, base_ch=64, depth_mult=0.33):
        super().__init__()
        self.backbone = CSPDarknet(base_ch=base_ch, depth_mult=depth_mult)
        self.neck = PANet(
            ch_p3=base_ch * 4, ch_p4=base_ch * 8, ch_p5=base_ch * 16,
            depth_mult=depth_mult
        )
        self.head = DetectHead(
            num_classes=num_classes,
            ch=(base_ch * 4, base_ch * 8, base_ch * 16)
        )
        self.num_classes = num_classes

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)
        predictions = self.head((p3_out, p4_out, p5_out))
        return predictions


if __name__ == "__main__":
    model = YOLOv8Scratch(num_classes=80)

    # Count parameters — useful to know model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    dummy = torch.randn(1, 3, 640, 640)
    outputs = model(dummy)

    print("\nOutput shapes:")
    for i, (box_pred, cls_pred) in enumerate(outputs):
        print(f"  Scale {i}: box {box_pred.shape}, cls {cls_pred.shape}")

    # Test on GPU too, since that's what we'll actually train on
    if torch.cuda.is_available():
        model = model.cuda()
        dummy = dummy.cuda()
        outputs = model(dummy)
        print("\n✅ GPU forward pass successful")
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
