import torch
import torch.nn as nn
from models.backbone import ConvBlock, C2f


class PANet(nn.Module):
    """
    Path Aggregation Network neck, as used in YOLOv8.
    Takes backbone features (P3, P4, P5) and fuses them:
      1. Top-down pathway: upsample + concat (P5 -> P4 -> P3)
      2. Bottom-up pathway: downsample + concat (P3 -> P4 -> P5)
    Outputs 3 fused feature maps ready for the detection head.
    """
    def __init__(self, ch_p3=256, ch_p4=512, ch_p5=1024, depth_mult=0.33):
        super().__init__()
        d = lambda n: max(1, round(n * depth_mult))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Top-down pathway ---
        # P5 -> upsample -> concat with P4 -> C2f
        self.c2f_p4_td = C2f(ch_p5 + ch_p4, ch_p4, n=d(3), shortcut=False)

        # fused P4 -> upsample -> concat with P3 -> C2f
        self.c2f_p3_out = C2f(ch_p4 + ch_p3, ch_p3, n=d(3), shortcut=False)

        # --- Bottom-up pathway ---
        # downsample P3_out, concat with fused P4 -> C2f
        self.down_p3 = ConvBlock(ch_p3, ch_p3, k=3, s=2)
        self.c2f_p4_out = C2f(ch_p3 + ch_p4, ch_p4, n=d(3), shortcut=False)

        # downsample fused P4_out, concat with P5 -> C2f
        self.down_p4 = ConvBlock(ch_p4, ch_p4, k=3, s=2)
        self.c2f_p5_out = C2f(ch_p4 + ch_p5, ch_p5, n=d(3), shortcut=False)

    def forward(self, p3, p4, p5):
        # Top-down
        p5_up = self.upsample(p5)
        p4_td = self.c2f_p4_td(torch.cat([p5_up, p4], dim=1))

        p4_up = self.upsample(p4_td)
        p3_out = self.c2f_p3_out(torch.cat([p4_up, p3], dim=1))

        # Bottom-up
        p3_down = self.down_p3(p3_out)
        p4_out = self.c2f_p4_out(torch.cat([p3_down, p4_td], dim=1))

        p4_down = self.down_p4(p4_out)
        p5_out = self.c2f_p5_out(torch.cat([p4_down, p5], dim=1))

        return p3_out, p4_out, p5_out  # same shapes as input, but fused


if __name__ == "__main__":
    import torch
    from models.backbone import CSPDarknet

    backbone = CSPDarknet()
    neck = PANet()

    dummy = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = backbone(dummy)
    p3_out, p4_out, p5_out = neck(p3, p4, p5)

    print("P3 out:", p3_out.shape)  # expect [1, 256, 80, 80]
    print("P4 out:", p4_out.shape)  # expect [1, 512, 40, 40]
    print("P5 out:", p5_out.shape)  # expect [1, 1024, 20, 20]
