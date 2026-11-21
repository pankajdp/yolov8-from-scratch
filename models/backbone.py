
import torch
import torch.nn as nn


def autopad(k, p=None):
    """Auto-calculate padding for 'same' output shape."""
    if p is None:
        p = k // 2
    return p


class ConvBlock(nn.Module):
    """Standard Conv -> BatchNorm -> SiLU block, the basic unit of YOLOv8."""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck block used inside C2f."""
    def __init__(self, in_ch, out_ch, shortcut=True):
        super().__init__()
        self.cv1 = ConvBlock(in_ch, out_ch, k=3)
        self.cv2 = ConvBlock(out_ch, out_ch, k=3)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (the core YOLOv8 building block).
    Splits input into two paths, processes one through bottlenecks,
    then concatenates — this is what gives YOLOv8 its efficiency.
    """
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        self.hidden_ch = out_ch // 2
        self.cv1 = ConvBlock(in_ch, 2 * self.hidden_ch, k=1)
        self.cv2 = ConvBlock((2 + n) * self.hidden_ch, out_ch, k=1)
        self.m = nn.ModuleList(
            Bottleneck(self.hidden_ch, self.hidden_ch, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast. Expands receptive field cheaply."""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = ConvBlock(in_ch, hidden_ch, k=1)
        self.cv2 = ConvBlock(hidden_ch * 4, out_ch, k=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPDarknet(nn.Module):
    """
    YOLOv8-style backbone. Outputs 3 feature maps at different scales
    (P3, P4, P5) — these feed into the neck (PANet) next.
    """
    def __init__(self, in_ch=3, base_ch=64, depth_mult=0.33):
        super().__init__()
        d = lambda n: max(1, round(n * depth_mult))  # scale depth for model size

        self.stem = ConvBlock(in_ch, base_ch, k=3, s=2)          # /2

        self.stage1 = nn.Sequential(
            ConvBlock(base_ch, base_ch * 2, k=3, s=2),            # /4
            C2f(base_ch * 2, base_ch * 2, n=d(3)),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4, k=3, s=2),        # /8  -> P3
            C2f(base_ch * 4, base_ch * 4, n=d(6)),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 8, k=3, s=2),        # /16 -> P4
            C2f(base_ch * 8, base_ch * 8, n=d(6)),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(base_ch * 8, base_ch * 16, k=3, s=2),       # /32
            C2f(base_ch * 16, base_ch * 16, n=d(3)),
            SPPF(base_ch * 16, base_ch * 16),                     # -> P5
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5  # multi-scale features for the neck


if __name__ == "__main__":
    model = CSPDarknet()
    dummy = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = model(dummy)
    print("P3 shape:", p3.shape)  # expect [1, 256, 80, 80]
    print("P4 shape:", p4.shape)  # expect [1, 512, 40, 40]
    print("P5 shape:", p5.shape)  # expect [1, 1024, 20, 20]
