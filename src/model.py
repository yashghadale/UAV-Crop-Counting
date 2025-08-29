# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Basic Blocks
# -----------------------
class CBR(nn.Module):
    """Conv -> BatchNorm -> ReLU"""
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class CBS(nn.Module):
    """Conv -> BatchNorm -> SiLU (Swish)"""
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

# -----------------------
# CBAM Attention
# -----------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.mlp(self.avgpool(x))
        m = self.mlp(self.maxpool(x))
        return self.sig(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        return self.sig(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# -----------------------
# Conv-CBAM(-Pool) blocks
# -----------------------
class ConvCBAM(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            CBR(c_in, c_out),
            CBAM(c_out)
        )
    def forward(self, x): return self.block(x)

class ConvCBAMPool(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            CBR(c_in, c_out),
            CBAM(c_out),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.block(x)

# -----------------------
# Simplified YOLOv5s-like backbone (Branch B)
# -----------------------
class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True, expansion=0.5):
        super().__init__()
        c_mid = int(c * expansion)
        self.cv1 = CBS(c, c_mid, k=1, s=1, p=0)
        self.cv2 = CBS(c_mid, c, k=3, s=1, p=1)
        self.add = shortcut
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class YOLOv5sLite(nn.Module):
    def __init__(self, in_ch=3, c_list=(32,64,128,256)):
        super().__init__()
        c1, c2, c3, c4 = c_list
        self.stem = CBS(in_ch, c1, k=3, s=2, p=1)   # /2
        self.stage1 = nn.Sequential(Bottleneck(c1), CBS(c1, c2, k=3, s=2, p=1)) # /4
        self.stage2 = nn.Sequential(Bottleneck(c2), CBS(c2, c3, k=3, s=2, p=1)) # /8
        self.stage3 = nn.Sequential(Bottleneck(c3), CBS(c3, c4, k=3, s=2, p=1)) # /16
    def forward(self, x):
        x = self.stem(x)   # /2
        x = self.stage1(x) # /4
        x = self.stage2(x) # /8
        x = self.stage3(x) # /16
        return x

# -----------------------
# Low-IFM (Low-stage Information Fusion Module)
# -----------------------
class LowIFM(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fuse = nn.Sequential(
            CBS(c_in, c_out, k=1, s=1, p=0),
            CBAM(c_out)
        )
    def forward(self, x): return self.fuse(x)

# -----------------------
# Counter Head & Normalizer
# -----------------------
class CounterHead(nn.Module):
    """
    Produces a redundant count map from fused features.
    Pipeline: AvgPool -> CBS -> CBAM -> CBR -> conv(1)
    """
    def __init__(self, c_in, c_mid=128):
        super().__init__()
        self.avg = nn.AvgPool2d(2,2)
        self.cbs = CBS(c_in, c_mid, k=3, s=1, p=1)
        self.cbam = CBAM(c_mid)
        self.cbr = CBR(c_mid, c_mid, k=3, s=1, p=1)
        self.out = nn.Conv2d(c_mid, 1, kernel_size=1, bias=True)  # redundant count map
    def forward(self, x):
        x = self.avg(x)
        x = self.cbs(x)
        x = self.cbam(x)
        x = self.cbr(x)
        return self.out(x)

class Normalizer(nn.Module):
    """
    Refine/re-normalize redundant map -> final density/count map.
    Here implemented as a light 1x1 conv (identity-like) but easy to extend.
    """
    def __init__(self):
        super().__init__()
        self.refine = nn.Conv2d(1, 1, kernel_size=1, bias=True)
    def forward(self, x): return self.refine(x)

# -----------------------
# TasselNetV2++ Model
# -----------------------
class TasselNetV2PP(nn.Module):
    """
    Dual-branch encoder -> fusion (GD) -> Counter -> Normalizer
    Returns: final density / count map
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # Branch A (shallow + attention)
        self.a1 = ConvCBAMPool(in_ch, base)       # /2
        self.a2 = ConvCBAMPool(base, base*2)      # /4
        self.a3 = ConvCBAMPool(base*2, base*4)    # /8
        self.a4 = ConvCBAM(base*4, base*4)
        self.a5 = ConvCBAM(base*4, base*4)        # final A features ~ /8

        # Branch B (deeper YOLO-lite)
        self.b = YOLOv5sLite(in_ch, c_list=(base, base*2, base*4, base*8)) # final B features ~ /16

        # Low-IFM: fuse A (~1/8) and upsampled B (~1/8) -> produce combined features
        self.low_ifm = LowIFM(base*4 + base*8, base*8)

        # Counter and normalizer
        self.counter = CounterHead(base*8, c_mid=base*8)
        self.norm = Normalizer()

    def forward(self, x):
        # Branch A features (~1/8)
        xa = self.a1(x)
        xa = self.a2(xa)
        xa = self.a3(xa)
        xa = self.a4(xa)
        xa = self.a5(xa)

        # Branch B deep features (~1/16)
        xb = self.b(x)

        # Upsample B to A's spatial size and concat (GD + fusion)
        xb_up = F.interpolate(xb, size=xa.shape[-2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([xa, xb_up], dim=1)

        xf = self.low_ifm(x_cat)   # fused features

        # Counter -> redundant map
        red = self.counter(xf)     # smaller stride map
        # Upsample redundant map to reduce stride misalignment (optional tuning)
        red_up = F.interpolate(red, scale_factor=2, mode='bilinear', align_corners=False)

        final = self.norm(red_up)  # final density/count map
        return final

# -----------------------
# Helper
# -----------------------
def create_model(in_ch=3, base=32, device='cpu'):
    model = TasselNetV2PP(in_ch=in_ch, base=base)
    return model.to(device)

# -----------------------
# Quick sanity check (prints shapes & param counts)
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = create_model(in_ch=3, base=32, device=device)
    print("Model params (M):", sum(p.numel() for p in m.parameters())/1e6)
    # test forward with a synthetic batch (B=2) and large input (3 x 1024 x 2048)
    x = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        y = m(x)
    print("Input:", x.shape, "-> Output:", y.shape)
