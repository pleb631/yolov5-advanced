from timm.layers import trunc_normal_
import torch
import torch.nn as nn

from models.common import *
from models.extra_module.models.yolov8 import C2f

__all__ = ["C3_Parc", "C2f_Parc"]

class ParC_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size, use_pe=True, groups=1):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)
        if use_pe:
            if self.type=='H':
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type=='W':
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        if self.use_pe:
            x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)

        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == 'H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = self.gcc_conv(x_cat)

        return x

class ParConv(nn.Module):
    def __init__(self, dim, fmapsize, use_pe=True, groups=1) -> None:
        super().__init__()
        
        self.parc_H = ParC_operator(dim // 2, 'H', fmapsize[0], use_pe, groups = groups)
        self.parc_W = ParC_operator(dim // 2, 'W', fmapsize[1], use_pe, groups = groups)
        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act
    
    def forward(self, x):
        out_H, out_W = torch.chunk(x, 2, dim=1)
        out_H, out_W = self.parc_H(out_H), self.parc_W(out_W)
        out = torch.cat((out_H, out_W), dim=1)
        out = self.bn(out)
        out = self.act(out)
        return out

class Bottleneck_ParC(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            self.cv2 = ParConv(c2, fmapsize, groups=g)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_Parc(C3):
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ParC(c_, c_, fmapsize, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_Parc(C2f):
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ParC(self.c, self.c, fmapsize, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))