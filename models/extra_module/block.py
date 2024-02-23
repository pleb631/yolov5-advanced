from models.common import Conv, Bottleneck, autopad

# v5的Bottleneck和v8的不太一样,要注意k参数，v5是(1,3).v8是(3,3)
from models.extra_module.conv import RepConv, GSConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yolo import fuse_conv_and_bn

__all__ = [
    "RepLightConv",
    "Rep_HGBlock",
    "HGStem",
    "DFL",
    "C1",
    "C2",
    "C2f",
    "RepC3",
    "ContextGuidedBlock_Down",
    "SPDConv",
    "DySample",
    "CARAFE",
    "HWD",
    "RepBlock",
    "BiFusion",
    "CSPStage",
    "RCSOSA",
    "RepNCSPELAN4",
    "DPBlock",
    "ES_Bottleneck",
    "Dense",
    "conv_bn_relu_maxpool",
    "Shuffle_Block",
    "ADD",
    "conv_bn_act",
]


class RepLightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = RepConv(c2, c2, k, g=c2, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class Rep_HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=True):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = RepLightConv if lightconv else Conv
        self.m = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(
            *(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


### ContextGuidedBlock_Down
class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        nOut = 2 * nIn
        self.conv1x1 = Conv(nIn, nOut, 3, s=2)  #  size/2, channel: nIn--->nOut

        self.F_loc = nn.Conv2d(nOut, nOut, 3, padding=1, groups=nOut)
        self.F_sur = nn.Conv2d(
            nOut,
            nOut,
            3,
            padding=autopad(3, None, dilation_rate),
            dilation=dilation_rate,
            groups=nOut,
        )

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = Conv.default_act
        self.reduce = Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


### SPDConv


class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )
        x = self.conv(x)
        return x


### DySample


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h]))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(
            1, 2, 1, 1, 1
        )
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_) * self.scope(x_).sigmoid(), self.scale
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = (
                F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
            )
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


### CARAFE


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")
        self.unfold = nn.Unfold(
            kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale
        )

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum("bkhw,bckhw->bchw", [W, X])  # b * c * h_ * w_
        return X


### HWD


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward

        self.wt = DWTForward(J=1, mode="zero", wave="haar")
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x


### RepBlock,BiFusion
## yolov6


class BottleRep(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block=RepConv, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class RepBlock(nn.Module):
    """
    RepBlock is a stage block with rep-style basic block
    """

    def __init__(
        self, in_channels, out_channels, n=1, block=RepConv, basic_block=RepConv
    ):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = (
            nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1)))
            if n > 1
            else None
        )
        if block == BottleRep:
            self.conv1 = BottleRep(
                in_channels, out_channels, basic_block=basic_block, weight=True
            )
            n = n // 2
            self.block = (
                nn.Sequential(
                    *(
                        BottleRep(
                            out_channels,
                            out_channels,
                            basic_block=basic_block,
                            weight=True,
                        )
                        for _ in range(n - 1)
                    )
                )
                if n > 1
                else None
            )

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BiFusion(nn.Module):
    """BiFusion Block in PAN"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channels[1], out_channels, 1, 1)
        self.cv2 = Conv(in_channels[2], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)

        self.upsample = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.downsample = Conv(out_channels, out_channels, 3, 2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


### CSPStage


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=size, stride=1, padding=size // 2, ceil_mode=False
            )
            self.add_module("pool{}".format(i), pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        n,
        block_fn="BasicBlock_3x3_Reverse",
        ch_hidden_ratio=1.0,
        act="silu",
        spp=False,
    ):
        super(CSPStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == "BasicBlock_3x3_Reverse":
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(
                        next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True
                    ),
                )
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y


### RCSOSA


class SR(nn.Module):
    # Shuffle RepVGG
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = int(c1 // 2)
        c2_ = int(c2 // 2)
        self.repconv = RepConv(c1_, c2_, bn=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.repconv(x2)), dim=1)
        out = self.channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class RCSOSA(nn.Module):
    # VoVNet with Res Shuffle RepVGG
    def __init__(self, c1, c2, n=1, se=False, g=1, e=0.5):
        super().__init__()
        n_ = n // 2
        from utils.general import make_divisible

        c_ = make_divisible(int(c1 * e), 8)
        self.conv1 = RepConv(c1, c_, bn=True)
        self.conv3 = RepConv(int(c_ * 3), c2, bn=True)
        self.sr1 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])

        self.se = None
        if se:
            from models.extra_module.attention import SEAttention

            self.se = SEAttention(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x = torch.cat((x1, x2, x3), 1)
        return self.conv3(x) if self.se is None else self.se(self.conv3(x))


### yolov9


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(
        self, c1, c2, n, c3, c4
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


### picodet


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


from models.common import GhostConv


class _GhostConv(GhostConv):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=3, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__(c1, c2, k, s, g, act)
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, k, s, None, c_, act=act)


class ES_SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out


class ES_Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ES_Bottleneck, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        # assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.Hardswish(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            ES_SEModule(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

        self.branch3 = nn.Sequential(
            _GhostConv(branch_features, branch_features, 3, 1),
            ES_SEModule(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

        self.branch4 = nn.Sequential(
            self.depthwise_conv(oup, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.Hardswish(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size=3, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    @staticmethod
    def conv1x1(i, o, kernel_size=1, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x3 = torch.cat((x1, self.branch3(x2)), dim=1)
            out = channel_shuffle(x3, 2)
        elif self.stride == 2:
            x1 = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            out = self.branch4(x1)

        return out


class conv_bn_act(nn.Module):
    def __init__(
        self, c1, c2, k=3, s=1, p=None, g=1, act="hard_swish"
    ):  # ch_in, ch_out
        super(conv_bn_act, self).__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act=False)

        if act == "hard_swish":
            self.act = nn.Hardswish()
        elif act == "relu6":
            self.act = nn.ReLU6()

    def forward(self, x):
        return self.act(self.conv(x))


class DPBlock(nn.Module):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, k, s, use_se=False):
        super(DPBlock, self).__init__()
        self.use_se = use_se
        self.dw_conv = conv_bn_act(in_channels, in_channels, k=k, s=s, g=in_channels)
        self.pw_conv = conv_bn_act(in_channels, out_channels, k=1, s=1)

        if use_se:
            self.se = LC_SEModule(in_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


### lite
class LC_SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # self.act = nn.SiLU()
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.act(x)
        out = identity * x
        return out

class Dense(nn.Module):
    def __init__(self, num_channels, num_filters, filter_size, dropout_prob):
        super().__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout_prob)
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # self.fc = nn.Linear(num_filters, num_filters)

    def forward(self, x):
        # x = self.avg_pool(x)
        # b, _, w, h = x.shape
        x = self.dense_conv(x)
        # b, _, w, h = x.shape
        x = self.hardswish(x)
        x = self.dropout(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        # x = x.reshape(b, self.c2, w, h)
        return x


class conv_bn_relu_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(conv_bn_relu_maxpool, self).__init__()
        self.conv = nn.Sequential(
            Conv(c1, c2, 3, 2, 1, act=False),
            nn.ReLU6(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

    def fuse(self):
        if hasattr(self, "branch1"):
            re_branch1 = nn.Sequential(
                nn.Conv2d(
                    self.branch1[0].in_channels,
                    self.branch1[0].out_channels,
                    kernel_size=self.branch1[0].kernel_size,
                    stride=self.branch1[0].stride,
                    padding=self.branch1[0].padding,
                    groups=self.branch1[0].groups,
                ),
                nn.Conv2d(
                    self.branch1[2].in_channels,
                    self.branch1[2].out_channels,
                    kernel_size=self.branch1[2].kernel_size,
                    stride=self.branch1[2].stride,
                    padding=self.branch1[2].padding,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )
            re_branch1[0] = fuse_conv_and_bn(self.branch1[0], self.branch1[1])
            re_branch1[1] = fuse_conv_and_bn(self.branch1[2], self.branch1[3])
            # pdb.set_trace()
            # print(m.branch1[0])
            self.branch1 = re_branch1
        if hasattr(self, "branch2"):
            re_branch2 = nn.Sequential(
                nn.Conv2d(
                    self.branch2[0].in_channels,
                    self.branch2[0].out_channels,
                    kernel_size=self.branch2[0].kernel_size,
                    stride=self.branch2[0].stride,
                    padding=self.branch2[0].padding,
                    groups=self.branch2[0].groups,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.branch2[3].in_channels,
                    self.branch2[3].out_channels,
                    kernel_size=self.branch2[3].kernel_size,
                    stride=self.branch2[3].stride,
                    padding=self.branch2[3].padding,
                    bias=False,
                ),
                nn.Conv2d(
                    self.branch2[5].in_channels,
                    self.branch2[5].out_channels,
                    kernel_size=self.branch2[5].kernel_size,
                    stride=self.branch2[5].stride,
                    padding=self.branch2[5].padding,
                    groups=self.branch2[5].groups,
                ),
                nn.ReLU(inplace=True),
            )
            re_branch2[0] = fuse_conv_and_bn(self.branch2[0], self.branch2[1])
            re_branch2[2] = fuse_conv_and_bn(self.branch2[3], self.branch2[4])
            re_branch2[3] = fuse_conv_and_bn(self.branch2[5], self.branch2[6])
            self.branch2 = re_branch2


class ADD(nn.Module):
    # Stortcut a list of tensors along dimension
    def __init__(self, alpha=0.5):
        super(ADD, self).__init__()
        self.a = alpha

    def forward(self, x):
        x1, x2 = x[0], x[1]
        return torch.add(x1, x2, alpha=self.a)
