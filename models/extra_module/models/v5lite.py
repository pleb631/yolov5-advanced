import torch
import torch.nn as nn

from models.common import Conv
from utils.torch_utils import fuse_conv_and_bn

__all__ = ["Dense", "conv_bn_relu_maxpool", "Shuffle_Block", "ADD"]

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


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
