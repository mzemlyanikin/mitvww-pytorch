import torch
import torch.nn as nn


class PlainConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(PlainConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, relu=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if relu:
            self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if hasattr(self, 'relu6'):
            x = self.relu6(x)
        return x


class InvBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, mid_stride, residual):
        super(InvBottleneck, self).__init__()
        self.inverted_bottleneck = ConvBN(in_channels, mid_channels, 1)
        self.depth_conv = ConvBN(mid_channels, mid_channels, kernel_size, mid_stride, groups=mid_channels, padding=(kernel_size-1)//2)
        self.point_linear = ConvBN(mid_channels, out_channels, 1, relu=False)
        self.residual = residual

    def forward(self, inp):
        x = self.inverted_bottleneck(inp)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        if self.residual:
            x += inp
        return x


#in_channels, mid_channels, out_channels, k_size, stride, residual_connection
CFG = [[8, 12, 8, 3, 2, False],
       [8, 20, 16, 5, 1, False],
       [16, 36, 16, 3, 1, True],
       [16, 32, 20, 7, 2, False],
       [20, 40, 20, 3, 1, True],
       [20, 40, 20, 5, 1, True],
       [20, 40, 20, 5, 1, True],
       [20, 80, 40, 7, 2, False],
       [40, 80, 40, 5, 1, True],
       [40, 80, 40, 5, 1, True],
       [40, 80, 40, 5, 1, True],
       [40, 160, 48, 5, 1, False],
       [48, 96, 48, 5, 1, True],
       [48, 96, 48, 5, 1, True],
       [48, 100, 48, 5, 1, True],
       [48, 200, 96, 7, 2, False],
       [96, 152, 96, 5, 1, True]
]


class MITVWW(nn.Module):
    def __init__(self):
        super(MITVWW, self).__init__()
        self.first_conv = ConvBN(3, 8, 3, stride=2, padding=1)

        blocks = [InvBottleneck(*layer_parameters) for layer_parameters in CFG]
        self.blocks = nn.Sequential(*blocks)

        self.feature_mix_layer = PlainConv(96, 640, 1)
        self.pool = nn.AvgPool2d(7, stride=7)
        self.linear = nn.Linear(640, 2)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.feature_mix_layer(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return torch.softmax(self.linear(x), 1)
