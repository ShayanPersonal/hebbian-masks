import torch
from torch import nn

class SeparatedConv2d(nn.Module):
    def __init__(self, in_channels, k, out_channels, kernel_size, stride, padding, dilation, bias):
        super(SeparatedConv2d, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels*k, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels*k, out_channels,
            kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        return x