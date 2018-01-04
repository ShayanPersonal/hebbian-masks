import torch
from torch import nn

class BetterConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(BetterConv2d, self).__init__()
        filters = []
        for channel in range(out_channels):
            filters.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, 1, 
                        kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
            ))
        self.filters = nn.ModuleList(filters)

    def forward(self, x):
        features = []
        for f in self.filters:
            features.append(f(x))
        return torch.cat(features, dim=-3)