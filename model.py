import torch
from torch import nn
import torch.nn.functional as F

from better_conv2d import BetterConv2d
from weight_hacks import HebbMask, PositiveConstraint

#def double_relu(x, dim=1):
#    return torch.cat((F.relu(x), F.relu(-x)), dim)

class RandomNet(nn.Module):
    def __init__(self):
        super(RandomNet, self).__init__()

        #self.bn1 = nn.BatchNorm2d(64)
        #self.bn2 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, 128, bias=True)
        self.hebb_classifier = HebbMask(nn.Linear(128, 10, bias=True), ['weight'])

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.hebb_classifier(x)
        return x
