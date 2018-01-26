import torch
from torch import nn
import torch.nn.functional as F

class WeightHacker(nn.Module):
    # Base class for wrapper classes that manipulate weights of the module
    def __init__(self, module, weights):
        super(WeightHacker, self).__init__()
        self.module = module
        self.weights = weights
        self._setup()

    def _setup(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = self.method(raw_w)
            setattr(self.module, name_w, w)

    def forward(self, x):
        self._setweights()
        return self.module.forward(x)

    def method(self, raw_w):
        # Override me
        raise NotImplementedError


class PositiveConstraint(WeightHacker):
    # Example usage of WeightHacker. This module forces all the weights to be positive.
    def method(self, raw_w):
        return F.relu(raw_w)


class HebbMask(WeightHacker):
    def __init__(self, module, weights, lr=0.1):
        super(HebbMask, self).__init__(module, weights)
        self.lr = lr
        self.register_buffer('permanence', torch.autograd.Variable(torch.ones_like(module.weight_raw.data), requires_grad=False))
    
    def _update(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        update = torch.addbmm(torch.autograd.Variable(torch.Tensor([0]).cuda()), y, x)
        self.permanence.data = (self.permanence.data + self.lr * update.data * self.module.weight_raw.data).clamp_(-1, 1)

    def method(self, raw_w):
        return raw_w * (self.permanence > 0).type_as(self.permanence)

    def forward(self, x):
        y = super(HebbMask, self).forward(x)
        if self.train:
            self._update(x.detach(), y.detach())
        return y
