import torch
import sys
import os
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import torch_function as PMX

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return PMX.rms_norm(x, self.weight, -1, self.eps)


class SkipRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, skip):
        return PMX.skip_rms_norm(x, self.weight, skip, -1, self.eps)

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_term: bool = True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor):
        return PMX.linear(X, self.weight, self.bias, self.in_features, self.out_features)


class GroupNorm(torch.nn.Module):
    def __init__(self, num_groups: int, num_channels: int = 1, eps: float = 1e-5,):
        super().__init__()
        self.num_groups =  num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return PMX.group_norm(x, self.weight, self.bias, self.num_groups, self.eps)
