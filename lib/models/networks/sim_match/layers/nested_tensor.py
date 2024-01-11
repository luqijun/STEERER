import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List
from torch import Tensor

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if self.mask is None:
            bs, c, h, w = tensors.shape
            self.mask = torch.zeros((bs, h , w), dtype=torch.int, device=tensors.device)

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)