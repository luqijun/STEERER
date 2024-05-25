import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
import numpy as np

class TransitionLayer(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, hidden_channels=128):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(*[nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                                          nn.BatchNorm2d(hidden_channels),
                                          nn.ReLU(True),
                                          nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                                          nn.BatchNorm2d(hidden_channels),
                                          nn.ReLU(True),
                                          nn.Conv2d(hidden_channels, out_channels, 3, padding=1)])
        self.load_model = None

    def forward(self, x):
        output = self.transition(x)
        return output