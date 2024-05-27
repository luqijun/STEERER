
import torch
import torch.nn as nn

class Segmentation_Head(nn.Module):

    def __init__(self, in_channels=256, out_channels=1):
        super(Segmentation_Head, self).__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//2, in_channels//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//4, out_channels, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.decoder(input)
        return output