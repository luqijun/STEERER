import torch.nn as nn

class SegmentationLayer(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, hidden_channels=128):
        super(SegmentationLayer, self).__init__()
        self.regression = nn.Sequential(*[nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                                          nn.ReLU(True),
                                          nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                                          nn.ReLU(True),
                                          nn.Conv2d(hidden_channels, out_channels, 3, padding=1)])
        # self.load_model = None
        # _initialize_weights(self)

    def forward(self, x):
        output = self.regression(x)
        return output