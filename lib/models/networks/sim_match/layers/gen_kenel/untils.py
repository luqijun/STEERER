import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.networks.sim_match.layers.position_encoding import PositionEmbeddingSine, PositionEmbeddingRandom
from typing import Tuple

def conv_with_kernel(input, kernel):
    sim_maps =[]
    for i in range(input.shape[0]):
        input_i = input[i:i+1]
        k_i = kernel[i].unsqueeze(dim=0)
        sim_map = F.conv2d(input_i, k_i, padding=(k_i.shape[-2] // 2, k_i.shape[-1] // 2))
        sim_maps.append(sim_map)
    sim_map = torch.cat(sim_maps, dim=0)
    return sim_map


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    win_h, win_w = window_size

    pad_h = (win_h - H % win_h) % win_h
    pad_w = (win_w - W % win_w) % win_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // win_h, win_h, Wp // win_w, win_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h, win_w, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: Tuple[int, int], pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    win_h, win_w = window_size
    B = windows.shape[0] // (Hp * Wp // win_h // win_w)
    x = windows.view(B, Hp // win_h, Wp // win_w, win_h, win_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def make_kernel_extractor_adaptive(hidden_channels, feature_size=(7, 7)):

    return nn.Sequential(*[nn.AdaptiveAvgPool2d(output_size=feature_size),
                           nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1),
                           nn.BatchNorm2d(hidden_channels // 2),
                           nn.ReLU(True),
                           nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 1, padding=1)])


def make_kernel_extractor_rect2x3(config, hidden_channels, dilation_num=3):

    kernel_blocks = [
                     nn.Conv2d(hidden_channels, hidden_channels, 3, 1, padding=1),
                     nn.BatchNorm2d(hidden_channels),
                     nn.ReLU(True),
                     nn.Conv2d(hidden_channels, hidden_channels, 3, 1, padding=p2)]

    # extract kernerls
    conv1 = make_conv_layer(hidden_channels, (2, 3), 2)  # 11*11
    conv2 = make_conv_layer(hidden_channels, 3, 1)  # 9*9
    conv3 = make_conv_layer(hidden_channels, 3, 1)  # 7*7
    conv4 = make_conv_layer(hidden_channels, 3, 1)  # 5*5
    conv5 = make_conv_layer(hidden_channels, 3, 1)  # 3*3

    # conv6 = self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=0)  # 1*1
    kernel_blocks = [conv1, conv2, conv3, conv4, conv5]

    # 创建不同的dialation kernel
    extra_conv_group = []
    for i in range(dilation_num - 1):
        cov_list = nn.Sequential(*[make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in
                                   enumerate(config.head.stages_channel)])
        extra_conv_group.append(cov_list)

    return kernel_blocks, extra_conv_group

def make_kernel_extractor(config, hidden_channels, dilation_num=3):
    # extract kernerls
    conv1 = make_conv_layer(hidden_channels, 3, 2)  # 11*11
    conv2 = make_conv_layer(hidden_channels, 3, 1)  # 9*9
    conv3 = make_conv_layer(hidden_channels, 3, 1)  # 7*7
    conv4 = make_conv_layer(hidden_channels, 3, 1)  # 5*5
    conv5 = make_conv_layer(hidden_channels, 3, 1)  # 3*3

    # conv6 = self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=0)  # 1*1
    kernel_blocks = [conv1, conv2, conv3, conv4, conv5]

    # 创建不同的dialation kernel
    extra_conv_group = []
    for i in range(dilation_num - 1):
        cov_list = nn.Sequential(*[make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in
                                   enumerate(config.head.stages_channel)])
        extra_conv_group.append(cov_list)

    return kernel_blocks, extra_conv_group


def make_conv_layer(hidden_channels, k, s, p1=0, p2=1):
    return nn.Sequential(*[nn.Conv2d(hidden_channels, hidden_channels, k, s, padding=p1),
                           nn.BatchNorm2d(hidden_channels),
                           nn.ReLU(True),
                           nn.Conv2d(hidden_channels, hidden_channels, 3, 1, padding=p2)])


def make_head_layer(input_cahnnels=1, hidden_channels=3):
    return nn.Sequential(*[nn.Conv2d(input_cahnnels, hidden_channels, 3, 1, 1),
                           nn.ReLU(True),
                           nn.Upsample(scale_factor=2, mode='bilinear'),
                           nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                           nn.ReLU(True),
                           nn.Upsample(scale_factor=2, mode='bilinear'),
                           nn.Conv2d(hidden_channels, 1, 3, 1, 1)])


def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False