import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.networks.sim_match.layers.dsnet import DenseScaleNet
from lib.models.networks.sim_match.layers.transition import TransitionLayer
from lib.models.networks.sim_match.layers.nested_tensor import NestedTensor
from lib.models.networks.sim_match.layers.position_encoding import PositionEmbeddingSine, PositionEmbeddingRandom
from lib.models.networks.sim_match.layers.trans_decoder_wrapper import TransDecoderWrapperLayer2
from lib.models.networks.sim_match.layers.twoway_transformer import TwoWayTransformer
from typing import Tuple
from .untils import *


# 使用相似特征库sim_feature_bank，使用rectangle window
class GenerateKernelLayer205(nn.Module):
    def __init__(self, config):
        super(GenerateKernelLayer205, self).__init__()
        hidden_channels = config.gen_kernel.get("hidden_channels", 256)
        self.load_model = None
        self.config = config
        self.hidden_channels = hidden_channels
        self.kernel_num = 1

        # two way transformer
        # self.twoway_trans = TwoWayTransformer(depth=2,
        #                                       embedding_dim=hidden_channels,
        #                                       mlp_dim=2048,
        #                                       num_heads=8,)

        window_sizes = config.gen_kernel.get("window_sizes", [(32, 16), (16, 8)])
        num_encoder_layers = config.gen_kernel.get("num_encoder_layers", 6)
        num_decoder_layers = config.gen_kernel.get("num_decoder_layers", 6)
        self.window_sizes = window_sizes

        # position encoding
        self.pe_layer = PositionEmbeddingRandom(hidden_channels // 2)

        # 提取卷积核
        self.conv_kernel_extrator = make_kernel_extractor_adaptive(hidden_channels, (5, 5))

        # sim_feature_bank
        num_sim_features = 9
        self.sim_feature_bank = nn.Parameter(torch.randn((num_sim_features, hidden_channels)), requires_grad=True)
        self.transformer = nn.Transformer(hidden_channels, 8, batch_first=True, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)

        # upsample
        self.upsample_block = make_head_layer(self.hidden_channels, self.hidden_channels // 2)
        self.upsample_block_2 = make_head_layer(1, 3)

    def forward(self, x, x_with_seg, level):

        src = x
        src_pos = self.pe_layer(src.shape[-2:]).unsqueeze(0)
        src = src + src_pos
        src = src.permute(0, 2, 3, 1)
        B, H, W, C = src.shape

        kernel = self.conv_kernel_extrator(x_with_seg)
        tgt_pos = self.pe_layer(kernel.shape[-2:]).unsqueeze(0).flatten(-2).permute(0, 2, 1)
        kernel = kernel.flatten(-2).permute(0, 2, 1) + tgt_pos

        output = src
        for i, window_size in enumerate(self.window_sizes):

            win_w = window_size[0] // 2**level
            win_h = window_size[1] // 2**level

            # Split into windows
            output, pad_hw = window_partition(output, (win_h, win_w)) # [B * num_windows, win_h, win_w, C].
            output = output.flatten(1, 2)

            num_wins = output.shape[0] // B
            tgt = self.sim_feature_bank.unsqueeze(0) + kernel
            tgt = tgt.unsqueeze(1).expand(-1, num_wins, -1, -1).flatten(0, 1)

            # Use transformer to update sim_feature_bank
            output = self.transformer(tgt, output)
            output = output.view(-1, win_h, win_w, C)
            output = window_unpartition(output, (win_h, win_w), pad_hw, (H, W))

        # get sim map by transformer ouput
        output = output.permute(0, 3, 1, 2)
        sim_map1 = self.upsample_block(output)

        # get sim map by conv
        # src = src.permute(0, 3, 1, 2)
        # kernel = (kernel + self.sim_feature_bank.unsqueeze(0)).permute(0, 2, 1).reshape(1, C, 3, 3)
        # output2 = F.conv2d(src, kernel, stride=1, padding=1)
        # sim_map2 = self.upsample_block_2(output2)

        return sim_map1, None # sim_map2

# GenerateKernelLayer205_v1：将不同层级的卷积核融合一下

class GenerateKernelLayer205_v1(GenerateKernelLayer205):
    def __init__(self, config):
        super(GenerateKernelLayer205_v1, self).__init__(config)
        # 提取卷积核
        # self.conv_kernel_extrator = make_kernel_extractor_downsample(self.hidden_channels)
        self.conv_kernel_extrator = make_kernel_extractor_adaptive(self.hidden_channels, (5, 5))
        self.sim_feature_banks = nn.Parameter(torch.randn((4, 9, self.hidden_channels)), requires_grad=True)
        self.kernels = [None, None, None, None]
        self.kernel_trans = nn.Sequential(*[
            nn.Conv2d(self.hidden_channels * (i + 2), self.hidden_channels, 1, 1) for i in range(3)
        ][::-1])

    def forward(self, x, x_with_seg, level):

        src = x
        src_pos = self.pe_layer(src.shape[-2:]).unsqueeze(0)
        src = src + src_pos
        src = src.permute(0, 2, 3, 1)
        B, H, W, C = src.shape

        kernel = self.extract_kernel(x_with_seg, level)
        tgt_pos = self.pe_layer(kernel.shape[-2:]).unsqueeze(0).flatten(-2).permute(0, 2, 1)
        kernel = kernel.flatten(-2).permute(0, 2, 1) + tgt_pos

        output = src
        for i, window_size in enumerate(self.window_sizes):
            win_w = window_size[0] // 2 ** level
            win_h = window_size[1] // 2 ** level

            # Split into windows
            output, pad_hw = window_partition(output, (win_h, win_w))  # [B * num_windows, win_h, win_w, C].
            output = output.flatten(1, 2)

            num_wins = output.shape[0] // B
            tgt = self.sim_feature_banks[level].unsqueeze(0) + kernel
            tgt = tgt.unsqueeze(1).expand(-1, num_wins, -1, -1).flatten(0, 1)

            # Use transformer to update sim_feature_bank
            output = self.transformer(tgt, output)
            output = output.view(-1, win_h, win_w, C)
            output = window_unpartition(output, (win_h, win_w), pad_hw, (H, W))

        # get sim map by transformer ouput
        output = output.permute(0, 3, 1, 2)
        sim_map1 = self.upsample_block(output)

        return sim_map1, None  # sim_map2

    def extract_kernel(self, input, level):
        if level == 3:
            kernel = self.conv_kernel_extrator(input)
        else:
            kernel = self.conv_kernel_extrator(input)
            kernel_list = [kernel]
            kernel_list.extend([k for i, k in enumerate(self.kernels) if i > level])
            kernel = torch.cat(kernel_list, dim=1)
            kernel = self.kernel_trans[level](kernel)
        self.kernels[level] = kernel
        return kernel
