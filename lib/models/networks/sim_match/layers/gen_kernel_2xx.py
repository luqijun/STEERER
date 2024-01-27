import torch
import torch.nn as nn
import torch.nn.functional as F
from .dsnet import DenseScaleNet
from .transition import TransitionLayer
from .nested_tensor import NestedTensor
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingRandom
from .trans_decoder_wrapper import TransDecoderWrapperLayer2
from .twoway_transformer import TwoWayTransformer
from typing import Tuple


class GenerateKernelLayer201(nn.Module):
    def __init__(self, config, kernel_size=3, hidden_channels=256):
        super(GenerateKernelLayer201, self).__init__()
        self.load_model = None
        self.config = config
        self.kernel_size = kernel_size
        self.hidden_channels = hidden_channels
        self.kernel_num = 1

        # dense net
        self.dsnet = DenseScaleNet(hidden_channels, hidden_channels, hidden_channels//2)

        # two way transformer
        self.twoway_trans = TwoWayTransformer(depth=2,
                                              embedding_dim=hidden_channels,
                                              mlp_dim=2048,
                                              num_heads=8,)

        # position encoding
        self.pe_layer = PositionEmbeddingRandom(hidden_channels // 2)

        self.relu = nn.ReLU(True)

        # 初级的提取
        self.dilation_num = 1
        kernel_blocks, extra_conv_kernels = make_kernel_extractor(config, hidden_channels, dilation_num=self.dilation_num)
        self.kernel_blocks = nn.Sequential(*kernel_blocks)
        self.extra_conv_kernels = nn.Sequential(*extra_conv_kernels)

        self.kernel_list = None

        # 转换kernel
        kernel_tran_layers = []
        kernel_tran_layers_list = []
        for i, c in enumerate(self.config.head.stages_channel):
            for _ in range(self.kernel_num):
                kernel_tran_layers.append(
                    TransitionLayer(hidden_channels, hidden_channels,
                                    hidden_channels=hidden_channels // 2))
            kernel_tran_layers_list.append(nn.Sequential(*kernel_tran_layers))
        self.kernel_tran_layers_list = nn.Sequential(*kernel_tran_layers_list)

        # upsample
        self.upsample_block = make_head_layer(self.dilation_num, self.dilation_num)
        self.upsample_block_copy = make_head_layer(self.dilation_num, self.dilation_num)

        # _initialize_weights(self)

    def forward(self, x):

        self.upsample_block_copy.load_state_dict(self.upsample_block.state_dict())
        freeze_model(self.upsample_block_copy)

        x = self.dsnet(x)
        src = x

        kernel_list = []
        for i, block in enumerate(self.kernel_blocks):
            x = block(x)

            # if i == len(self.kernel_blocks) - 1:
            #     memory = NestedTensor(origin_map, None)
            #     pos_embedding = self.position_embed(memory)
            #     x = self.decoder_wrapper(x, origin_map, None, self.query_embed.weight, pos_embedding)

            if i == len(self.kernel_blocks)-1:
                src_pos = self.pe_layer(src.shape[-2:]).unsqueeze(0)
                b, c, h, w = x.shape
                query = x.flatten(2).permute(0, 2, 1)
                x, src = self.twoway_trans(src, src_pos, query, None, None)
                x = x.permute(0, 2, 1).view(b, c, h, w)

            kernel_list.append(x)
            x = self.relu(x)

        result = kernel_list[-self.kernel_num:]
        self.kernel_list = result

        return result

    def conv_with_kernels(self, x, level):
        sim_maps = []
        for i, kernel in enumerate(self.kernel_list):
            kernel = self.kernel_tran_layers_list[level][i](kernel)
            act_kernel = self.relu(kernel)

            extra_kernels = []
            for extra_kernel_list in self.extra_conv_kernels:
                extra_kenel = extra_kernel_list[level](act_kernel)
                extra_kernels.append(extra_kenel)

            single_sim_maps = []
            for j in range(0, kernel.shape[0]):
                single_map = x[j, ...].unsqueeze(dim=0)

                temp_sim_maps = []
                single_kernel = kernel[j].unsqueeze(dim=0)
                sim_map = F.conv2d(single_map, single_kernel, padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2))
                temp_sim_maps.append(sim_map)

                # 不同的dilation
                for k, extra_kenel_k in enumerate(extra_kernels):
                    extra_single_kernel = extra_kenel_k[j].unsqueeze(dim=0)
                    r1 = (k+2) * (extra_single_kernel.shape[-2] - 1) + 1 # k+2代表dialation
                    r2 = (k+2) * (extra_single_kernel.shape[-1] - 1) + 1
                    extra_sim_map = F.conv2d(single_map, extra_single_kernel, padding=(r1//2, r2//2), dilation=(r1//2, r2//2))
                    temp_sim_maps.append(extra_sim_map)

                map = torch.cat(temp_sim_maps, dim=1)
                single_sim_maps.append(map)

            sim_map = torch.cat(single_sim_maps, dim=0)
            sim_map = self.upsample_block(sim_map) if level == 3 else self.upsample_block_copy(sim_map)
            sim_maps.append(sim_map)

        output = torch.cat(sim_maps, dim=1)

        return output





class GenerateKernelLayer202(GenerateKernelLayer201):
    def __init__(self, config, kernel_size=3, hidden_channels=256):
        super(GenerateKernelLayer202, self).__init__(config, kernel_size, hidden_channels)
        # 初级的提取
        self.dilation_num = 1
        kernel_blocks, extra_conv_kernels = self.make_kernel_extractor(config, hidden_channels,
                                                                  dilation_num=self.dilation_num)
        self.kernel_blocks = nn.Sequential(*kernel_blocks)
        self.extra_conv_kernels = nn.Sequential(*extra_conv_kernels)
        pass

    def make_kernel_extractor(self, config, hidden_channels, dilation_num=3):

        # channels = hidden_channels // 3
        # pass

        # extract kernerls
        avg_pool =  nn.AvgPool2d(2, 2)
        conv1 = make_conv_layer(hidden_channels, 3, 1, 1)  # 3*3
        # conv2 = make_conv_layer(hidden_channels, 3, 1)  # 9*9
        # conv3 = make_conv_layer(hidden_channels, 3, 1)  # 7*7
        # conv4 = make_conv_layer(hidden_channels, 3, 1)  # 5*5
        # conv5 = make_conv_layer(hidden_channels, 3, 1)  # 3*3

        # conv6 = self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=0)  # 1*1
        # kernel_blocks = [conv1, conv2, conv3, conv4, conv5]
        kernel_blocks = [avg_pool, conv1]

        # 创建不同的dialation kernel
        extra_conv_group = []
        for i in range(dilation_num - 1):
            cov_list = nn.Sequential(*[make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in
                                       enumerate(config.head.stages_channel)])
            extra_conv_group.append(cov_list)

        return kernel_blocks, extra_conv_group


# 201基础上，提取一个rectangle卷积核作为人头的参考特征
class GenerateKernelLayer204(nn.Module):
    def __init__(self, config, kernel_size=3, hidden_channels=256):
        super(GenerateKernelLayer204, self).__init__()
        self.load_model = None
        self.config = config
        self.kernel_size = kernel_size
        self.hidden_channels = hidden_channels

        # two way transformer
        # self.twoway_trans = TwoWayTransformer(depth=2,
        #                                       embedding_dim=hidden_channels,
        #                                       mlp_dim=2048,
        #                                       num_heads=8,)

        self.window_size = (32, 16)

        # position encoding
        self.pe_layer = PositionEmbeddingRandom(hidden_channels // 2)

        # sim_feature_bank
        num_sim_features = 9
        self.sim_feature_bank = nn.Parameter(torch.randn((num_sim_features, hidden_channels)), requires_grad=True)
        self.transformer = nn.Transformer(hidden_channels, 8, batch_first=True)

        # upsample
        self.upsample_block = make_head_layer(self.hidden_channels, self.hidden_channels // 2)

    def forward(self, x, i):

        win_w = self.window_size[0] // 2**i
        win_h = self.window_size[1] // 2**i

        src = x
        src_pos = self.pe_layer(src.shape[-2:]).unsqueeze(0)
        src = src + src_pos

        # Split into windows
        src = src.permute(0, 2, 3, 1)
        B, H, W, C = src.shape
        src, pad_hw = window_partition(src, (win_h, win_w)) # [B * num_windows, win_h, win_w, C].
        src = src.flatten(1, 2)

        # Use transformer to update sim_feature_bank
        output = self.transformer(self.sim_feature_bank.unsqueeze(0).expand(src.shape[0], -1, -1), src)

        # Reshape output to match input shape
        output = output.view(-1, win_h, win_w, C)
        output = window_unpartition(output, (win_h, win_w), pad_hw, (H, W))
        output = output.permute(0, 3, 1, 2)

        # upsample
        output = self.upsample_block(output)
        return output



# ==================================================================================分隔线==============================================================================================================

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