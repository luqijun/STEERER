import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.networks.sim_match.layers.dsnet import DenseScaleNet
from lib.models.networks.sim_match.layers.transition import TransitionLayer
from lib.models.networks.sim_match.layers.nested_tensor import NestedTensor
from lib.models.networks.sim_match.layers.position_encoding import PositionEmbeddingSine
from lib.models.networks.sim_match.layers.trans_decoder_wrapper import TransDecoderWrapperLayer2

# 原来的GenerateKernelLayer21

class GenerateKernelLayer21(nn.Module):
    def __init__(self, config, kernel_size=3, hidden_channels=256):
        super(GenerateKernelLayer21, self).__init__()
        self.load_model = None
        self.config = config
        self.kernel_size = kernel_size
        self.hidden_channels = hidden_channels
        self.kernel_num = 1

        self.dsnet = DenseScaleNet(hidden_channels, hidden_channels, hidden_channels//2)

        self.relu = nn.ReLU(True)

        # 初级的提取
        self.dilation_num = 3
        kernel_blocks, extra_conv_kernels = self.make_kernel_extractor(hidden_channels, dilation_num=self.dilation_num)
        self.kernel_blocks = nn.Sequential(*kernel_blocks)
        self.extra_conv_kernels = nn.Sequential(*extra_conv_kernels)

        # 高级的提取
        # kernel_blocks, extra_conv_kernels = self.make_kernel_extractor(self.dilation_num, dilation_num=self.dilation_num)
        # self.kernel_blocks_advanced = kernel_blocks
        # self.extra_conv_kernels_advaced = extra_conv_kernels

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

        # transformer
        self.decoder_wrapper = TransDecoderWrapperLayer2(self.config, hidden_channels)
        self.position_embed = PositionEmbeddingSine(hidden_channels // 2)
        self.query_embed = nn.Embedding(9, hidden_channels)

        # upsample
        self.upsample_block = self.make_head_layer(self.dilation_num, self.dilation_num)
        self.upsample_block_copy = self.make_head_layer(self.dilation_num, self.dilation_num)

        # _initialize_weights(self)

    def make_conv_layer(self, hidden_channels, k, s, p1=0, p2=1):
        return nn.Sequential(*[nn.Conv2d(hidden_channels, hidden_channels, k, s, padding=p1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.ReLU(True),
                               nn.Conv2d(hidden_channels, hidden_channels, 3, 1, padding=p2)])

    def make_head_layer(self, input_cahnnels=1, hidden_channels=3):
        return nn.Sequential(*[nn.Conv2d(input_cahnnels, hidden_channels, 3, 1, 1),
                               nn.ReLU(True),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                               nn.ReLU(True),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.Conv2d(hidden_channels, 1, 3, 1, 1)])
    def forward(self, x):

        # self.upsample_block_copy.load_state_dict(self.upsample_block.state_dict())
        # freeze_model(self.upsample_block_copy)

        x = self.dsnet(x)
        origin_map = x

        kernel_list = []
        for i, block in enumerate(self.kernel_blocks):
            x = block(x)

            if i == len(self.kernel_blocks)-1:
                memory = NestedTensor(origin_map, None)
                pos_embedding = self.position_embed(memory)
                x = self.decoder_wrapper(x, origin_map, None, self.query_embed.weight, pos_embedding)

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


    def make_kernel_extractor(self, hidden_channels, dilation_num = 3):

        # extract kernerls
        conv1 = self.make_conv_layer(hidden_channels, 3, 2)  # 11*11
        conv2 = self.make_conv_layer(hidden_channels, 3, 1)  # 9*9
        conv3 = self.make_conv_layer(hidden_channels, 3, 1)  # 7*7
        conv4 = self.make_conv_layer(hidden_channels, 3, 1)  # 5*5
        conv5 = self.make_conv_layer(hidden_channels, 3, 1)  # 3*3

        # conv6 = self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=0)  # 1*1
        kernel_blocks = [conv1, conv2, conv3, conv4, conv5]

        # 创建不同的dialation kernel
        extra_conv_group = []
        for i in range(dilation_num-1):
            cov_list = nn.Sequential(*[self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in
                                            enumerate(self.config.head.stages_channel)])
            extra_conv_group.append(cov_list)

        return kernel_blocks, extra_conv_group


# 包含densenet 、transition，仅仅在最后一层训练head参数
class GenerateKernelLayer18(nn.Module):
    def __init__(self, config, kernel_size=3, hidden_channels=256):
        super(GenerateKernelLayer18, self).__init__()
        self.load_model = None
        self.config = config
        self.kernel_size = kernel_size
        self.hidden_channels = hidden_channels
        self.kernel_num = 1

        self.dsnet = DenseScaleNet(hidden_channels, hidden_channels, hidden_channels//2)

        # extract kernerls
        conv1 = self.make_conv_layer(hidden_channels, 3, 2)  # 11*11
        conv2 = self.make_conv_layer(hidden_channels, 3, 1)  # 9*9
        conv3 = self.make_conv_layer(hidden_channels, 3, 1)  # 7*7
        conv4 = self.make_conv_layer(hidden_channels, 3, 1)  # 5*5
        conv5 = self.make_conv_layer(hidden_channels, 3, 1)  # 3*3

        # conv6 = self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=0)  # 1*1
        self.relu = nn.ReLU(True)
        self.blocks = nn.Sequential(*[conv1, conv2, conv3, conv4, conv5])

        self.conv5_2_list = nn.Sequential(*[self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in enumerate(self.config.head.stages_channel)])   # 3*3 dialation2
        self.conv5_3_list = nn.Sequential(*[self.make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in enumerate(self.config.head.stages_channel)])   # 3*3 dialation3

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

        # transformer
        self.decoder_wrapper = TransDecoderWrapperLayer2(self.config, hidden_channels)
        self.position_embed = PositionEmbeddingSine(hidden_channels // 2)
        self.query_embed = nn.Embedding(9, hidden_channels)

        # upsample
        self.upsample_block = self.make_head_layer(3, 3)
        self.upsample_block_copy = self.make_head_layer(3, 3)

        _initialize_weights(self)

    def make_conv_layer(self, hidden_channels, k, s, p1=0, p2=1):
        return nn.Sequential(*[nn.Conv2d(hidden_channels, hidden_channels, k, s, padding=p1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.ReLU(True),
                               nn.Conv2d(hidden_channels, hidden_channels, 3, 1, padding=p2)])

    def make_head_layer(self, input_cahnnels=1, hidden_channels=3):
        return nn.Sequential(*[nn.Conv2d(input_cahnnels, hidden_channels, 3, 1, 1),
                               nn.ReLU(True),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                               nn.ReLU(True),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.Conv2d(hidden_channels, 1, 3, 1, 1)])

    def forward(self, x):

        x  = self.dsnet(x)
        origin_map = x

        kernel_list = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            if i == len(self.blocks)-1:
                memory = NestedTensor(origin_map, None)
                pos_embedding = self.position_embed(memory)
                x = self.decoder_wrapper(x, origin_map, None, self.query_embed.weight, pos_embedding)

            kernel_list.append(x)
            x = self.relu(x)

        result = kernel_list[-self.kernel_num:]
        self.kernel_list = result

        return result

    def conv_with_kernels(self, x, level):
        sim_maps = []
        for i, kernel in enumerate(self.kernel_list):
            kernel = self.kernel_tran_layers_list[level][i](kernel)

            kernel_2 = self.relu(kernel)
            kernel_2 = self.conv5_2_list[level](kernel_2)

            kernel_3 = self.relu(kernel_2)
            kernel_3 = self.conv5_3_list[level](kernel_3)

            single_sim_maps = []
            for j in range(0, kernel.shape[0]):
                single_map = x[j, ...].unsqueeze(dim=0)

                single_kernel = kernel[j].unsqueeze(dim=0)
                single_kernel_2 = kernel_2[j].unsqueeze(dim=0)
                single_kernel_3 = kernel_3[j].unsqueeze(dim=0)

                sim_map = F.conv2d(single_map, single_kernel, padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2))

                r1 = 2 * (single_kernel_2.shape[-2] - 1) + 1
                r2 = 2 * (single_kernel_2.shape[-1] - 1) + 1
                sim_map_2 = F.conv2d(single_map, single_kernel_2, padding=(r1, r2), dilation=(r1, r2))

                r1 = 3 * (single_kernel_3.shape[-2] - 1) + 1
                r2 = 3 * (single_kernel_3.shape[-1] - 1) + 1
                sim_map_3 = F.conv2d(single_map, single_kernel_3, padding=(r1, r2), dilation=(r1, r2))

                map = torch.cat([sim_map, sim_map_2, sim_map_3], dim=1)
                single_sim_maps.append(map)

            sim_map = torch.cat(single_sim_maps, dim=0)
            sim_map = self.upsample_block(sim_map) if level == 3 else self.upsample_block_copy(sim_map)
            sim_maps.append(sim_map)

        output = torch.cat(sim_maps, dim=1)

        return output




import torchvision.models as models

def _initialize_weights(model):
    self_dict = model.state_dict()
    pretrained_dict = dict()
    _random_initialize_weights(model)
    if not model.load_model:
        vgg16 = models.vgg16(pretrained=True)
        print('vgg imported')
        for k, v in vgg16.named_parameters():
            # print(k)
            # print(v)
            if k in self_dict and self_dict[k].size() == v.size():
                print('layer %s is updated with pretrained vgg-16' % k)
                pretrained_dict[k] = v
        self_dict.update(pretrained_dict)
        model.load_state_dict(self_dict)
    else:
        model.load_state_dict(torch.load(model.load_model))



def _random_initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.01)
            # nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)