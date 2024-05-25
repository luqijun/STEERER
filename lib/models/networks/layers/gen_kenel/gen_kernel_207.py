from lib.models.networks.layers.position_encoding import PositionEmbeddingRandom
from .untils import *


class GenerateKernelLayer207(nn.Module):
    def __init__(self, config):
        super(GenerateKernelLayer207, self).__init__()
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

        # kernel转换
        kernel_tran_layers = []
        for i in range(3):
            kernel_tran_layers.append(nn.Conv2d(self.hidden_channels * 2,
                                                self.hidden_channels,
                                                1, 1))
        self.kernel_tran_layers = nn.Sequential(*kernel_tran_layers)

        # sim_feature_bank
        num_sim_features = 9
        self.sim_feature_bank = nn.Parameter(torch.randn((num_sim_features, hidden_channels)), requires_grad=True)
        self.transformer = nn.Transformer(hidden_channels, 8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)

        # upsample
        self.upsample_block = make_head_layer(self.hidden_channels, self.hidden_channels // 2)
        # self.upsample_block_2 = make_head_layer(1, 3)

    def forward(self, x, x_with_seg, level):

        kernel = self.conv_kernel_extrator(x_with_seg)

        output, kernel = self.rectangle_trans(x, kernel.flatten(-2).permute(0, 2, 1), level)
        # if level==3 :
        #    output, kernel = self.rectangle_trans(x, kernel.flatten(-2).permute(0, 2, 1), level)
        #    self.kernel = kernel
        # else:
        #     output = x
        #     kernel = torch.cat([self.kernel, kernel], dim=1)
        #     kernel = self.kernel_tran_layers[level](kernel)

        # output = conv_with_kernel(output, kernel)

        # get sim map by transformer ouput
        sim_map1 = self.upsample_block(output)

        # get sim map by conv
        # src = src.permute(0, 3, 1, 2)
        # kernel = (kernel + self.sim_feature_bank.unsqueeze(0)).permute(0, 2, 1).reshape(1, C, 3, 3)
        # output2 = F.conv2d(src, kernel, stride=1, padding=1)
        # sim_map2 = self.upsample_block_2(output2)

        return sim_map1, None # sim_map2


    def rectangle_trans(self, src, kernel, level):

        src_pos = self.pe_layer(src.shape[-2:]).unsqueeze(0)
        src = src + src_pos
        src = src.permute(0, 2, 3, 1)
        B, H, W, C = src.shape

        output = src
        tgt_pos = self.pe_layer((3, 3)).unsqueeze(0).flatten(-2).permute(0, 2, 1)
        tgt = self.sim_feature_bank.unsqueeze(0) + kernel + tgt_pos
        for i, window_size in enumerate(self.window_sizes):

            win_w = window_size[0] // 2**level
            win_h = window_size[1] // 2**level

            # Split into windows
            output, pad_hw = window_partition(output, (win_h, win_w)) # [B * num_windows, win_h, win_w, C].
            output = output.flatten(1, 2)

            num_wins = output.shape[0] // B

            query = tgt.unsqueeze(1).expand(-1, num_wins, -1, -1).flatten(0, 1)
            # Use transformer to update sim_feature_bank
            output = self.transformer(query, output)
            output = output.view(-1, win_h, win_w, C)
            output = window_unpartition(output, (win_h, win_w), pad_hw, (H, W))

        output = output.permute(0, 3, 1, 2)
        tgt = tgt.permute(0, 2, 1).view(B, C, 3, 3)
        return output, tgt