from lib.models.networks.layers.position_encoding import PositionEmbeddingRandom
from .untils import *


# 使用相似特征库sim_feature_bank，使用rectangle window
class GenerateKernelLayer204(nn.Module):
    def __init__(self, config):
        super(GenerateKernelLayer204, self).__init__()
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