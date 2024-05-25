# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData
from torch import Tensor
from .positional_encoding import SinePositionalEncoding
from mmengine.config import ConfigDict
from .detr.mask2former_layers import Mask2FormerTransformerDecoder
from .gen_kenel.untils import *

# 使用卷积提取人头特征，尝试最后一层加上relu，优化显示效果

class Mask2FormerHead3_1(nn.Module):

    def __init__(self, decoder,
                 num_transformer_feat_level=3,
                 feat_channels=256,
                 out_channels=256,
                 num_queries: int = 1,
                 num_classes: int = 1,
                 positional_encoding=dict(num_feats=128, normalize=True),
                 transformer_decoder: ConfigDict = ...,
                 ):
        super(Mask2FormerHead3_1, self).__init__()
        self.decoder = decoder
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())

        self.decoder_positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        # decoder to get sim_map
        self.sim_map_decoders = nn.ModuleList()
        for _ in range(self.num_transformer_decoder_layers + 1):
            decoder_layer = nn.TransformerDecoderLayer(d_model=feat_channels, nhead=self.num_heads, batch_first=True)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.sim_map_decoders.append(decoder)

        self.window_sizes = [(32, 16), (16, 8)]

        # position encoding
        self.pe_layer = PositionEmbeddingRandom(feat_channels // 2)

        # 提取卷积核
        self.conv_kernel_extrator = self.make_kernel_extractor_adaptive_3_1(feat_channels, (7, 7))

        # upsample
        self.upsample_block = self.make_head_layer_3_1(feat_channels, feat_channels // 2)


    def forward(self, x: List[Tensor]):

        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.decoder(x)

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        sim_map_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

        mask = decoder_input.new_zeros((batch_size,) + mask_features.shape[-2:], dtype=torch.bool)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                                               (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        # 最后一层
        sim_map = self.get_sim_map(mask_features,
                                   self.decoder_positional_encoding(mask),
                                   mask_pred,
                                   self.num_transformer_decoder_layers)
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        sim_map_list.append(sim_map)

        return cls_pred_list, mask_pred_list, sim_map_list


    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask


    def get_sim_map(self, x, x_pos, mask_pred, layer_num):

        if layer_num==9:
            scale_index = 0
        else:
            scale_index = 3 - layer_num % 3

        # x = self.dsnet(x)
        B, C, H, W = x.shape

        seg_mask = (mask_pred.sigmoid() > 0.5).float()
        if seg_mask.shape[-2:] != x.shape[-2:]:
            seg_mask = F.interpolate(seg_mask, size=x.shape[-2:])
        x_with_seg = x * seg_mask
        kernel = self.conv_kernel_extrator(x_with_seg)
        tgt_pos = self.pe_layer(kernel.shape[-2:]).unsqueeze(0).flatten(-2).permute(0, 2, 1)
        kernel = kernel.flatten(-2).permute(0, 2, 1) + tgt_pos
        sim_features = kernel

        src = (x + x_pos).permute(0, 2, 3, 1)
        output = src
        for i, window_size in enumerate(self.window_sizes):
            win_w = window_size[0] // 2 ** scale_index
            win_h = window_size[1] // 2 ** scale_index

            # Split into windows
            output, pad_hw = window_partition(output, (win_h, win_w))  # [B * num_windows, win_h, win_w, C].
            output = output.flatten(1, 2)

            num_wins = output.shape[0] // B
            # assert num_wins == 8 or num_wins == 32
            tgt =  sim_features
            tgt = tgt.unsqueeze(1).expand(-1, num_wins, -1, -1).flatten(0, 1)

            # Use transformer to update sim_feature_bank
            output = self.sim_map_decoders[layer_num](output, tgt)
            output = output.view(-1, win_h, win_w, C)
            output = window_unpartition(output, (win_h, win_w), pad_hw, (H, W))

        # get sim map by transformer ouput
        output = output.permute(0, 3, 1, 2)
        sim_map = self.upsample_block(output)


        return sim_map

    def make_kernel_extractor_adaptive_3_1(self, hidden_channels, feature_size=(7, 7)):

        # return nn.Sequential(*[nn.AdaptiveAvgPool2d(output_size=feature_size),
        #                        nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1),
        #                        nn.BatchNorm2d(hidden_channels // 2),
        #                        nn.ReLU(True),
        #                        nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 1, padding=1)])
        return nn.Sequential(*[nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, padding=1),
                               nn.AdaptiveAvgPool2d(output_size=feature_size),
                               nn.InstanceNorm2d(hidden_channels // 2),
                               nn.ReLU(True),
                               nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 3, 1),
                               nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 1)])

    def make_head_layer_3_1(self, input_cahnnels=1, hidden_channels=3):
        return nn.Sequential(*[nn.Conv2d(input_cahnnels, hidden_channels, 3, 1, 1),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.InstanceNorm2d(hidden_channels),
                               nn.ReLU(True),
                               nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.Conv2d(hidden_channels // 2, 3, 3, 1, 1),
                               nn.Conv2d(3, 1, 3, 1, 1),],
                               nn.ReLU(True))