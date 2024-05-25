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
from .regression import RegressionModel, ClassificationModel
from .anchor_points import AnchorPoints

from lib.models.backbones.hrnet.fpn import FPN_Decoder

# 使用点查询的的方法：复现P2PNet，使用vgg19_bn
# 使用32*32的特征生成anchor points，使用卷积预测偏置和得分支
# 使用前3个scale传入fpn，然后使用第2个特征图

class Mask2FormerHead_v3_2_6(nn.Module):

    def __init__(self, decoder,
                 num_transformer_feat_level=3,
                 feat_channels=256,
                 out_channels=256,
                 num_queries: int = 1,
                 num_classes: int = 1,
                 positional_encoding=dict(num_feats=128, normalize=True),
                 transformer_decoder: ConfigDict = ...,
                 **kwargs):
        super(Mask2FormerHead_v3_2_6, self).__init__()
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

        # predict
        # self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        # decoder to get sim_map
        # self.sim_map_decoders = nn.ModuleList()
        # for _ in range(self.num_transformer_decoder_layers + 1):
        #     decoder_layer = nn.TransformerDecoderLayer(d_model=feat_channels, nhead=self.num_heads, batch_first=True)
        #     decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        #     self.sim_map_decoders.append(decoder)
        self.window_sizes = [(32, 16), (16, 8)]

        decoder_layer = nn.TransformerDecoderLayer(d_model=feat_channels, nhead=self.num_heads, batch_first=True)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.points_query_decoder = decoder

        # self.class_embed = nn.Linear(feat_channels, num_classes + 1)
        # self.coord_embed = MLP(feat_channels, feat_channels, 2, 3)

        num_anchor_points = 4
        self.regression = RegressionModel(num_features_in=feat_channels, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=feat_channels, \
                                                  num_classes=self.num_classes + 1, \
                                                  num_anchor_points=num_anchor_points)

        row, line = 2, 2
        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)
        # self.origin_features_conv = nn.Conv2d(48, feat_channels, (3, 3), 1, padding=1)
        in_channels = kwargs['in_channels']
        self.fpn = FPN_Decoder(in_channels[2], in_channels[1], in_channels[0], feat_channels)


    def forward(self, x: List[Tensor], **kwargs):

        batch_size = x[0].shape[0]
        # mask_features, multi_scale_memorys = self.decoder(x)

        # decoder_inputs = []
        # decoder_positional_encodings = []
        # for i in range(self.num_transformer_feat_level):
        #     decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
        #     # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
        #     decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
        #     level_embed = self.level_embed.weight[i].view(1, 1, -1)
        #     decoder_input = decoder_input + level_embed
        #     # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
        #     mask = decoder_input.new_zeros(
        #         (batch_size, ) + multi_scale_memorys[i].shape[-2:],
        #         dtype=torch.bool)
        #     decoder_positional_encoding = self.decoder_positional_encoding(
        #         mask)
        #     decoder_positional_encoding = decoder_positional_encoding.flatten(
        #         2).permute(0, 2, 1)
        #     decoder_inputs.append(decoder_input)
        #     decoder_positional_encodings.append(decoder_positional_encoding)
        # # shape (num_queries, c) -> (batch_size, num_queries, c)
        # query_feat = self.query_feat.weight.unsqueeze(0).repeat(
        #     (batch_size, 1, 1))
        # query_embed = self.query_embed.weight.unsqueeze(0).repeat(
        #     (batch_size, 1, 1))
        #
        cls_pred_list = []
        mask_pred_list = []
        # cls_pred, mask_pred, attn_mask = self._forward_head(
        #     query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        #
        # cls_pred_list.append(cls_pred)
        # mask_pred_list.append(mask_pred)
        #
        # for i in range(self.num_transformer_decoder_layers):
        #     level_idx = i % self.num_transformer_feat_level
        #     # if a mask is all True(all background), then set it all False.
        #     mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
        #     attn_mask = attn_mask & mask_sum
        #     # cross_attn + self_attn
        #     layer = self.transformer_decoder.layers[i]
        #     query_feat = layer(
        #         query=query_feat,
        #         key=decoder_inputs[level_idx],
        #         value=decoder_inputs[level_idx],
        #         query_pos=query_embed,
        #         key_pos=decoder_positional_encodings[level_idx],
        #         cross_attn_mask=attn_mask,
        #         query_key_padding_mask=None,
        #         # here we do not apply masking on padded region
        #         key_padding_mask=None)
        #
        #     cls_pred, mask_pred, attn_mask = self._forward_head(
        #         query_feat, mask_features, multi_scale_memorys[
        #                                        (i + 1) % self.num_transformer_feat_level].shape[-2:])
        #
        #     cls_pred_list.append(cls_pred)
        #     mask_pred_list.append(mask_pred)
        #
        # # 获取查询向量, 预测偏置
        # mask = decoder_input.new_zeros((batch_size,) + mask_features.shape[-2:], dtype=torch.bool)
        # mask_pos = self.decoder_positional_encoding(mask)
        # origin_features = self.origin_features_conv(x[0]) # 转换channel

        inputs = kwargs['inputs']
        fpn_features = self.fpn(x[0:3])
        down_features = fpn_features[1]
        # run the regression and classification branch
        regression = self.regression(down_features) * 100  # 8x
        classification = self.classification(down_features)
        anchor_points = self.anchor_points(inputs).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_coord = output_coord.flip(dims=(-1,)) # (w, h) -> (h, w)
        img_h, img_w = inputs.shape[-2:]
        output_coord[..., 0] /= img_h
        output_coord[..., 1] /= img_w

        output_class = classification
        out = {'pred_logits': output_class,
               'pred_points': output_coord,
               'pred_offsets': regression,
               'point_queries': anchor_points,
               'img_shape': inputs.shape[-2:],}
        return cls_pred_list, mask_pred_list, out

    def query_crowd_head(self, query_embed, query_pos, src_embed, src_pos):

        query_feas = (query_embed + query_pos).permute(0, 2, 3, 1)
        src_feas = (src_embed + src_pos).permute(0, 2, 3, 1)

        scale_index = 0
        B, C, H, W = query_embed.shape
        tgt = query_feas
        src = src_feas
        for i, window_size in enumerate(self.window_sizes):
            win_w = window_size[0] // 2 ** scale_index
            win_h = window_size[1] // 2 ** scale_index

            # Split into windows
            tgt, pad_hw = window_partition(tgt, (win_h, win_w))  # [B * num_windows, win_h, win_w, C].
            tgt = tgt.flatten(1, 2)

            src, pad_hw2 = window_partition(src, (win_h, win_w))  # [B * num_windows, win_h, win_w, C].
            src = src.flatten(1, 2)

            # Use transformer to update sim_feature_bank
            tgt = self.points_query_decoder(tgt, src)

            # restore
            tgt = tgt.view(-1, win_h, win_w, C)
            tgt = window_unpartition(tgt, (win_h, win_w), pad_hw, (H, W))

            src = src.view(-1, win_h, win_w, C)
            src = window_unpartition(src, (win_h, win_w), pad_hw2, (H, W))

        return tgt


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
        cls_pred = None # self.cls_embed(decoder_out)
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


    def get_sim_map(self, x, x_pos, sim_features, sim_features_pos, layer_num):

        if layer_num==9:
            scale_index = 0
        else:
            scale_index = 3 - layer_num % 3

        # x = self.dsnet(x)
        B, C, H, W = x.shape
        src = (x + x_pos).permute(0, 2, 3, 1)

        sim_features = sim_features + sim_features_pos

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


