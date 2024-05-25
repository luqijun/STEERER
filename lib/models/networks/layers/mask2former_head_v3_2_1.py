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

# 使用点查询的的方法：不使用查询，直接使用64*64的msdeter特征预测偏置值
class Mask2FormerHead_v3_2_1(nn.Module):

    def __init__(self, decoder,
                 num_transformer_feat_level=3,
                 feat_channels=256,
                 out_channels=256,
                 num_queries: int = 1,
                 num_classes: int = 1,
                 positional_encoding=dict(num_feats=128, normalize=True),
                 transformer_decoder: ConfigDict = ...,
                 **kwargs):
        super(Mask2FormerHead_v3_2_1, self).__init__()
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

        self.class_embed = nn.Linear(feat_channels, num_classes + 1)
        self.coord_embed = MLP(feat_channels, feat_channels, 2, 3)
        self.origin_features_conv = nn.Conv2d(48, feat_channels, (3, 3), 1, padding=1)



    def forward(self, x: List[Tensor], **kwargs):

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
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

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

        # 获取查询向量, 预测偏置
        mask = decoder_input.new_zeros((batch_size,) + mask_features.shape[-2:], dtype=torch.bool)
        mask_pos = self.decoder_positional_encoding(mask)
        origin_features = self.origin_features_conv(x[0]) # 转换channel

        # query 要不要拼接query_feat?
        points_queries, query_features = self.get_point_queries(mask_features, mask_features.shape[-2:], 4)

        img_shape = mask_features.shape[-2:]
        out = self.predict([img_shape[0] * 4 , img_shape[1] * 4], points_queries, query_features.permute(0, 2, 3, 1).flatten(1, 2), **kwargs)

        return cls_pred_list, mask_pred_list, out


    def get_point_queries(self, features, shape, stride):

        # generate point queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1, 0)  # 2xN --> Nx2
        h, w = shift_x.shape

        # get point queries embedding
        # query_embed = features[:, :, points_queries[:, 0], points_queries[:, 1]]
        # bs, c = query_embed.shape[:2]
        # query_embed = query_embed.view(bs, c, h, w)

        query_embed = features

        return points_queries, query_embed

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

    def predict(self, img_shape, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """

        # points_queries = torch.cat(points_queries, dim=0).cuda()
        # hs = torch.cat(hs, dim=0).cuda()

        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        img_h, img_w = img_shape
        # rescale offset range during testing
        if kwargs['mode'] == 'val':
            outputs_offsets[..., 0] /= (img_h / 256)
            outputs_offsets[..., 1] /= (img_w / 256)

        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        outputs_points = outputs_offsets + points_queries
        out = {'pred_logits': outputs_class, 'pred_points': outputs_points, 'img_shape': img_shape,
               'pred_offsets': outputs_offsets}

        out['points_queries'] = points_queries
        # out['pq_stride'] = self.pq_stride
        return out


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


