import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
import numpy as np
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .nested_tensor import NestedTensor
from .position_encoding import PositionEmbeddingSine

class TransDecoderWrapperLayer2(nn.Module):
    def __init__(self, config=None, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                 num_decoder_layers=6, reg_hidden_channels=128,
                 activation="relu", normalize_before=False, return_intermediate_dec=False, use_pos=True):
        super(TransDecoderWrapperLayer2, self).__init__()
        self.use_pos = use_pos
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, use_pos=use_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model)

        self.config = config
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

        self.tgt_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.src_conv = nn.Conv2d(d_model, d_model, kernel_size=1)

        # self.regression_head = nn.Sequential(*[nn.Conv2d(d_model, reg_hidden_channels, 3, padding=1),
        #                                        nn.BatchNorm2d(reg_hidden_channels),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(reg_hidden_channels, reg_hidden_channels, 3, padding=1),
        #                                        nn.BatchNorm2d(reg_hidden_channels),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(reg_hidden_channels, 1, 3, padding=1),
        #                                        ])

        self.position_embed = PositionEmbeddingSine(128)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed_position(self, tensors):
        nest_tensors = NestedTensor(tensors, None)
        pos_embedding = self.position_embed(nest_tensors)
        return tensors + pos_embedding

    def forward(self, tgt, src, mask, query_embed, pos_embed):

        if not self.use_pos:
            tgt = self.embed_position(tgt)
            src = self.embed_position(src)

        bs, c, h_tgt, w_tgt = tgt.shape

        # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape

        tgt = self.tgt_conv(tgt).flatten(2).permute(2, 0, 1)
        src = self.src_conv(src).flatten(2).permute(2, 0, 1)

        if self.use_pos:
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)

        memory = src
        # tgt = torch.zeros_like(query_embed)
        hs: torch.Tensor = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        hs = hs.squeeze(0).permute(1, 2, 0).contiguous().view(bs, self.d_model, h_tgt, w_tgt)
        # output =  self.regression_head(hs)

        return hs