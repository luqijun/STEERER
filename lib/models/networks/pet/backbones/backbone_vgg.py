"""
Backbone modules
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from ...layers import NestedTensor
from .vgg import *
from ..position_encoding import build_position_encoding
from typing import Dict

def get_activation(act_type=None):
    if act_type is None:
        return nn.Identity()
    elif act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)

# Basic conv layer
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act_type='relu', depthwise=False, bias=False):
        super(Conv, self).__init__()
        if depthwise:
            assert c1 == c2
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=c1, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(act_type),
                nn.Conv2d(c2, c2, kernel_size=1, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(act_type)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(act_type)
            )


    def forward(self, x):
        return self.convs(x)

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 dilation=[4, 8, 12, 16],
                 expand_ratio=0.25,
                 act_type='relu'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch0 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[0], d=dilation[0], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        self.branch1 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[1], d=dilation[1], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        self.branch2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[2], d=dilation[2], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        self.branch3 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[3], d=dilation[3], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        # self.branch4 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[4], d=dilation[4], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )
        # self.branch5 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[5], d=dilation[5], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )
        # self.branch6 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[6], d=dilation[6], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )
        # self.branch7 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[7], d=dilation[7], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )

    def forward(self, x):
        x1 = self.branch0(x) + x
        x2 = self.branch1(x1 + x) + x1
        x3 = self.branch2(x2 + x1 + x) + x2
        x4 = self.branch3(x3 + x2 + x1 + x) + x3
        return x4


class FeatsFusion(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, hidden_size=256, out_size=256, out_kernel=3):
        super(FeatsFusion, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        # self.dense_net = Bottleneck(hidden_size, dilation=[1, 2, 3, 4])

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3_shape, C4_shape, C5_shape = C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        # P4_x = self.dense_net(P4_x) # Dense Net
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]
    

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.fpn = FeatsFusion(256, 512, 512, hidden_size=num_channels, out_size=num_channels, out_kernel=3)

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                feats.append(xs)
                        
            # feature fusion
            features_fpn = self.fpn([feats[1], feats[2], feats[3]])
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

            # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
            mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
            out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out = [xs]

        return out


class Backbone_VGG(BackboneBase_VGG):
    """
    VGG backbone
    """
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = vgg16_bn(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[NestedTensor] = {}
        pos = {}
        for name, x in xs.items():            
            out[name] = x
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_vgg(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_VGG(args.sub_arch, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == '__main__':
    Backbone_VGG('vgg16_bn', True)