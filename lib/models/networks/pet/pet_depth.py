import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.heads.head_selector import HeadSelector
from lib.models.heads.moe import upsample_module
from lib.utils.Gaussianlayer import Gaussianlayer
import math
# from .layers import Gaussianlayer, DenseScaleNet, TransitionLayer, SegmentationLayer, build_gen_kernel
from ...losses import CHSLoss2, build_criterion
# from .layers.gen_kenel.untils import multi_apply
import logging
from mmengine.utils import is_list_of
from typing import Dict, Optional, Tuple, Union, OrderedDict
from ..layers import *
from .transformer import  build_encoder, build_decoder
from .pet_decoder import PETDecoder
from .backbones import build_backbone_vgg

# p2pnet
class PET_Depth(nn.Module):
    def __init__(self, config=None, *args, **kwargs):
        super(PET_Depth, self).__init__()
        self.config = config
        self.device = config.device

        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)
        self.gaussian_maximum = self.gaussian.gaussian.gkernel.weight.max()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # backbone
        # backbone = BackboneSelector(self.config).get_backbone()
        # self.backbone = backbone
        # self.fpn = FPN_Decoder(256, 512, 512)
        self.backbone = build_backbone_vgg(config)

        # positional embedding
        self.pos_embed = build_position_encoding(config)

        # feature projection
        hidden_dim = config.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(256, hidden_dim, kernel_size=1),
            nn.Conv2d(256, hidden_dim, kernel_size=1),
        ]
        )

        # context encoder
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        config.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(config, enc_win_list=enc_win_list)

        # segmentation
        # self.seg_head = Segmentation_Head(config.hidden_dim, 1)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = context_patch[0] // int(self.encode_feats[:-1]), context_patch[1] // int(
            self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        num_classes = config.num_classes

        # point-query quadtree
        config.sparse_stride, config.dense_stride = 8, 4  # point-query stride
        transformer = build_decoder(config)
        self.quadtree_sparse = PETDecoder(self.backbone, num_classes, quadtree_layer='sparse', args=config,
                                          transformer=transformer)
        self.quadtree_dense = PETDecoder(self.backbone, num_classes, quadtree_layer='dense', args=config,
                                         transformer=transformer)

        # depth adapt
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.split_depth_th = 0.4

        # level embeding
        # self.level_embed = nn.Parameter(
        #     torch.Tensor(2, backbone.num_channels))
        # normal_(self.level_embed)



        # criterion
        matcher = build_matcher(config.matcher)
        self.criterion = build_criterion(config.criterion, matcher)

    def forward(self, samples: NestedTensor, labels=None, mode='train', *args, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if mode == 'train':
            kwargs[mode] = True
        else:
            kwargs['test'] = True
        kwargs['criterion'] = self.criterion

        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        if 'targets' not in kwargs:
            return features['4x'].tensors

        # positional embedding
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # depth embedding
        depth_embed = torch.cat([pos2posemb1d(tgt['depth']) for tgt in kwargs['targets']])
        # kwargs['depth_input_embed'] = depth_embed.permute(0, 3, 1, 2)
        kwargs['depth_input_embed'] = self.adapt_pos1d(depth_embed).permute(0, 3, 1, 2)

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)
        return out

    def test_forward(self, samples, features, pos, **kwargs):
        thrs = 0.5  # inference threshold
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']

        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat(
                        [out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]

        gt_split_map = 1 - (torch.cat([tgt['depth'] for tgt in kwargs['targets']], dim=0) > self.split_depth_th).long()
        div_out['gt_split_map'] = gt_split_map
        div_out['gt_seg_head_map'] = torch.cat([tgt['seg_map'].unsqueeze(0) for tgt in kwargs['targets']], dim=0)
        div_out['pred_split_map'] = F.interpolate(outputs['split_map_raw'], size=gt_split_map.shape[-2:]).squeeze(1)
        if outputs['seg_map'] is not None:
            div_out['pred_seg_head_map'] = F.interpolate(outputs['seg_map'], size=div_out['gt_seg_head_map'].shape[-2:]).squeeze(1)

        div_out['targets'] = kwargs['targets']
        return div_out

    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)

        result = { }
        thres = 0.5
        sparse_index = outputs['split_map_sparse'] > thres
        dense_index = outputs['split_map_dense'] > thres

        pred_points_sparse = outputs['sparse']['pred_points'][0][sparse_index[0]]
        pred_points_dense = outputs['dense']['pred_points'][0][dense_index[0]]
        pred_points = torch.cat([pred_points_sparse, pred_points_dense])
        result['pred_points'] = pred_points.unsqueeze(0)

        pred_logits_sparse = outputs['sparse']['pred_logits'][0][sparse_index[0]]
        pred_logits_dense = outputs['dense']['pred_logits'][0][dense_index[0]]
        pred_logits = torch.cat([pred_logits_sparse, pred_logits_dense])
        result['pred_logits'] = pred_logits.unsqueeze(0)

        losses['outputs'] = result
        losses['targets'] = kwargs['targets']
        losses['loss_dict']['loss_ce'] = losses['loss_dict']['loss_ce_sp'] + losses['loss_dict']['loss_ce_ds']
        losses['loss_dict']['loss_points'] = losses['loss_dict']['loss_points_sp'] + losses['loss_dict']['loss_points_ds']
        return losses

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        context_info = (encode_src, src_pos_embed, mask)

        # apply seg head
        seg_map = None # self.seg_head(encode_src)

        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)

        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            # level embeding
            # kwargs['level_embed'] = self.level_embed[0]
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
        else:
            outputs_sparse = None

        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            # level embeding
            # kwargs['level_embed'] = self.level_embed[1]
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
        else:
            outputs_dense = None

        # format outputs
        outputs = dict(seg_map=None)
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        outputs['seg_map'] = seg_map
        return outputs

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # sparse point queries loss
        loss_dict_sparse = {k + '_sp': v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k + '_sp': v for k, v in weight_dict.items()}
        loss_pq_sparse = sum(
            loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k + '_ds': v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k + '_ds': v for k, v in weight_dict.items()}
        loss_pq_dense = sum(
            loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)

        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # seg head loss
        seg_map = outputs['seg_map']
        if seg_map is not None:
            gt_seg_map = torch.stack([tgt['seg_map'] for tgt in targets], dim=0)
            gt_seg_map = F.interpolate(gt_seg_map.unsqueeze(1), size=seg_map.shape[-2:]).squeeze(1)
            loss_seg_map = self.bce_loss(seg_map.float().squeeze(1), gt_seg_map)
            losses += loss_seg_map * 0.1
            loss_dict['loss_seg_map'] = loss_seg_map * 0.1

        # splitter depth loss
        pred_depth_levels = outputs['split_map_raw']
        gt_depth_levels = []
        for tgt in targets:
            depth = F.adaptive_avg_pool2d(tgt['depth'], pred_depth_levels.shape[-2:])
            depth_level = torch.ones(depth.shape, device=depth.device)
            depth_level[depth > self.split_depth_th] = 0
            gt_depth_levels.append(depth_level)
        gt_depth_levels = torch.cat(gt_depth_levels, dim=0)
        # gt_depth_levels = torch.cat([target['depth_level'] for target in targets], dim=0)
        # pred_depth_levels = F.interpolate(outputs['split_map_raw'], size=gt_depth_levels.shape[-2:])
        loss_split_depth = F.binary_cross_entropy(pred_depth_levels.float().squeeze(1), gt_depth_levels)
        loss_split = loss_split_depth
        losses += loss_split * 0.1
        loss_dict['loss_split_depth'] = loss_split

        return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}


    def cal_lc_loss(self, output, target, sizes=(1, 2, 4)):
        criterion_L1 = nn.L1Loss(reduction='sum')
        Lc_loss = None
        for s in sizes:
            pool = nn.AdaptiveAvgPool2d(s)
            est = pool(output)
            gt = pool(target)
            if Lc_loss:
                Lc_loss += criterion_L1(est, gt) / s ** 2
            else:
                Lc_loss = criterion_L1(est, gt) / s ** 2
        return Lc_loss

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

import torchvision

def _max_by_axis_pad(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    block = 256
    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])

        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb