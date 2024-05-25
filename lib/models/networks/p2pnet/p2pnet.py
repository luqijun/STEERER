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

# p2pnet
class P2PNet(nn.Module):
    def __init__(self, config=None, *args, **kwargs):
        super(P2PNet, self).__init__()
        self.config = config
        self.device = config.device

        # backbone
        self.backbone = BackboneSelector(self.config).get_backbone()
        self.fpn = FPN_Decoder(256, 512, 512)

        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)
        self.gaussian_maximum = self.gaussian.gaussian.gkernel.weight.max()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        row = config.row
        line = config.line
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        # criterion
        matcher = build_matcher(config.matcher)
        self.criterion = build_criterion(config.criterion, matcher)


    def forward(self, inputs, labels=None, mode='train', *args, **kwargs):

        kwargs['mode'] = mode
        kwargs['inputs'] = inputs

        result = {'pre_den': {}, 'gt_den': {}}
        result.update({'acc1': {'gt': 0, 'error': 0}})

        samples = inputs

        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[1]) * 100  # 8x
        output_class = self.classification(features_fpn[1])

        # generate anchor points
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)

        # decode the points as prediction
        output_coord = regression + anchor_points
        output_coord = output_coord.flip(dims=(-1,))  # (w, h) -> (h, w)
        img_h, img_w = inputs.shape[-2:]
        output_coord[..., 0] /= img_h
        output_coord[..., 1] /= img_w

        outputs = {
            'pred_logits': output_class,
            'pred_points': output_coord,
            'pred_offsets': regression,
            'point_queries': anchor_points,
            'img_shape': inputs.shape[-2:]
        }
        if labels is None:
            return output_coord

        # ============================高斯卷积处理目标图=============================
        gt_den_map_list = []
        gt_mask_level = 1
        fg_mask = None
        fg_mask_list = []
        th = 1e-5

        # 是否预测掩码
        pred_mask = self.config.get('pred_mask', True)
        if pred_mask:
            labels = labels[self.label_start:self.label_end]
            for i, label in enumerate(labels):
                label = self.gaussian(label.unsqueeze(1)) * self.weight
                gt_den_map_list.append(label)
                if i == gt_mask_level:
                    fg_mask = torch.where(label > th, torch.tensor(1), torch.tensor(0)).float()

            for i, label in enumerate(labels):
                if i == gt_mask_level:
                    fg_mask_list.append(fg_mask)
                else:
                    cur_fg_mask = F.interpolate(fg_mask, size=label.shape[-2:], mode='nearest')
                    fg_mask_list.append(cur_fg_mask)

            gt_masks = [fg_mask_list[1] for _ in range(len(labels))]

        if mode == 'train' or mode == 'val':
            final_loss = torch.tensor(0.).cuda()
            criterion, targets, epoch = self.criterion, kwargs['targets'], kwargs['epoch']
            if mode == 'train':
                # compute split loss
                # if pred_mask:
                #     loss = self.loss_by_feat(mask_pred_list, gt_masks)
                #     parsed_losses, log_vars = self.parse_losses(loss)
                #     final_loss += parsed_losses

                # compute point loss and classification loss
                losses, loss_dict = self.compute_loss(outputs, criterion, targets, epoch)
                result.update({'loss_dict': loss_dict})
                final_loss += losses

            for i in ['x4', 'x8', 'x16', 'x32']:
                if i not in result.keys():
                    result.update({i: {'gt': 0, 'error': 0}})

            result.update({'losses': torch.unsqueeze(final_loss, 0)})
            result.update({'outputs': outputs})
            result.update({'targets': targets})
            # return result
            # result['pre_den'].update({'1': out_list[0] / self.weight})
            # result['pre_den'].update({'8': out_list[-1] / self.weight})

            # result['pre_den'].update({'1': sim_map_list[-1] / self.weight})
            # if mode == 'val':
            #     result['pre_den'].update({'2': sim_map_list[-2] / self.weight})
            #     result['pre_den'].update({'4': sim_map_list[-3] / self.weight})
            # result['pre_den'].update({'8': sim_map_list[-4] / self.weight})
            # result['gt_den'].update({'1': gt_den_map_list[0] / self.weight})
            # if mode == 'val':
            #     result['gt_den'].update({'2': gt_den_map_list[-3] / self.weight})
            #     result['gt_den'].update({'4': gt_den_map_list[-2] / self.weight})
            # result['gt_den'].update({'8': gt_den_map_list[-1] / self.weight})
            #

            # if len(mask_pred_list) > 0:
            #     result['pre_seg_crowd'] = {}
            #     result['pre_seg_crowd'].update({'1': F.interpolate(mask_pred_list[-1], size=fg_mask_list[0].shape[-2:]) > 0.5})
            #     # result['pre_seg_crowd'].update({'2': F.interpolate(mask_pred_list[-1], size=fg_mask_list[1].shape[-2:]) > 0.5})
            #     # result['pre_seg_crowd'].update({'4': F.interpolate(mask_pred_list[-1], size=fg_mask_list[2].shape[-2:]) > 0.5})
            #     # result['pre_seg_crowd'].update({'8': F.interpolate(mask_pred_list[-1], size=fg_mask_list[3].shape[-2:]) > 0.5})
            #     result['gt_seg_crowd'] = {}
            #     result['gt_seg_crowd'].update({'1': fg_mask_list[0]})
            #     # result['gt_seg_crowd'].update({'2': fg_mask_list[1]})
            #     # result['gt_seg_crowd'].update({'4': fg_mask_list[2]})
            #     # result['gt_seg_crowd'].update({'8': fg_mask_list[3]})
            #     #
            return result


    def compute_loss(self, outputs, criterion, targets, epoch):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = weight_dict["loss_ce"] * loss_dict["loss_ce"] + weight_dict["loss_points"] * loss_dict["loss_points"]
        return losses, loss_dict

    def loss_by_feat(self, mask_pred_list, gt_masks):

        mask_losses = multi_apply(self._loss_mask_single, mask_pred_list, gt_masks)
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_mask'] = mask_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_mask_i in zip(
                mask_losses[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            num_dec_layer += 1
        return loss_dict

    def _loss_mask_single(self, mask, gt_mask):
        loss = self.bce_loss(mask, gt_mask)
        return loss,

    def _loss_sim_map_single(self, sim_map, gt_den_map):
        loss = self.mse_loss(sim_map, gt_den_map)
        return loss,

    def build_encoder(self, cfg):

        from .layers.msdetr import MSDeformAttnPixelDecoder
        args = cfg.copy()
        args.pop('type')
        decoder = MSDeformAttnPixelDecoder(in_channels=args.in_channels,
                                           strides=args.strides,
                                           feat_channels=args.feat_channels,
                                           out_channels=args.out_channels,
                                           num_outs=args.num_outs,
                                           norm_cfg=args.norm_cfg,
                                           act_cfg=args.act_cfg,
                                           encoder=args.encoder)
        return decoder

    def build_transformer_encoder_head(self, decoder, cfg):

        args = cfg.mask2former_head.copy()
        type = args.pop('type')
        kwargs = {}
        kwargs['in_channels'] = cfg.pixel_decoder.in_channels
        encoder = eval(type)(decoder=decoder,
                             positional_encoding=args.positional_encoding,
                             transformer_decoder=args.transformer_decoder, **kwargs)
        return encoder


    def set_num_iters(self, num_iters):
        self.num_iters = num_iters

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