from lib.models.backbones.backbone_selector import BackboneSelector
from lib.utils.Gaussianlayer import Gaussianlayer
from ...losses import CHSLoss2, build_criterion
from mmengine.utils import is_list_of
from typing import Dict, OrderedDict
from lib.models.networks.layers import *

# 使用sim_feature_bank存储相似特征  输入特征支持（256, 256）
# level_map + rentangle window attention

def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False

# 分割特征加入了低层级的分割特征

class SMC210(nn.Module):
    def __init__(self, config=None, weight=200, route_size=(64, 64), device=None):
        super(SMC210, self).__init__()
        self.config = config
        self.device = device
        self.resolution_num = config.resolution_num

        self.backbone = BackboneSelector(self.config).get_backbone()
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)

        self.decoder = self.build_encoder(config.pixel_decoder)
        self.transformer_decoder_head = self.build_transformer_encoder_head(decoder=self.decoder, cfg=config)

        self.gaussian_maximum = self.gaussian.gaussian.gkernel.weight.max()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.weight = weight
        self.route_size = (route_size[0] // (2 ** self.resolution_num[0]),
                           route_size[1] // (2 ** self.resolution_num[0]))
        self.label_start = self.resolution_num[0]
        self.label_end = self.resolution_num[-1] + 1

        # 转换上采样的数据
        self.hidden_channels = 64
        transition_layers = []
        for i, c in enumerate(self.config.head.stages_channel):
            if i == 0:
                transition_layers.append(TransitionLayer(self.config.head.stages_channel[i], self.hidden_channels))
            else:
                transition_layers.append(TransitionLayer(self.config.head.stages_channel[i] + self.hidden_channels, self.hidden_channels))
        self.transitions1 = nn.Sequential(*transition_layers)

        seg_layers = []
        seg_level_layers = []
        for i, c in enumerate(self.config.head.stages_channel):
            seg_layers.append(
                SegmentationLayer(self.hidden_channels, 1))
            seg_level_layers.append(
                SegmentationLayer(self.hidden_channels, 6))
        self.seg_crowds = nn.Sequential(*seg_layers)
        self.seg_levels = nn.Sequential(*seg_level_layers)

        self.kernel_size = 3
        self.kernel_extractor = build_gen_kernel(self.config)

        channel_last_layer = self.config.head.stages_channel[0]

        # self.decoder_wrapper = TransDecoderWrapperLayer(self.config, 256)
        # self.position_embed = PositionEmbeddingSine(128)
        # self.query_embed = nn.Embedding(576, 256)

        self.criterion_chs = CHSLoss2(size=config.CHSLoss.dcsize, max_noisy_ratio=config.CHSLoss.max_noisy_ratio,
                                 max_weight_ratio=config.CHSLoss.max_weight_ratio)

        self.criterion_mse = nn.MSELoss()
        self.sgm = nn.Sigmoid()
        self.iter_count = 0
        self.num_iters = None


        matcher = build_matcher(config.matcher)
        self.criterion = build_criterion(config.criterion, matcher)


    def forward(self, inputs, labels=None, mode='train', *args, **kwargs):
        result = {'pre_den': {}, 'gt_den': {}}
        result.update({'acc1': {'gt': 0, 'error': 0}})

        in_list = self.backbone(inputs)
        in_list = in_list[self.resolution_num[0]:self.resolution_num[-1] + 1]

        kwargs['mode'] = mode
        kwargs['inputs'] = inputs
        cls_pred_list, mask_pred_list, outputs = self.transformer_decoder_head(in_list, **kwargs)
        if labels is None:
            return mask_pred_list

        # ============================高斯卷积处理目标图=============================
        gt_den_map_list = []
        gt_mask_level = 1
        fg_mask = None
        fg_mask_list = []
        th = 1e-5

        pred_mask = self.config.get('pred_mask', True)
        # if len(labels) > 0:
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

            gt_masks = [fg_mask_list[1] for _ in range(len(mask_pred_list))]



        if mode == 'train' or mode == 'val':
            final_loss = torch.tensor(0.).cuda()
            criterion, targets, epoch = self.criterion, kwargs['targets'], kwargs['epoch']
            if mode == 'train':
                # 计算损失
                if pred_mask:
                    loss = self.loss_by_feat(mask_pred_list, gt_masks)
                    parsed_losses, log_vars = self.parse_losses(loss)
                    final_loss += parsed_losses

                # compute loss
                if epoch >= self.config.get('warm_points_loss_epoch', 10):
                    losses, loss_dict = self.compute_loss(outputs, criterion, targets, epoch)
                    result.update({'loss_dict': loss_dict})
                else:
                    losses = 0.
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
            if len(mask_pred_list) > 0:
                result['pre_seg_crowd'] = {}
                result['pre_seg_crowd'].update({'1': F.interpolate(mask_pred_list[-1], size=fg_mask_list[0].shape[-2:]) > 0.5})
                # result['pre_seg_crowd'].update({'2': F.interpolate(mask_pred_list[-1], size=fg_mask_list[1].shape[-2:]) > 0.5})
                # result['pre_seg_crowd'].update({'4': F.interpolate(mask_pred_list[-1], size=fg_mask_list[2].shape[-2:]) > 0.5})
                # result['pre_seg_crowd'].update({'8': F.interpolate(mask_pred_list[-1], size=fg_mask_list[3].shape[-2:]) > 0.5})
                result['gt_seg_crowd'] = {}
                result['gt_seg_crowd'].update({'1': fg_mask_list[0]})
                # result['gt_seg_crowd'].update({'2': fg_mask_list[1]})
                # result['gt_seg_crowd'].update({'4': fg_mask_list[2]})
                # result['gt_seg_crowd'].update({'8': fg_mask_list[3]})
                #
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
        # output_sparse, output_dense = outputs['sparse'], outputs['dense']
        # warmup_ep = 0
        #
        #
        #
        # # compute loss
        # if epoch >= warmup_ep:
        #     loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_level_sparse'])
        #     loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_level_dense'])
        # else:
        #     loss_dict_sparse = criterion(output_sparse, targets)
        #     loss_dict_dense = criterion(output_dense, targets)
        #
        # # sparse point queries loss
        # loss_dict_sparse = {k + '_sp': v for k, v in loss_dict_sparse.items()}
        # weight_dict_sparse = {k + '_sp': v for k, v in weight_dict.items()}
        # loss_pq_sparse = sum(
        #     loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)
        #
        # # dense point queries loss
        # loss_dict_dense = {k + '_ds': v for k, v in loss_dict_dense.items()}
        # weight_dict_dense = {k + '_ds': v for k, v in weight_dict.items()}
        # loss_pq_dense = sum(
        #     loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
        #
        # # point queries loss
        # losses = loss_pq_sparse + loss_pq_dense
        #
        # # update loss dict and weight dict
        # loss_dict = dict()
        # loss_dict.update(loss_dict_sparse)
        # loss_dict.update(loss_dict_dense)
        #
        # weight_dict = dict()
        # weight_dict.update(weight_dict_sparse)
        # weight_dict.update(weight_dict_dense)
        #
        # # 分割损失
        # split_map = outputs['split_map_raw']
        # bs, num_levels, h, w = split_map.shape
        # split_map = split_map.view(bs, num_levels, -1)
        # tgt_level_map = torch.stack([tgt['level_label_8x'] for tgt in targets], dim=0).view(bs, -1).type(
        #     torch.LongTensor).cuda()
        # loss_split = F.cross_entropy(split_map, tgt_level_map, ignore_index=-1)
        #
        # # quadtree splitter loss
        # # den = torch.tensor([target['density'] for target in targets])   # crowd density
        # # bs = len(den)
        # # ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        # # ds_div = outputs['split_map_raw'][ds_idx]
        # # sp_div = 1 - outputs['split_map_raw']
        # #
        # # # constrain sparse regions
        # # loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()
        # #
        # # # constrain dense regions
        # # if sum(ds_idx) > 0:
        # #     ds_num = ds_div.shape[0]
        # #     loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        # # else:
        # #     loss_split_ds = outputs['split_map_raw'].sum() * 0.0
        #
        # # update quadtree splitter loss
        # # loss_split = loss_split_sp + loss_split_ds
        # weight_split = 0.1 if epoch >= warmup_ep else 0.0
        # loss_dict['loss_split'] = loss_split
        # weight_dict['loss_split'] = weight_split
        #
        # # final loss
        # losses += loss_split * weight_split
        # return {'loss_dict': loss_dict, 'weight_dict': weight_dict, 'losses': losses}

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

        from lib.models.networks.layers import MSDeformAttnPixelDecoder
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