from lib.models.backbones.backbone_selector import BackboneSelector
from lib.utils.Gaussianlayer import Gaussianlayer
from ...losses import CHSLoss2
from mmengine.utils import is_list_of
from typing import Dict, OrderedDict
from lib.models.networks.layers import *

# 使用sim_feature_bank存储相似特征  输入特征支持（256, 256）
# level_map + rentangle window attention

def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False

# 分割特征加入了低层级的分割特征

class SMC209(nn.Module):
    def __init__(self, config=None, weight=200, route_size=(64, 64), device=None):
        super(SMC209, self).__init__()
        self.config = config
        self.device = device
        self.resolution_num = config.resolution_num

        self.backbone = BackboneSelector(self.config).get_backbone()
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)

        self.decoder = self.build_encoder(config.pixel_decoder)
        self.transformer_decoder_head = self.build_transformer_encoder_head(decoder=self.decoder, cfg=config.mask2former_head)

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

    def forward(self, inputs, labels=None, mode='train', *args, **kwargs):
        result = {'pre_den': {}, 'gt_den': {}}
        result.update({'acc1': {'gt': 0, 'error': 0}})

        in_list = self.backbone(inputs)
        in_list = in_list[self.resolution_num[0]:self.resolution_num[-1] + 1]

        cls_pred_list, mask_pred_list, sim_map_list = self.transformer_decoder_head(in_list)
        if labels is None:
            return sim_map_list[-4:]

        # ============================高斯卷积处理目标图=============================
        gt_den_map_list = []
        gt_mask_level = 1
        fg_mask_list = []
        th = 1e-5
        labels = labels[self.label_start:self.label_end]
        for i, label in enumerate(labels):
            label = self.gaussian(label.unsqueeze(1)) * self.weight
            gt_den_map_list.append(label)
            fg_mask = torch.where(label > th, torch.tensor(1), torch.tensor(0)).float()
            fg_mask_list.append(fg_mask)

        gt_masks = []
        for i, mask in enumerate(fg_mask_list):
            mask = F.interpolate(mask, size=mask_pred_list[-1].shape[-2:])
            gt_masks.append([mask for _ in range(len(mask_pred_list))])
        gt_masks = [torch.cat(ms, dim=1) for ms in zip(*gt_masks)]
        gt_den_maps = []
        for sim_map in sim_map_list:
            for gt_den_map in gt_den_map_list:
                if sim_map.shape[-2:] == gt_den_map.shape[-2:]:
                    gt_den_maps.append(gt_den_map)

        # 计算损失
        loss = self.loss_by_feat(mask_pred_list, gt_masks, sim_map_list, gt_den_maps)
        parsed_losses, log_vars = self.parse_losses(loss)

        if mode == 'train' or mode == 'val':
            loss = 0
            loss += parsed_losses

            for i in ['x4', 'x8', 'x16', 'x32']:
                if i not in result.keys():
                    result.update({i: {'gt': 0, 'error': 0}})
            result.update({'losses': torch.unsqueeze(loss, 0)})
            # result['pre_den'].update({'1': out_list[0] / self.weight})
            # result['pre_den'].update({'8': out_list[-1] / self.weight})

            if len(sim_map_list) == 1:
                sim_map_list.insert(0, F.interpolate(sim_map_list[-1], scale_factor=0.5))
                sim_map_list.insert(0, F.interpolate(sim_map_list[-1], scale_factor=0.25))
                sim_map_list.insert(0, F.interpolate(sim_map_list[-1], scale_factor=0.125))
            elif len(sim_map_list) == 2:
                sim_map_list.insert(0, F.interpolate(sim_map_list[-1], scale_factor=0.25))
                sim_map_list.insert(0, F.interpolate(sim_map_list[-1], scale_factor=0.125))
            elif len(sim_map_list) == 3:
                sim_map_list.insert(0, F.interpolate(sim_map_list[-1], scale_factor=0.125))

            shape = sim_map_list[-1].shape[-2:]
            if sim_map_list[-2].shape[-2] != shape[0] // 2:
                sim_map_list[-2] = F.interpolate(sim_map_list[-2], scale_factor=0.5)
            if sim_map_list[-3].shape[-2] != shape[0] // 4:
                sim_map_list[-3] = F.interpolate(sim_map_list[-3], scale_factor=0.25)
            if sim_map_list[-4].shape[-2] != shape[0] // 8:
                sim_map_list[-4] = F.interpolate(sim_map_list[-4], scale_factor=0.125)

            result['pre_den'].update({'1': sim_map_list[-1] / self.weight})
            if mode == 'val':
                result['pre_den'].update({'2': sim_map_list[-2] / self.weight})
                result['pre_den'].update({'4': sim_map_list[-3] / self.weight})
            result['pre_den'].update({'8': sim_map_list[-4] / self.weight})
            result['gt_den'].update({'1': gt_den_map_list[0] / self.weight})
            if mode == 'val':
                result['gt_den'].update({'2': gt_den_map_list[-3] / self.weight})
                result['gt_den'].update({'4': gt_den_map_list[-2] / self.weight})
            result['gt_den'].update({'8': gt_den_map_list[-1] / self.weight})

            result['pre_seg_crowd'] = {}
            result['pre_seg_crowd'].update({'1': F.interpolate(mask_pred_list[-1], size=fg_mask_list[0].shape[-2:])[:, 0:1] > 0.5})
            result['pre_seg_crowd'].update({'2': F.interpolate(mask_pred_list[-1], size=fg_mask_list[1].shape[-2:])[:, 1:2] > 0.5})
            result['pre_seg_crowd'].update({'4': F.interpolate(mask_pred_list[-1], size=fg_mask_list[2].shape[-2:])[:, 2:3] > 0.5})
            result['pre_seg_crowd'].update({'8': F.interpolate(mask_pred_list[-1], size=fg_mask_list[3].shape[-2:])[:, 3:4] > 0.5})
            result['gt_seg_crowd'] = {}
            result['gt_seg_crowd'].update({'1': fg_mask_list[0]})
            result['gt_seg_crowd'].update({'2': fg_mask_list[1]})
            result['gt_seg_crowd'].update({'4': fg_mask_list[2]})
            result['gt_seg_crowd'].update({'8': fg_mask_list[3]})

            return result

    def loss_by_feat(self, mask_pred_list, gt_masks, sim_map_list, gt_den_maps):

        mask_losses = multi_apply(self._loss_mask_single, mask_pred_list, gt_masks)
        sim_map_losses = multi_apply(self._loss_sim_map_single, sim_map_list, gt_den_maps)

        sim_map_loss = 0.
        weight = self.config.loss_weight[::-1]
        for i, s_loss in enumerate(sim_map_losses[0]):
            sim_map_loss += s_loss * weight[3]

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_mask'] = mask_losses[-1]
        loss_dict['loss_sim_map'] = sim_map_loss
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_mask_i, loss_sim_map_i in zip(
                mask_losses[:-1], sim_map_losses[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_sim_map'] = loss_sim_map_i
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

        args = cfg.copy()
        type = args.pop('type')
        decoder = eval(type)(decoder=decoder,
                                  positional_encoding=args.positional_encoding,
                                  transformer_decoder=args.transformer_decoder)
        return decoder


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