import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.utils.Gaussianlayer import Gaussianlayer
from lib.models.networks.layers import Gaussianlayer, TransitionLayer, SegmentationLayer, build_gen_kernel
from ...losses import CHSLoss2


# 使用sim_feature_bank存储相似特征  输入特征支持（256, 256）
# level_map + rentangle window attention

def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False

class SMC205(nn.Module):
    def __init__(self, config=None, weight=200, route_size=(64, 64), device=None):
        super(SMC205, self).__init__()
        self.config = config
        self.device = device
        self.resolution_num = config.resolution_num

        self.backbone = BackboneSelector(self.config).get_backbone()
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)

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

    def forward(self, inputs, labels=None, mode='train', *args, **kwargs):
        self.iter_count += 1
        if self.config.counter_type == 'single_resolution':
            pass
        elif self.config.counter_type == 'withMOE':
            result = {'pre_den': {}, 'gt_den': {}}
            in_list = self.backbone(inputs)
            in_list = in_list[self.resolution_num[0]:self.resolution_num[-1] + 1]

            out_list = []
            fg_list = []
            seg_level_map_list = []
            sim_map_list = []
            trans_output_list = []
            last_map = None
            seg_crowd_map = None
            seg_level_map = None
            kernel =  None
            for i, in_map in enumerate(in_list[::-1]):
                # in_map = self.DSNets[i](in_map)
                if last_map is not None:
                    last_map = F.interpolate(last_map, scale_factor=2, mode='nearest')
                    in_map_cat = torch.cat([last_map, in_map], dim=1)
                    in_map = self.transitions1[i](in_map_cat)
                else:
                    in_map = self.transitions1[i](in_map)

                # seg crowd
                seg_crowd_map = self.sgm(self.seg_crowds[i](in_map))
                fg_list.insert(0, seg_crowd_map)
                crop_seg_crowd = in_map * seg_crowd_map
                last_map = in_map + crop_seg_crowd

                # seg level
                # seg_level_map = self.seg_levels[i](in_map)
                # seg_level_map_list.insert(0, seg_level_map)

                # 直接预测一个Density Map
                out_list.insert(0, None)

                # 提取共性特征 获得一个卷积核
                sim_map1, sim_map2 = self.kernel_extractor(last_map, crop_seg_crowd, len(in_list)- i - 1)
                sim_map_list.insert(0, sim_map1)
                # sim_map = self.kernel_extractor.conv_with_kernels(last_map, i)

                trans_output_list.insert(0, sim_map2)

            if labels is None:
                return out_list

            # ============================高斯卷积处理目标图=============================
            label_list = []
            fg_mask_list = []
            th = 1e-5
            labels = labels[self.label_start:self.label_end]
            for i, label in enumerate(labels):
                label = self.gaussian(label.unsqueeze(1)) * self.weight
                label_list.append(label)
                fg_mask_list.append(torch.where(label > th, torch.tensor(1), torch.tensor(0)))

            loss_list = []
            result.update({'acc1': {'gt': 0, 'error': 0}})
            lbda = 0 # 1000

            lbda1 = 1.0
            lbda2 = 1.0

            multi_outputs_list = []

            # =============================计算分割Level map损失==============================
            seg_level_loss = 0.
            # if mode == 'train':
            #     level_nums = 5
            #     gt_seg_level_map = args[0][0]
            #     gt_seg_level_map = (gt_seg_level_map * level_nums).long()  # 将标签值乘以5（因为有6个类别）然后向下取整，得到每个像素的类别标签
            #
            #     for i in range(len(out_list)):
            #         temp_gt_seg_level_map = gt_seg_level_map.float()
            #         if temp_gt_seg_level_map.shape[2:] != seg_level_map_list[i].shape[i]:
            #             temp_gt_seg_level_map = F.interpolate(temp_gt_seg_level_map.unsqueeze(1), seg_level_map_list[i].shape[2:],
            #                                                   mode='nearest').squeeze(1)
            #         seg_level_loss += self.ce_loss(seg_level_map_list[i], temp_gt_seg_level_map.long())  # 使用交叉熵损失函数
            #         gt_seg_level_map = torch.clamp(gt_seg_level_map, 0, level_nums)

            # =============================计算分割Crowd map损失==============================
            # 分割图损失
            seg_crowd_loss = 0.
            for i in range(len(out_list)):
                if fg_list[i].shape[2:] != fg_mask_list[i].shape[2:]:
                    fg_list[i] = F.interpolate(fg_list[i], fg_mask_list[i].shape[2:], mode='nearest')
                pre_seg_crowd = fg_list[i]
                gt_seg_crowd = fg_mask_list[i].float()
                ce_loss = (gt_seg_crowd * torch.log(pre_seg_crowd + 1e-10) + (1 - gt_seg_crowd) * torch.log(1 - pre_seg_crowd + 1e-10)) * -1
                seg_crowd_loss += torch.mean(ce_loss)
                # seg_loss += self.bce_loss(fg_list[i], amp_gt_us)

            # =============================计算输出图MSE损失==============================
            for i in range(len(out_list)):
                tmp_loss =0

                label = label_list[i]
                out_map_list = []

                # 计算直接输入损失
                out_map = out_list[i]
                if out_map is not None:
                    Le_Loss = self.mse_loss(out_map, label_list[i])
                    Lc_Loss = self.cal_lc_loss(out_map, label_list[i])
                    hard_loss1 = Le_Loss + lbda * Lc_Loss
                    tmp_loss += hard_loss1
                    out_map_list.append(out_map)

                # 计算sim_map损失
                sim_map = sim_map_list[i]
                if sim_map is not None:

                    if sim_map.shape[2:] != label.shape[2:]:
                        sim_map = F.interpolate(sim_map, label.shape[2:], mode='bilinear')
                        # label = F.interpolate(label, sim_map.shape[2:], mode='bilinear')

                    # sim_map_loss = 0
                    # _ , seg_mask = torch.max(seg_level_map_list[i], dim=1)
                    # seg_mask =  F.interpolate(seg_mask.unsqueeze(1).float(), label.shape[2:], mode='nearest')
                    # for j in range(seg_level_map_list[i].shape[1]):
                    #     # mask = torch.zeros_like(seg_mask)
                    #     # mask[seg_mask == j] = 1
                    #     # sim_map_loss += self.mse_loss(sim_map * mask, label * mask)
                    #     mask = seg_mask == j
                    #     pred = sim_map[mask]
                    #     target = label[mask]
                    #     if len(pred) > 0:
                    #         sim_map_loss += self.mse_loss(pred, target)
                    # tmp_loss += sim_map_loss
                    # out_map_list.append(sim_map)

                    Le_Loss2 = self.mse_loss(sim_map, label)
                    Lc_Loss2 = self.cal_lc_loss(sim_map, label)
                    hard_loss2 = Le_Loss2 + lbda * Lc_Loss2
                    tmp_loss += hard_loss2
                    out_map_list.append(sim_map)

                # 计算trans_map损失
                trans_map = trans_output_list[i]
                if trans_map is not None:
                    if trans_map.shape[2:] != label.shape[2:]:
                        # label = F.interpolate(label, trans_map.shape[2:], mode='bilinear')
                        trans_map = F.interpolate(trans_map, label.shape[2:], mode='bilinear')
                    Le_Loss3 = self.mse_loss(trans_map, label)
                    Lc_Loss3 = self.cal_lc_loss(trans_map, label)
                    hard_loss3 = Le_Loss3 + lbda * Lc_Loss3
                    tmp_loss += hard_loss3
                    out_map_list.append(trans_map)

                multi_outputs_list.append(torch.stack(out_map_list, dim=0) )

                # 计算交叉交叉监督损失
                chs_loss = 0
                # if len(out_map_list)>1:
                #     rate = min(1, (self.iter_count) / 6000)
                #     chs_loss = self.criterion_chs(out_map_list, label, rate) / (label.shape[-2] * label.shape[-1])

                loss_list.append((lbda1 * chs_loss + lbda2 * tmp_loss) * self.config.loss_weight[i])

            if mode == 'train' or mode == 'val':
                loss = 0
                loss += seg_crowd_loss
                if self.config.baseline_loss:
                    loss = loss_list[0]
                else:
                    for i in range(len(self.resolution_num)):
                        # if self.config.loss_weight:
                        loss += loss_list[i] * self.config.loss_weight[i]
                        # else:
                        #     loss += loss_list[i] /(2**(i))

                for i in ['x4', 'x8', 'x16', 'x32']:
                    if i not in result.keys():
                        result.update({i: {'gt': 0, 'error': 0}})
                result.update({'losses': torch.unsqueeze(loss, 0)})
                # result['pre_den'].update({'1': out_list[0] / self.weight})
                # result['pre_den'].update({'8': out_list[-1] / self.weight})
                result['pre_den'].update({'1': torch.mean(multi_outputs_list[0], dim=0) / self.weight})
                if mode == 'val':
                    result['pre_den'].update({'2': torch.mean(multi_outputs_list[-3], dim=0) / self.weight})
                    result['pre_den'].update({'4': torch.mean(multi_outputs_list[-2], dim=0) / self.weight})
                result['pre_den'].update({'8': torch.mean(multi_outputs_list[-1], dim=0) / self.weight})
                result['gt_den'].update({'1': label_list[0] / self.weight})
                if mode == 'val':
                    result['gt_den'].update({'2': label_list[-3] / self.weight})
                    result['gt_den'].update({'4': label_list[-2] / self.weight})
                result['gt_den'].update({'8': label_list[-1] / self.weight})

                result['pre_seg_crowd'] = {}
                result['pre_seg_crowd'].update({'1': fg_list[0] > 0.5})
                result['pre_seg_crowd'].update({'2': fg_list[1] > 0.5})
                result['pre_seg_crowd'].update({'4': fg_list[2] > 0.5})
                result['pre_seg_crowd'].update({'8': fg_list[3] > 0.5})
                result['gt_seg_crowd'] = {}
                result['gt_seg_crowd'].update({'1': fg_mask_list[0]})
                result['gt_seg_crowd'].update({'2': fg_mask_list[1]})
                result['gt_seg_crowd'].update({'4': fg_mask_list[2]})
                result['gt_seg_crowd'].update({'8': fg_mask_list[3]})

                return result

            elif mode == 'test':
                outputs = torch.mean(multi_outputs_list[-1], dim=0)
                return outputs / self.weight