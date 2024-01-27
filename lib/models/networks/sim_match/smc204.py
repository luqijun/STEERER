import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.heads.head_selector import HeadSelector
from lib.models.heads.moe import upsample_module
from lib.utils.Gaussianlayer import Gaussianlayer
import math
from .layers import Gaussianlayer, DenseScaleNet, TransitionLayer, SegmentationLayer, GenerateKernelLayer204
import logging

# 使用sim_feature_bank存储相似特征  输入特征支持（256, 256）
# level_map + rentangle window attention

def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False

class SMC204(nn.Module):
    def __init__(self, config=None, weight=200, route_size=(64, 64), device=None):
        super(SMC204, self).__init__()
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
        self.kernel_extractor = GenerateKernelLayer204(self.config, self.kernel_size, self.hidden_channels)

        channel_last_layer = self.config.head.stages_channel[0]

        # self.decoder_wrapper = TransDecoderWrapperLayer(self.config, 256)
        # self.position_embed = PositionEmbeddingSine(128)
        # self.query_embed = nn.Embedding(576, 256)

        # self.criterion_chs = CHSLoss2(size=config.CHSLoss.dcsize, max_noisy_ratio=config.CHSLoss.max_noisy_ratio,
        #                          max_weight_ratio=config.CHSLoss.max_weight_ratio)

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

                # crowd seg
                seg_crowd_map = self.seg_crowds[i](in_map)
                last_map = in_map * seg_crowd_map
                fg_list.insert(0, seg_crowd_map)

                # level seg
                # seg_level_map = self.seg_levels[i](in_map)
                # seg_level_map_list.insert(0, seg_level_map)

                # 直接预测一个Density Map
                out_list.insert(0, None)

                # 提取共性特征 获得一个卷积核
                sim_map = self.kernel_extractor(last_map, len(in_list)- i - 1)
                sim_map_list.insert(0, sim_map)
                # sim_map = self.kernel_extractor.conv_with_kernels(last_map, i)

                trans_output_list.insert(0, None)

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

            # =============================计算MSE、分割、等级损失==============================
            if mode == 'train':
                level_nums = 5
                gt_seg_level_map = args[0][0]
                gt_seg_level_map = (gt_seg_level_map * level_nums).long()  # 将标签值乘以5（因为有6个类别）然后向下取整，得到每个像素的类别标签
                gt_seg_level_map = torch.clamp(gt_seg_level_map, 0, level_nums)

            for i in range(len(out_list)):
                tmp_loss =0

                # level map损失
                # if mode == 'train':
                #     temp_gt_seg_level_map = gt_seg_level_map.float()
                #     if temp_gt_seg_level_map.shape[2:] != seg_level_map_list[i].shape[i]:
                #         temp_gt_seg_level_map = F.interpolate(temp_gt_seg_level_map.unsqueeze(1), seg_level_map_list[i].shape[2:],
                #                                               mode='nearest').squeeze(1)
                #     tmp_loss += self.ce_loss(seg_level_map_list[i], temp_gt_seg_level_map.long())  # 使用交叉熵损失函数

                # 分割图损失
                amp = self.sgm(fg_list[i])
                amp_gt_us = fg_mask_list[i].float()
                if amp_gt_us.shape[2:] != amp.shape[2:]:
                    amp_gt_us = F.interpolate(amp_gt_us, amp.shape[2:], mode='nearest')
                ce_loss = (amp_gt_us * torch.log(amp + 1e-10) + (1 - amp_gt_us) * torch.log(1 - amp + 1e-10)) * -1
                tmp_loss += torch.mean(ce_loss)
                #tmp_loss += self.bce_loss(fg_list[i], amp_gt_us)

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
                if len(out_map_list)>1:
                    rate = min(1, (self.iter_count) / self.num_iters)
                    chs_loss = 0 # self.criterion_chs(out_map_list, label, rate)

                loss_list.append((lbda1 * chs_loss + lbda2 * tmp_loss) * self.config.loss_weight[i])

            if mode == 'train' or mode == 'val':
                loss = 0

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

                return result

            elif mode == 'test':
                outputs = torch.mean(multi_outputs_list[-1], dim=0)
                return outputs / self.weight

    def get_moe_label(self, out_list, label_list, route_size):
        """
        :param out_list: (N,resolution_num,H, W) tensor
        :param in_list:  (N,resolution_num,H, W) tensor
        :param route_size: 256
        :return:
        """
        B_num, C_num, H_num, W_num =  out_list[0].size()
        patch_h, patch_w = H_num // route_size[0], W_num // route_size[1]
        errorInslice_list = []

        for i, (pre, gt) in enumerate(zip(out_list, label_list)):
            pre, gt= pre.detach(), gt.detach()
            kernel = (int(route_size[0]/(2**i)), int(route_size[1]/(2**i)))

            weight = torch.full(kernel,1/(kernel[0]*kernel[1])).expand(1,pre.size(1),-1,-1)
            weight =  nn.Parameter(data=weight, requires_grad=False).to(self.device)

            error= (pre - gt)**2
            patch_mse=F.conv2d(error, weight,stride=kernel)

            weight = torch.full(kernel,1.).expand(1,pre.size(1),-1,-1)
            weight =  nn.Parameter(data=weight, requires_grad=False).to(self.device)

            # mask = (gt>0).float()
            # mask = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
            patch_error=F.conv2d(error, weight,stride=kernel)  #(pre-gt)*(gt>0)
            fractions = F.conv2d(gt, weight, stride=kernel)

            instance_mse = patch_error/(fractions+1e-10)

            errorInslice_list.append(patch_mse + instance_mse)


        score = torch.cat(errorInslice_list, dim=1)
        moe_label = score.argmin(dim=1, keepdim=True)


        # mask = mask.view(mask.size(0),mask.size(1),patch_h, patch_w).float()
        # import pdb
        # pdb.set_trace()

        return  moe_label, score