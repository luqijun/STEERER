import torch
import torch.nn as nn
from einops import rearrange

class CHSLoss(nn.Module):
    def __init__(self, size=8, max_noisy_ratio=0.1, max_weight_ratio=1):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.max_noisy_ratio = max_noisy_ratio
        self.max_weight_ratio = max_weight_ratio

    def forward_single(self, dmap_conv, dmap_tran, gt_density, process):
        weight = self.max_weight_ratio * process
        noisy_ratio = self.max_noisy_ratio * process

        gt_density = self.avgpooling(gt_density) * self.tot
        assert dmap_conv.size() == dmap_tran.size(), f'{dmap_conv.size()},{dmap_tran.size}'
        b, c, h, w = dmap_conv.size()
        dmap_conv = rearrange(dmap_conv, 'b c h w -> b (c h w)')
        dmap_tran = rearrange(dmap_tran, 'b c h w -> b (c h w)')
        dmap_gt = rearrange(gt_density, 'b c h w -> b (c h w)')

        error_conv_gt = torch.abs(dmap_gt - dmap_conv)
        error_tran_gt = torch.abs(dmap_gt - dmap_tran)

        # weight: dmap_gt ---> dmap_conv/tran
        combine_conv_gt = weight * dmap_conv + (1 - weight) * dmap_gt
        combine_tran_gt = weight * dmap_tran + (1 - weight) * dmap_gt

        num = int(h * w * noisy_ratio)
        if num < 1:
            loss = torch.sum((dmap_conv - dmap_gt) ** 2) + torch.sum((dmap_tran - dmap_gt) ** 2)
            return loss

        # conv-branch use tran+gt to supervise
        v, _ = torch.topk(error_conv_gt, num, dim=-1, largest=True)
        v_min = v.min(dim=-1).values
        v_min = v_min.unsqueeze(-1)
        supervision_from_tran = torch.where(torch.ge(error_conv_gt, v_min), combine_tran_gt, dmap_gt)
        mse_conv = (dmap_conv - supervision_from_tran) ** 2

        # tran-branch use conv+gt to supervise
        v, _ = torch.topk(error_tran_gt, num, dim=-1, largest=True)
        v_min = v.min(dim=-1).values
        v_min = v_min.unsqueeze(-1)
        supervision_from_conv = torch.where(torch.ge(error_tran_gt, v_min), combine_conv_gt, dmap_gt)
        mse_tran = (dmap_tran - supervision_from_conv) ** 2

        loss = torch.sum(mse_conv) + torch.sum(mse_tran)
        return loss

    def forward(self, output_map_list, gt_density, process):
        weight = self.max_weight_ratio * process
        noisy_ratio = self.max_noisy_ratio * process

        output_size = output_map_list[0].size()
        b, c, h, w = output_size
        for i, output_map in enumerate(output_map_list):
            assert output_map.size() == output_size, f'{output_size.size()},{output_size}'
            output_map_list[i] = rearrange(output_map, 'b c h w -> b (c h w)')

        gt_density = self.avgpooling(gt_density) * self.tot
        dmap_gt = rearrange(gt_density, 'b c h w -> b (c h w)')

        error_map_gt_list = []
        combine_map_gt_list = []
        for i, output_map in enumerate(output_map_list):
            error_map_gt = torch.abs(dmap_gt - output_map)
            error_map_gt_list.append(error_map_gt)

            # weight: dmap_gt ---> dmap_conv/tran
            combine_map_gt = weight * output_map + (1 - weight) * dmap_gt
            combine_map_gt_list.append(combine_map_gt)

        num = int(h * w * noisy_ratio)
        if num < 1:
            loss = 0
            for i, output_map in enumerate(output_map_list):
                loss += torch.sum((output_map - dmap_gt) ** 2)
            return loss

        loss = 0
        for i, error_map_gt in enumerate(error_map_gt_list) :
            output_map = output_map_list[i]
            # conv-branch use tran+gt to supervise
            v, _ = torch.topk(error_map_gt, num, dim=-1, largest=True)
            v_min = v.min(dim=-1).values
            v_min = v_min.unsqueeze(-1)

            for j, combine_map_gt in enumerate(combine_map_gt_list):
                if j == i:
                    continue
                supervision_from_other = torch.where(torch.ge(error_map_gt, v_min), combine_map_gt, dmap_gt)
                mse_conv = (output_map - supervision_from_other) ** 2
                loss += torch.sum(mse_conv)

        return loss