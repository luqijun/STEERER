import torch
import torch.nn as nn
import torch.nn.functional as F
from ....utils.utils import get_world_size
from torchvision.ops.focal_loss import sigmoid_focal_loss

class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, args):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.args = args
        self.loss_type = args.get('loss_type', 'cross_entropy_loss')
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # self.num_levels = args.num_levels
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef  # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4: 0.5}

    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_leghths = [len(t["labels"][J]) for t, (_, J) in zip(targets, indices)]
        target_classes_o = [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        # target_classes = torch.zeros(src_logits.shape[:1], dtype=torch.int64, device=src_logits.device)

        idx = self._get_src_permutation_idx(indices)[1]
        idx = list(idx.split(target_leghths, dim=0))
        # target_classes = list(target_classes.split(outputs["point_query_lengths"], dim=0))
        src_logits = list(src_logits.split(outputs["point_query_lengths"], dim=0))

        loss_ce = 0.
        for idx_one, tc_o, src_logit in zip(idx, target_classes_o, src_logits):
            if len(src_logit) == 0 :
                continue
            tc = torch.zeros(src_logit.shape[:1], dtype=torch.int64, device=src_logit.device)
            tc[idx_one] = tc_o
            if self.loss_type == 'focal_loss':
                logit = (src_logit.softmax(-1) - 0.5) * 5
                loss_ce += sigmoid_focal_loss(logit[..., 1], tc.float(), reduction="mean", alpha=0.8, gamma=2)
            if self.loss_type == 'cross_entropy_loss':
                loss_ce += F.cross_entropy(src_logit, tc, self.empty_weight, ignore_index=-1)
        # loss_ce /= len(target_leghths)

        # cum_sum_offsets = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(outputs["point_query_lengths"]), dim=0)[:-1]])
        # cum_sum_offsets = torch.cat([torch.full([len(indices[i][0])], v) for i, v in enumerate(cum_sum_offsets)])
        # idx = idx + cum_sum_offsets
        #
        #
        # target_classes[idx] = target_classes_o
        #
        # # compute classification loss
        # if 'div' in kwargs:
        #     # get sparse / dense image index
        #     den = torch.tensor([target['density'] for target in targets])
        #     den_sort = torch.sort(den)[1]
        #     ds_idx = den_sort[:len(den_sort) // 2]
        #     sp_idx = den_sort[len(den_sort) // 2:]
        #     eps = 1e-5
        #
        #     # raw cross-entropy loss
        #     weights = target_classes.clone().float()
        #     weights[weights == 0] = self.empty_weight[0]
        #     weights[weights == 1] = self.empty_weight[1]
        #     raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')
        #
        #     # binarize split map
        #     split_map = kwargs['div'].view(raw_ce_loss.shape[0], -1)
        #
        #     loss_ce = 0
        #     levels = []
        #     if self.args.loss_mode == 'mode1':
        #         levels = range(0, self.num_levels)
        #     elif self.args.loss_mode == 'mode2':
        #         levels = range(0, self.num_levels // 2) if outputs['pq_stride'] == 4 else range(self.num_levels // 2,
        #                                                                                         self.num_levels)
        #     elif self.args.loss_mode == 'mode3':
        #         levels = range(0, self.num_levels // 2) if outputs['pq_stride'] == 4 else range(0, self.num_levels)
        #
        #     for level in levels:
        #         div_mask = split_map == level
        #         loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / (
        #                 (weights * div_mask)[sp_idx].sum() + eps)
        #         loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / (
        #                 (weights * div_mask)[ds_idx].sum() + eps)
        #         loss_ce_level = loss_ce_sp + loss_ce_ds
        #         loss_ce += loss_ce_level
        #     # div_thrs = self.div_thrs_dict[outputs['pq_stride']]
        #     # div_mask = split_map > div_thrs
        #     #
        #     # # dual supervision for sparse/dense images
        #     # loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
        #     # loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
        #     # loss_ce = loss_ce_sp + loss_ce_ds
        #     #
        #     # # loss on non-div regions
        #     # non_div_mask = split_map <= div_thrs
        #     # loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
        #     # loss_ce = loss_ce + loss_ce_nondiv
        # else:
        #     loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices

        target_leghths = [len(t["labels"][J]) for t, (_, J) in zip(targets, indices)]

        idx = self._get_src_permutation_idx(indices)[1]
        idx = list(idx.split(target_leghths, dim=0))

        target_points = [t['points'][i] for t, (_, i) in zip(targets, indices)]
        pred_points = list(outputs['pred_points'].split(outputs["point_query_lengths"], dim=0))

        img_shape = outputs['img_shape']
        img_h, img_w = img_shape

        losses = {}
        loss_points = 0.
        for idx_one, tgt_point, pred_point in zip(idx, target_points, pred_points):
            if len(tgt_point) == 0:
                continue
            pred_point = pred_point[idx_one]
            tgt_point[:, 0] /= img_h
            tgt_point[:, 1] /= img_w

            loss_raw = F.smooth_l1_loss(pred_point, tgt_point, reduction='none')
            loss_points += loss_raw.sum() / len(tgt_point)

        losses['loss_points'] = loss_points

        # cum_sum_offsets = torch.cat(
        #     [torch.tensor([0]), torch.cumsum(torch.tensor(outputs["point_query_lengths"]), dim=0)[:-1]])
        # cum_sum_offsets = torch.cat([torch.full([len(indices[i][0])], v) for i, v in enumerate(cum_sum_offsets)])
        # idx = idx + cum_sum_offsets
        #
        # src_points = outputs['pred_points'][idx]
        # target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # # compute regression loss
        #
        # loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')
        #
        # if 'div' in kwargs:
        #     # get sparse / dense index
        #     den = torch.tensor([target['density'] for target in targets])
        #     den_sort = torch.sort(den)[1]
        #     img_ds_idx = den_sort[:len(den_sort) // 2]
        #     img_sp_idx = den_sort[len(den_sort) // 2:]
        #     pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
        #     pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])
        #
        #     # dual supervision for sparse/dense images
        #     eps = 1e-5
        #     split_map = kwargs['div']
        #     split_map = split_map.view(split_map.shape[0], -1)
        #     div_thrs = self.div_thrs_dict[outputs['pq_stride']]
        #     div_mask = split_map > div_thrs
        #     loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
        #     loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
        #     loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)
        #
        #     # loss on non-div regions
        #     non_div_mask = split_map <= div_thrs
        #     loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (
        #                 non_div_mask[idx].sum() + eps)
        #
        #     # final point loss
        #     losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        # else:
        #     losses['loss_points'] = loss_points_raw.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


class SetCriterion_P2PNet(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, args):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.args = args
        self.type = args.get('loss_type', 'cross_entropy_loss')
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef  # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        src_points[:, 0] *= img_h
        src_points[:, 1] *= img_w

        loss_points_raw = F.mse_loss(src_points, target_points, reduction='none') # F.smooth_l1_loss(src_points, target_points, reduction='none')
        losses['loss_points'] = loss_points_raw.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses

def build_criterion(args, matcher):
    type_name = args.get('type', 'SetCriterion')

    num_classes = args.num_classes
    eos_coef = args.eos_coef
    losses = ['labels', 'points']
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}

    return eval(type_name)(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=eos_coef, losses=losses, args=args)