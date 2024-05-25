# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from lib.utils.utils import *
from lib.utils.points_from_den import local_maximum_points
from lib.eval.eval_loc_count import eval_loc_MLE_point, eval_loc_F1_boxes

def build_tester(config):
    name = config.test.get('tester', 'TesterBase')
    return eval(name)

def get_seg_map_result(result, key, level_ley):
    seg =  result.get(key, None)
    if seg is not None:
        return seg[level_ley]
    return None

class TesterBase:

    def test_cc(config, test_dataset, testloader, model
                , mean, std, sv_dir='', sv_pred=False, logger=None):

        model.eval()
        save_count_txt = ''
        device = torch.cuda.current_device()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
        with torch.no_grad():
            for index, batch in enumerate(tqdm(testloader)):
                image, label, _, name = batch

                image, label, _, name = batch
                image = image.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)

                result = model(image, label, 'val')

                # result = patch_forward(model, image, label,
                #                                       config.test.patch_batch_size, mode='val')

                losses = result['losses']
                pre_den = result['pre_den']['1']
                gt_den = result['gt_den']['1']
                #    -----------Counting performance------------------
                gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item()  # pre_data['num'] #

                save_count_txt += '{} {}\n'.format(name[0], pred_cnt)
                # import pdb
                # pdb.set_trace()
                msg = '{} {}'.format(gt_count, pred_cnt)
                logger.info(msg)
                s_mae = abs(gt_count - pred_cnt)
                s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                cnt_errors['mae'].update(s_mae)
                cnt_errors['mse'].update(s_mse)
                if gt_count != 0:
                    s_nae = (abs(gt_count - pred_cnt) / gt_count)
                    cnt_errors['nae'].update(s_nae)

                image = image[0]
                if sv_pred:
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)
                    save_results_more(name, sv_dir, image.cpu().data, \
                                      pre_den[0].detach().cpu(), gt_den[0].detach().cpu(), pred_cnt, gt_count,
                                      )

                if index % 100 == 0:
                    logging.info('processing: %d images' % index)
                    mae = cnt_errors['mae'].avg
                    mse = np.sqrt(cnt_errors['mse'].avg)
                    nae = cnt_errors['nae'].avg
                    msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                           nae: {: 4.4f}, Class IoU: '.format(mae,
                                                              mse, nae)
                    logging.info(msg)
            mae = cnt_errors['mae'].avg
            mse = np.sqrt(cnt_errors['mse'].avg)
            nae = cnt_errors['nae'].avg

        return mae, mse, nae, save_count_txt


class Tester_Points:

    def test_cc(config, test_dataset, testloader, model
                , mean, std, sv_dir='', sv_pred=False, logger=None):

        model.eval()
        save_count_txt = ''
        device = torch.cuda.current_device()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
        with torch.no_grad():
            for index, batch in enumerate(tqdm(testloader)):

                image, label, targets, ratio, name, *args = batch
                image = image.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)

                kwargs = {}
                kwargs["epoch"] = -1
                kwargs["targets"] = targets
                result = model(image, label, 'val', *args, **kwargs)
                outputs = result["outputs"]
                targets = result["targets"]

                outputs_points = outputs['pred_points']
                outputs_offsets = outputs['pred_offsets']

                thrs = 0.5
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[..., 1]
                valid_indexs = out_scores > thrs
                out_scores = out_scores[valid_indexs]
                pre_points = outputs_points[valid_indexs]

                #    -----------Counting performance-----------------
                losses = result['losses']

                # process predicted points
                pred_cnt = torch.tensor(len(out_scores))
                gt_cnt = torch.tensor(targets['points'][0].shape[0])

                # compute error
                mae = abs(pred_cnt - gt_cnt)
                mse = (pred_cnt - gt_cnt) * (pred_cnt - gt_cnt)

                # print(f" rank: {rank} gt: {gt_count} pre: {pred_cnt}")
                s_mae = mae
                s_mse = mse

                save_count_txt += '{} {}\n'.format(name[0], pred_cnt)

                msg = '{} {}'.format(gt_cnt, pred_cnt)
                logger.info(msg)
                cnt_errors['mae'].update(s_mae)
                cnt_errors['mse'].update(s_mse)
                if gt_cnt != 0:
                    s_nae = (abs(gt_cnt - pred_cnt) / gt_cnt)
                    cnt_errors['nae'].update(s_nae)

                if sv_pred:

                    vis_dir = os.path.join(sv_dir, 'vis')
                    os.makedirs(vis_dir, exist_ok=True)

                    image = image[0]
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)

                    gt_points = targets['points'][0]
                    pre_seg_crowd = get_seg_map_result(result, 'pre_seg_crowd', '1')
                    gt_seg_crowd = get_seg_map_result(result, 'gt_seg_crowd', '1')
                    save_results_points_with_seg_map(name[0], vis_dir, image.cpu().data, \
                                                     pre_points, gt_points,
                                                     pre_seg_map0=pre_seg_crowd[0].detach().cpu(),
                                                     gt_seg_map0=gt_seg_crowd[0].detach().cpu())

                if index % 100 == 0:
                    logging.info('processing: %d images' % index)
                    mae = cnt_errors['mae'].avg
                    mse = np.sqrt(cnt_errors['mse'].avg)
                    nae = cnt_errors['nae'].avg
                    msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                           nae: {: 4.4f}, Class IoU: '.format(mae,
                                                              mse, nae)
                    logging.info(msg)
            mae = cnt_errors['mae'].avg
            mse = np.sqrt(cnt_errors['mse'].avg)
            nae = cnt_errors['nae'].avg

        return mae, mse, nae, save_count_txt