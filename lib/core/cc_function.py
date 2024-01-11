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

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
def allreduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return  None
    dist.all_reduce(inp,op=dist.ReduceOp.SUM)


def test_cc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None):

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

            losses=result['losses']
            pre_den=result['pre_den']['1']
            gt_den = result['gt_den']['1']
            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item() #pre_data['num'] #

            save_count_txt+='{} {}\n'.format(name[0], pred_cnt)
            # import pdb
            # pdb.set_trace()
            msg = '{} {}' .format(gt_count,pred_cnt)
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
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),pred_cnt,gt_count,
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

    return  mae, mse, nae,save_count_txt

def test_loc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None,loc_gt=None):

    model.eval()
    device = torch.cuda.current_device()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    num_classes = 6
    max_dist_thresh = 100
    metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}

    loc_100_metrics = {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}

    MLE_metric = AverageMeter()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, size_factor, name = batch
            # if name[0] != '1202':
            #     continue

            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            b, c, h, w = image.size()

            result = model(image, label, 'val')
            # result = patch_forward(model, image, label,
            #                        config.test.patch_batch_size, mode='val')
            # import pdb
            # pdb.set_trace()

            losses=result['losses']
            pre_den=result['pre_den']['1']
            # pre_den_x2=result['pre_den']['2']
            pre_den_x4=result['pre_den']['4']
            pre_den_x8=result['pre_den']['8']

            gt_den = result['gt_den']['1']
            # gt_den_x8 = result['gt_den']['8']

            gt_data = loc_gt[int(name[0])]

            pred_data = local_maximum_points(pre_den.detach(),model.gaussian_maximum, patch_size=32,threshold=config.test.loc_threshold)
            # pred_data_x2 = local_maximum_points(pre_den_x2.detach(),model.gaussian_maximum,patch_size=64,den_scale=2)
            pred_data_x4 = local_maximum_points(pre_den_x4.detach(),model.gaussian_maximum,patch_size=32,den_scale=4,threshold=config.test.loc_threshold)
            pred_data_x8 = local_maximum_points(pre_den_x8.detach(),model.gaussian_maximum,patch_size=16,den_scale=8,threshold=config.test.loc_threshold)

            def nms4points(pred_data, pred_data_x8, threshold):
                points = torch.from_numpy(pred_data['points']).unsqueeze(0)
                points_x8 =  torch.from_numpy(pred_data_x8['points']).unsqueeze(0)


                dist = torch.cdist(points,points_x8)     #torch.Size([1, 16, 16])
                dist = dist.squeeze(0)
                min_val, min_idx = torch.min(dist,0)
                keep_idx_bool = (min_val>threshold)


                keep_idx=torch.where(keep_idx_bool==1)[0]
                if keep_idx.size(0)>0:
                    app_points = (pred_data_x8['points'][keep_idx]).reshape(-1,2)
                    pred_data['points'] = np.concatenate([pred_data['points'], app_points],0)
                    pred_data['num'] =  pred_data['num'] +keep_idx_bool.sum().item()
                return pred_data
            #
            # if name[0] == '3613':
            #     import pdb
            #     pdb.set_trace()
            for idx, down_scale in enumerate([pred_data_x4,pred_data_x8]):
                if pred_data['points'].shape[0]==0 and down_scale['points'].shape[0]>0:
                    pred_data = down_scale
                if pred_data['points'].shape[0]>0  and down_scale['points'].shape[0]>0:
                    pred_data = nms4points(pred_data, down_scale,threshold=(2**(idx+1))*16)

            pred_data_4val  = pred_data.copy()
            pred_data_4val['points'] = pred_data_4val['points']/size_factor.numpy()
            tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_loc_F1_boxes(num_classes, pred_data_4val, gt_data)

            tp_100, fp_100, fn_100 =  0,0,0 #eval_loc_F1_point(pred_data['points'],gt_data['points'],max_dist_thresh = max_dist_thresh)
            Distance_Sum = eval_loc_MLE_point(pred_data['points'], gt_data['points'], 16)

            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item() #
            msg = '{}: gt:{} pre:{}' .format(name, gt_count,pred_cnt)
            logger.info(msg)
            # print(name,':', gt_count, pred_cnt)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)

            MLE_metric.update(Distance_Sum/(gt_data['num']+1e-20), gt_data['num'])

            metrics_l['tp'].update(tp_l)
            metrics_l['fp'].update(fp_l)
            metrics_l['fn'].update(fn_l)
            metrics_l['tp_c'].update(tp_c_l)
            metrics_l['fn_c'].update(fn_c_l)

            metrics_s['tp'].update(tp_s)
            metrics_s['fp'].update(fp_s)
            metrics_s['fn'].update(fn_s)
            metrics_s['tp_c'].update(tp_c_s)
            metrics_s['fn_c'].update(fn_c_s)

            loc_100_metrics['tp_100'].update(tp_100)
            loc_100_metrics['fp_100'].update(fp_100)
            loc_100_metrics['fn_100'].update(fn_100)


            image = image[0]
            if sv_pred:
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)

                save_results_more(name, sv_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),pred_cnt,gt_count,
                                  pred_data['points'],gt_data['points']*size_factor.numpy() )

        # confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        # reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        #
        # confusion_matrix = reduced_confusion_matrix.cpu().numpy()
        # pos = confusion_matrix.sum(1)
        # res = confusion_matrix.sum(0)
        # tp = np.diag(confusion_matrix)
        # IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # mean_IoU = IoU_array.mean()
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mae = cnt_errors['mae'].avg
                mse = np.sqrt(cnt_errors['mse'].avg)
                nae = cnt_errors['nae'].avg
                msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                       nae: {: 4.4f}, Class IoU: '.format(mae,
                                                          mse, nae)
                logging.info(msg)

        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l + 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)

        ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
        ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
        f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s)
        ar_c_s = metrics_s['tp_c'].sum / (metrics_s['tp_c'].sum + metrics_s['fn_c'].sum + 1e-20)

        pre_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum + loc_100_metrics['fp_100'].sum + 1e-20)
        rec_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum + loc_100_metrics['fn_100'].sum + 1e-20)  # True pos rate
        f1_100 = 2 * (pre_100 * rec_100) / (pre_100 + rec_100 + + 1e-20)

        logging.info('-----Localization performance with box annotations-----')
        logging.info('AP_small: '+str(ap_s))
        logging.info('AR_small: '+str(ar_s))
        logging.info('F1m_small: '+str(f1m_s))
        logging.info('AR_small_category: '+str(ar_c_s))
        logging.info('    avg: '+str(ar_c_s.mean()))
        logging.info('AP_large: '+str(ap_l))
        logging.info('AR_large: '+str(ar_l))
        logging.info('F1m_large: '+str(f1m_l))
        logging.info('AR_large_category: '+str(ar_c_l))
        logging.info('    avg: '+str(ar_c_l.mean()))

        logging.info('-----Localization performance with points annotations-----')
        logging.info('avg precision_overall:{}'.format(pre_100.mean()))
        logging.info('avg recall_overall:{}'.format(rec_100.mean()))
        logging.info('avg F1_overall:{}'.format(f1_100.mean()))
        logging.info('Mean Loclization Error:{}'.format(MLE_metric.avg))


        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

        logging.info('-----Counting performance-----')
        logging.info('MAE: ' + str(mae))
        logging.info('MSE: ' + str(mse))
        logging.info('NAE: ' + str(nae))
            # pred = test_dataset.multi_scale_inference(
            #             model,
            #             image,
            #             scales=config.TEST.SCALE_LIST,
            #             flip=config.TEST.FLIP_TEST)
            #

    return  mae, mse, nae

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
