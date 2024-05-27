import torch
import torch.nn as nn
import numpy as np
from .cc_function import get_rank, get_world_size, AverageMeter, \
    allreduce_tensor, save_results_more_with_seg_map, reduce_tensor, save_results_points_with_seg_map
import time
import logging

from ..models.networks.layers import NestedTensor


def build_trainer(config):
    name = config.train.get('trainer', None)
    if not name:
        name = "TrainerBase"
    return eval(name)()


class TrainerBase:
    def train(self, config, epoch, num_epoch, epoch_iters, num_iters,
              trainloader, optimizer, scheduler, model, writer_dict, device, img_vis_dir, mean, std, task_KPI,
              train_dataset):

        # Training
        model.train()
        batch_time = AverageMeter()
        avg_loss = AverageMeter()
        tic = time.time()
        cur_iters = epoch * epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        rank = get_rank()
        world_size = get_world_size()

        for i_iter, batch in enumerate(trainloader):
            images, label, size, name_idx, *args = batch
            images = images.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            result = model(images, label, 'train', *args)
            losses = result['losses']
            # import pdb
            # pdb.set_trace()
            pre_den = result['pre_den']['1']
            gt_den = result['gt_den']['1']

            for i in range(len(name_idx[0])):

                _name = name_idx[0][i]
                if _name not in train_dataset.resize_memory_pool.keys():
                    p_h = int(np.ceil(size[i][0] / config.train.route_size[0]))
                    p_w = int(np.ceil(size[i][1] / config.train.route_size[1]))
                    train_dataset.resize_memory_pool.update({_name: {"avg_size": np.ones((p_h, p_w)),
                                                                     "load_num": np.zeros((p_h, p_w)),
                                                                     'size': np.array(size)}})

            loss = losses.mean()

            model.zero_grad()
            loss.backward()
            optimizer.step()

            task_KPI.add({
                'acc1': {'gt': result['acc1']['gt'], 'error': result['acc1']['error']},
                'x4': {'gt': result['x4']['gt'], 'error': result['x4']['error']},
                'x8': {'gt': result['x8']['gt'], 'error': result['x8']['error']},
                'x16': {'gt': result['x16']['gt'], 'error': result['x16']['error']},
                'x32': {'gt': result['x32']['gt'], 'error': result['x32']['error']}

            })

            KPI = task_KPI.query()
            reduced_loss = reduce_tensor(loss)
            x4_acc = reduce_tensor(KPI['x4']) / world_size
            x8_acc = reduce_tensor(KPI['x8']) / world_size
            x16_acc = reduce_tensor(KPI['x16']) / world_size
            x32_acc = reduce_tensor(KPI['x32']) / world_size
            acc1 = reduce_tensor(KPI['acc1']) / world_size
            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss
            avg_loss.update(reduced_loss.item())
            # lr rate
            no_adjust = config.lr_config.get('no_adjust', False)
            if not no_adjust:
                scheduler.step_update(epoch * epoch_iters + i_iter)

            lr = optimizer.param_groups[0]['lr']
            gt_cnt, pred_cnt = label[0].sum().item(), pre_den.sum().item()
            if i_iter % config.print_freq == 0 and rank == 0:
                print_loss = avg_loss.average() / world_size
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.4f}, Loss: {:.8f}, pre: {:.1f}, gt: {:.1f},' \
                      'acc:{:.2f}, accx8:{:.2f},  accx16:{:.2f},accx32:{:.2f},acc1:{:.2f}'.format(
                    epoch, num_epoch, i_iter, epoch_iters,
                    batch_time.average(), lr * 1e5, print_loss,
                    pred_cnt, gt_cnt,
                    x4_acc.item(), x8_acc.item(), x16_acc.item(), x32_acc.item(), acc1.item())
                logging.info(msg)

                writer.add_scalar('train_loss', print_loss, global_steps)
                global_steps = writer_dict['train_global_steps']
                writer_dict['train_global_steps'] = global_steps + 1
                image = images[0]

                save_vis = config.train.get('save_vis', False)
                save_vis_freq = config.train.get('save_vis_freq', 20)
                if save_vis and i_iter % save_vis_freq == 0:
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)

                    pre_seg_crowd = get_seg_map_result(result, 'pre_seg_crowd')
                    gt_seg_crowd = get_seg_map_result(result, 'gt_seg_crowd')
                    pre_seg_level = get_seg_map_result(result, 'pre_seg_level')
                    gt_seg_level = get_seg_map_result(result, 'gt_seg_level')
                    if pre_seg_level is None:
                        pre_seg_level = torch.zeros_like(pre_seg_crowd)
                    if gt_seg_level is None:
                        gt_seg_level = torch.zeros_like(gt_seg_crowd)
                    save_results_more_with_seg_map(global_steps, img_vis_dir, image.cpu().data, \
                                                   pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                                   pre_den[0].sum().item(), label[0][0].sum().item(),
                                                   pre_seg_map0=pre_seg_crowd[0].detach().cpu(),
                                                   gt_seg_map0=gt_seg_crowd[0].detach().cpu(),
                                                   pre_seg_level_map0=pre_seg_level[0].detach().cpu(),
                                                   gt_seg_level_map0=gt_seg_level[0].detach().cpu())


def get_seg_map_result(result, key):
    seg =  result.get(key, None)
    if seg is not None:
        return seg['1']
    return None

# class Trainer_Adaptive_Kernel:
#     def train(config, epoch, num_epoch, epoch_iters, num_iters,
#               trainloader, optimizer, scheduler, model, writer_dict, device, img_vis_dir, mean, std, task_KPI,
#               train_dataset):
#
#         # Training
#         model.train()
#         batch_time = AverageMeter()
#         avg_loss = AverageMeter()
#         tic = time.time()
#         cur_iters = epoch * epoch_iters
#         writer = writer_dict['writer']
#         global_steps = writer_dict['train_global_steps']
#         rank = get_rank()
#         world_size = get_world_size()
#
#         for i_iter, batch in enumerate(trainloader):
#             images, label, size, name_idx, *args = batch
#             images = images.to(device)
#             for i in range(len(label)):
#                 label[i] = label[i].to(device)
#
#             result = model(images, label, 'train', args)
#             losses = result['losses']
#             # import pdb
#             # pdb.set_trace()
#             pre_den = result['pre_den']['1']
#             gt_den = result['gt_den']['1']
#
#             for i in range(len(name_idx[0])):
#
#                 _name = name_idx[0][i]
#
#                 if _name not in train_dataset.resize_memory_pool.keys():
#                     p_h = int(np.ceil(size[i][0] / config.train.route_size[0]))
#                     p_w = int(np.ceil(size[i][1] / config.train.route_size[1]))
#                     train_dataset.resize_memory_pool.update({_name: {"avg_size": np.ones((p_h, p_w)),
#                                                                      "load_num": np.zeros((p_h, p_w)),
#                                                                      'size': np.array(size)}})
#
#             loss = losses.mean()
#
#             model.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             task_KPI.add({
#                 'acc1': {'gt': result['acc1']['gt'], 'error': result['acc1']['error']},
#                 'x4': {'gt': result['x4']['gt'], 'error': result['x4']['error']},
#                 'x8': {'gt': result['x8']['gt'], 'error': result['x8']['error']},
#                 'x16': {'gt': result['x16']['gt'], 'error': result['x16']['error']},
#                 'x32': {'gt': result['x32']['gt'], 'error': result['x32']['error']}
#
#             })
#
#             KPI = task_KPI.query()
#             reduced_loss = reduce_tensor(loss)
#             x4_acc = reduce_tensor(KPI['x4']) / world_size
#             x8_acc = reduce_tensor(KPI['x8']) / world_size
#             x16_acc = reduce_tensor(KPI['x16']) / world_size
#             x32_acc = reduce_tensor(KPI['x32']) / world_size
#             acc1 = reduce_tensor(KPI['acc1']) / world_size
#             # measure elapsed time
#             batch_time.update(time.time() - tic)
#             tic = time.time()
#
#             # update average loss
#             avg_loss.update(reduced_loss.item())
#             #
#             scheduler.step_update(epoch * epoch_iters + i_iter)
#
#             lr = optimizer.param_groups[0]['lr']
#             gt_cnt, pred_cnt = label[0].sum().item(), pre_den.sum().item()
#             if i_iter % config.print_freq == 0 and rank == 0:
#                 print_loss = avg_loss.average() / world_size
#                 msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
#                       'lr: {:.4f}, Loss: {:.4f}, pre: {:.1f}, gt: {:.1f},' \
#                       'acc:{:.2f}, accx8:{:.2f},  accx16:{:.2f},accx32:{:.2f},acc1:{:.2f}'.format(
#                     epoch, num_epoch, i_iter, epoch_iters,
#                     batch_time.average(), lr * 1e5, print_loss,
#                     pred_cnt, gt_cnt,
#                     x4_acc.item(), x8_acc.item(), x16_acc.item(), x32_acc.item(), acc1.item())
#                 logging.info(msg)
#
#                 writer.add_scalar('train_loss', print_loss, global_steps)
#                 global_steps = writer_dict['train_global_steps']
#                 writer_dict['train_global_steps'] = global_steps + 1
#                 image = images[0]
#
#                 if i_iter % 20 * config.print_freq == 0:
#                     for t, m, s in zip(image, mean, std):
#                         t.mul_(s).add_(m)
#
#                     save_results_more(global_steps, img_vis_dir, image.cpu().data, \
#                                       pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
#                                       pre_den[0].sum().item(), label[0][0].sum().item())

# 返回points
class Trainer_Points:
    def train(self, config, epoch, num_epoch, epoch_iters, num_iters,
              trainloader, optimizer, scheduler, model, writer_dict, device, img_vis_dir, mean, std, task_KPI,
              train_dataset):

        # Training
        model.train()
        batch_time = AverageMeter()
        avg_loss = AverageMeter()
        tic = time.time()
        cur_iters = epoch * epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        rank = get_rank()
        world_size = get_world_size()

        for i_iter, batch in enumerate(trainloader):
            images, label, targets, size, name_idx, *args = batch
            images = images.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            kwargs = {}
            kwargs["epoch"] = epoch
            kwargs["targets"] = targets
            result = model(images, label, 'train', *args, **kwargs)
            losses = result['losses']
            # import pdb
            # pdb.set_trace()
            # pre_den = result['pre_den']['1']
            # gt_den = result['gt_den']['1']

            # for i in range(len(name_idx[0])):
            #
            #     _name = name_idx[0][i]
            #     if _name not in train_dataset.resize_memory_pool.keys():
            #         p_h = int(np.ceil(size[i][0] / config.train.route_size[0]))
            #         p_w = int(np.ceil(size[i][1] / config.train.route_size[1]))
            #         train_dataset.resize_memory_pool.update({_name: {"avg_size": np.ones((p_h, p_w)),
            #                                                          "load_num": np.zeros((p_h, p_w)),
            #                                                          'size': np.array(size)}})

            loss = losses.mean()

            model.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()

            reduced_loss = reduce_tensor(loss)
            # update average loss
            avg_loss.update(reduced_loss.item())
            # continue

            outputs = result["outputs"]
            targets = result["targets"]

            outputs_points = outputs['pred_points']
            outputs_offsets = outputs['pred_offsets']

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # lr rate
            no_adjust = config.lr_config.get('no_adjust', False)
            if not no_adjust:
                scheduler.step_update(epoch * epoch_iters + i_iter)
            # lr = optimizer.param_groups[0]['lr']
            # if not no_adjust or epoch <= config.lr_config.WARMUP_EPOCHS \
            #         or lr !=config.optimizer.BASE_LR:


            thrs = 0.5
            if 'point_query_lengths' in outputs:
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[..., 1]
                valid_indexs = out_scores > thrs
                valid_indexs_0 = valid_indexs.split(outputs['point_query_lengths'], 0)[0]
                out_scores_0 = out_scores[:len(valid_indexs_0)][valid_indexs_0]
                pre_points = outputs_points[:len(valid_indexs_0)][valid_indexs_0]
            else:
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'][0], -1)[..., 1]
                valid_indexs = out_scores > thrs
                out_scores_0 = out_scores[valid_indexs]
                pre_points = outputs_points[0][valid_indexs]

            lr = optimizer.param_groups[0]['lr']
            gt_cnt, pred_cnt =targets[0]['points'].shape[0], len(out_scores_0) # label[0].sum().item(), pre_den.sum().item()
            if i_iter % config.print_freq == 0 and rank == 0:
                print_loss = avg_loss.average() / world_size
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.6f}, Loss: {:.6f}, loss_ce:{:.6f}, loss_points:{:.6f}, pre: {:.1f}, gt: {:.1f}'.format(
                    epoch, num_epoch, i_iter, epoch_iters,
                    batch_time.average(), lr / config.optimizer.BASE_LR, print_loss, result['loss_dict']['loss_ce'], result['loss_dict']['loss_points'],
                    pred_cnt, gt_cnt)
                logging.info(msg)

                writer.add_scalar('train_loss', print_loss, global_steps)
                global_steps = writer_dict['train_global_steps']
                writer_dict['train_global_steps'] = global_steps + 1

                image = images[0]

                save_vis = config.train.get('save_vis', False)
                save_vis_freq = config.train.get('save_vis_freq', 20)
                if save_vis and i_iter % save_vis_freq == 0:
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)

                    gt_points = targets[0]['points']
                    pre_seg_crowd = get_seg_map_result(result, 'pre_seg_crowd')
                    gt_seg_crowd = get_seg_map_result(result, 'gt_seg_crowd')
                    save_results_points_with_seg_map(config, global_steps, img_vis_dir, image.cpu().data, \
                                                   pre_points, gt_points,
                                                   pre_seg_map0= None if pre_seg_crowd is None else pre_seg_crowd[0].detach().cpu()  ,
                                                   gt_seg_map0= None if gt_seg_crowd is None else gt_seg_crowd[0].detach().cpu())


class Trainer_Points_P2PNet:
    def train(self, config, epoch, num_epoch, epoch_iters, num_iters,
              trainloader, optimizer, scheduler, model, writer_dict, device, img_vis_dir, mean, std, task_KPI,
              train_dataset):

        # Training
        model.train()
        batch_time = AverageMeter()
        avg_loss = AverageMeter()
        tic = time.time()
        cur_iters = epoch * epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        rank = get_rank()
        world_size = get_world_size()

        for i_iter, batch in enumerate(trainloader):
            samples, targets, *args = batch
            images = samples.to(device)

            kwargs = {}
            kwargs["epoch"] = epoch
            kwargs["targets"] = targets
            label_maps = [tgt['label_map'] for tgt in kwargs["targets"]]
            kwargs["label_map"] = [torch.stack(lm, dim=0).cuda() for lm in zip(*label_maps)]
            result = model(images, kwargs["label_map"], 'train', *args, **kwargs)
            losses = result['losses']
            loss = losses.mean()

            model.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()

            reduced_loss = reduce_tensor(loss)
            # update average loss
            avg_loss.update(reduced_loss.item())
            # continue

            outputs = result["outputs"]
            targets = result["targets"]

            outputs_points = outputs['pred_points']
            outputs_offsets = outputs['pred_offsets']

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # lr rate
            no_adjust = config.lr_config.get('no_adjust', False)
            if not no_adjust:
                scheduler.step_update(epoch * epoch_iters + i_iter)

            thrs = 0.5
            if 'point_query_lengths' in outputs:
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[..., 1]
                valid_indexs = out_scores > thrs
                valid_indexs_0 = valid_indexs.split(outputs['point_query_lengths'], 0)[0]
                out_scores_0 = out_scores[:len(valid_indexs_0)][valid_indexs_0]
                pre_points = outputs_points[:len(valid_indexs_0)][valid_indexs_0]
            else:
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'][0], -1)[..., 1]
                valid_indexs = out_scores > thrs
                out_scores_0 = out_scores[valid_indexs]
                pre_points = outputs_points[0][valid_indexs]

            lr = optimizer.param_groups[0]['lr']
            gt_cnt, pred_cnt =targets[0]['points'].shape[0], len(out_scores_0) # label[0].sum().item(), pre_den.sum().item()
            if i_iter % config.print_freq == 0 and rank == 0:
                print_loss = avg_loss.average() / world_size
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.6f}, Loss: {:.6f}, loss_ce:{:.6f}, loss_points:{:.6f}, pre: {:.1f}, gt: {:.1f}'.format(
                    epoch, num_epoch, i_iter, epoch_iters,
                    batch_time.average(), lr / config.optimizer.BASE_LR, print_loss, result['loss_dict']['loss_ce'], result['loss_dict']['loss_points'],
                    pred_cnt, gt_cnt)
                logging.info(msg)

                writer.add_scalar('train_loss', print_loss, global_steps)
                global_steps = writer_dict['train_global_steps']
                writer_dict['train_global_steps'] = global_steps + 1

                image = images[0]

                save_vis = config.train.get('save_vis', False)
                save_vis_freq = config.train.get('save_vis_freq', 20)
                if save_vis and i_iter % save_vis_freq == 0:
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)

                    gt_points = targets[0]['points']
                    pre_seg_crowd = get_seg_map_result(result, 'pre_seg_crowd')
                    gt_seg_crowd = get_seg_map_result(result, 'gt_seg_crowd')
                    save_results_points_with_seg_map(config, global_steps, img_vis_dir, image.cpu().data, \
                                                   pre_points, gt_points,
                                                   pre_seg_map0= None if pre_seg_crowd is None else pre_seg_crowd[0].detach().cpu()  ,
                                                   gt_seg_map0= None if gt_seg_crowd is None else gt_seg_crowd[0].detach().cpu())


class Trainer_Points_PET:
    def train(self, config, epoch, num_epoch, epoch_iters, num_iters,
              trainloader, optimizer, scheduler, model, writer_dict, device, img_vis_dir, mean, std, task_KPI,
              train_dataset):

        # Training
        model.train()
        batch_time = AverageMeter()
        avg_loss = AverageMeter()
        tic = time.time()
        cur_iters = epoch * epoch_iters
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        rank = get_rank()
        world_size = get_world_size()

        for i_iter, batch in enumerate(trainloader):
            samples, targets, *args = batch
            images = samples.to(device)

            kwargs = {}
            kwargs["epoch"] = epoch
            kwargs["targets"] = targets
            label_maps = [tgt['label_map'] for tgt in kwargs["targets"]]
            kwargs["label_map"] = [torch.stack(lm, dim=0).cuda() for lm in zip(*label_maps)]
            result = model(images, kwargs["label_map"], 'train', *args, **kwargs)
            losses = result['losses']
            loss = losses.mean()

            model.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()

            reduced_loss = reduce_tensor(loss)
            # update average loss
            avg_loss.update(reduced_loss.item())
            # continue

            outputs = result["outputs"]
            targets = result["targets"]

            outputs_points = outputs['pred_points']
            # outputs_offsets = outputs['pred_offsets']

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # lr rate
            no_adjust = config.lr_config.get('no_adjust', False)
            if not no_adjust:
                scheduler.step_update(epoch * epoch_iters + i_iter)

            thrs = 0.5
            if 'point_query_lengths' in outputs:
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[..., 1]
                valid_indexs = out_scores > thrs
                valid_indexs_0 = valid_indexs.split(outputs['point_query_lengths'], 0)[0]
                out_scores_0 = out_scores[:len(valid_indexs_0)][valid_indexs_0]
                pre_points = outputs_points[:len(valid_indexs_0)][valid_indexs_0]
            else:
                out_scores = torch.nn.functional.softmax(outputs['pred_logits'][0], -1)[..., 1]
                valid_indexs = out_scores > thrs
                out_scores_0 = out_scores[valid_indexs]
                pre_points = outputs_points[0][valid_indexs]

            lr = optimizer.param_groups[0]['lr']
            gt_cnt, pred_cnt =targets[0]['points'].shape[0], len(out_scores_0) # label[0].sum().item(), pre_den.sum().item()
            if i_iter % config.print_freq == 0 and rank == 0:
                print_loss = avg_loss.average() / world_size
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.6f}, Loss: {:.6f}, loss_ce:{:.6f}, loss_points:{:.6f}, pre: {:.1f}, gt: {:.1f}'.format(
                    epoch, num_epoch, i_iter, epoch_iters,
                    batch_time.average(), lr / config.optimizer.BASE_LR, print_loss, result['loss_dict']['loss_ce'], result['loss_dict']['loss_points'],
                    pred_cnt, gt_cnt)
                logging.info(msg)

                writer.add_scalar('train_loss', print_loss, global_steps)
                global_steps = writer_dict['train_global_steps']
                writer_dict['train_global_steps'] = global_steps + 1

                image = images.tensors[0] if  isinstance(images, NestedTensor) else images[0]

                save_vis = config.train.get('save_vis', False)
                save_vis_freq = config.train.get('save_vis_freq', 20)
                if save_vis and i_iter % save_vis_freq == 0:
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)

                    gt_points = targets[0]['points']
                    pre_seg_crowd = get_seg_map_result(result, 'pre_seg_crowd')
                    gt_seg_crowd = get_seg_map_result(result, 'gt_seg_crowd')
                    save_results_points_with_seg_map(config, global_steps, img_vis_dir, image.cpu().data, \
                                                   pre_points, gt_points,
                                                   pre_seg_map0= None if pre_seg_crowd is None else pre_seg_crowd[0].detach().cpu()  ,
                                                   gt_seg_map0= None if gt_seg_crowd is None else gt_seg_crowd[0].detach().cpu())