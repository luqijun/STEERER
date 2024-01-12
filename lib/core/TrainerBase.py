import torch
import torch.nn as nn
import numpy as np
from .cc_function import get_rank, get_world_size, AverageMeter, \
    allreduce_tensor, save_results_more, reduce_tensor
import time
import logging


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

            result = model(images, label, 'train', args)
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
            #
            scheduler.step_update(epoch * epoch_iters + i_iter)

            lr = optimizer.param_groups[0]['lr']
            gt_cnt, pred_cnt = label[0].sum().item(), pre_den.sum().item()
            if i_iter % config.print_freq == 0 and rank == 0:
                print_loss = avg_loss.average() / world_size
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                      'lr: {:.4f}, Loss: {:.4f}, pre: {:.1f}, gt: {:.1f},' \
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

                if i_iter % 20 * config.print_freq == 0:
                    for t, m, s in zip(image, mean, std):
                        t.mul_(s).add_(m)

                    save_results_more(global_steps, img_vis_dir, image.cpu().data, \
                                      pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                      pre_den[0].sum().item(), label[0][0].sum().item())

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