# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import json
import os
import _init_paths
import pprint
import logging
import timeit
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from lib.utils.dist_utils import (
    get_dist_info,
    init_dist)
from mmcv import Config, DictAction
import lib.datasets as datasets
from lib.core.criterion import MSE, CrossEntropy, OhemCrossEntropy
from lib.core.TrainerBase import build_trainer
from lib.core.ValidatorBase import build_validator
from lib.utils.modelsummary import get_model_summary
from lib.utils.KPI_pool import Task_KPI_Pool
from lib.utils.utils import create_logger, random_seed_setting, copy_cur_env
from lib.core.Counter import *
from bisect import bisect_right
from lib.datasets.build_dataset import build_dataset
from lib.datasets.utils.collate import default_collate
from lib.models.build_counter import build_counter
import argparse
from lib.solver.build import build_optimizer_cls
from lib.solver.lr_scheduler_cls import build_scheduler
from fvcore.nn.flop_count import flop_count

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Crowd Counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi','torchrun'],
                        default='none',
                        help='job launcher')
    parser.add_argument(
                        '--debug',
                        type=bool,
                        default=False,
                        help='set debug mode to avoid copying the current codebase')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    config = Config.fromfile(args.cfg)

    # cudnn related and random seed setting
    random_seed_setting(config)

    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)

    logger, train_log_dir = create_logger(
        config, args.cfg, 'train')



    writer_dict = {
        'writer': SummaryWriter(train_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    logger.info('GPU idx:'+os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = config.gpus
    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://",
        # )
        init_dist(args.launcher)
        # torch.cuda.set_device(args.local_rank)
        if args.launcher == 'pytorch':
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank, world_size = get_dist_info()
            args.local_rank = rank
    device = torch.device('cuda:{}'.format(args.local_rank))
    config.device = device

    # build model

    model = build_counter(config)

      # provide the summary of model
    if args.local_rank == 0:
        logger.info(pprint.pformat(args))
        logger.info(config)

        from fvcore.nn.flop_count import flop_count
        from fvcore.nn.parameter_count import parameter_count_table
        print(parameter_count_table(model))
        h, w = config.train.image_size
        dump_input = torch.rand(
            (1, 3, h, w)
        ).cuda()
        logger.info(flop_count(model.cuda(), dump_input.cuda(),))
        # import pdb
        # pdb.set_trace()

        if config.train.resume_path is None and args.debug is False:
            work_dir = os.path.join(os.path.dirname(__file__), '../')
            backup_dir = os.path.join(train_log_dir, 'code')
            copy_cur_env(work_dir, backup_dir, ['exp'])

    optimizer = build_optimizer_cls(config.optimizer, model)
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)

    best_mae = 1e20
    best_mse = 1e20
    last_epoch = 0

    if config.train.resume_path is not None:
        model_state_file = os.path.join(config.train.resume_path,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mae = checkpoint['best_MAE']
            best_mse = checkpoint['best_MSE']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'],strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    # prepare data
    train_dataset, test_dataset = build_dataset(config)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size_per_gpu,
        shuffle=config.train.shuffle and train_sampler is None,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=default_collate,
        sampler=train_sampler)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        collate_fn=default_collate,
        sampler=test_sampler)


        # torch.optim.SGD([{'params':
        #                               filter(lambda p: p.requires_grad,
        #                                      model.parameters()),
        #                               'lr': config.train.LR}],
        #                             lr=config.train.LR,
        #                             momentum=config.train.MOMENTUM,
        #                             weight_decay=config.train.WD,
        #                             nesterov=config.train.NESTEROV,
        #                             )


    epoch_iters = len(trainloader)
    start = timeit.default_timer()
    end_epoch = config.train.end_epoch + config.train.extra_epoch
    num_iters = config.train.end_epoch * epoch_iters


    if config.dataset.extra_train_set:
        epoch_iters = len(trainloader)
        extra_iters = config.train.extra_epoch * epoch_iters

    task_KPI = Task_KPI_Pool(
        task_setting={
            'x4': ['gt', 'error'],
            'x8': ['gt', 'error'],
            'x16': ['gt', 'error'],
            'x32': ['gt', 'error'],
            'acc1': ['gt', 'error']},
        maximum_sample=1000, device=device)
    scheduler = build_scheduler(config.lr_config, optimizer, epoch_iters, config.train.end_epoch)
    trainer = build_trainer(config)
    validator = build_validator(config)
    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= config.train.end_epoch:
            trainer.train(config, epoch - config.train.end_epoch,
                  config.train.extra_epoch, epoch_iters,
                  extra_iters,
                  extra_trainloader, optimizer, scheduler ,
            model, writer_dict, device)
        else:
            vis_train_dir = train_log_dir + '/vis_train'
            os.makedirs(vis_train_dir, exist_ok=True)
            trainer.train(config, epoch, config.train.end_epoch,
                  epoch_iters, num_iters,
                  trainloader, optimizer,scheduler, model, writer_dict,
                  device, vis_train_dir, train_dataset.mean,
                  train_dataset.std, task_KPI,train_dataset)
        if epoch >=5:
            train_dataset.AI_resize =False
        if (epoch+1) % bisect_right(config.train.val_span, -epoch) == 0:
            vis_val_dir = train_log_dir + '/vis_val'
            os.makedirs(vis_val_dir, exist_ok=True)
            valid_loss, mae, mse, nae = validator.validate(config,
                                                 testloader, model, writer_dict, device,
                                                 config.test.patch_batch_size,
                                                 vis_val_dir, test_dataset.mean,
                                                 test_dataset.std)
            with open(train_log_dir + '/scale_statistic.json', mode='w') as f:
                f.write(json.dumps(train_dataset.resize_memory_pool, cls=NpEncoder))

            if args.local_rank == 0:
                logger.info('=> saving checkpoint to {}'.format(
                    train_log_dir + 'checkpoint.pth.tar'))

                def save_checkpoint(filename):
                    torch.save({
                        'epoch': epoch + 1,
                        'best_MAE': best_mae,
                        'best_MSE': best_mse,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(train_log_dir, filename))

                save_checkpoint('checkpoint.pth.tar')

                pth_file_name = os.path.join(
                    train_log_dir,
                    'Ep_' + str(epoch) + '_mae_' + str(mae) + '_mse_' + str(mse) + '.pth')

                if mae < best_mae:
                    best_mae = mae
                    torch.save(model.module.state_dict(), pth_file_name)
                    save_checkpoint('checkpoint.pth.tar_best_mae')
                    best_mae_vis_val_dir = train_log_dir + '/vis_val_best_mae'
                    copy_cur_env(vis_val_dir, best_mae_vis_val_dir, [])

                if mse < best_mse:
                    best_mse = mse
                    torch.save(model.module.state_dict(), pth_file_name)
                    save_checkpoint('checkpoint.pth.tar_best_mse')
                    best_mse_vis_val_dir = train_log_dir + '/vis_val_best_mse'
                    copy_cur_env(vis_val_dir, best_mse_vis_val_dir, [])


                # if mae < best_mae:
                #     best_mae = mae
                #     torch.save(
                #         model.module.state_dict(),
                #         os.path.join(
                #             train_log_dir,
                #             'Ep_' +
                #             str(epoch) +
                #             '_mae_' +
                #             str(mae) +
                #             '_mse_' +
                #             str(mse) +
                #             '.pth'))
                # if mse < best_mse:
                #     best_mse = mse
                #     torch.save(
                #         model.module.state_dict(),
                #         os.path.join(
                #             train_log_dir,
                #             'Ep_' +
                #             str(epoch) +
                #             '_mae_' +
                #             str(mae) +
                #             '_mse_' +
                #             str(mse) +
                #             '.pth'))


                msg = 'Loss: {:.3f}, MAE: {: 4.2f}, Best_MAE: {: 4.4f} ' \
                      'MSE: {: 4.4f},Best_MSE: {: 4.4f}'.format(
                          valid_loss, mae, best_mae, mse, best_mse)
                logging.info(msg)
                # logging.info(IoU_array)

                if epoch == end_epoch - 1:
                    torch.save(model.module.state_dict(),
                               os.path.join(train_log_dir, 'final_state.pth'))

                    writer_dict['writer'].close()
                    end = timeit.default_timer()
                    logger.info('Hours: %d' % np.int32((end - start) / 3600))
                    logger.info('Done')


if __name__ == '__main__':
    main()
