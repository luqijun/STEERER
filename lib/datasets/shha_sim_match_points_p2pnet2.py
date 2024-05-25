# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import heapq
from PIL import Image
import json
import torch
from torch.nn import functional as F
import random
from .base_dataset import BaseDataset
import torchvision.transforms as standard_transforms
from .utils import dataprocessor as dp
import logging


def load_data(img_gt_path, train):
    img_path, img_depth_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_depth = Image.open(img_depth_path)
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    return img, img_depth, np.array(points)

class SHHA_Sim_Match_Points_P2PNet2(BaseDataset):
    def __init__(self,
                 config,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 min_unit = (32,32),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=(0.5,1/0.5),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(SHHA_Sim_Match_Points_P2PNet2, self).__init__(ignore_label, base_size,
                                   crop_size, downsample_rate, scale_factor, mean, std)

        self.config=config
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([1]).cuda()

        self.train = 'train' in self.list_path
        self.patch = config.dataset.get('patch', True)
        self.patch_size = config.dataset.get('patch_size', 4)
        self.flip = flip

        self.multi_scale = multi_scale
        self.scale_factor =scale_factor
        a = np.arange(scale_factor[0],1.,0.05)
        # b = np.linspace(1, scale_factor[1],a.shape[0])

        self.scale_factor = np.concatenate([a,1/a],0)

        # the pre-proccssing transform
        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
        ])
        self.pil_to_tensor = standard_transforms.ToTensor()


        self.train_lists = "train.list"
        self.eval_list = "test.list"

        self.img_map = {}
        self.img_list = []
        self.img_list_depth = []
        data_dir =''
        if self.train:
            data_dir = 'train'
            self.img_list_file = self.train_lists.split(',')
        else:
            data_dir = 'test'
            self.img_list_file = self.eval_list.split(',')


        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root, line[0].strip())] = os.path.join(self.root, line[1].strip())
                    self.img_list_depth.append(os.path.join(self.root, f'{data_dir}_images_depth', os.path.basename(line[0].strip())))
        self.img_list = sorted(list(self.img_map.keys()))

        # depth imgs


        # number of samples
        self.nSamples = len(self.img_list)

        self.box_gt = []
        self.min_unit = min_unit

        self.resize_memory_pool = {}
        self.AI_resize = False

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        img_depth_path = self.img_list_depth[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, img_depth, point = load_data((img_path, img_depth_path, gt_path), self.train)
        size = img.size
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)
            img_depth = self.pil_to_tensor(img_depth)

        if self.train:
            crop_h, crop_w = self.crop_size[0], self.crop_size[1]
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > min(crop_h, crop_w):
                img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                img_depth = torch.nn.functional.interpolate(img_depth.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale

            # random crop augumentaiton
            if self.patch:
                img, img_depth, point = self.random_crop(img, img_depth, point, self.patch_size)
                for i, _ in enumerate(point):
                    point[i] = torch.Tensor(point[i])

            # random flipping
            if random.random() > 0.5 and self.flip:
                # random flip
                img = torch.Tensor(img[:, :, :, ::-1].copy())
                img_depth = torch.Tensor(img_depth[:, :, :, ::-1].copy())
                for i, _ in enumerate(point):
                    point[i][:, 0] = crop_w - point[i][:, 0]
        else:
            # test
            point = torch.Tensor(point).unsqueeze(0) # [point]
            img_depth = torch.Tensor(img_depth).unsqueeze(0)

        img = torch.Tensor(img)
        img_depth = torch.Tensor(img_depth)
        h, w = img_depth.shape[-2:]
        # pack up related infos
        target = [{} for i in range(len(point))]
        points_format = self.config.dataset.get('points_format', "hw")
        gen_label_map = self.config.dataset.get('gen_label_map', False)
        for i, _ in enumerate(point):

            if gen_label_map:
                label_maps = self.label_transform(point[i], img[i].shape[-2:])
                target[i]['label_map'] = label_maps
            else:
                target[i]['label_map'] = torch.tensor([])

            if points_format == 'hw':
                point[i] = point[i].flip(dims=(-1,))
            target[i]['points'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

            # depth info
            h_coords = torch.clamp(point[i][:, 0].int(), min=0, max=h-1) if points_format == 'hw' else torch.clamp(point[i][:, 1].int(), min=0, max=h-1)
            w_coords = torch.clamp(point[i][:, 1].int(), min=0, max=w-1) if points_format == 'hw' else torch.clamp(point[i][:, 0].int(), min=0, max=w-1)
            depth = img_depth[i][:, h_coords, w_coords]
            depth_weight = torch.clamp(1 - depth, min=0.05, max=1)
            target[i]['depth_weight'] = depth_weight / 7

        return img, target


    def random_crop(self, img, img_depth, den, num_patch=4):
        assert img.shape[-2] == img_depth.shape[-2]
        half_h, half_w = self.crop_size[0], self.crop_size[1]
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
        result_img_depth = np.zeros([num_patch, img_depth.shape[0], half_h, half_w])
        result_den = []
        # crop num_patch for each image
        for i in range(num_patch):
            start_h = random.randint(0, img.size(1) - half_h)
            start_w = random.randint(0, img.size(2) - half_w)
            end_h = start_h + half_h
            end_w = start_w + half_w
            # copy the cropped rect
            result_img[i] = img[:, start_h:end_h, start_w:end_w]
            result_img_depth[i] = img_depth[:, start_h:end_h, start_w:end_w]
            # copy the cropped points
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h

            result_den.append(record_den)

        return result_img, result_img_depth, result_den

    def label_transform(self, points, shape):

        label = torch.zeros(shape).float()
        labelx2= torch.zeros((shape[0]//2,shape[1]//2) ).float()
        labelx4 = torch.zeros((shape[0]//4,shape[1]//4) ).float()
        labelx8 = torch.zeros((shape[0]//8,shape[1]//8) ).float()

        # index = np.round(points).astype('int32')
        # index[:, 0]=np.clip(index[:, 0], 0,  shape[1]-1)
        # index[:, 1] = np.clip(index[:, 1], 0, shape[0]-1)

        for i in range(points.shape[0]):
            point = points[i]
            w_idx = torch.clip(point[0].round().int(), 0, shape[1] - 1)
            h_idx = torch.clip(point[1].round().int(), 0, shape[0] - 1)
            label [h_idx,w_idx] +=1

            w_idx = torch.clip((point[0]/2).round().int(), 0, shape[1]//2 - 1)
            h_idx = torch.clip((point[1]/2).round().int(), 0, shape[0]//2 - 1)
            labelx2[h_idx, w_idx] += 1

            w_idx = torch.clip((point[0]/4).round().int(), 0, shape[1]//4 - 1)
            h_idx = torch.clip((point[1]/4).round().int(), 0, shape[0]//4 - 1)
            labelx4[h_idx, w_idx] += 1

            w_idx = torch.clip((point[0]/8).round().int(), 0, shape[1]//8 - 1)
            h_idx = torch.clip((point[1]/8).round().int(), 0, shape[0]//8 - 1)
            labelx8[h_idx, w_idx] += 1


        # import pdb
        # pdb.set_trace()
        return [label, labelx2, labelx4, labelx8]