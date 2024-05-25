import math
import os
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat
from transformers import pipeline
from PIL import Image
import scipy.ndimage as ndimage
import cv2
from tqdm import tqdm
import h5py


import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from tqdm import tqdm
from transformers import pipeline
import scipy.ndimage as ndimage

num = 12
# dataset = './ProcessedData/SHHA'
dataset = '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/SHHA_Custom2'

def read_files():

    train_set = 'train.txt'
    test_set = 'test.txt'
    train_imgs_txt = os.path.join(dataset, train_set)
    test_imgs_txt = os.path.join(dataset, test_set)

    train_img_list = [line.strip().split() for line in open(train_imgs_txt)]
    test_img_list = [line.strip().split() for line in open(test_imgs_txt)]
    img_list = train_img_list + test_img_list
    # box_gt_Info = self.read_box_gt(os.path.join(self.root, 'val_gt_loc.txt'))
    files = []
    for item in img_list:
        # import pdb
        # pdb.set_trace()
        image_id = item[0]
        # if 'val' in self.list_path:
        #     self.box_gt.append(box_gt_Info[int(image_id)])
        files.append({
            "img": 'images/' + image_id + '.jpg',
            "label": 'jsons/' + image_id + '.json',
            "level_map": 'level_maps/' + image_id + '.h5',
            "name": image_id,
            "weight": 1
        })
    return files

def compute_image_density(points, h, w, depth):
    """
    Compute crowd density:
        - defined as the average nearest distance between ground-truth points
    """
    density = np.zeros((h, w), dtype=np.float32)
    for point in points:
        pt2d = np.zeros((h, w), dtype=np.float32)
        x, y = point
        x = round(x) - 1
        y = round(y) - 1

        pt2d[y, x] = 1.

        depth_value = depth[y, x]

        sigma = math.sqrt(depth_value * 0.2)
        sigma = max(sigma, 1)
        sigma = min(sigma, 6)
        density += ndimage.filters.gaussian_filter(pt2d, sigma)

    return density

def main():

    images_depth_path = os.path.join(dataset, 'images_depth')
    den_maps_dir = os.path.join(dataset, 'den_maps_depth')

    os.makedirs(images_depth_path, exist_ok=True)
    os.makedirs(den_maps_dir, exist_ok=True)
    files = read_files()
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

    for item in tqdm(files):
        img_path = os.path.join(dataset, item["img"])
        image = Image.open(img_path)

        depth_save_path = os.path.join(images_depth_path, item["name"] + '.jpg')
        den_map_save_path = os.path.join(den_maps_dir, item["name"] + '.h5')
        den_heat_save_path = os.path.join(den_maps_dir, item["name"] + '.jpg')

        if os.path.exists(depth_save_path) and os.path.exists(den_map_save_path) and os.path.exists(den_heat_save_path):
            continue

        # 获取深度信息
        if os.path.exists(depth_save_path):
            depth = Image.open(depth_save_path)
        else:
            depth = pipe(image)["depth"]
            depth.save(depth_save_path)

        depth_img = np.array(depth)

        # 获取点信息
        H = image.height
        W = image.width
        with open(os.path.join(dataset, item["label"]), 'r') as f:
            info = json.load(f)
        points =  np.array(info['points']).astype('float32').reshape(-1,2) # n * (w,h)

        # points_hw = points[:, ::-1] # n * (h, w)
        density = compute_image_density(points, H, W, depth_img)
        with h5py.File(den_map_save_path, 'w') as hf:
            hf['den_map'] = density

        # 创建热度图
        density_heat_map = cv2.applyColorMap((255 * density / (density.max() + 1e-10)).astype(np.uint8).squeeze(),
                                             cv2.COLORMAP_JET)
        density_img = Image.fromarray(cv2.cvtColor(density_heat_map, cv2.COLOR_BGR2RGB))
        density_img.save(den_heat_save_path)



if __name__ == '__main__':
    
    # root_test = '/mnt/c/Users/lqjun/Desktop'
    # root = '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final'
    # generate_depth_image(root)
    # print('Generate Success!')
    
    # get_max_min_value(root)
    
    main()
    pass
