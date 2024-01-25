import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from tqdm import tqdm

num = 12
dataset = './ProcessedData/SHHA'

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

def compute_image_density(points, H, W, num):
    """
    Compute crowd density:
        - defined as the average nearest distance between ground-truth points
    """
    y = torch.arange(0, H)
    x = torch.arange(0, W)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_points = torch.vstack([grid_y.flatten(), grid_x.flatten()]).permute(1,0).float().cuda()
    points_tensor = torch.from_numpy(points[:, ::-1].copy()).float().cuda()

    # tmp_grid_points = grid_points.unsqueeze(1)
    # points_tensor = points_tensor.unsqueeze(0)
    # indexes =  points[:, 0] > grid_points[:, 0] + 100

    dist = torch.cdist(grid_points, points_tensor, p=2)

    if points_tensor.shape[0] > 1:
        nearest_num = min(points_tensor.shape[0], num)

        if len(grid_points) > 300000:
            dens=[]
            chunks = torch.chunk(dist, 100)
            for chunk in chunks:
                den =  chunk.sort(dim=1)[0][:, 0:nearest_num].mean(dim=1)
                dens.append(den)
            density = torch.cat(dens, dim=0).cpu().reshape(H, W).numpy()
        else:
            density = dist.sort(dim=1)[0][:, 0:nearest_num].mean(dim=1).reshape(H, W).cpu().numpy()

    return density

def read_max_min_level(num):

    level_maps_dir = os.path.join(dataset, 'level_maps', f'{num}_nearest')
    os.makedirs(level_maps_dir, exist_ok=True)
    files = read_files()

    max_level = 0
    min_level = 1e9
    for item in tqdm(files):
        level_map_path = os.path.join(level_maps_dir, item["name"] + '.h5')
        with h5py.File(level_map_path, 'r') as hf:
            max_v = hf['level_map'][:].max()
            min_v = hf['level_map'][:].min()
            if max_v > max_level:
                max_level = max_v
            if min_v < min_level:
                min_level = min_v
    print("最大值为：", str(max_level))
    print("最小值为：", str(min_level))

def main():

    level_maps_dir = os.path.join(dataset, 'level_maps', f'{num}_nearest')
    os.makedirs(level_maps_dir, exist_ok=True)
    files = read_files()

    for item in tqdm(files):
        image = cv2.imread(os.path.join(dataset, item["img"]),
                           cv2.IMREAD_COLOR)
        H, W = image.shape[:2]
        with open(os.path.join(dataset, item["label"]), 'r') as f:
            info = json.load(f)
        points =  np.array(info['points']).astype('float32').reshape(-1,2) # n * (w,h)
        # points_hw = points[:, ::-1] # n * (h, w)
        density = compute_image_density(points, H, W, num)
        level_map_path = os.path.join(level_maps_dir, item["name"] + '.h5')
        with h5py.File(level_map_path, 'w') as hf:
            hf['level_map'] = density

        # 创建热度图
        data = density
        # 将数组归一化到0-255
        arr_normalized = cv2.normalize(data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(arr_normalized, cv2.COLORMAP_JET)
        heatmap_path = os.path.join(level_maps_dir, item["name"] + '.png')
        cv2.imwrite(heatmap_path, heatmap)


if __name__ == '__main__':
    # main()
    read_max_min_level()

    # 最大值为： 1413.0303
    # 最小值为： 3.5070856
