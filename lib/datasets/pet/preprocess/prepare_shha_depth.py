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


def generate_depth_image(path):
     
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
     
    for folder in ['train_data', 'test_data']:
        images_path = os.path.join(path, folder, 'images')
        images_depth_path = os.path.join(path, folder, 'images_depth')
        os.makedirs(images_depth_path, exist_ok=True)
        for filename in os.listdir(images_path):
            image = Image.open(os.path.join(images_path, filename))
            depth_save_path = os.path.join(images_depth_path, filename)
            width, height = image.size
            
           
            depth = pipe(image)["depth"]
            depth.save(depth_save_path)
            # depth_img = np.array(depth)


min_value = 1e9
max_value = -1e9

def get_max_min_value(path):
    global min_value
    global max_value
    
    for folder in ['train_data', 'test_data']:
        images_path = os.path.join(path, folder, 'images_depth')
        annotations_path = os.path.join(path, folder, 'ground-truth')
        for filename in tqdm(os.listdir(images_path)):
            image = Image.open(os.path.join(images_path, filename))
            image = np.array(image)
            
            mat = loadmat(os.path.join(annotations_path, 'GT_' + filename.replace('.jpg', '.mat')))
            points = mat["image_info"][0, 0][0, 0][0]
            for point in points:
                x, y = point
                x = round(x) - 1
                y = round(y) - 1
                depth_value = image[y, x]
                
                if min_value > depth_value:
                    min_value = depth_value
                if max_value < depth_value:
                    max_value = depth_value
            
    print("min value = ", str(min_value)) # 0
    print("max value = ", str(max_value)) # 254


def generate_gassian_map_origin(path):
    
    for folder in ['train_data', 'test_data']:
        images_path = os.path.join(path, folder, 'images_depth')
        annotations_path = os.path.join(path, folder, 'ground-truth')
        den_maps_path = os.path.join(path, folder, 'den_maps_depth')
        os.makedirs(den_maps_path, exist_ok=True)
        for filename in tqdm(os.listdir(images_path)):
            image = Image.open(os.path.join(images_path, filename))
            den_save_path = os.path.join(den_maps_path, filename)
            den_map_save_path = os.path.join(den_maps_path, filename)
            image = np.array(image)
            h, w = image.shape[-2:]
            
            density = np.zeros((h, w), dtype=np.float32)
            
            mat = loadmat(os.path.join(annotations_path, 'GT_' + filename.replace('.jpg', '.mat')))
            points = mat["image_info"][0, 0][0, 0][0]
            for point in points:
                pt2d = np.zeros((h, w), dtype=np.float32)
                x, y = point
                x = round(x) - 1
                y = round(y) - 1
                
                pt2d[y, x] = 1.
                
                depth_value = image[y, x]
                
                sigma = max(depth_value * 0.1, 1)
                density += ndimage.filters.gaussian_filter(pt2d, sigma)

            with h5py.File(den_save_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = density

            density_heat_map = cv2.applyColorMap((255 * density / (density.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            density_img = Image.fromarray(cv2.cvtColor(density_heat_map, cv2.COLOR_BGR2RGB))
            density_img.save(den_map_save_path)




if __name__ == '__main__':
    
    # root_test = '/mnt/c/Users/lqjun/Desktop'
    
    root = '/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final'
    generate_depth_image(root)
    # print('Generate Success!')
    
    # get_max_min_value(root)
    
    # generate_gassian_map_origin(root)
    pass
