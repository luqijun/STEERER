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
import glob


def generate_depth_image(path):
     
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
     
    for folder in ['train', 'val', 'test']:
        images_path = os.path.join(path, folder)
        images_depth_path = os.path.join(path, f'{folder}_Depth')
        os.makedirs(images_depth_path, exist_ok=True)
        
        image_files = glob.glob(os.path.join(images_path, '*.jpg'))
        for full_file_name in tqdm(image_files):
            image = Image.open(full_file_name)
            file_name = os.path.basename(full_file_name)
            depth_save_path = os.path.join(images_depth_path, file_name)
            width, height = image.size
            
            if os.path.exists(depth_save_path):
                continue
           
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
        for filename in os.listdir(images_path):
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


if __name__ == '__main__':
    
    # root_test = '/mnt/c/Users/lqjun/Desktop'
    
    root = '/mnt/e/MyDocs/Code/Datasets/UCF-QNRF/UCF-QNRF_ECCV18-Processed'
    # root = '/mnt/e/MyDocs/Code/Datasets/UCF-QNRF/qnrf_.1024_a'
    generate_depth_image(root)
    # print('Generate Success!')
    
    # get_max_min_value(root)
    pass
