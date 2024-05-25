import os
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat

def convert_shanghaitech_to_coco(path):

    ids = 1
    ann_ids = 1
    for folder in ['train_data', 'test_data']:

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}]
        }

        images_path = os.path.join(path, folder, 'images')
        annotations_path = os.path.join(path, folder, 'ground-truth')

        for filename in os.listdir(images_path):
            if filename.endswith('.jpg'):
                image = Image.open(os.path.join(images_path, filename))
                width, height = image.size

                image_info = {
                    "id": ids,
                    "file_name": filename,
                    "width": width,
                    "height": height
                }
                coco_data["images"].append(image_info)

                mat = loadmat(os.path.join(annotations_path, 'GT_' + filename.replace('.jpg', '.mat')))
                points = mat["image_info"][0, 0][0, 0][0]

                for point in points:
                    x, y = point
                    ann = {
                        "id": ann_ids,
                        "image_id": ids,
                        "category_id": 1,
                        "segmentation": [],
                        "area": 1,
                        "bbox": [x, y, 1, 1],
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(ann)
                    ann_ids += 1

                ids += 1

        save_path = os.path.join(path, f'coco_{folder}.json')
        with open(save_path, 'w') as f:
            json.dump(coco_data, f)

if __name__ == '__main__':
    convert_shanghaitech_to_coco("/mnt/e/MyDocs/Code/Datasets/ShangHaiTech/ShanghaiTech/part_A_final")
