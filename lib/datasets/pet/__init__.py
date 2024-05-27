import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .SHB import build as build_shb
from .UCF_QNRF_1024 import build as build_ucf_qnrf_1024
from .UCF_QNRF_2048 import build as build_ucf_qnrf_2048
from .JHU import build as build_jhu

data_path = {
    'SHA': './ProcessedData/ShanghaiTech/part_A/',
    'SHB': './ProcessedData/ShanghaiTech/part_B/',
    'UCF_QNRF_1024': './ProcessedData/UCF_QNRF_1024/',
    'UCF_QNRF_2048': './ProcessedData/UCF-QNRF_ECCV18-Processed/',
    'JHU': './ProcessedData/JHU/',
}

def build_pet_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == 'SHB':
        return build_shb(image_set, args)
    if args.dataset_file == 'UCF_QNRF_1024':
        return build_ucf_qnrf_1024(image_set, args)
    if args.dataset_file == 'UCF_QNRF_2048':
        return build_ucf_qnrf_2048(image_set, args)
    if args.dataset_file == 'JHU':
        return build_jhu(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
