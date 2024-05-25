r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import numpy as np
import re
from torch.utils.data._utils.collate import default_collate
from typing import Optional, List
from torch import Tensor

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, string_classes, int_classes
else:
    import collections.abc as container_abcs
    int_classes = int
    string_classes = str

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def assign_storage(batch_tensors, stack=True):
    elem = batch_tensors[0]
    numel = sum(x.numel() for x in batch_tensors)
    storage = elem.storage()._new_shared(numel, device=elem.device)
    if stack:
        out = elem.new(storage).resize_(len(batch_tensors), *list(elem.size()))
        return torch.stack(batch_tensors, dim=0, out=out)
    else:
        length = sum(x.shape[0] for x in batch_tensors)
        out = elem.new(storage).resize_(length, *list(elem.size()[1:]))
        return torch.cat(batch_tensors, dim=0, out=out)


def get_collate(config):
    collate_name = config.dataset.get('collate', 'default_collate')
    return eval(collate_name)

def default_collate2(batch):

    images = []
    labels = []
    sizes = []
    names = []
    point_lens = []
    points_list = []
    target_list = []
    for i, item in enumerate(batch):
        images.append(torch.as_tensor(item[0]))
        labels.append([torch.as_tensor(l) for l in item[1]])
        sizes.append(item[2])
        names.append(item[3])
        point_lens.append(item[4])
        points_list.append(torch.as_tensor(item[5]))
        target_list.append(torch.as_tensor(item[6]))

    if torch.utils.data.get_worker_info() is not None:
        images = assign_storage(images)
        labels = [assign_storage(l) for l in zip(*labels)]
        points_list = assign_storage(points_list, False)
        target_list = assign_storage(target_list, False)
    else:
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)

    sizes = torch.as_tensor(np.array(sizes))

    names_arr = []
    if len(names) == 1:
        names_arr = names
    else:
        name_list = [name[0] for name in names]
        name_index = [name[1] for name in names]
        name_resize_factor = [name[2] for name in names]
        names_arr = [name_list, name_index, name_resize_factor]

    return images, labels, sizes, names_arr, point_lens, points_list, target_list

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # import os
    # if int(os.environ["LOCAL_RANK"])==0:
    #     import pdb
    #     pdb.set_trace()

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)

            out = elem.new(storage).resize_(len(batch), *list(elem.size()))

        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(
                        elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, list):
        # import pdb
        # pdb.set_trace()
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return batch
        # return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples)
                         for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def p2pnet_collate(batch):
    # re-organize the batch
    batch_new = []
    for b in batch:
        imgs, points = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i, :, :, :], points[i]))
    batch = batch_new
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:

        # TODO make it support different-sized images
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor

def _max_by_axis_pad(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    block = 128

    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes