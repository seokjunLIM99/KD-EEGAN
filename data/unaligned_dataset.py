import os
import random

import torch
from torch import nn

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import store_dataset

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)


        def pair_key(path, root):
            relative = os.path.relpath(path, root)
            return os.path.splitext(relative)[0].replace('\\', '/').lower()

        A_dict = {pair_key(p, self.dir_A): (img, p)
                  for img, p in zip(self.A_imgs, self.A_paths)}
        B_dict = {pair_key(p, self.dir_B): (img, p)
                  for img, p in zip(self.B_imgs, self.B_paths)}

        common_keys = sorted(set(A_dict) & set(B_dict))
        if not common_keys:
            raise RuntimeError(
                'No paired images were found. Put matching filenames under %s and %s.'
                % (self.dir_A, self.dir_B))

        self.A_imgs = [A_dict[k][0] for k in common_keys]
        self.A_paths = [A_dict[k][1] for k in common_keys]
        self.B_imgs = [B_dict[k][0] for k in common_keys]
        self.B_paths = [B_dict[k][1] for k in common_keys]

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
    def __getitem__(self, index):
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        if A_img.size != B_img.size:
            raise ValueError(
                'Paired images must have the same size: %s (%s) vs %s (%s)'
                % (A_path, A_img.size, B_path, B_img.size))

        params = get_params(self.opt, A_img.size)
        transform = get_transform(self.opt, params)
        A_img = transform(A_img)
        B_img = transform(B_img)

        if self.opt.vary == 1 and random.random() < 0.5:
            times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
            input_img = (A_img + 1) / 2. / times
            input_img = input_img * 2 - 1
        else:
            input_img = A_img
        if self.opt.lighten:
            B_img = (B_img + 1) / 2.
            value_range = torch.max(B_img) - torch.min(B_img)
            B_img = (B_img - torch.min(B_img)) / value_range.clamp_min(1e-8)
            B_img = B_img * 2. - 1

        r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedDataset'



