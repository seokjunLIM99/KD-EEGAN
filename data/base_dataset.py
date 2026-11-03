import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h, new_w = h, w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = int(opt.loadSize * h / w)

    max_x = max(0, new_w - opt.fineSize)
    max_y = max(0, new_h - opt.fineSize)
    return {
        'crop_pos': (random.randint(0, max_x), random.randint(0, max_y)),
        'flip': opt.isTrain and not opt.no_flip and random.random() < 0.5,
    }


def get_transform(opt, params=None):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        transform_list.append(transforms.Resize(
            [opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(_crop_transform(opt, params))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(_crop_transform(opt, params))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(_crop_transform(opt, params))

    if opt.isTrain and not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def _crop_transform(opt, params):
    if params is None:
        return transforms.RandomCrop(opt.fineSize)
    return transforms.Lambda(
        lambda img: __crop(img, params['crop_pos'], opt.fineSize))


def __crop(img, pos, size):
    ow, oh = img.size
    x, y = pos
    if ow < size or oh < size:
        raise ValueError(
            'Image size %sx%s is smaller than fineSize=%s' % (ow, oh, size))
    return img.crop((x, y, x + size, y + size))


def __flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
