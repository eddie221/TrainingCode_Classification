#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:31:40 2021

@author: mmplab603
"""

import random
from PIL import Image
import math
import torch
import numpy as np
import numbers
from torchvision.transforms import Pad
from torchvision.transforms import functional as F

class Normalize(object):
    '''
        Normalize the tensors
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __repr__(self):
        s = "Normalize(mean={}, std={})".format(self.mean, self.std)
        return s
    
    def __call__(self, rgb_img, label_img=None):
        rgb_img = F.to_tensor(rgb_img) # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, self.mean, self.std) # normalize the tensor
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))
        return rgb_img, label_img


class RandomFlip(object):
    '''
        Random Flipping
    '''
    
    def __repr__(self):
        s = "RandomFlip()"
        return s
    
    def __call__(self, rgb_img, label_img = None):
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            if label_img is not None:
                label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
                return rgb_img, label_img
            else:
                return rgb_img


class RandomScale(object):
    '''
    Random scale, where scale is logrithmic
    '''
    def __init__(self, scale=(0.5, 1.0)):
        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)
    
    def __repr__(self):
        s = "RandomScale(scale={})".format(self.scale)
        return s

    def __call__(self, rgb_img, label_img = None):
        w, h = rgb_img.size
        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        rgb_img = rgb_img.resize(new_size, Image.ANTIALIAS)
        if label_img is not None:
            label_img = label_img.resize(new_size, Image.NEAREST)
            return rgb_img, label_img
        else:
            return rgb_img


class RandomCrop(object):
    '''
    Randomly crop the image
    '''
    def __init__(self, crop_size, ignore_idx=255):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.ignore_idx = ignore_idx
    
    def __repr__(self):
        s = "RandomCrop(crop_size={}, ignore_idx={})".format(self.crop_size, self.ignore_idx)
        return s

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img = None):
        w, h = rgb_img.size
        pad_along_w = max(0, int((1 + self.crop_size[0] - w) / 2))
        pad_along_h = max(0, int((1 + self.crop_size[1] - h) / 2))
        # padd the images
        rgb_img = Pad(padding=(pad_along_w, pad_along_h), fill=0, padding_mode='constant')(rgb_img)
        if label_img is not None:
            label_img = Pad(padding=(pad_along_w, pad_along_h), fill=self.ignore_idx, padding_mode='constant')(label_img)

        i, j, h, w = self.get_params(rgb_img, self.crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        if label_img is not None:
            label_img = F.crop(label_img, i, j, h, w)
            return rgb_img, label_img
        else:
            return rgb_img


class RandomResizedCrop(object):
    '''
    Randomly crop the image and then resize it
    '''
    def __init__(self, size, scale=(0.5, 1.0), ignore_idx=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.ignore_idx = ignore_idx
        
    def __repr__(self):
        s = "RansomResizedCrop(size={}, scale={}, ignor_idx={})".format(self.size, self.scale, self.ignore_idx)
        return s

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, rgb_img, label_img = None):
        w, h = rgb_img.size

        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (
                    math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        crop_size = (int(round(w * random_scale)), int(round(h * random_scale)))

        i, j, h, w = self.get_params(rgb_img, crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        if label_img is not None:
            label_img = F.crop(label_img, i, j, h, w)

        rgb_img = rgb_img.resize(self.size, Image.ANTIALIAS)
        if label_img is not None:
            label_img = label_img.resize(self.size, Image.NEAREST)
            return rgb_img, label_img
        else:
            return rgb_img


class Resize(object):
    '''
        Resize the images
    '''
    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (int(size), int(size))
            
    def __repr__(self):
        s = "Resize(size={})".format(self.size)
        return s

    def __call__(self, rgb_img, label_img = None):
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        if label_img is not None:
            label_img = label_img.resize(self.size, Image.NEAREST)
            return rgb_img, label_img
        else:
            return rgb_img


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        
    def __repr__(self):
        s = "Compose : {\n"
        for step, t in enumerate(self.transforms):
            s += "    ({}): {},\n".format(step, t.__repr__())
        s += "}"
        return s
        

    def __call__(self, **args):
        for t in self.transforms:
            args = t(**args)
        return args
    
def prepare_transforms(**args):
    train_transforms = []
    validation_transforms = []
    # Resize image
    if "IMAGE_SIZE" in args:
        train_transforms.append(Resize(args.TRAIN_IMAGE_SIZE))
    if "IMAGE_SIZE" in args.VALIDATION:
        validation_transforms.append(Resize(args.VALIDATION["IMAGE_SIZE"]))
    
    # RandomCrop image
    if "RANDOM_CROP_SIZE" in args:
        train_transforms.append(RandomCrop(args.TRAIN_RANDOM_CROP_SIZE))
    if "RANDOM_CROP_SIZE" in args.VALIDATION:
        validation_transforms.append(RandomCrop(args.VALIDATION.getint("RANDOM_CROP_SIZE")))
        
    # Random flip
    if "RANDOM_FILP" in args and args.TRAIN_RANDOM_FILP:
        train_transforms.append(RandomFlip())
    if "RANDOM_FILP" in args.VALIDATION and args.VALIDATION["RANDOM_FILP"]:
        validation_transforms.append(RandomFlip())
        
# =============================================================================
#     # Change type to Tensor
#     train_transforms.append(RandomFlip(args.TRAIN_RANDOM_FILP))
#     validation_transforms.append(RandomFlip(args.VALIDATION["RANDOM_FILP"]))
# =============================================================================

    if args.TRAIN_Normalize:
        assert "MEAN" in args, "Missing nomalize parameter \"MEAN\" in training."
        assert "STD" in args, "Missing nomalize parameter \"STD\" in training."   
        train_transforms.append(Normalize(args.TRAIN_MEAN, args.TRAIN_STD))
        
    if args.VALIDATION["Normalize"]:
        assert "MEAN" in args.VALIDATION, "Missing nomalize parameter \"MEAN\" in validation."
        assert "STD" in args.VALIDATION, "Missing nomalize parameter \"STD\" in validation." 
        validation_transforms.append(Normalize(args.VALIDATION["MEAN"], args.VALIDATION["STD"]))

    data_transforms = {
        "train": Compose(train_transforms),
        "val": Compose(validation_transforms),
    }   
    return data_transforms

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    trans = [RandomCrop(224), Resize(224)]
    t = Compose(trans)
    a = Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))
    a = t(a)