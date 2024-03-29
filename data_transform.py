#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:35:41 2021

@author: mmplab603
"""

import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def prepare_transforms(args):
    train_transforms = []
    validation_transforms = []
    # Resize image
    if "TRAIN_IMAGE_SIZE" in args and args["TRAIN_IMAGE_SIZE"]:
        train_transforms.append(transforms.Resize(args.TRAIN_IMAGE_SIZE))
    if "VAL_IMAGE_SIZE" in args and args["VAL_IMAGE_SIZE"]:
        validation_transforms.append(transforms.Resize(args.VAL_IMAGE_SIZE))
    
    # RandomCrop image
    if "TRAIN_RANDOM_CROP_SIZE" in args and args["TRAIN_RANDOM_CROP_SIZE"]:
        train_transforms.append(transforms.RandomCrop(args.TRAIN_RANDOM_CROP_SIZE))
    if "VAL_RANDOM_CROP_SIZE" in args and args["VAL_RANDOM_CROP_SIZE"]:
        validation_transforms.append(transforms.RandomCrop(args.VAL_RANDOM_CROP_SIZE))
        
    # CenterCrop image
    if "TRAIN_CENTER_CROP_SIZE" in args and args["TRAIN_CENTER_CROP_SIZE"]:
        train_transforms.append(transforms.CenterCrop(args.TRAIN_CENTER_CROP_SIZE))
    if "VAL_CENTER_CROP_SIZE" in args and args["VAL_CENTER_CROP_SIZE"]:
        validation_transforms.append(transforms.CenterCrop(args.VAL_CENTER_CROP_SIZE))

    # Random flip
    if "TRAIN_RANDOM_FILP" in args and args["TRAIN_RANDOM_FILP"]:
        train_transforms.append(transforms.RandomHorizontalFlip())
    if "VAL_RANDOM_FILP" in args and args["VAL_RANDOM_FILP"]:
        validation_transforms.append(transforms.RandomHorizontalFlip())
        
# =============================================================================
#     # Change type to Tensor
#     train_transforms.append(RandomFlip(args["TRAIN"]["RANDOM_FILP"]))
#     validation_transforms.append(RandomFlip(args["VALIDATION"]["RANDOM_FILP"]))
# =============================================================================
    # To tensor
    train_transforms.append(transforms.ToTensor())
    validation_transforms.append(transforms.ToTensor())

    # Normalize
    if "TRAIN_NORM" in args and args["TRAIN_NORM"]:
        train_transforms.append(transforms.Normalize(args.TRAIN_MEAN, args.TRAIN_STD,))
    if "VAL_NORM" in args and args["VAL_NORM"]:
        validation_transforms.append(transforms.Normalize(args.VAL_MEAN, args.VAL_STD,))

# =============================================================================
#     if "Normalize" in args["TRAIN"] and args["TRAIN"]["Normalize"]:
#         assert "MEAN" in args["TRAIN"], "Missing nomalize parameter \"MEAN\" in training."
#         assert "STD" in args["TRAIN"], "Missing nomalize parameter \"STD\" in training."   
#         train_transforms.append(transforms.Normalize(args["TRAIN"]["MEAN"], args["TRAIN"]["STD"]))
#     if "Normalize" in args["VALIDATION"] and args["VALIDATION"]["Normalize"]:
#         assert "MEAN" in args["VALIDATION"], "Missing nomalize parameter \"MEAN\" in validation."
#         assert "STD" in args["VALIDATION"], "Missing nomalize parameter \"STD\" in validation." 
#         validation_transforms.append(transforms.Normalize(args["VALIDATION"]["MEAN"], args["VALIDATION"]["STD"]))
# =============================================================================

    data_transforms = {
        "train": transforms.Compose(train_transforms),
        "val": transforms.Compose(validation_transforms),
    }   
    return data_transforms

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    trans = [transforms.RandomCrop(224), transforms.Resize(224)]
    t = transforms.Compose(trans)
    a = Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))
    a = t(a)