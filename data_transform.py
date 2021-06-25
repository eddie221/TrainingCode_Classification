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
    if "IMAGE_SIZE" in args["TRAIN"]:
        train_transforms.append(transforms.Resize(args["TRAIN"].getint("IMAGE_SIZE")))
    if "IMAGE_SIZE" in args["VALIDATION"]:
        validation_transforms.append(transforms.Resize(args["VALIDATION"].getint("IMAGE_SIZE")))
    
    # RandomCrop image
    if "RANDOM_CROP_SIZE" in args["TRAIN"]:
        train_transforms.append(transforms.RandomCrop(args["TRAIN"].getint("RANDOM_CROP_SIZE")))
    if "RANDOM_CROP_SIZE" in args["VALIDATION"]:
        validation_transforms.append(transforms.RandomCrop(args["VALIDATION"].getint("RANDOM_CROP_SIZE")))
        
    # Random flip
    if "RANDOM_FILP" in args["TRAIN"] and args["TRAIN"]["RANDOM_FILP"]:
        train_transforms.append(transforms.RandomHorizontalFlip())
    if "RANDOM_FILP" in args["VALIDATION"] and args["VALIDATION"]["RANDOM_FILP"]:
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
    train_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    validation_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

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