#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:34:42 2021

@author: mmplab603
"""

import torchvision
from torch.utils.data import DataLoader
from data_transform import prepare_transforms

def load_data(args):
    dataloader = []
    dataset_sizes = []
    
    data_transforms = prepare_transforms(args)
    
    train_dataset = torchvision.datasets.ImageFolder(args["DEFAULT"].get("TRAIN_DATASET_PATH"), data_transforms["train"])
    trainloader = DataLoader(train_dataset,
                             batch_size = args["TRAIN"].getint("BATCH_SIZE"),
                             shuffle = args["TRAIN"].getboolean("SHFFLE"),
                             num_workers = args["TRAIN"].getint("NUMBER_WORKDERS"))  
    
    val_dataset = torchvision.datasets.ImageFolder(args["DEFAULT"].get("VAL_DATASET_PATH"), data_transforms["val"])
    valloader = DataLoader(val_dataset,
                           batch_size = args["VALIDATION"].getint("BATCH_SIZE"),
                           shuffle = args["VALIDATION"].getboolean("SHFFLE"),
                           num_workers = args["VALIDATION"].getint("NUMBER_WORKDERS"))  
    
    # combine
    dataloader.append({"train" : trainloader, "val" : valloader})
    dataset_sizes.append({"train" : len(trainloader), "val" : len(valloader)})
    return dataloader, dataset_sizes
    
    
    
    