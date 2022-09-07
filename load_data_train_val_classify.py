#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:34:42 2021

@author: mmplab603
"""

import torchvision
from torch.utils.data import DataLoader
from data_transform import prepare_transforms
import torchvision.transforms as transforms

def load_data(args):
    dataloader = []
    dataset_sizes = []
    
    data_transforms = prepare_transforms(args)
    train_dataset = torchvision.datasets.ImageFolder(args.TRAIN_DATASET_PATH, data_transforms["train"])
    trainloader = DataLoader(train_dataset,
                             batch_size = args.TRAIN_BATCH_SIZE,
                             shuffle = args.TRAIN_SHUFFLE,
                             num_workers = args.TRAIN_NUMBER_WORKDERS)  
    
    val_dataset = torchvision.datasets.ImageFolder(args.VAL_DATASET_PATH, data_transforms["val"])
    valloader = DataLoader(val_dataset,
                           batch_size = args.VAL_BATCH_SIZE,
                           shuffle = args.VAL_SHUFFLE,
                           num_workers = args.VAL_NUMBER_WORKDERS)  
    
    # combine
    dataloader.append({"train" : trainloader, "val" : valloader})
    dataset_sizes.append({"train" : len(trainloader), "val" : len(valloader)})
    return dataloader, dataset_sizes, None

def push_load_data(args):
    
    train_dataset = torchvision.datasets.ImageFolder(args.TRAIN_DATASET_PATH, 
                                                    transforms.Compose([
                                                        transforms.Resize(size=(args.TRAIN_IMAGE_SIZE, args.TRAIN_IMAGE_SIZE)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(args.TRAIN_MEAN, args.TRAIN_STD)
                                                    ]))
    trainloader = DataLoader(train_dataset,
                             batch_size = args.TRAIN_BATCH_SIZE * 2,
                             shuffle = False,
                             num_workers = args.TRAIN_NUMBER_WORKDERS)  
    
    # combine
    
    return trainloader
    
    
    
    