#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:59:13 2021

@author: mmplab603
"""

import torchvision
from torch.utils.data import DataLoader
from data_transform import prepare_transforms


def load_data(args):
    data_transforms = prepare_transforms(args)
    all_image_datasets = torchvision.datasets.ImageFolder(args["DEFAULT"]["TRAIN_DATASET_PATH"], data_transforms["train"])
    
    dataloader = []
    dataset_sizes = []
    if int(args["DEFAULT"]["KFOLD"]) != 1:
        # package only use in this situation
        from sklearn.model_selection import KFold
        from torch.utils.data import Subset
        
        kf = KFold(args["DEFAULT"].getint("KFOLD"), shuffle = True)
        for train_idx, val_idx in kf.split(all_image_datasets):
            # training set
            train_dataset = Subset(all_image_datasets, train_idx)
            trainloader = DataLoader(train_dataset,
                                     batch_size = args["TRAIN"].getint("BATCH_SIZE"),
                                     shuffle = args["TRAIN"].getboolean("SHFFLE"),
                                     num_workers = args["TRAIN"].getint("NUMBER_WORKDERS"))    
            # validation set
            val_dataset = Subset(all_image_datasets, val_idx)
            valloader = DataLoader(val_dataset,
                                   batch_size = args["TRAIN"].getint("BATCH_SIZE"),
                                     shuffle = args["TRAIN"].getboolean("SHFFLE"),
                                     num_workers = args["TRAIN"].getint("NUMBER_WORKDERS"))    
            
            # combine
            dataloader.append({"train" : trainloader, "val" : valloader})
            dataset_sizes.append({"train" : len(trainloader), "val" : len(valloader)})
    else:
        # package only use in this situation
        import numpy as np
        from torch.utils.data.sampler import SubsetRandomSampler
        
        indices = list(range(len(all_image_datasets)))
        dataset_size = len(all_image_datasets)
        split = int(np.floor(args["DEFAULT"].getfloat("VAL_SPLIT")* dataset_size))
        # shuffle the dataset
        np.random.seed(0)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        trainloader = DataLoader(all_image_datasets,
                                 batch_size = args["TRAIN"].getint("BATCH_SIZE"),
                                 sampler = train_sampler,
                                 num_workers = args["TRAIN"].getint("NUMBER_WORKDERS")) 
        
        valloader = DataLoader(all_image_datasets,
                               batch_size = args["VALIDATION"].getint("BATCH_SIZE"),
                               sampler = valid_sampler,
                               num_workers = args["VALIDATION"].getint("NUMBER_WORKDERS"))    
        
        dataloader.append({"train" : trainloader, "val" : valloader})
        dataset_sizes.append({"train" : len(trainloader), "val" : len(valloader)})
        
        
    return dataloader, dataset_sizes, all_image_datasets