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
from utils.general import info_log, main_process_first
import os
import torch

def load_data(args):
    dataloader = []
    dataset_sizes = []
    
    with main_process_first(args.global_rank):
        data_transforms = prepare_transforms(args)
        info_log("Image preprocess : {}".format(data_transforms), args.global_rank, type = args.INFO_SHOW)
        train_dataset = torchvision.datasets.ImageFolder(args.TRAIN_DATASET_PATH, data_transforms["train"])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.global_rank != -1 else None
        train_num_workers = min([os.cpu_count() // args.world_size, args.TRAIN_BATCH_SIZE if args.TRAIN_BATCH_SIZE > 1 else 0, args.TRAIN_NUMBER_WORKDERS])  # number of workers
        trainloader = DataLoader(train_dataset,
                                batch_size = args.TRAIN_BATCH_SIZE,
                                sampler = train_sampler if args.global_rank != -1 else None,
                                shuffle = args.TRAIN_SHUFFLE if args.global_rank == -1 else None,
                                num_workers = train_num_workers)  
        
        val_dataset = torchvision.datasets.ImageFolder(args.VAL_DATASET_PATH, data_transforms["val"])
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.global_rank != -1 else None
        val_num_workers = min([os.cpu_count() // args.world_size, args.VAL_BATCH_SIZE if args.VAL_BATCH_SIZE > 1 else 0, args.VAL_NUMBER_WORKDERS])  # number of workers
        valloader = DataLoader(val_dataset,
                            batch_size = args.VAL_BATCH_SIZE,
                            sampler = val_sampler if args.global_rank != -1 else None,
                            shuffle = args.VAL_SHUFFLE if args.global_rank == -1 else None,
                            num_workers = val_num_workers)
        
        # combine
        dataloader.append({"train" : trainloader, "val" : valloader})
        dataset_sizes.append({"train" : len(trainloader), "val" : len(valloader)})

    return dataloader, dataset_sizes, None
    