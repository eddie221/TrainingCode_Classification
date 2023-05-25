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

def create_dataset(batch_size, data_path, data_transforms, shuffle, num_workers, world_size, global_rank):
    dataset = torchvision.datasets.ImageFolder(data_path, data_transforms)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle) if global_rank != -1 else None
    num_workers = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, num_workers])  # number of workers
    loader = DataLoader(dataset,
                            batch_size = batch_size,
                            sampler = sampler if global_rank != -1 else None,
                            shuffle = shuffle if global_rank == -1 else None,
                            num_workers = num_workers) 
    return dataset, loader

def load_data(args):
    dataloader = []
    dataset_sizes = []
    
    with main_process_first(args.global_rank):
        data_transforms = prepare_transforms(args)
        info_log("Image preprocess : {}".format(data_transforms), args.global_rank, type = args.INFO_SHOW)
        train_dataset, trainloader = create_dataset(batch_size = args.TRAIN_BATCH_SIZE,
                                                    data_path = args.TRAIN_DATASET_PATH,
                                                    data_transforms = data_transforms["train"],
                                                    shuffle = args.TRAIN_SHUFFLE,
                                                    num_workers = args.TRAIN_NUMBER_WORKDERS,
                                                    world_size = args.world_size,
                                                    global_rank = args.global_rank)
        
        val_dataset, valloader = create_dataset(batch_size = args.VAL_BATCH_SIZE,
                                                    data_path = args.VAL_DATASET_PATH,
                                                    data_transforms = data_transforms["val"],
                                                    shuffle = args.VAL_SHUFFLE,
                                                    num_workers = args.VAL_NUMBER_WORKDERS,
                                                    world_size = args.world_size,
                                                    global_rank = args.global_rank)
        
        # combine
        dataloader.append({"train" : trainloader, "val" : valloader})
        dataset_sizes.append({"train" : len(trainloader), "val" : len(valloader)})

    return dataloader, dataset_sizes, None
    