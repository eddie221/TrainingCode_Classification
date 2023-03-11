#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:46:39 2021

@author: mmplab603
"""

import torchvision.models.resnet as resnet
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch
import importlib
from utils.general import info_log
import logging

# =============================================================================
# Return model
# =============================================================================
def load_model(args):
    # import model module
    if "resnet" in args.MODEL.lower():
        model_module = importlib.import_module("ResNet")

    # load model
    model = model_module.load_model(args.BASIC_MODEL, args.CATEGORY)

    use_cuda = args.device_id != "cpu"
    
    # DP mode
    if use_cuda and args.global_rank == -1 and torch.cuda.device_count() >= 1:
        info_log("DP mode", rank = args.local_rank, type = args.INFO_SHOW)
        model = model.to(args.device_id)
        model = torch.nn.DataParallel(model, device_ids = args.DEVICE)

    # use SyncBatchNorm
    if use_cuda and args.global_rank != -1:
        info_log("Using SyncBatchNorm", rank = args.local_rank, type = args.INFO_SHOW)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device_id)
    
    # DDP mode
    if use_cuda and args.global_rank != -1:
        info_log("DDP mode", rank = args.local_rank, type = args.INFO_SHOW)
        model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank)

    if "PRETRAINED_PATH" in args:
        info_log("Load pretrained weight !", rank = args.local_rank, type = args.INFO_SHOW)
        model_module.load_pretrained(args.PRETRAINED_PATH, model)

    return model

# =============================================================================
# Custom model
# =============================================================================
def custom_model():
    pass

# =============================================================================
# Load model weight
# =============================================================================
def load_param(model):
    # load resnet
    params = torch.load("../pretrain/resnet50.pth")
    load = []
    not_load = []
    for name, param in params.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except:
                not_load.append(name)
        else:
            not_load.append(name)
    print("Load {} layers".format(len(load)))
    print("Not load {} layers".format(len(not_load)))
            
    return model