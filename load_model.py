#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:46:39 2021

@author: mmplab603
"""

import torchvision.models.resnet as resnet
import torch.nn as nn
import torch

# =============================================================================
# Return model
# =============================================================================
def load_model(args):
    model = resnet.resnet50(num_classes = args.getint("CATEGORY"))
    if "PARAMETER_PATH" in args:
        model = load_parameter(args, model)
    model = nn.DataParallel(model, device_ids = list(map(int, args.get("DEVICE").split(","))))
    return model


# =============================================================================
# Load pretrain model parameter
# =============================================================================
def load_parameter(args, model):
    import torch
    params = torch.load(args.get("PARAMETER_PATH"))
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