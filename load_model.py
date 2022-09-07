#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:46:39 2021

@author: mmplab603
"""

import torchvision.models.resnet as resnet
import torch.nn as nn
import torch
import importlib

# =============================================================================
# Return model
# =============================================================================
def load_model(args):
    #model = importlib.import_module(args.MODEL)
    #model = model.load_model(args.CATEGORY)
    model = resnet.resnet50(num_classes = args.CATEGORY)
    if "PARAMETER_PATH" in args:
        model = load_parameter(model, args)
    model = nn.DataParallel(model, device_ids = args.DEVICE)
    return model


# =============================================================================
# Load pretrain model parameter
# =============================================================================
def load_parameter(model, args):
    import torch
    params = torch.load(args.PARAMETER_PATH)
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