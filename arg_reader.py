#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:19:56 2021

@author: mmplab603
"""

import argparse
import configparser
from easydict import EasyDict as edict

def read_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Copy multiple Files from a specified data file')
    parser.add_argument('-c', '--configfile', default="./config.cfg", help='file to read the config from')
    args = vars(parser.parse_args())
    cfg.update(args)

    return edict(cfg)

# =============================================================================
# def read_cfg(path):
#     config = configparser.ConfigParser()
#     config.read(path)
#     if 'TRAIN' not in config:
#         print("Missing training section.")
#     if 'VALIDATION' not in config:
#         print("Missing validation section.")
#     
#     return config
# =============================================================================
