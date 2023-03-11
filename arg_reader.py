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
    parser = argparse.ArgumentParser(description="Copy multiple Files from a specified data file")
    parser.add_argument("-c", "--configfile", type = str, default = "./config.py", help = "file to read the config from")
    parser.add_argument("--local_rank", type = int, default = -1, help = "DDP parameter. (Don't modify !!)")
    args = vars(parser.parse_args())
    cfg.update(args)

    return edict(cfg)