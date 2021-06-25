#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:09:48 2021

@author: mmplab603
"""

import torch
import arg_reader
import numpy as np
import importlib
from load_model import load_model
import logging
import tqdm
import os

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('../tensorflow/logs/cub_{}'.format(INDEX), comment = "224_64")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_nn_model():
    pass

def create_opt_loss(model):
    pass

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

# run one iteration
def one_step(args, model, data, label, loss_func, optimizers, phase):
    if args.getint("DEVICE") != -1:
        b_data = data.cuda()
        b_label = label.cuda()
    else:
        b_data = data
        b_label = label
        
    for optimizer in optimizers:
        optimizer.zero_grad() 
        
    output_1 = model(b_data)
    
    _, predicted = torch.max(output_1.data, 1)
    #_, predicted5 = torch.topk(output_1.data, 5, dim = 1)
    
    cls_loss = loss_func[0](output_1, b_label)
    loss = cls_loss
    return loss.item(), predicted.detach().cpu()#, predicted5.detach().cpu()

# load data, load model (pretrain if needed), define loss function, define optimizer, 
# define learning rate scheduler (if needed), training and validation
def runs(args):
    
    # load dataset ------------------------------------------------------------
    if "DATALOADER" in args["DEFAULT"]:
        if "KFOLD" in args["DEFAULT"]:
            dataloader = importlib.import_module(args["DEFAULT"].get("DATALOADER"))
            dataset, dataset_sizes, all_image_datasets = dataloader.load_data(args)
    else:
        dataloader = importlib.import_module("load_data_train_val_classify")
        dataset, dataset_sizes = dataloader.load_data(args)
        all_image_datasets = None
    # -------------------------------------------------------------------------
    
    for index, image_data in enumerate(dataset):

        # load model (load pretrain if needed) ------------------------------------
        model = load_model(args["DEFAULT"])
        # -------------------------------------------------------------------------
        
        # define loss -------------------------------------------------------------
        loss_funcs = []
        if args["DEFAULT"].get("LOSS") == "CE":
            loss_funcs.append(torch.nn.CrossEntropyLoss())
        assert len(loss_funcs) != 0, "Miss define loss"
        # -------------------------------------------------------------------------
        
        # define optimizer --------------------------------------------------------
        optimizers = []
        if args["DEFAULT"].get("OPTIMIZER") == "ADAM":
            optimizers.append(torch.optim.Adam(model.parameters(), lr = args["DEFAULT"].getfloat("LR"), weight_decay = args["DEFAULT"].getfloat("WD")))
        assert len(optimizers) != 0, "Miss define optimizer"
        # -------------------------------------------------------------------------
        
        # define learning rate scheduler ------------------------------------------
        lr_schedulers = []
        if "LR_SCHEUDLER" in args["DEFAULT"]:
            lr_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[75, 150, 225, 300, 375], gamma=0.1))
        # -------------------------------------------------------------------------
        
        # Define Meters -------------------------------------------------------
        ACCMeters = []
        ACCMeters5 = []
        LOSSMeters = []
        for i in range(args["DEFAULT"].getint("KFOLD")):
            ACCMeters.append(AverageMeter())
            ACCMeters5.append(AverageMeter())
            LOSSMeters.append(AverageMeter())
        max_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
        max_acc5 = {'train' : AverageMeter(), 'val' : AverageMeter()}
        min_loss = {'train' : AverageMeter(), 'val' : AverageMeter()}
        last_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
        last_acc5 = {'train' : AverageMeter(), 'val' : AverageMeter()}
        # ---------------------------------------------------------------------
        
        # Train ---------------------------------------------------------------
        for epoch in range(1, args["DEFAULT"].getint("EPOCH") + 1):
            print('Fold {}/{} Epoch {}/{}'.format(index + 1, args["DEFAULT"].getint("KFOLD"), epoch, args["DEFAULT"].getint("EPOCH")))
            logging.info("-" * 15)
            logging.info('Fold {}/{} Epoch {}/{}'.format(index + 1, args["DEFAULT"].getint("KFOLD"), epoch, args["DEFAULT"].getint("EPOCH")))
            print('-' * 10)
            for phase in ["train", "val"]:
                correct_t = AverageMeter()
                correct_t5 = AverageMeter()
                loss_t = AverageMeter()
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)
                step = 0
                for data, label in tqdm.tqdm(image_data[phase]):
                    #loss, predicted, predicted5 = one_step(args["DEFAULT"], model, data, label, loss_funcs, optimizers, phase)
                    loss, predicted = one_step(args["DEFAULT"], model, data, label, loss_funcs, optimizers, phase)
                    
                    loss_t.update(loss, data.size(0))
                    correct_t.update((predicted == label).sum().item() / label.shape[0], label.shape[0])
                    #correct_t5.update((predicted5 == label.unsqueeze(1)).sum().item() / label.shape[0], label.shape[0])
                    step += 1
                # save model --------------------------------------------------
# =============================================================================
#                 # top5 
#                 if max_acc5[phase].avg < correct_t5.avg:
#                     last_acc5[phase] = max_acc5[phase]
#                     max_acc5[phase] = correct_t5
#                     
#                     if phase == 'val':
#                         ACCMeters5[index] = correct_t
#                         save_data = model.state_dict()
#                         print('save')
#                         torch.save(save_data, './pkl/{}/fold_{}_best5_{}.pkl'.format(args["DEFAULT"].get("INDEX"), index, args["DEFAULT"].get("INDEX")))
# =============================================================================
                        
                # top1
                if max_acc[phase].avg < correct_t.avg:
                    last_acc[phase] = max_acc[phase]
                    max_acc[phase] = correct_t
                    
                    if phase == 'val':
                        ACCMeters[index] = correct_t
                        LOSSMeters[index] = loss_t
                        save_data = model.state_dict()
                        print('save')
                        torch.save(save_data, './pkl/{}/fold_{}_best_{}.pkl'.format(args["DEFAULT"].get("INDEX"), index, args["DEFAULT"].get("INDEX")))
                # -------------------------------------------------------------
                logging.info("{} set loss : {:.6f}".format(phase, loss_t.avg))        
                logging.info("{} set top-1 acc : {:.6f}%".format(phase, correct_t.avg * 100.))  
                logging.info("{} set top-5 acc : {:.6f}%".format(phase, correct_t5.avg * 100.))  
                print('Index : {}'.format(args["DEFAULT"].get("INDEX")))
                print("dataset : {}".format(args["DEFAULT"].get("DATASET_NAME") if "DATASET_NAME" in args["DEFAULT"] else args["DEFAULT"]["DATASET_PATH"].split('/')[-1]))
                print("Model name : {}".format(args["DEFAULT"].get("Model")))
                print("{} set loss : {:.6f}".format(phase, loss_t.avg))
                print("{} set acc : {:.6f}%".format(phase, correct_t.avg * 100.))
                print("{} last update : {:.6f}%".format(phase, (max_acc[phase].avg - last_acc[phase].avg) * 100.))
                print("{} set max acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.))
# =============================================================================
#                 print("{} set acc(5) : {:.6f}%".format(phase, correct_t5.avg * 100.))
#                 print("{} last update(5) : {:.6f}%".format(phase, (max_acc5[phase].avg - last_acc5[phase].avg) * 100.))
#                 print("{} set max acc(5) : {:.6f}%".format(phase, max_acc5[phase].avg * 100.))
# =============================================================================
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        # ---------------------------------------------------------------------
    acc = 0
    acc5 = 0
    loss = 0
    for idx in range(1, len(ACCMeters) + 1):
        print("Fold {} best acc : {:.6f} acc(5) : {:.6f} loss : {:.6f}".format(idx, ACCMeters[idx - 1].avg, ACCMeters5[idx - 1].avg, LOSSMeters[idx - 1].avg))
        acc += ACCMeters[idx - 1].avg
        acc5 += ACCMeters5[idx - 1].avg
        loss += LOSSMeters[idx - 1].avg
    print("Avg. ACC : {:.6f} Avg. ACC(5) : {:.6f} Avg. Loss : {:.6f}".format(acc / args["DEFAULT"].getint("KFOLD"), acc5 / args["DEFAULT"].getint("KFOLD"),loss / args["DEFAULT"].getint("KFOLD")))
    pass    

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count

if __name__ == '__main__':
    args = arg_reader.read_args()
    if args.configfile is not None:
        args = arg_reader.read_cfg(args.configfile)
    if not os.path.exists("./pkl"):
        os.mkdir("./pkl")
    if not os.path.exists("./pkl/{}".format(args["DEFAULT"].get("INDEX"))):
        os.mkdir("./pkl/{}".format(args["DEFAULT"].get("INDEX")))
    logging.basicConfig(filename = './pkl/{}/logging.txt'.format(args["DEFAULT"].get("INDEX")), level=logging.DEBUG)
    logging.info('Index : {}'.format(args["DEFAULT"].get("INDEX")))
    logging.info("dataset : {}".format(args["DEFAULT"].get("DATASET_NAME") if "DATASET_NAME" in args["DEFAULT"] else args["DEFAULT"]["DATASET_PATH"].split('/')[-1] ))
    
    runs(args)
        