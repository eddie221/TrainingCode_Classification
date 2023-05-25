#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:09:48 2021

@author: mmplab603
"""

import torch
import arg_reader
import importlib
from load_model import load_model
import logging
import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from config import cfg
import sys
from utils.general import info_log
from utils.env_check import check_device
from easydict import EasyDict as edict
import shutil
import time

# =============================================================================
# Get optimizer learning rate
# =============================================================================
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Train one iteration
def train(model, data, label, loss_func, optimizers, args):
    if args.DEVICE != -1:
        b_data = data.to(args.device_id)
        b_label = label.to(args.device_id)
    else:
        b_data = data
        b_label = label
    for optimizer in optimizers:
        optimizer.zero_grad() 
    
    # Model forward
    output = model(b_data)

    # Get prediction 
    _, predicted = torch.max(output.data, 1)
    #_, predicted5 = torch.topk(output_1.data, 5, dim = 1)
    
    # calculate loss
    cls_loss = loss_func["CE"](output, b_label)

    loss = cls_loss
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    losses = {
                "cls_loss" : cls_loss.detach(), 
             }
    
    return losses, predicted.detach().cpu()#, predicted5.detach().cpu()

# Test one iteration
def test(model, data, label, loss_func, optimizers, args):
    with torch.no_grad():
        if args.DEVICE != -1:
            b_data = data.to(args.device_id)
            b_label = label.to(args.device_id)
        else:
            b_data = data
            b_label = label
            
        # Model forward
        output = model(b_data)

        # Get prediction 
        _, predicted = torch.max(output.data, 1)
        #_, predicted5 = torch.topk(output_1.data, 5, dim = 1)
        
        # calculate loss
        cls_loss = loss_func["CE"](output, b_label)

        loss = cls_loss
        losses = {
                    "cls_loss" : cls_loss.detach(), 
                }
    
    return losses, predicted.detach().cpu()#, predicted5.detach().cpu()

# =============================================================================
# Load data, load model (pretrain if needed), define loss function, define optimizer, 
# define learning rate scheduler (if needed), training and validation
# =============================================================================
def runs(args):
    # Load dataset ------------------------------------------------------------
    dataloader = importlib.import_module(args.DATALOADER)
    dataset, dataset_sizes, all_image_datasets = dataloader.load_data(args)
    # -------------------------------------------------------------------------
    
    # Define tensorboard for recording ----------------------------------------
    if args.global_rank in [-1, 0]:
        writer = SummaryWriter('./logs/{}'.format(args.INDEX))
    # -------------------------------------------------------------------------
    
    for index, image_data in enumerate(dataset):
        # resume training process ------------------------------------------------
        start_epoch = 1
        if args.RESUME:
            resume_data = torch.load(args.WEIGHT_PATH)
            args.m = resume_data['m']
            args.m_is_concept_num = resume_data['m_is_concept_num']
            start_epoch = resume_data["Epoch"] + 1
        # ------------------------------------------------------------------------

        # Load model (load pretrain if needed) ------------------------------------
        model = load_model(args)
        # -------------------------------------------------------------------------
        
        # Define loss -------------------------------------------------------------
        loss_funcs = {}
        if args.LOSS == "CE":
            loss_funcs[args.LOSS] = torch.nn.CrossEntropyLoss()
        assert len(loss_funcs) != 0, "Miss define loss"
        # -------------------------------------------------------------------------
        
        # Define optimizer --------------------------------------------------------
        train_optimizers = []
        if args.OPTIMIZER == "ADAM":
            train_optimizers.append(torch.optim.Adam(model.parameters(), lr = args.LR, weight_decay = args.WD))

        if args.RESUME:
            for i in range(len(resume_data["Optimizer"])):
                train_optimizers[i].load_state_dict(resume_data["Optimizer"][i])
        assert len(train_optimizers) != 0, "Miss define optimizer"
        # -------------------------------------------------------------------------
        
        # Define learning rate scheduler ------------------------------------------
        lr_schedulers = []
        if "LR_SCHEUDLER" in args:
            lr_schedulers.append(torch.optim.lr_scheduler.StepLR(train_optimizers[0], step_size = args.joint_lr_step_size, gamma = 0.1))
        if args.RESUME:
            for i in range(len(resume_data["LR_scheduler"])):
                lr_schedulers[i].load_state_dict(resume_data["LR_scheduler"][i])
        # -------------------------------------------------------------------------
        
        # Define Meters -------------------------------------------------------
        ACCMeters = []
        ACCMeters5 = []
        LOSSMeters = []
        for i in range(args.KFOLD):
            ACCMeters.append(AverageMeter())
            ACCMeters5.append(AverageMeter())
            LOSSMeters.append(AverageMeter())
        max_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
        max_acc5 = {'train' : AverageMeter(), 'val' : AverageMeter()}
        min_loss = {'train' : AverageMeter(), 'val' : AverageMeter()}
        last_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
        last_acc5 = {'train' : AverageMeter(), 'val' : AverageMeter()}
        # ---------------------------------------------------------------------
        
        # Start training process ---------------------------------------------------------------
        for epoch in range(start_epoch, args.EPOCH + 1):
            info_log('Fold {}/{} Epoch {}/{}'.format(index + 1, args.KFOLD, epoch, args.EPOCH), type = args.INFO_SHOW)
            info_log("-" * 15, type = args.INFO_SHOW)

            for phase in ["train", "val"]:
                correct_t = AverageMeter()
                correct_t5 = AverageMeter()
                loss_t = AverageMeter()
                loss_detail_t = {}
                if phase == 'train':
                    model.train(True)
                    optimizers = train_optimizers
                else:
                    model.train(False)
                
                if args.global_rank != -1:
                    image_data["train"].sampler.set_epoch(epoch)
                    image_data["val"].sampler.set_epoch(epoch)
                    
                data_bar = enumerate(image_data[phase])
                if args.global_rank in [-1, 0]:
                    data_bar = tqdm.tqdm(data_bar, total = len(image_data[phase]))

                for step, (data, label) in data_bar:
                    #loss, predicted, predicted5 = one_step(args["DEFAULT"], model, data, label, loss_funcs, optimizers, phase)
                    if phase == "train":
                        losses, predicted = train(model, data, label, loss_funcs, optimizers, args)
                    else:
                        losses, predicted = test(model, data, label, loss_funcs, optimizers, args)
                    loss = 0
                    for key in losses.keys():
                        loss += losses[key]
                        if step == 0:
                            loss_detail_t[key] = AverageMeter()
                        loss_detail_t[key].update(losses[key], data.size(0))
                        
                    loss_t.update(loss, data.size(0))
                    correct_t.update((predicted == label).sum().item() / label.shape[0], label.shape[0])
                    #correct_t5.update((predicted5 == label.unsqueeze(1)).sum().item() / label.shape[0], label.shape[0])
                    for lr_scheduler in lr_schedulers:
                        lr_scheduler.step()
                if args.global_rank in [-1, 0]:
                    # Recording loss and accuracy ---------------------------------
                    writer.add_scalar('Loss/{}'.format(phase), loss_t.avg, epoch)
                    writer.add_scalar('Accuracy/{}'.format(phase), correct_t.avg, epoch)
                    # -------------------------------------------------------------
                    
                    # Save model --------------------------------------------------
                    # top5 
                    # if max_acc5[phase].avg < correct_t5.avg:
                    #     last_acc5[phase] = max_acc5[phase]
                    #     max_acc5[phase] = correct_t5
                        
                    #     if phase == 'val':
                    #         ACCMeters5[index] = correct_t
                    #         save_data = model.state_dict()
                    #         print('save')
                    #         torch.save(save_data, './pkl/{}/fold_{}_best5_{}.pkl'.format(args.INDEX, index, args.INDEX))
                            
                    # top1
                    if max_acc[phase].avg < correct_t.avg:
                        last_acc[phase] = max_acc[phase]
                        max_acc[phase] = correct_t

                        optimizers_state_dict = []
                        for tmp in train_optimizers:
                            optimizers_state_dict.append(tmp.state_dict())
                        lr_state_dict = []
                        for tmp in lr_schedulers:
                            lr_state_dict.append(tmp.state_dict())
                        if phase == 'val':
                            ACCMeters[index] = correct_t
                            LOSSMeters[index] = loss_t
                            save_data = {"Model" : model.state_dict(),
                                        "Epoch" : args.EPOCH,
                                        "Optimizer" : optimizers_state_dict,
                                        "LR_scheduler" : lr_state_dict,
                                        "Best ACC" : max_acc[phase].avg,
                                        "Time" : args.start,
                                        "Loss" : loss_t,
                                        "ACC" : max_acc}
                            torch.save(save_data, './pkl/{}/{}_{}/fold_{}_best_{}.pkl'.format(args.INDEX, args.MODEL.lower(), args.BASIC_MODEL.lower(), index, args.INDEX))
                    
                    optimizers_state_dict = []
                    for tmp in train_optimizers:
                        optimizers_state_dict.append(tmp.state_dict())
                    lr_state_dict = []
                    for tmp in lr_schedulers:
                        lr_state_dict.append(tmp.state_dict())
                    if phase == 'val':
                        ACCMeters[index] = correct_t
                        LOSSMeters[index] = loss_t
                        save_data = {"Model" : model.state_dict(),
                                    "Epoch" : args.EPOCH,
                                    "Optimizer" : optimizers_state_dict,
                                    "LR_scheduler" : lr_state_dict,
                                    "Best ACC" : max_acc[phase].avg,
                                    "Time" : args.start,
                                    "Loss" : loss_t,
                                    "ACC" : max_acc}
                        torch.save(save_data, './pkl/{}/{}_{}/fold_{}_last_{}.pkl'.format(args.INDEX, args.MODEL.lower(), args.BASIC_MODEL.lower(), index, args.INDEX))
                    # -------------------------------------------------------------
                    info_log('Index : {}'.format(args.INDEX), type = args.INFO_SHOW)
                    info_log("dataset : {}".format(args.DATASET_NAME), type = args.INFO_SHOW)
                    info_log("Model name : {}_{}".format(args.MODEL, args.BASIC_MODEL), type = args.INFO_SHOW)
                    info_log("{} set loss : {:.6f}".format(phase, loss_t.avg), type = args.INFO_SHOW)
                    for key in loss_detail_t.keys():
                        info_log("    {} set {} : {:.6f}".format(phase, key, loss_detail_t[key].avg), type = args.INFO_SHOW)
                    info_log("{} set top-1 acc : {:.6f}%".format(phase, correct_t.avg * 100.), type = args.INFO_SHOW)
                    info_log("{} last update : {:.6f}%".format(phase, (max_acc[phase].avg - last_acc[phase].avg) * 100.), type = args.INFO_SHOW)
                    info_log("{} set best acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.), type = args.INFO_SHOW)
                    # print("{} set acc(5) : {:.6f}%".format(phase, correct_t5.avg * 100.))
                    # print("{} last update(5) : {:.6f}%".format(phase, (max_acc5[phase].avg - last_acc5[phase].avg) * 100.))
                    # print("{} set max acc(5) : {:.6f}%".format(phase, max_acc5[phase].avg * 100.))
            # for lr_scheduler in lr_schedulers:
            #     lr_scheduler.step()
        # ---------------------------------------------------------------------
    
    # Show the best result ----------------------------------------------------
    if args.global_rank in [-1, 0]:
        acc = 0
        acc5 = 0
        loss = 0
        for idx in range(1, len(ACCMeters) + 1):
            info_log("Fold {} best acc : {:.6f} acc(5) : {:.6f} loss : {:.6f}".format(idx, ACCMeters[idx - 1].avg, ACCMeters5[idx - 1].avg, LOSSMeters[idx - 1].avg), type = args.INFO_SHOW)
            acc += ACCMeters[idx - 1].avg
            acc5 += ACCMeters5[idx - 1].avg
            loss += LOSSMeters[idx - 1].avg
        info_log("Avg. ACC : {:.6f} Avg. ACC(5) : {:.6f} Avg. Loss : {:.6f}".format(acc / args.KFOLD, acc5 / args.KFOLD,loss / args.KFOLD), type = args.INFO_SHOW)

# =============================================================================
# Templet for recording values
# =============================================================================
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
    args = arg_reader.read_args(**cfg)
    
    # Set DDP variables
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # check if it can run on gpu
    device_id = check_device(args.DEVICE, args.TRAIN_BATCH_SIZE, args.VAL_BATCH_SIZE)
    args.TRAIN_TOTAL_BATCH_SIZE = args.TRAIN_BATCH_SIZE
    args.VAL_TOTAL_BATCH_SIZE = args.VAL_BATCH_SIZE
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device_id = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='gloo', init_method='env://')  # distributed backend
        assert args.TRAIN_IMAGE_SIZE % args.world_size == 0, 'TRAIN_IMAGE_SIZE must be multiple of CUDA device count'
        assert args.VAL_IMAGE_SIZE % args.world_size == 0, 'VAL_IMAGE_SIZE size be multiple of CUDA device count'
        args.TRAIN_BATCH_SIZE = args.TRAIN_TOTAL_BATCH_SIZE // args.world_size
        args.VAL_BATCH_SIZE = args.VAL_TOTAL_BATCH_SIZE // args.world_size

    if args.global_rank in [-1, 0]:
        if not os.path.exists("./pkl"):
            os.mkdir("./pkl")
        if not os.path.exists("./pkl/{}/".format(args.INDEX)):
            os.mkdir("./pkl/{}/".format(args.INDEX))
        
        if not os.path.exists("./pkl/{}/{}_{}".format(args.INDEX, args.MODEL.lower(), cfg.BASIC_MODEL.lower())):
            os.mkdir("./pkl/{}/{}_{}".format(args.INDEX, args.MODEL.lower(), cfg.BASIC_MODEL.lower()))
        elif not args.RESUME:
            response = input("The experiment already exist ({}/{}_{}). Are you sure you want replace it? (y/n)".format(args.INDEX, args.MODEL.lower(), cfg.BASIC_MODEL.lower())).lower()
            while response != 'y' and response != 'n':
                response = input("The experiment already exist ({}/{}_{}). Are you sure you want replace it? (y/n)".format(args.INDEX, args.MODEL.lower(), cfg.BASIC_MODEL.lower())).lower()
            if response == 'n':
                import sys
                sys.exit()
        
            with open("./pkl/{}/{}_{}/logging.txt".format(args.INDEX, args.MODEL.lower(), cfg.BASIC_MODEL.lower()), "w") as f:
                print(args, file = f)

            info_log("Args : {}".format(args), type = args.INFO_SHOW)

            # save file to specific direction -----------------------------------------
            dst = "./pkl/{}/{}_{}".format(args.INDEX, args.MODEL.lower(), cfg.BASIC_MODEL.lower())
            shutil.copy(src = os.path.join(os.getcwd(), __file__), dst = dst)
            shutil.copy(src = os.path.join(os.getcwd(), "config.py"), dst = dst)
            if "resnet" in args.MODEL.lower():
                shutil.copy(src = os.path.join(os.getcwd(), "ResNet.py"), dst = dst)
            else:
                shutil.copy(src = os.path.join(os.getcwd(), "{}.py".format(args.MODEL)), dst = dst)
                shutil.copy(src = os.path.join(os.getcwd(), "ResNet.py"), dst = dst)
                shutil.copy(src = os.path.join(os.getcwd(), "loss.py"), dst = dst)
            shutil.copy(src = os.path.join(os.getcwd(), "load_model.py"), dst = dst)
            # -------------------------------------------------------------------------
            info_log('Index : {}'.format(args.INDEX), type = args.INFO_SHOW)
            info_log("dataset : {}".format(args.DATASET_NAME), type = args.INFO_SHOW)

            args.start = time.time()

    args.device_id = device_id
    runs(args)
    if args.global_rank in [-1, 0]:
        info_log("Total training time : {:.2f} hours".format((time.time() - args.start) / 3600), type = args.INFO_SHOW)