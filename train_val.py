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
from config import cfg

# =============================================================================
# Get optimizer learning rate
# =============================================================================
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# =============================================================================
# Run one iteration
# =============================================================================
def one_step(model, data, label, loss_func, optimizers, phase, args):
    if args.DEVICE != -1:
        b_data = data.cuda()
        b_label = label.cuda()
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
    cls_loss = loss_func[0](output, b_label)

    loss = cls_loss
    
    if phase == 'train':
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    
    return loss.item(), predicted.detach().cpu()#, predicted5.detach().cpu()

# =============================================================================
# Load data, load model (pretrain if needed), define loss function, define optimizer, 
# define learning rate scheduler (if needed), training and validation
# =============================================================================
def runs(args):
    # Load dataset ------------------------------------------------------------
    dataloader = importlib.import_module(args.DATALOADER)
    dataset, dataset_sizes, all_image_datasets = dataloader.load_data(args)
    push_dataset = dataloader.push_load_data(args)
    # -------------------------------------------------------------------------
    
    # 
    # Define tensorboard for recording ----------------------------------------
    writer = SummaryWriter('./logs/{}'.format(args.INDEX))
    # -------------------------------------------------------------------------
    
    for index, image_data in enumerate(dataset):

        # Load model (load pretrain if needed) ------------------------------------
        model = load_model(args)
        # -------------------------------------------------------------------------
        
        # Define loss -------------------------------------------------------------
        loss_funcs = []
        if args.LOSS == "CE":
            loss_funcs.append(torch.nn.CrossEntropyLoss())
        assert len(loss_funcs) != 0, "Miss define loss"
        # -------------------------------------------------------------------------
        
        # Define optimizer --------------------------------------------------------

        train_optimizers = []
        if args.OPTIMIZER == "ADAM":
            train_optimizers.append(torch.optim.Adam(model.parameters(), lr = args.LR, weight_decay = args.WD))
        
        assert len(train_optimizers) != 0, "Miss define optimizer"
        # -------------------------------------------------------------------------
        
        # Define learning rate scheduler ------------------------------------------
        lr_schedulers = []
        if "LR_SCHEUDLER" in args:
            #lr_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=[75, 150, 225, 300, 375], gamma=0.1))
            lr_schedulers.append(torch.optim.lr_scheduler.StepLR(train_optimizers[0], step_size = args.joint_lr_step_size, gamma = 0.1))
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
        
        # Train ---------------------------------------------------------------
        for epoch in range(1, args.EPOCH + 1):
            print('Fold {}/{} Epoch {}/{}'.format(index + 1, args.EPOCH, epoch, args.EPOCH))
            logging.info("-" * 15)
            logging.info('Fold {}/{} Epoch {}/{}'.format(index + 1, args.KFOLD, epoch, args.EPOCH))
            print('-' * 10)
            for phase in ["train", "val"]:
                correct_t = AverageMeter()
                correct_t5 = AverageMeter()
                loss_t = AverageMeter()
                if phase == 'train':
                    model.train(True)
                    optimizers = train_optimizers
                else:
                    model.train(False)

                for step, (data, label) in enumerate(tqdm.tqdm(image_data[phase])):
                    #loss, predicted, predicted5 = one_step(args["DEFAULT"], model, data, label, loss_funcs, optimizers, phase)
                    loss, predicted = one_step(model, data, label, loss_funcs, optimizers, phase, args)
                    
                    loss_t.update(loss, data.size(0))
                    correct_t.update((predicted == label).sum().item() / label.shape[0], label.shape[0])
                    #correct_t5.update((predicted5 == label.unsqueeze(1)).sum().item() / label.shape[0], label.shape[0])
                    
                # Recording loss and accuracy ---------------------------------
                writer.add_scalar('Loss/{}'.format(phase), loss_t.avg, epoch)
                writer.add_scalar('Accuracy/{}'.format(phase), correct_t.avg, epoch)
                # -------------------------------------------------------------
                    
                # Save model --------------------------------------------------
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
#                         torch.save(save_data, './pkl/{}/fold_{}_best5_{}.pkl'.format(args.INDEX, index, args.INDEX))
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
                        torch.save(save_data, './pkl/{}/fold_{}_best_{}.pkl'.format(args.INDEX, index, args.INDEX))
                # -------------------------------------------------------------
                logging.info("{} set loss : {:.6f}".format(phase, loss_t.avg))        
                logging.info("{} set top-1 acc : {:.6f}%".format(phase, correct_t.avg * 100.))  
                logging.info("{} set top-5 acc : {:.6f}%".format(phase, correct_t5.avg * 100.))  
                print('Index : {}'.format(args.INDEX))
                print("dataset : {}".format(args.DATASET_NAME))
                print("Model name : {}".format(args.MODEL))
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
    
    # Show the best result ----------------------------------------------------
    acc = 0
    acc5 = 0
    loss = 0
    for idx in range(1, len(ACCMeters) + 1):
        print("Fold {} best acc : {:.6f} acc(5) : {:.6f} loss : {:.6f}".format(idx, ACCMeters[idx - 1].avg, ACCMeters5[idx - 1].avg, LOSSMeters[idx - 1].avg))
        acc += ACCMeters[idx - 1].avg
        acc5 += ACCMeters5[idx - 1].avg
        loss += LOSSMeters[idx - 1].avg
    print("Avg. ACC : {:.6f} Avg. ACC(5) : {:.6f} Avg. Loss : {:.6f}".format(acc / args.KFOLD, acc5 / args.KFOLD,loss / args.KFOLD))

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
    if not os.path.exists("./pkl"):
        os.mkdir("./pkl")
    print("Args : ", args)
    if not os.path.exists("./pkl/{}".format(args.INDEX)):
        os.mkdir("./pkl/{}".format(args.INDEX))
    
    logging.basicConfig(filename = './pkl/{}/logging.txt'.format(args.INDEX), level=logging.DEBUG)
    logging.info('Index : {}'.format(args.INDEX))
    logging.info("dataset : {}".format(args.DATASET_NAME))
    
    runs(args)
        