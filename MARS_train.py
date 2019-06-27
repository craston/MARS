from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import *
#from utils import AverageMeter, calculate_accuracy
import pdb
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

if __name__=="__main__":
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
    print("Length of validation data = ", len(val_data))
    
    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))   

    log_path = os.path.join(opt.result_path, opt.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.log == 1:
        if opt.pretrain_path != '':
            epoch_logger = Logger_MARS(os.path.join(log_path, 'PreKin_MARS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'loss_MSE', 'loss_MARS', 'acc', 'lr'], opt.MARS_resume_path, opt.begin_epoch)
            val_logger   = Logger_MARS(os.path.join(log_path, 'PreKin_MARS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset,opt.split,  opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)
        else:
            epoch_logger = Logger_MARS(os.path.join(log_path, 'MARS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'loss_MSE', 'loss_MARS', 'acc', 'lr'], opt.MARS_resume_path, opt.begin_epoch)
            val_logger   = Logger_MARS(os.path.join(log_path, 'MARS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)

    if opt.pretrain_path!='' and opt.dataset!= 'Kinetics': 
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001
        
    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening

       
    # define the model 
    print("Loading MARS model... ", opt.model, opt.model_depth)
    opt.input_channels =3
    model_MARS, parameters_MARS = generate_model(opt)

    print("Loading Flow model... ", opt.model, opt.model_depth) 
    opt.input_channels =2 
    if opt.pretrain_path != '':
        opt.pretrain_path = ''
        if opt.dataset == 'HMDB51':
            opt.n_classes = 51
        elif opt.dataset == 'Kinetics':
            opt.n_classes = 400 
        elif opt.dataset == 'UCF101':
            opt.n_classes = 101 

    model_Flow, parameters_Flow = generate_model(opt)
    
    criterion_MARS  = nn.CrossEntropyLoss().cuda()
    criterion_Flow = nn.MSELoss().cuda()
    
    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        model_Flow.load_state_dict(checkpoint['state_dict'])
           
        
    if opt.MARS_resume_path:
        print('loading MARS checkpoint {}'.format(opt.MARS_resume_path))
        checkpoint = torch.load(opt.MARS_resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model_MARS.load_state_dict(checkpoint['state_dict'])

    
    print("Initializing the optimizer ...")
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience) 

    optimizer = optim.SGD(
        parameters_MARS,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    
    if opt.MARS_resume_path != '':
        print("Loading optimizer checkpoint state")
        optimizer.load_state_dict(torch.load(opt.MARS_resume_path)['optimizer'])
    
   
    model_Flow.eval()
    print('run')
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        
        model_MARS.train()
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses     = AverageMeter()
        losses_MARS = AverageMeter()
        losses_MSE = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            inputs_MARS  = inputs[:,0:3,:,:,:]
            inputs_Flow = inputs[:,3:,:,:,:]
            
            targets = targets.cuda(non_blocking=True)
            # pdb.set_trace()
            inputs_MARS  = Variable(inputs_MARS)
            inputs_Flow = Variable(inputs_Flow)
            targets     = Variable(targets)
            
            outputs_MARS  = model_MARS(inputs_MARS)
            outputs_Flow = model_Flow(inputs_Flow)[1].detach()
           
            loss_MARS = criterion_MARS(outputs_MARS[0], targets)
            loss_MSE = opt.MARS_alpha*criterion_Flow(outputs_MARS[1], outputs_Flow)
            loss     = loss_MARS + loss_MSE
            acc = calculate_accuracy(outputs_MARS[0], targets)

            losses.update(loss.data, inputs.size(0))
            losses_MARS.update(loss_MARS.data, inputs.size(0))
            losses_MSE.update(loss_MSE.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_MARS {loss_MARS.val:.4f} ({loss_MARS.avg:.4f})\t'
                  'Loss_MSE {loss_MSE.val:.4f} ({loss_MSE.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_MARS=losses_MARS,
                      loss_MSE=losses_MSE,
                      acc=accuracies))
                      
        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'loss_MSE' : losses_MSE.avg,
                'loss_MARS': losses_MARS.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        if epoch % opt.checkpoint == 0:
            if opt.pretrain_path != '':
                save_file_path = os.path.join(log_path, 'MARS_preKin_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index, 
                                opt.output_layers[0], opt.MARS_alpha, epoch))
            else:
                save_file_path = os.path.join(log_path, 'MARS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index, 
                                opt.output_layers[0], opt.MARS_alpha, epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model_MARS.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
        model_MARS.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):
                
                data_time.update(time.time() - end_time)
                inputs_MARS  = inputs[:,0:3,:,:,:]
                
                targets = targets.cuda(non_blocking=True)
                inputs_MARS  = Variable(inputs_MARS)
                targets     = Variable(targets)
                
                outputs_MARS  = model_MARS(inputs_MARS)
                
                loss = criterion_MARS(outputs_MARS[0], targets)
                acc  = calculate_accuracy(outputs_MARS[0], targets)

                losses.update(loss.data, inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Val_Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch,
                        i + 1,
                        len(val_dataloader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accuracies))
                          
        if opt.log == 1:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        scheduler.step(losses.avg)
        


