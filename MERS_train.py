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
    # print("pytorch cuda version = ", torch.version.cuda)
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
            epoch_logger = Logger_MARS(os.path.join(log_path, 'PreKin_MERS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'loss_MSE', 'loss_MERS', 'acc', 'lr'], opt.MARS_resume_path, opt.begin_epoch)
            val_logger   = Logger_MARS(os.path.join(log_path, 'PreKin_MERS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset,opt.split,  opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)
        else:
            epoch_logger = Logger_MARS(os.path.join(log_path, 'MERS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'loss_MSE', 'loss_MERS',  'acc', 'lr'], opt.MARS_resume_path, opt.begin_epoch)
            val_logger   = Logger_MARS(os.path.join(log_path, 'MERS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha))
                            ,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)

    print("Initializing the optimizer ...")
    if opt.pretrain_path: 
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001
        
    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience) 

    # define the model 
    print("Loading MERS model... ", opt.model, opt.model_depth)
    opt.input_channels =3
    model_MERS, parameters_MERS = generate_model(opt)

    print("Loading Flow model... ", opt.model, opt.model_depth) 
    opt.input_channels =2 
    if opt.pretrain_path != '':
        opt.pretrain_path = ''
        if opt.dataset == 'HMDB51':
            opt.n_classes = 51
        elif opt.dataset == 'UCF101':
            opt.n_classes = 101
        elif opt.dataset == 'Mini_Kinetics':
            opt.n_classes = 200
        elif opt.dataset == 'Kinetics':
            opt.n_classes = 400 

    model_Flow, parameters_Flow = generate_model(opt)
    
    criterion_MERS  = nn.CrossEntropyLoss().cuda()
    criterion_Flow = nn.MSELoss().cuda()
    
    if opt.resume_path1:
        print('loading Flow checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        model_Flow.load_state_dict(checkpoint['state_dict'])

    if opt.MARS_resume_path:
        print('loading MERS checkpoint {}'.format(opt.MARS_resume_path))
        checkpoint = torch.load(opt.MARS_resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model_MERS.load_state_dict(checkpoint['state_dict'])

    
    
    optimizer = optim.SGD(
        parameters_MERS,
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
        
        model_MERS.train()
        
        
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses     = AverageMeter()
        losses_MERS = AverageMeter()
        losses_MSE = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            inputs_MERS  = inputs[:,0:3,:,:,:]
            inputs_Flow = inputs[:,3:,:,:,:]
            
            targets = targets.cuda(non_blocking=True)
            inputs_MERS  = Variable(inputs_MERS)
            inputs_Flow = Variable(inputs_Flow)
            targets     = Variable(targets)
            
            outputs_MERS  = model_MERS(inputs_MERS)[1]
            outputs_Flow = model_Flow(inputs_Flow)[1].detach()
           
            loss_MSE = opt.MARS_alpha*criterion_Flow(outputs_MERS, outputs_Flow)
            
            optimizer.zero_grad()
            loss_MSE.backward(retain_graph=True)
            
            #Only training the last layer of MERS for classification
            for name, param in model_MERS.named_parameters():    
                if name=="module.fc.weight" or name=="module.fc.bias":
                    param.requires_grad=True
                else:
                    param.requires_grad = False

            outputs_MERS  = model_MERS(inputs_MERS)[0]
            loss_MERS = criterion_MERS(outputs_MERS, targets)
            
            loss_MERS.backward()
            optimizer.step()
            
            loss = loss_MERS +  loss_MSE
            
            acc = calculate_accuracy(outputs_MERS, targets)
            losses.update(loss.data, inputs.size(0))
            losses_MERS.update(loss_MERS.data, inputs.size(0))
            losses_MSE.update(loss_MSE.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            for name, param in model_MERS.named_parameters():
                param.requires_grad = True
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_MERS {loss_MERS.val:.4f} ({loss_MERS.avg:.4f})\t'
                  'Loss_MSE {loss_MSE.val:.4f} ({loss_MSE.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_MERS=losses_MERS,
                      loss_MSE=losses_MSE,
                      acc=accuracies))
                      
        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'loss_MSE' : losses_MSE.avg,
                'loss_MERS': losses_MERS.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        if epoch % opt.checkpoint == 0:
            if opt.pretrain_path:
                save_file_path = os.path.join(log_path, 'preKin_MERS_{}_{}_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha, epoch))
            else:
                save_file_path = os.path.join(log_path, 'MERS_{}_{}_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.MARS_alpha, epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model_MERS.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
        model_MERS.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):
                
                data_time.update(time.time() - end_time)
                inputs_MERS  = inputs[:,0:3,:,:,:]
                
                targets = targets.cuda(non_blocking=True)
                #pdb.set_trace()
                inputs_MERS  = Variable(inputs_MERS)
                targets     = Variable(targets)
                
                outputs_MERS  = model_MERS(inputs_MERS)
                
                loss = criterion_MERS(outputs_MERS[0], targets)
                acc  = calculate_accuracy(outputs_MERS[0], targets)

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
        


