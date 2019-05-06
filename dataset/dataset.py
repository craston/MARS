from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from .preprocess_data import *
from PIL import Image, ImageFilter
import pickle
#import dircache
import pdb


def get_test_video(opt, frame_path, Total_frames):
    """
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames
        """

    clip = []
    i = 0
    loop = 0
    if Total_frames < opt.sample_duration: loop = 1
    if opt.modality == 'RGB': 
        while len(clip) < max(opt.sample_duration, Total_frames):
            try:
                im = Image.open(os.path.join(frame_path, '%05d.jpg'%(i+1)))
                clip.append(im.copy())
                im.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0

    elif opt.modality == 'Flow':  
        while len(clip) < 2*max(opt.sample_duration, Total_frames):
            try:
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(i+1)))
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
                
    elif  opt.modality == 'RGB_Flow':
        while len(clip) < 3*max(opt.sample_duration, Total_frames):
            try:
                im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(i+1)))
                im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(i+1)))
                im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(i+1)))
                clip.append(im.copy())
                clip.append(im_x.copy())
                clip.append(im_y.copy())
                im.close()
                im_x.close()
                im_y.close()
            except:
                pass
            i += 1
            
            if loop==1 and i == Total_frames:
                i = 0
    return clip

class HMDB51_test(Dataset):
    """HMDB51 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : Should be 0 while testing
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_test = train
        self.opt = opt
        
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2])for file in os.listdir(opt.annotation_path)]))

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 51

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.annotation_path) if file.strip('.txt')[-1] ==str(split)])
       
        self.data = []                                     # (filename , lab_id)
        
        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            f = open(os.path.join(opt.annotation_path, file), 'r')
            for line in f:
                if line.split(' ')[1] == '2':
                    frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                    if opt.only_RGB and os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
                    elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
            f.close()

    def __len__(self):
        '''
        returns number of test set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        frame_path = os.path.join(self.opt.frame_dir, video[1], video[0])

        if self.opt.only_RGB:
            Total_frames = int(len(os.listdir(frame_path))) - 2    
        else:
            Total_frames = int((len(os.listdir(frame_path)) + 1)/3) -1 

        clip = get_test_video(self.opt, frame_path, Total_frames)
                    
        return((scale_crop(clip, self.train_test, self.opt), label_id))


class UCF101_test(Dataset):
    """UCF101 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : Should be 0 while testing
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_test = train
        self.opt = opt
        
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2])for file in os.listdir(opt.annotation_path)]))

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.annotation_path) if file.strip('.txt')[-1] ==str(split)])
       
        self.data = []                                     # (filename , lab_id)
        
        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            f = open(os.path.join(opt.annotation_path, file), 'r')
            for line in f:
                if line.split(' ')[1] == '2':
                    frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                    if opt.only_RGB and "done" in os.listdir(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
                    elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
            f.close()

    def __len__(self):
        '''
        returns number of test set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        frame_path = os.path.join(self.opt.frame_dir, video[1], video[0])

        if self.opt.only_RGB:
            Total_frames = int(len(os.listdir(frame_path))) - 2    
        else:
            Total_frames = int((len(os.listdir(frame_path)) + 1)/3) -1 

        clip = get_test_video(self.opt, frame_path, Total_frames)
                    
        return((scale_crop(clip, self.train_test, self.opt), label_id))

class Kinetics_test(Dataset):
    def __init__(self, split, train, opt):
        """
        Args:
            opt   : config options
            train : Should be 0 while testing
            split : 'val'
        Returns:
            (tensor(frames), class_id ) : Shape of tensor C x T x H x W
        """
        self.split = split
        self.opt = opt
        self.train_val = train
              
        # joing labnames with underscores
        self.lab_names = sorted([f for f in os.listdir(os.path.join(self.__root_dir, "train"))])        
       
        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 400
        
        # indexes for validation set
        label_file = self.opt.annotation_path
     
        self.data = []                                     # (filename , lab_id)
    
        f = open(label_file, 'r')
        for line in f:
            class_id = int(line.strip('\n').split(' ')[-2])
            nb_frames = int(line.strip('\n').split(' ')[-1])
            self.data.append((' '.join(line.strip('\n').split(' ')[:-2]), class_id, nb_frames))
        f.close()
            
    def __len__(self):
        '''
        returns number of test set
        '''          
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = video[0]
        Total_frames = video[2]

        clip = get_test_video(self.opt, frame_path, Total_frames)

        return((scale_crop(clip, self.train_val, self.opt), label_id))

    
