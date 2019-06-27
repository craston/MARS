from __future__ import division
from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
#from torchvision.transforms import *
import torch
import random
import numbers
import pdb
import time

try:
    import accimage
except ImportError:
    accimage = None
    
scale_choice = [1, 1/2**0.25, 1/2**0.5, 1/2**0.75, 0.5]
crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass
        
class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output sizeself.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self. will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass
    
class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass    
    
class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, scale, size, crop_position, interpolation=Image.BILINEAR):
        self.scale = scale
        self.size = size
        self.interpolation = interpolation
        self.crop_position = crop_position
   
    def __call__(self, img):
        
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

        
def get_mean( dataset='HMDB51'):
    #assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [114.7748, 107.7354, 99.4750 ]
    elif dataset == 'kinetics':
    # Kinetics (10 videos for each class)
        return [110.63666788, 103.16065604,  96.29023126]
    elif dataset == "HMDB51":
        return [0.36410178082273*255, 0.36032826208483*255, 0.31140866484224*255]

def get_std(dataset = 'HMDB51'):
# Kinetics (10 videos for each class)
    if dataset == 'kinetics':
        return [38.7568578, 37.88248729, 40.02898126]
    elif dataset == 'HMDB51':
        return [0.20658244577568*255, 0.20174469333003*255, 0.19790770088352*255]


def scale_crop(clip, train, opt): 
    """Preprocess list(frames) based on train/test and modality.
    Training:
        - Multiscale corner crop
        - Random Horizonatal Flip (change direction of Flow accordingly)
        - Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor
        - Normalize R,G,B based on mean and std of ``ActivityNet``
    Testing/ Validation:
        - Scale frame
        - Center crop
        - Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor
        - Normalize R,G,B based on mean and std of ``ActivityNet``
    Args:
        clip (list(frames)): list of RGB/Flow frames
        train : 1 for train, 0 for test
    Return:
        Tensor(frames) of shape C x T x H x W
    """
    if opt.modality == 'RGB':
        processed_clip = torch.Tensor(3, len(clip), opt.sample_size, opt.sample_size)
    elif opt.modality == 'Flow':
        processed_clip = torch.Tensor(2, int(len(clip)/2), opt.sample_size, opt.sample_size)
    elif opt.modality == 'RGB_Flow':
        processed_clip = torch.Tensor(5, int(len(clip)/3), opt.sample_size, opt.sample_size)
    
    flip_prob     = random.random()
    scale_factor  = scale_choice[random.randint(0, len(scale_choice) - 1)]
    crop_position = crop_positions[random.randint(0, len(crop_positions) - 1)] 
   
    if train == 1:
        j = 0
        for i, I in enumerate(clip):
            I = MultiScaleCornerCrop(scale = scale_factor, size = opt.sample_size, crop_position = crop_position)(I)
            I = RandomHorizontalFlip(p = flip_prob)(I)
            if opt.modality == 'RGB':
                I = ToTensor(1)(I)
                I = Normalize(get_mean('activitynet'), [1,1,1])(I)
                processed_clip[:, i, :, :] = I

            elif opt.modality == 'Flow':
                if i%2 == 0 and flip_prob<0.5:
                    I = ImageChops.invert(I)                    # Flipping x-direction
                I = ToTensor(1)(I)
                I = Normalize([127.5, 127.5, 127.5], [1,1,1])(I)
                if i%2 == 0:
                    processed_clip[0, int(i/2), :, :] = I
                elif i%2 == 1:
                    processed_clip[1, int((i-1)/2), :, :] = I

            elif opt.modality == 'RGB_Flow':
                if j == 1 and flip_prob<0.5:                # Flipping x-direction
                    I = ImageChops.invert(I)
                I = ToTensor(1)(I)                          
                if j == 0:
                    I = Normalize(get_mean('activitynet'), [1,1,1,])(I)
                    processed_clip[0:3, int(i/3), :, :] = I
                else:
                    I = Normalize([127.5, 127.5, 127.5], [1,1,1])(I)
                    if j == 1:
                        processed_clip[3, int((i-1)/3), :, :] = I
                    elif j == 2:
                        processed_clip[4, int((i-2)/3), :, :] = I
                j += 1
                if j == 3:
                    j = 0

    else:
        j = 0
        for i, I in enumerate(clip):
            I = Scale(opt.sample_size)(I)
            I = CenterCrop(opt.sample_size)(I)
            I = ToTensor(1)(I)

            if opt.modality == 'RGB':
                I = Normalize(get_mean('activitynet'), [1,1,1])(I)
                processed_clip[:, i, :, :] = I

            elif opt.modality == 'Flow':
                I = Normalize([127.5, 127.5, 127.5], [1,1,1])(I)
                if i%2 == 0:
                    processed_clip[0, int(i/2), :, :] = I
                elif i%2 == 1:
                    processed_clip[1, int((i-1)/2), :, :] = I

            elif opt.modality == 'RGB_Flow':
                if j == 0:
                    I = Normalize(get_mean('activitynet'), [1,1,1,])(I)
                    processed_clip[0:3, int(i/3), :, :] = I
                else:
                    I = Normalize([127.5, 127.5, 127.5], [1,1,1])(I)                     
                    if j == 1:
                        processed_clip[3, int((i-1)/3), :, :] = I
                    elif j == 2:
                        processed_clip[4, int((i-2)/3), :, :] = I
                j += 1
                if j == 3:
                    j = 0
                    
    return(processed_clip)