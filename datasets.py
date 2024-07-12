import os
from os.path import join 
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.LUT import *


def augment(img_input, img_target):
    try:
        W, H = img_input.size
    except:
        H,W = img_input.shape[1:]
    crop_h = round(H * np.random.uniform(0.6,1.))
    crop_w = round(W * np.random.uniform(0.6,1.))
    b = np.random.uniform(0.8,1.2)
    s = np.random.uniform(0.8,1.2)
    img_input = TF.adjust_brightness(img_input,b)
    img_input = TF.adjust_saturation(img_input,s)
    i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
    img_input = TF.resized_crop(img_input, i, j, h, w, (256, 256))
    img_target = TF.resized_crop(img_target, i, j, h, w, (256, 256))
    if np.random.random() > 0.5:
        img_input = TF.hflip(img_input)
        img_target = TF.hflip(img_target)
    if np.random.random() > 0.5:
        img_input = TF.vflip(img_input)
        img_target = TF.vflip(img_target)
    return img_input, img_target


def augment_quan(img_input, img_input_msb, img_input_lsb, img_target):
    try:
        W, H = img_input.size
    except:
        H,W = img_input.shape[1:]
    crop_h = round(H * np.random.uniform(0.6,1.))
    crop_w = round(W * np.random.uniform(0.6,1.))
    range_list = [32, 64, 128, 256]
    selected_values = np.random.choice(range_list, size=1)[0]
    
    i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
    img_input_msb = TF.resized_crop(img_input_msb, i, j, h, w, (selected_values, selected_values))
    img_input_lsb = TF.resized_crop(img_input_lsb, i, j, h, w, (selected_values, selected_values))
    img_input = TF.resized_crop(img_input, i, j, h, w, (selected_values, selected_values))
    img_target = TF.resized_crop(img_target, i, j, h, w, (selected_values, selected_values))
    if np.random.random() > 0.5:
        img_input = TF.hflip(img_input)
        img_input_msb = TF.hflip(img_input_msb)
        img_input_lsb = TF.hflip(img_input_lsb)
        img_target = TF.hflip(img_target)
    if np.random.random() > 0.5:
        img_input = TF.vflip(img_input)
        img_input_msb = TF.vflip(img_input_msb)
        img_input_lsb = TF.vflip(img_input_lsb)
        img_target = TF.vflip(img_target)
    return img_input, img_input_msb, img_input_lsb, img_target


class FiveK(Dataset):
    def __init__(self, data_root, split, model): 
        self.split = split
        self.model = model
        input_dir = join(data_root, "fiveK/input_"+split)
        target_dir = join(data_root, "fiveK/target_"+split)
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        self.input_files = [join(input_dir, file_name) for file_name in input_files]
        self.target_files = [join(target_dir, file_name) for file_name in target_files]


    def __getitem__(self, index):
        res = {}
        input_path = self.input_files[index]
        target_path = self.target_files[index]

        if 'ICELUT' == self.model:
            img_input = cv2.cvtColor(cv2.imread(input_path, -1), cv2.COLOR_BGR2RGB)                
            if self.split == "train":
                img_input_msb = TF.to_tensor((img_input//16)/16)
                img_input_lsb = TF.to_tensor((img_input%16)/16)
                img_input = TF.to_tensor(img_input)
                img_target = TF.to_tensor(cv2.cvtColor(cv2.imread(target_path, -1), cv2.COLOR_BGR2RGB)/255) 
                img_input, img_input_msb, img_input_lsb, img_target = augment_quan(img_input, img_input_msb, img_input_lsb, img_target) 
                res["input_org"] = img_input.type(torch.FloatTensor)
                res["input_msb"] = img_input_msb.type(torch.FloatTensor)
                res["input_lsb"] = img_input_lsb.type(torch.FloatTensor)
                res["target"] = img_target.type(torch.FloatTensor)
                res["target_org"] = img_target.type(torch.FloatTensor)
            else:
                scale = 32
                img_input_msb = TF.to_tensor((img_input//16)/16)
                img_input_lsb = TF.to_tensor((img_input%16)/16)
                img_input = TF.to_tensor(img_input)
                img_target = TF.to_tensor(cv2.cvtColor(cv2.imread(target_path, -1), cv2.COLOR_BGR2RGB)/255) 
                img_input_resize_msb, img_input_resize_lsb, img_target_resize = TF.resize(img_input_msb, (scale, scale)),TF.resize(img_input_lsb, (scale, scale)), TF.resize(img_target, (scale, scale))
                img_input_resize_msb = torch.round(img_input_resize_msb*16) / 16
                img_input_resize_lsb = torch.round(img_input_resize_lsb*16) / 16
                res["target_org"] = img_target.type(torch.FloatTensor)
                res["input_msb"] = img_input_resize_msb.type(torch.FloatTensor)
                res["input_lsb"] = img_input_resize_lsb.type(torch.FloatTensor)
                res["input_org"] = img_input.type(torch.FloatTensor)
                res["target"] = img_target_resize.type(torch.FloatTensor)


        else:
            raise 
            
        img_name = os.path.split(self.input_files[index])[-1]
        res["name"] = img_name
        return res 

    def f(self, img, mode='chw'):
        if mode == 'hwc':
            return np.array(img, dtype=np.float32)/255 # hwc
        elif mode == 'chw':
            return TF.to_tensor(img) # chw
        else:
            raise
    
    def __len__(self):
        return len(self.input_files)
