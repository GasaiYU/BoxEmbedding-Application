from torch.utils.data import Dataset
import numpy as np
import torch
import cv2

import os
import re

class SketchImageDataset(Dataset):
    def __init__(self, img_dir, info_path, ratio):
        super().__init__()
        circle_dir = os.path.join(img_dir, 'circle')
        rectangle_dir = os.path.join(img_dir, 'rectangle')
        triangle_dir = os.path.join(img_dir, 'triangle')

        circle_len, circle_imgs = self.load_imgs(circle_dir, ratio)
        rect_len, rectangle_imgs = self.load_imgs(rectangle_dir, ratio)
        tria_len, triangle_imgs = self.load_imgs(triangle_dir, ratio)
        
        self.imgs = circle_imgs + rectangle_imgs + triangle_imgs
        self.infos = self.load_info(info_path, ratio, circle_len, rect_len, tria_len)
    
    def load_imgs(self, shape_dir, ratio):
        imgs = []
        dir_list = os.listdir(shape_dir)
        dir_list.sort(key=lambda x: int(x.split('.')[0]))
        for path in dir_list:
            imgs.append(torch.from_numpy(cv2.imread(os.path.join(shape_dir ,path))).permute(2,0,1) / 255.0)
        ac_len = int(len(dir_list) * ratio)
        return len(dir_list), imgs[:ac_len]
    
    def load_info(self, info_path, ratio, circle_len, rect_len, tria_len):
        infos = []
        with open(info_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.replace('\n', '')
                line_arr = line.split(' ')
                line_arr.append(i)
                infos.append(line_arr)
        ac_circle_len = int(circle_len * ratio)
        ac_rect_len = int(rect_len * ratio)
        ac_tria_len = int(tria_len * ratio)
        res = infos[:ac_circle_len] + infos[circle_len:circle_len + ac_rect_len] + \
            infos[circle_len + rect_len:circle_len + rect_len + ac_tria_len]
        return res
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        return self.imgs[index], self.infos[index]
    
    