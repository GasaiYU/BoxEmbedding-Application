import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json

import os

CIRCLE_PATH = '/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/dataset/data/circle'
RECTANGLE_PATH = '/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/dataset/data/rectangle'
TRIANGLE_PATH = '/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/dataset/data/triangle'

class ImageDataset(Dataset):
    def __init__(self, circle_path, rect_path, tria_path):
        self.circle_dict = {}
        self.rect_dict = {}
        self.tria_dict = {}
        self.all_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.all_shape = ['circle', 'rectangle', 'triangle']
        
        with open(circle_path, 'r') as f_circle:
            circle_json = json.load(f_circle)
        for file_name, file_attr in circle_json.items():
            file_name_arr = file_name.split('/')
            num = int(file_name_arr[-1].split(r'.')[0])
            color = file_attr['color']
            self.circle_dict[num] = {'color' : self.all_color.index(color), 'shape': 0, 'index': num}
            
        with open(rect_path, 'r') as f_rect:
            rect_json = json.load(f_rect)
        for file_name, file_attr in rect_json.items():
            file_name_arr = file_name.split('/')
            num = int(file_name_arr[-1].split(r'.')[0])
            color = file_attr['color']
            self.rect_dict[num] = {'color' : self.all_color.index(color), 'shape': 1, 'index': num+len(self.circle_dict.keys())}
        
        with open(tria_path, 'r') as f_tria:
            tria_json = json.load(f_tria)
        for file_name, file_attr in tria_json.items():
            file_name_arr = file_name.split('/')
            num = int(file_name_arr[-1].split(r'.')[0])
            color = file_attr['color']
            self.tria_dict[num] = {'color': self.all_color.index(color), 'shape': 2, 'index': num+len(self.circle_dict.keys())+len(self.rect_dict.keys())}
            

    def __len__(self):
        return len(self.circle_dict.keys()) + len(self.rect_dict.keys()) + len(self.tria_dict.keys())
    
    def __getitem__(self, index):
        if index >= 0 and index < len(self.circle_dict.keys()):
            return self.circle_dict[index]
        elif index >= len(self.circle_dict.keys()) and index < len(self.circle_dict.keys()) + len(self.rect_dict.keys()):
            return self.rect_dict[index-len(self.circle_dict.keys())]
        elif index < len(self.circle_dict.keys()) + len(self.rect_dict.keys()) + len(self.tria_dict.keys()):
            return self.tria_dict[index-len(self.circle_dict.keys())-len(self.rect_dict.keys())]
        else:
            raise KeyError('Out of range')

