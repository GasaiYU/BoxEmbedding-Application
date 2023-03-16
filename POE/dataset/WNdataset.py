import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os


class WNDataSet(Dataset):
    def __init__(self, data_path):
        self.data1 = []
        self.data2 = []
        self.label = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line_arr = line.split(',')
                self.data1.append(int(line_arr[0]))
                self.data2.append(int(line_arr[1]))
                self.label.append(int(line_arr[2]))
        
    def __len__(self):
        return len(self.data1)
    
    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.label[index]
    
    
    
    