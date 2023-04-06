from torch.utils.data import DataLoader, Dataset
import numpy as np

class TokenDatesetTrain(Dataset):
    def __init__(self, token_path, token_info_path):
        token_arr_all = np.loadtxt(token_path, dtype=np.int32)
        token_arr_all_len = token_arr_all.shape[0]
        train_len = int(token_arr_all_len * 0.8)
        self.token_arr = token_arr_all[:train_len, :]
        token_info = []
        with open(token_info_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i >= train_len:
                    break
                line = line.replace('\n', '')
                line_arr = line.split(' ')
                token_info.append(line_arr)
        self.token_info = token_info
    
    def __len__(self):
        return self.token_arr.shape[0]
    
    def __getitem__(self, index):
        return self.token_arr[index], self.token_info[index]


class TokenDatesetTest(Dataset):
    def __init__(self, token_path, token_info_path):
        token_arr_all = np.loadtxt(token_path, dtype=np.int32)
        token_arr_all_len = token_arr_all.shape[0]
        train_len = int(token_arr_all_len * 0.8)
        self.token_arr = token_arr_all[train_len:, :]
        token_info = []
        with open(token_info_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i >= token_arr_all.shape[0] - train_len:
                    break
                line = line.replace('\n', '')
                line_arr = line.split(' ')
                line_arr.append(i)
                token_info.append(line_arr)
        self.token_info = token_info
    
    def __len__(self):
        return self.token_arr.shape[0]
    
    def __getitem__(self, index):
        return self.token_arr[index], self.token_info[index]