from torch.utils.data import Dataset
import numpy as np

class OperatorDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        token1s = []
        token2s = []
        all_ops = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_arr = line.split(',')
                ops = []
                for i in range(1, 13):
                    ops.append(int(line_arr[i]))
                all_ops.append(ops)
               
                token1_arr = line_arr[0].split(' ')
                token1_line_res = []
                for e in token1_arr:
                    token1_line_res.append(int(e))
                token2_arr = line_arr[-1].split(' ')
                token2_line_res = []
                for e in token2_arr:
                    token2_line_res.append(int(e))
                token1s.append(token1_line_res)
                token2s.append(token2_line_res)
        self.all_ops, self.token1s, self.token2s = \
            all_ops, token1s, token2s
    
    def __len__(self):
        return 150
    
    def __getitem__(self, index):
        return np.asarray(self.token1s[index]), np.asarray(self.token2s[index]), np.asarray(self.all_ops[index])
