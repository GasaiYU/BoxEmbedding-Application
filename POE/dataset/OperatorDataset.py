from torch.utils.data import Dataset
import numpy as np

class OperatorDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        token1s = []
        token2s = []
        op1s = []
        op2s = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_arr = line.split(',')
                op1s.append(int(line_arr[1]))
                op2s.append(int(line_arr[2]))
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
        self.op1s, self.op2s, self.token1s, self.token2s = \
            op1s, op2s, token1s, token2s
    
    def __len__(self):
        return len(self.op1s)
    
    def __getitem__(self, index):
        return np.asarray(self.token1s[index]), np.asarray(self.token2s[index]), self.op1s[index], self.op2s[index]
