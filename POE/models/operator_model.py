import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MyMLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, embed_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MyOperator(nn.Module):
    def __init__(self, len_prog, dsl_num, embed_dim, batch_size, masks=None):
        super().__init__()
        layers = nn.ModuleList()
        for i in range(len_prog):
            layer = nn.ModuleList()
            for j in range(dsl_num):
                layer.append(MyMLP(embed_dim, embed_dim))
            layers.append(layer)
        self.layers = layers
        self.batch_size = batch_size
        self.len_prog = len_prog
        self.dsl_num = dsl_num
        self.masks = Parameter(torch.zeros(self.len_prog, self.batch_size, self.dsl_num),
                               requires_grad=True)
        
    
    def forward(self, src_embeddings, masks=None, tau=0.3):
        if masks is not None:
            assert masks.shape[0] == self.len_prog
            masks = masks.float()
            flag = True
        else:
            masks = self.masks
            flag = False
        x = src_embeddings

        for i in range(self.len_prog):
            layer_res = []
            for j in range(self.dsl_num):
                layer_res.append(self.layers[i][j](x.clone()))
            
            layer_res = torch.stack(layer_res, dim=1)
            if i == 1:
                layer_res[:, :9, :] = 0.0
            else:
                layer_res[:, 9:, :] = 0.0
            
            mask = masks[i].unsqueeze(1)
            if not flag:
                mask = F.gumbel_softmax(mask, tau=tau, hard=True, dim=2)
            
            x = (mask @ layer_res).squeeze(1)
        return x