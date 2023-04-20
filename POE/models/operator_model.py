import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MyMLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, embed_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class IdentifyMap(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
class MyOperator(nn.Module):
    def __init__(self, len_prog, dsl_num, embed_dim, batch_size, identity_idx=[]):
        super().__init__()
        layers = nn.ModuleList()
        for i in range(len_prog):
            layer = nn.ModuleList()
            for j in range(dsl_num[i]):
                if sum(dsl_num[:i]) + j not in identity_idx:
                    layer.append(MyMLP(embed_dim, embed_dim))
                else:
                    layer.append(IdentifyMap())
            layers.append(layer)
        self.embed_dim = embed_dim
        self.layers = layers
        self.batch_size = batch_size
        self.len_prog = len_prog
        self.dsl_num = dsl_num
        self.masks = Parameter(torch.zeros(self.len_prog, self.batch_size, max(self.dsl_num)),
                               requires_grad=True)
    
    def forward(self, src_embeddings, masks=None, tau=0.5):
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
            for j in range(self.dsl_num[i]):
                # breakpoint()
                layer_res.append(self.layers[i][j](x.clone()))

            for j in range(max(self.dsl_num) - len(layer_res)):
                layer_res.append(torch.zeros(self.batch_size, self.embed_dim, requires_grad=True))
            layer_res = torch.stack(layer_res, dim=1)
            # breakpoint()
            # if i == 1:
            #     layer_res[:, :9, :] = 0.0
            # else:
            #     layer_res[:, 9:, :] = 0.0

            mask = masks[i].unsqueeze(1)
            if not flag:
                mask = F.softmax(mask, dim=2)

            x = (mask @ layer_res).squeeze(1)
            
        return x