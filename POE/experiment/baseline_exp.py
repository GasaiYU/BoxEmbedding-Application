import torch
import numpy as np

from torch.optim import Adam
import torch.nn as nn
import sys
sys.path.append('..')

PAD = 32

from models.transformer_baseline import BaseLineTransformer
from dataset.OperatorDataset import OperatorDataset

from torch.utils.data import DataLoader

OP_PATH = '../config/token_config/multi_op.txt'

def train_loop(model, dataloader, epoch_num, device):
    total_loss = 0
    optimizer = Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epoch_num):
        for i, batch in enumerate(dataloader):
            token1, token2, ops = batch
            src = torch.cat([token1, token2], dim=1)

            ops = ops + 100
            ops = torch.cat([torch.ones(ops.shape[0], 1, dtype=torch.int32) * 21, ops], dim=1)
            ops = torch.cat([ops, torch.ones(ops.shape[0], 1, dtype=torch.int32) * 22], dim=1)
            tgt = ops[:, :-1]
            tgt_y = ops[:, 1:]
            n_tokens = (tgt != PAD).sum()

            out = model(src, tgt)

            loss = loss_fn(out.contiguous().view(-1, out.shape[-1]), tgt_y.contiguous().view(-1)) / n_tokens
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if True:
                print(f'Loss {total_loss} Epoch {epoch} Batch {i}')
                total_loss = 0
                

    torch.save(model.state_dict(), 'baseline.pth.tar')    
            
    
    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaseLineTransformer(num_tokens=360, embed_dim=512)
    dataset = OperatorDataset(OP_PATH)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)
    
    train_loop(model, dataloader, 1000, device)