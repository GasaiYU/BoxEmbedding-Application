import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from dataset.WNdataset import WNDataSet
from models.model import torch_model

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
VOCAB_SIZE = 82115

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_loop():
    dataloader = DataLoader(WNDataSet('/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/train.txt'), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    poe_model = torch_model(VOCAB_SIZE, device).to(device)
    # poe_model.load_state_dict(torch.load('/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/experiment/ckpt.pth.tar'))
    optimizer = Adam(poe_model.parameters(), lr=1e-3)
    
    running_loss = last_loss = 0

    for epoch in range(10):
        for i, (t1x, t2x, label) in enumerate(dataloader):
            t1x.to(device)
            t2x.to(device)
            train_pos_prob, train_neg_prob = poe_model.forward((t1x, t2x))
            
            train_pos_prob.to(device)
            train_neg_prob.to(device)
            label = label.cuda()
            pos_prob = torch.mul(train_pos_prob, label)
            neg_prob = torch.mul(train_neg_prob, 1-label)
            loss = torch.sum(pos_prob) / (BATCH_SIZE / 2) + \
                torch.sum(neg_prob) / (BATCH_SIZE / 2)
            # pos = poe_model.forward((t1x, t2x))
            # loss = -torch.mul(pos, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100
                print(f"epoch {epoch} batch {i} Loss {last_loss}")
                with open("log.txt", "a") as f:
                    f.write(f"epoch {epoch} batch {i} Loss {last_loss}\n")
                running_loss = 0
        
    torch.save(poe_model.state_dict(), 'ckpt.pth.tar')
            
if __name__ == "__main__":
    train_loop()
         
    