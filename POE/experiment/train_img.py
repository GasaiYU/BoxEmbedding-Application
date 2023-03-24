import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from dataset.ImageDataset import ImageDataset
from models.model import torch_model

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
FEATURE_SIZE = 10
DATA_SIZE = 1500

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_loop():
    dataset = ImageDataset('/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/dataset/data/labels/circle_label.json',\
                                         '/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/dataset/data/labels/rectangle_label.json',\
                                         '/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/dataset/data/labels/triangle_label.json')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    poe_model = torch_model(DATA_SIZE+FEATURE_SIZE, device).to(device)
    optimizer = Adam(poe_model.parameters(), lr=1e-3)
    all_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    running_loss = last_loss = 0
    with open('log_image_train.txt', 'w')as f:
        f.write('Begin Training.\n')
        
    for epoch in range(50):
        for i, image_dict in enumerate(dataloader):
            t1x = torch.tensor(image_dict['index']).to(device)
            color_idx = torch.tensor(image_dict['color']).to(device)
            shape_idx = torch.tensor(image_dict['shape'] + len(all_color)).to(device)
            pos_prob_color, _ = poe_model.forward((t1x, color_idx))
            pos_prob_shape, _ = poe_model.forward((t1x, shape_idx))
            
            loss = (torch.sum(pos_prob_color) + torch.sum(pos_prob_shape)) / BATCH_SIZE
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                last_loss = running_loss / 100
                print(f"epoch {epoch} batch {i} Loss {last_loss}")
                with open("log_image_train.txt", "a") as f:
                    f.write(f"epoch {epoch} batch {i} Loss {last_loss}\n")
                running_loss = 0
            pass
    
    torch.save(poe_model.state_dict(), 'image_ckpt.pth.tar')
            
if __name__ == "__main__":
    train_loop()
         
    