import torch
import numpy as np

from torch.optim import Adam, AdamW

import sys
sys.path.append('..')

from models.model import torch_model
from models.image_cnn import ImageCNN, ImageCNNWithBoxEmbedding
from dataset.SketchImageDataset import SketchImageDataset

from torch.utils.data import DataLoader

OKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'
IMAGE_DIR = '../dataset/data'


import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 15
EMBED_DIM = 4

FEATURE_DICT = {'Circle':0, 'Rectangle':1, 'Triangle':2, 'RED':3, "BLUE":4, 'GREEN':5, 'PURPLE':6, 'BLACK':7}

def parse_labels(label):
    shape_idx = []
    for e in label[0]:
        shape_idx.append(FEATURE_DICT[e])
    
    color_idx = []
    for e in label[1]:
        color_idx.append(FEATURE_DICT[e])
        
    truth_label = []

    for e in label[2]:
        if e == 'True':
            truth_label.append(1)
        else:
            truth_label.append(0)
    
    idx = []      
    for e in label[3]:
        idx.append(e)

    return shape_idx, color_idx, truth_label, idx


def train_loop(model, dataloader, epoch_num, device):
    model.train()
    running_loss = last_loss = 0
    optimizer = Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(epoch_num):
        # if epoch % 10 == 0:
        #     eval_loop(epoch, model, device)
        running_loss = 0
        
        for i, batch in enumerate(dataloader):
            img, labels = batch
            img = img.to(device)

            shape_idx, color_idx, truth_label, idx = parse_labels(labels)
            
            truth_label = torch.tensor(truth_label).to(device)
            train_pos_prob_color, train_neg_prob_color, train_pos_prob_shape, train_neg_prob_shape, mi_loss \
                = model(img, color_idx, shape_idx, [truth_label, shape_idx, color_idx], epoch, i)
            
            pos_prob_color = torch.mul(train_pos_prob_color, truth_label)
            neg_prob_color = torch.mul(train_neg_prob_color, 1-truth_label)
            pos_prob_shape = torch.mul(train_pos_prob_shape, truth_label)
            neg_prob_shape = torch.mul(train_neg_prob_shape, 1-truth_label)
            
            loss = torch.sum(pos_prob_color) / (BATCH_SIZE) + torch.sum(pos_prob_shape) / (BATCH_SIZE)\
                    + torch.sum(neg_prob_color) / (BATCH_SIZE) + torch.sum(neg_prob_shape) / (BATCH_SIZE) \

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 15 == 14 or True:
                last_loss = running_loss
                print(f"[Train] epoch {epoch} Batch {i} Loss {last_loss}")
                with open("logs/log_image_train.txt", "a") as f:
                    f.write(f"[Train] epoch {epoch} Batch {i} Loss {last_loss}\n")
                running_loss = 0 
                
    torch.save(model.state_dict(), 'image_cnn.pth.tar')        
    
    
def eval_loop(epoch, model ,device):
    pass
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = ImageCNN(num_layers=3, embed_dim=EMBED_DIM, in_c=3, \
                         out_c=16)
    box_embedding_model = torch_model(10, device, feature_size=len(FEATURE_DICT.keys()))
    model = ImageCNNWithBoxEmbedding(cnn_model, box_embedding_model, device)
    
    image_dataset_train = SketchImageDataset(IMAGE_DIR, TOKEN_INFO_PATH, ratio=0.8)
    dataloader = DataLoader(image_dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    with open('logs/log_image_train.txt', 'r+') as f:
        f.truncate(0)
    with open('logs/log_image_test.txt', 'r+') as f:
        f.truncate(0)
        
    train_loop(model, dataloader, 100, device)