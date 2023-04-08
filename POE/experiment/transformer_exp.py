import torch
import numpy as np

from torch.optim import Adam

import sys
sys.path.append('..')

from models.model import torch_model
from models.token_transformer import TokenBoxEmbeddingModel, TransformerEncoder
from dataset.TokenDataset import TokenDatesetTrain, TokenDatesetTest

from torch.utils.data import DataLoader

TOKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 1

FEATURE_DICT = {'Circle':0, 'Rectangle':1, 'Triangle':2, 'RED':3, "BLUE":4, 'GREEN':5, 'PURPLE':6, 'BLACK':7}

def train_loop(model, dataloader, epoch_num, device):
    model.train()
    running_loss = last_loss = 0
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epoch_num):
        # if epoch % 5 == 0:
        #     eval_model(epoch, model, device)   
        running_loss = 0
        for i, batch in enumerate(dataloader):

            x, label = batch
            x = x.to(device)
            
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
            
            truth_label = torch.tensor(truth_label).to(device)
            train_pos_prob_color, train_neg_prob_color, train_pos_prob_shape, train_neg_prob_shape \
                = model(x, color_idx, shape_idx, [truth_label, shape_idx, color_idx], idx, epoch, i)
                
            pos_prob_color = torch.mul(train_pos_prob_color, truth_label)
            neg_prob_color = torch.mul(train_neg_prob_color, 1-truth_label)
            pos_prob_shape = torch.mul(train_pos_prob_shape, truth_label)
            neg_prob_shape = torch.mul(train_neg_prob_shape, 1-truth_label)
            
            loss = torch.sum(pos_prob_color) / (BATCH_SIZE) + torch.sum(pos_prob_shape) / (BATCH_SIZE)\
                    + torch.sum(neg_prob_color) / (BATCH_SIZE) + torch.sum(neg_prob_shape) / (BATCH_SIZE)
            # if truth_label[0] == 1:
            #     breakpoint()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if True:
                last_loss = running_loss
                print(f"[Train] epoch {epoch} Batch {i} Loss {last_loss}")
                with open("log_image_train.txt", "a") as f:
                    f.write(f"[Train] epoch {epoch} Batch {i} Loss {last_loss}\n")
                running_loss = 0 
            
    torch.save(model.state_dict(), 'program_transform.pth.tar')
    
def eval_model(epoch, model, device):
    model.eval()
    token_dataset = TokenDatesetTest(TOKEN_PATH, TOKEN_INFO_PATH)
    dataloader = DataLoader(token_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    running_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, label = batch
            x = x.to(device)
            
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
            
            truth_label = torch.tensor(truth_label).to(device)
            
            train_pos_prob_color, train_neg_prob_color, train_pos_prob_shape, train_neg_prob_shape \
                = model(x, color_idx, shape_idx)

            pos_prob_color = torch.mul(train_pos_prob_color, truth_label)
            neg_prob_color = torch.mul(train_neg_prob_color, 1-truth_label)
            pos_prob_shape = torch.mul(train_pos_prob_shape, truth_label)
            neg_prob_shape = torch.mul(train_neg_prob_shape, 1-truth_label)
            
            loss = torch.sum(pos_prob_color) + torch.sum(pos_prob_shape)\
                    + torch.sum(neg_prob_color) + torch.sum(neg_prob_shape)
                    
                    
            running_loss += loss.item()
                
        print(f"[TEST] epoch {epoch} Loss {running_loss/6}")
        with open("log_image_test.txt", "a") as f:
            f.write(f"[TEST] epoch {epoch} Loss {running_loss/6}\n")

    pass
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model = TransformerEncoder(num_tokens=35, dim_model=4, num_heads=2, num_encoder_layers=2, \
                        dropout_p=0.1)
    box_embedding_model = torch_model(30, device, feature_size=len(FEATURE_DICT.keys()))
    model = TokenBoxEmbeddingModel(encoder_model, box_embedding_model, device)
    # model.load_state_dict(torch.load('program_transform.pth.tar'))
    token_dataset = TokenDatesetTrain(TOKEN_PATH, TOKEN_INFO_PATH)
    dataloder = DataLoader(token_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    with open('log_image_train.txt', 'r+') as f:
        f.truncate(0)
    with open('log_image_test.txt', 'r+') as f:
        f.truncate(0)
    
    train_loop(model, dataloder, 300, device)
    