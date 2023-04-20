import torch
import numpy as np

from torch.optim import Adam, AdamW

import json

import sys
sys.path.append('..')

from models.model import torch_model
from models.token_transformer import TokenBoxEmbeddingModel, TransformerEncoder, SimpleTokenBoxEmbeddingModel, SimpleTransformerEncoder
from dataset.TokenDataset import TokenDatesetTrain, TokenDatesetTest

from torch.utils.data import DataLoader

TOKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 15
TOKEN_LEN = 35
EMBED_DIM = 24
REAL_EMBED_DIM = 4

LINE_COLOR_ARR = ['NULL' ,'RED', "BLUE", 'GREEN', 'PURPLE', 'BLACK']
BOLD = ['BOLD_NULL' ,'THIN', 'THICK']

def gen_feature_dict():
    FEATURE_DICT = {'Circle':0, 'Rectangle':1, 'Triangle':2, 'RED':3, "BLUE":4, 'GREEN':5, 'PURPLE':6, 'BLACK':7}
    for i, e in enumerate(LINE_COLOR_ARR):
        FEATURE_DICT[f'CIRCLE_LINE_{e}'] = 8 + i
        
    for i, e in enumerate(BOLD):
        FEATURE_DICT[f'CIRCLE_LINE_{e}'] = 14 + i
    
    for i in range(4):
        for j, e in enumerate(LINE_COLOR_ARR):
            FEATURE_DICT[f'LINE{i+1}_{e}'] = 17 + i * 6 + j
    
    for i in range(4):
        for j, e in enumerate(BOLD):
            FEATURE_DICT[f'LINE{i+1}_{e}'] = 41 + i * 3 + j

    return FEATURE_DICT


def train_loop(model, dataloader, epoch_num, feature_dict, device):
    model.train()
    running_loss = last_loss = 0
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epoch_num):
        # if epoch % 10 == 0:
        #     eval_model(epoch, model, device)   
        running_loss = 0
        for i, batch in enumerate(dataloader):

            x, label = batch
            x = x.to(device)
            # shape_idx = []
            # for e in label[0]:
            #     shape_idx.append(FEATURE_DICT[e])
            
            # color_idx = []
            # for e in label[1]:
            #     color_idx.append(FEATURE_DICT[e])
                
            # truth_label = []
        
            # for e in label[2]:
            #     if e == 'True':
            #         truth_label.append(1)
            #     else:
            #         truth_label.append(0)
            
            # idx = []      
            # for e in label[3]:
            #     idx.append(e)
            features = []
            for batch_feature in label[:-1]:
                batch_feature_idx = []
                for feature in batch_feature:
                    batch_feature_idx.append(feature_dict[feature])
                features.append(batch_feature_idx)
                
            truth_label = []
            for e in label[-1]:
                if e == 'True':
                    truth_label.append(1)
                else:
                    truth_label.append(0)
            
            truth_label = torch.tensor(truth_label).to(device)
            train_pos_prob, train_neg_prob, mi_loss \
                = model(x, features, truth_label, epoch, i)
                
            pos_prob = torch.mul(train_pos_prob, truth_label)
            neg_prob = torch.mul(train_neg_prob, 1-truth_label)
            
            loss = torch.sum(pos_prob) / (BATCH_SIZE) + torch.sum(neg_prob) / (BATCH_SIZE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 15 == 14 or True:
                last_loss = running_loss
                print(f"[Train] epoch {epoch} Batch {i} Loss {last_loss}")
                with open(f"logs/log_token_train_{REAL_EMBED_DIM}.txt", "a") as f:
                    f.write(f"[Train] epoch {epoch} Batch {i} Loss {last_loss}\n")
                running_loss = 0 
            
    torch.save(model.state_dict(), 'multi_program_transform.pth.tar')
    
def eval_model(epoch, model, feature_dict ,device):
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
                shape_idx.append(feature_dict[e])
            
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
            
            idx = []      
            for e in label[3]:
                idx.append(e)
            
            train_pos_prob_color, train_neg_prob_color, train_pos_prob_shape, train_neg_prob_shape, mi_loss \
                = model(x, color_idx, shape_idx, [truth_label, shape_idx, color_idx], idx, epoch, i)

            pos_prob_color = torch.mul(train_pos_prob_color, truth_label)
            neg_prob_color = torch.mul(train_neg_prob_color, 1-truth_label)
            pos_prob_shape = torch.mul(train_pos_prob_shape, truth_label)
            neg_prob_shape = torch.mul(train_neg_prob_shape, 1-truth_label)
            
            loss = torch.sum(pos_prob_color) + torch.sum(neg_prob_color) \
                    + torch.sum(pos_prob_shape) + torch.sum(neg_prob_shape)
                    
                    
            running_loss += loss.item()
                
        print(f"[TEST] epoch {epoch} Loss {running_loss/120}")
        with open(f"logs/log_token_test_{REAL_EMBED_DIM}.txt", "a") as f:
            f.write(f"[TEST] epoch {epoch} Loss {running_loss/120}\n")

    pass
    

if __name__ == "__main__":
    feature_dict = gen_feature_dict()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model = TransformerEncoder(num_tokens=100, dim_model=EMBED_DIM, num_heads=1, num_encoder_layers=1, \
                        dropout_p=0)
    features = [3,5,6,3,6,6,6,6,3,3,3,3]
    box_embedding_model = torch_model(10, device, features=features)
    model = TokenBoxEmbeddingModel(encoder_model, box_embedding_model, device, \
                                   token_len=TOKEN_LEN, embed_dim=EMBED_DIM)
    
    # encoder_model = SimpleTransformerEncoder(num_tokens=35, dim_model=EMBED_DIM)
    # model = SimpleTokenBoxEmbeddingModel(encoder_model, box_embedding_model, device, TOKEN_LEN, EMBED_DIM)
    # model.load_state_dict(torch.load('fixed_program_transform.pth.tar'))
    token_dataset = TokenDatesetTrain(TOKEN_PATH, TOKEN_INFO_PATH)
    dataloder = DataLoader(token_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    # with open('logs/log_token_train.txt', 'r+') as f:
    #     f.truncate(0)
    # with open('logs/log_token_test.txt', 'r+') as f:
    #     f.truncate(0)
    
    train_loop(model, dataloder, 500, feature_dict, device)
    