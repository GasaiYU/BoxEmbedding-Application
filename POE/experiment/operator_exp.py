import torch
import numpy as np

from torch.optim import Adam
import torch.nn as nn
import sys
sys.path.append('..')

from models.model import torch_model
from models.token_transformer import TokenBoxEmbeddingModel, TransformerEncoder
from dataset.TokenDataset import TokenDatesetTrain, TokenDatesetTest
from dataset.OperatorDataset import OperatorDataset
from models.operator_model import MyOperator

from torch.utils.data import DataLoader

from torch.nn.parameter import Parameter

import json

TOKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'
OP_PATH = '../config/token_config/multi_op.txt'

BATCH_SIZE = 15
TOKEN_LEN = 35
EMBED_DIM = 24


DSL_NUM = [9, 36, 9, 36, 9, 36, 9, 36, 9, 36, 9, 25]
LEN_PROG = len(DSL_NUM)
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

def train_loop(embed_model, op_model, dataloader, epoch_num, device):
    op_model.train()
    embed_model.eval()
    running_loss = last_loss = 0
    for name, parameter in op_model.named_parameters():
        if name == 'masks':
            parameter.requries_grad = False
        else:
            parameter.requries_grad = True
    optimizer = Adam(op_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epoch_num):
        for c, batch in enumerate(dataloader):
            token1, token2, ops = batch
            # ops = ops[:, :3]
            with torch.no_grad():
                src_embeddings = embed_model.get_embedding(token1)
                tgt_embeddings = embed_model.get_embedding(token2)
                
            masks = torch.zeros(LEN_PROG, BATCH_SIZE, max(DSL_NUM))
    
            sum_dsl_num = []
            for i in range(len(DSL_NUM)):
                sum_dsl_num.append(sum(DSL_NUM[:i]))
               
            for i in range(ops.shape[0]):
                for j in range(ops.shape[1]):
                    op = ops[i][j]
                    for k in range(len(sum_dsl_num)):
                        if op < sum_dsl_num[k]:
                            ops[i][j] -= sum_dsl_num[k-1]
                            break
                    if ops[i][j] > max(DSL_NUM):
                        ops[i][j] -= sum_dsl_num[-1]
               
            for i in range(ops.shape[0]):
                for j in range(ops.shape[1]):
                    masks[j, i, ops[i][j]] = 1.0

            predicted = op_model(src_embeddings, masks)
            
            loss = loss_fn(predicted, tgt_embeddings)
          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if True:
                print(f"[Train] Loss {running_loss/1} Epoch {epoch} Batch {c}")
                # with open('logs/op_train.txt', 'a') as f:
                #     f.write(f"[Train] Loss {running_loss/1} Epoch {epoch} Batch {c}\n")
                running_loss = 0
    
    torch.save(op_model.state_dict(), '3_multi_fixed_op_model.pth.tar')
 
 
def model_infer(embed_model, dataloader, epoch_num, identity_idx, device):
    embed_model.eval()
    loss_fn = nn.MSELoss()
    running_loss = last_loss = 0
    
    total_top1_acc = 0
    total_top3_acc = 0
    for i, batch in enumerate(dataloader):
        op_model = MyOperator(LEN_PROG, DSL_NUM, EMBED_DIM, BATCH_SIZE, identity_idx)
        op_model.load_state_dict(torch.load('multi_fixed_op_model.pth.tar'))
        optimizer = Adam(op_model.parameters(), lr=1e-2)
        op_model.train() 
        tau = 2.0
        for name, parameter in op_model.named_parameters():
            op_model_t = op_model
            if name != 'masks':
                name_arr = name.split(r'.')
                for e in name_arr:
                    op_model_t = op_model_t.__getattr__(e)
                op_model_t.requires_grad = False
        loss_arr = []
        for epoch in range(epoch_num):
            token1, token2, ops = batch
            # ops = ops[:, :3]
            with torch.no_grad():
                src_embeddings = embed_model.get_embedding(token1)
                tgt_embeddings = embed_model.get_embedding(token2)

            predicted = op_model(src_embeddings, masks=None, tau=tau)
            if tau > 0 and tau > 2 / epoch_num:
                tau = tau - 2 / epoch_num

            if epoch != epoch_num - 1:
                
                loss = loss_fn(predicted, tgt_embeddings)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if epoch % 10 == 9:
                    print(f"[TEST] Loss {running_loss/10} Epoch {epoch} Batch {i} Tau {tau}")
                    loss_arr.append(running_loss/10)
                    if len(loss_arr) >= 40:    
                        if sum(loss_arr[20:]) >= sum(loss_arr[:20]) * 0.9 and tau > 0.1:       
                            tau -= 0.1
                        loss_arr = []
                    # with open('logs/op_test.txt', 'a') as f:
                    #     f.write(f"[TEST] Loss {running_loss} Epoch {epoch} Batch {i}\n")
                    running_loss = 0
            else:
                top1_acc = 0
                top3_acc = 0
                top5_acc = 0
                my_sorted, indices = op_model.masks.sort(dim=2, descending=True)
                sum_dsl_num = []
                for i in range(len(DSL_NUM)):
                    sum_dsl_num.append(sum(DSL_NUM[:i]))
                
                for i in range(ops.shape[0]):
                    for j in range(ops.shape[1]):
                        op = ops[i][j]
                        for k in range(len(sum_dsl_num)):
                            if op < sum_dsl_num[k]:
                                ops[i][j] -= sum_dsl_num[k-1]
                                break
                        if ops[i][j] > max(DSL_NUM):
                            ops[i][j] -= sum_dsl_num[-1]
                           
                for j in range(indices.shape[0]):
                    for k in range(indices.shape[1]):
                        if ops[k][j] in indices[j][k][:1]:
                            top1_acc += 1
                        if ops[k][j] in indices[j][k][:3]:
                            top3_acc += 1   
                        if ops[k][j] in indices[j][k][:5]:
                            top5_acc += 1       
                
                top1_acc = top1_acc / (indices.shape[0] * indices.shape[1])
                top3_acc = top3_acc / (indices.shape[0] * indices.shape[1])
                top5_acc = top5_acc / (indices.shape[0] * indices.shape[1])
                print(f'[INFER] Batch {i} top1_acc {top1_acc} top3_acc {top3_acc} top5_acc {top5_acc}')    
        total_top1_acc += top1_acc
        total_top3_acc += top3_acc
    print(f'[INFER] Total top1_acc: {total_top1_acc/8} Total top3_acc: {total_top3_acc/8}')    
    
            
            
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FEATURE_DICT = gen_feature_dict()
    
    with open('../config/token_config/multi_op_dict.json') as f:
        op_dict = json.load(f)

    identity_idx = []
    for e in list(op_dict.keys()):
        e_arr = e.split(' ')
        if e_arr[0] == e_arr[1]:
            identity_idx.append(op_dict[e])
            
    encoder_model = TransformerEncoder(num_tokens=100, dim_model=EMBED_DIM, num_heads=1, num_encoder_layers=1, \
                        dropout_p=0)
    features = [3,5,6,3,6,6,6,6,3,3,3,3]
    box_embedding_model = torch_model(10, device, features=features)
    embed_model = TokenBoxEmbeddingModel(encoder_model, box_embedding_model, device, \
                                   token_len=TOKEN_LEN, embed_dim=EMBED_DIM)
    embed_model.load_state_dict(torch.load('multi_program_transform.pth.tar'))
    
    
    op_dataset = OperatorDataset(OP_PATH)
    dataloader = DataLoader(op_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    

    op_model = MyOperator(LEN_PROG, DSL_NUM, EMBED_DIM, BATCH_SIZE, identity_idx)
    # op_model.load_state_dict(torch.load('multi_fixed_op_model.pth.tar'))
    # train_loop(embed_model, op_model, dataloader, 2000, device)

    model_infer(embed_model, dataloader, 20000, identity_idx, device)
    