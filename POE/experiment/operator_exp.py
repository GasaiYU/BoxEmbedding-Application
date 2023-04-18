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

TOKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'
OP_PATH = '../config/token_config/op.txt'

BATCH_SIZE = 15
TOKEN_LEN = 27
EMBED_DIM = 4
FINAL_EMBED_DIM = 4
LEN_PROG = 2
DSL_NUM=34
FEATURE_DICT = {'Circle':0, 'Rectangle':1, 'Triangle':2, 'RED':3, "BLUE":4, 'GREEN':5, 'PURPLE':6, 'BLACK':7}

def train_loop(embed_model, op_model, dataloader, epoch_num, device):
    op_model.train()
    embed_model.eval()
    running_loss = last_loss = 0
    for name, parameter in op_model.named_parameters():
        if name == 'masks':
            parameter.requries_grad = False
        else:
            parameter.requries_grad = True
    optimizer = Adam(op_model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(epoch_num):
        for i, batch in enumerate(dataloader):
            token1, token2, op1, op2 = batch
            with torch.no_grad():
                src_embeddings = embed_model.get_embedding(token1)
                tgt_embeddings = embed_model.get_embedding(token2)
                
            masks = torch.zeros(LEN_PROG, BATCH_SIZE, DSL_NUM)
            
            for j in range(len(op1)):
                masks[0, j, op1[j]] = 1
                
            for j in range(len(op2)):
                masks[1, j, op2[j]] = 1
            predicted = op_model(src_embeddings, masks)
            
            loss = loss_fn(predicted, tgt_embeddings)
          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 8 == 7:
                print(f"[Train] Loss {running_loss/8} Epoch {epoch} Batch {i}")
                with open('logs/op_train.txt', 'a') as f:
                    f.write(f"[Train] Loss {running_loss/8} Epoch {epoch} Batch {i}\n")
                running_loss = 0
    
    torch.save(op_model.state_dict(), 'fixed_op_model.pth.tar')
 
 
def model_infer(embed_model, dataloader, epoch_num, device):
    embed_model.eval()
    loss_fn = nn.MSELoss()
    running_loss = last_loss = 0
    
    total_top1_acc = 0
    total_top3_acc = 0
    for i, batch in enumerate(dataloader):
        op_model = MyOperator(LEN_PROG, DSL_NUM, FINAL_EMBED_DIM, BATCH_SIZE)
        op_model.load_state_dict(torch.load('op_model.pth.tar'))
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
            token1, token2, op1, op2 = batch
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
                op1_acc = 0
                op2_acc = 0
                my_sorted, indices = op_model.masks.sort(dim=2, descending=True)

                for i in range(indices.shape[1]):
                    if op1[i] in indices[0][i][:1]:
                        top1_acc += 1
                        op1_acc += 1
                for i in range(indices.shape[1]):
                    if op2[i] in indices[1][i][:1]:
                        top1_acc += 1
                        op2_acc += 1
                        
                for i in range(indices.shape[1]):
                    if op1[i] in indices[0][i][:3]:
                        top3_acc += 1
                for i in range(indices.shape[1]):
                    if op2[i] in indices[1][i][:3]:
                        top3_acc += 1
  
                top1_acc = top1_acc / (len(indices[0]) + len(indices[1]))
                top3_acc = top3_acc / (len(indices[0]) + len(indices[1]))
                print(f'[INFER] Batch {i} top1_acc {top1_acc} top3_acc {top3_acc} OP1 ACC: {op1_acc/indices.shape[1]} OP2 ACC: {op2_acc/indices.shape[1]}')    
        total_top1_acc += top1_acc
        total_top3_acc += top3_acc
    print(f'[INFER] Total top1_acc: {total_top1_acc/8} Total top3_acc: {total_top3_acc/8}')    
    
            
            
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model = TransformerEncoder(num_tokens=35, dim_model=EMBED_DIM, num_heads=1, num_encoder_layers=1, \
                        dropout_p=0)
    box_embedding_model = torch_model(10, device, feature_size=len(FEATURE_DICT.keys()))
    embed_model = TokenBoxEmbeddingModel(encoder_model, box_embedding_model, device, \
                                   token_len=TOKEN_LEN, embed_dim=EMBED_DIM)
    embed_model.load_state_dict(torch.load('program_transform.pth.tar'))
    
    
    op_dataset = OperatorDataset(OP_PATH)
    dataloader = DataLoader(op_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    op_model = MyOperator(LEN_PROG, DSL_NUM, FINAL_EMBED_DIM, BATCH_SIZE)
    op_model.load_state_dict(torch.load('op_model.pth.tar'))
    # train_loop(embed_model, op_model, dataloader, 2000, device)
    
    model_infer(embed_model, dataloader, 20000, device)
    