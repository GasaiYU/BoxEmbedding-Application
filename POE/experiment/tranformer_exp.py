import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import sys
sys.path.append('..')

from models.model import torch_model

TOKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'

import warnings
warnings.filterwarnings("ignore")

PAD = 32
BATCH_SIZE = 16

FEATURE_DICT = {'Circle':0, 'Rectangle':1, 'Triangle':2, 'RED':3, "BLUE":4, 'GREEN':5, 'PURPLE':6, 'BLACK':7}

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
        
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layes, dropout_p):
        super().__init__()
        
        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        
        # Layers
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layes,
            dropout=dropout_p
        )
        
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # to obtain size (sequence length, batch_size, dim_model)
        src = src.permute(1,0,2)
        tgt = src.permute(1,0,2)
        
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, \
                                           tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out

    
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask==0, float('-inf'))
        mask = mask.masked_fill(mask==1, float(0.0))
        pass
    
    def create_pad_mask(self, matrix, pad_token):
        return (matrix == pad_token)


class TransformerEncoder(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, dropout_p):
        super().__init__()
        
        # Info
        self.model_name = "Transformer Encoder"
        self.dim_model = dim_model
        
        # Layers
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                                    nhead=num_heads,
                                                                    dropout=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src, src_pad_mask=None):
        # Src size must be (batch_size, src sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        
        # to convert to shape (sequence length, batch_size, dim_model)
        src = src.permute(1, 0, 2)
        
        # encoder. Out size: (sequence length, batch_size, dim_model)
        out = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
        
        return out

    @staticmethod
    def create_pad_mask(matrix, pad_token=PAD):
        return (matrix == pad_token)

class TokenDatesetTrain(Dataset):
    def __init__(self, token_path, token_info_path):
        token_arr_all = np.loadtxt(token_path, dtype=np.int32)
        token_arr_all_len = token_arr_all.shape[0]
        train_len = int(token_arr_all_len * 0.8)
        self.token_arr = token_arr_all[:train_len, :]
        token_info = []
        with open(token_info_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i >= train_len:
                    break
                line = line.replace('\n', '')
                line_arr = line.split(' ')
                token_info.append(line_arr)
        self.token_info = token_info
    
    def __len__(self):
        return self.token_arr.shape[0]
    
    def __getitem__(self, index):
        return self.token_arr[index], self.token_info[index]
    
class TokenDatesetTest(Dataset):
    def __init__(self, token_path, token_info_path):
        token_arr_all = np.loadtxt(token_path, dtype=np.int32)
        token_arr_all_len = token_arr_all.shape[0]
        train_len = int(token_arr_all_len * 0.8)
        self.token_arr = token_arr_all[train_len:, :]
        breakpoint()
        token_info = []
        with open(token_info_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i >= token_arr_all.shape[0] - train_len:
                    break
                line = line.replace('\n', '')
                line_arr = line.split(' ')
                token_info.append(line_arr)
        self.token_info = token_info
    
    def __len__(self):
        return self.token_arr.shape[0]
    
    def __getitem__(self, index):
        return self.token_arr[index], self.token_info[index]


class TokenBoxEmbeddingModel(nn.Module):
    def __init__(self, transformer_encoder, box_embedding_model, device):
        super().__init__()
        self.transformer_encoder = transformer_encoder.to(device)
        self.box_embedding_model = box_embedding_model.to(device)
        pass
    
    def forward(self, x, color_idx, shape_idx):
        src_padding_mask = TransformerEncoder.create_pad_mask(x.clone().detach())
        encoder_out = self.transformer_encoder(x, src_padding_mask)
        # change (S, N, E) -> (N, S, E)
        encoder_out = encoder_out.permute(1,0,2)
        box_embedding_vector = encoder_out[:, 0, :]
    
        x1, x2 = TokenBoxEmbeddingModel.split_dim(box_embedding_vector)
        pos_prob_color, _ = self.box_embedding_model((color_idx, x1, x2))
        pos_prob_shape, _ = self.box_embedding_model((shape_idx, x1, x2))
        
        return pos_prob_color, pos_prob_shape       
    
    @staticmethod
    def split_dim(x):
        if len(x.shape) == 1:
            x.unsqueeze(0)
        dim1 = x.shape[1] // 2
        return x[:, :dim1], x[:, dim1:]


def train_loop(model, dataloader, epoch_num, device):
    model.train()
    running_loss = last_loss = 0
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epoch_num):
        for i, batch in enumerate(dataloader):
            x, label = batch
            x = x.to(device)
            
            shape_idx = []
            for e in label[0]:
                shape_idx.append(FEATURE_DICT[e])
            
            color_idx = []
            for e in label[1]:
                color_idx.append(FEATURE_DICT[e])

            pos_prob_color, pos_prob_shape = model(x, color_idx, shape_idx)
            
            loss = (torch.sum(pos_prob_color) + torch.sum(pos_prob_shape)) / BATCH_SIZE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 49:
                last_loss = running_loss / 50
                print(f"[Train] epoch {epoch} batch {i} Loss {last_loss}")
                with open("log_image_train.txt", "a") as f:
                    f.write(f"[Train] epoch {epoch} batch {i} Loss {last_loss}\n")
                running_loss = 0
            
        if epoch % 10 == 0:
            eval_model(epoch, model, device)    
            
    torch.save(model.state_dict(), 'program_transform.pth.tar')
    
def eval_model(epoch, model, device):
    model.eval()
    token_dataset = TokenDatesetTest(TOKEN_PATH, TOKEN_INFO_PATH)
    dataloader = DataLoader(token_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
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
                
            pos_prob_color, pos_prob_shape = model(x, color_idx, shape_idx)
            loss = (torch.sum(pos_prob_color) + torch.sum(pos_prob_shape)) / BATCH_SIZE
            
            print(f"[TEST] epoch {epoch} batch {i} Loss {loss.item()}")
            with open("log_image_test.txt", "a") as f:
                f.write(f"[TEST] epoch {epoch} batch {i} Loss {loss.item()}\n")
        
    pass
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model = TransformerEncoder(num_tokens=35, dim_model=512, num_heads=8, num_encoder_layers=6, \
                        dropout_p=0.1)
    box_embedding_model = torch_model(len(FEATURE_DICT.keys()), device)
    model = TokenBoxEmbeddingModel(encoder_model, box_embedding_model, device)
    
    token_dataset = TokenDatesetTrain(TOKEN_PATH, TOKEN_INFO_PATH)
    dataloder = DataLoader(token_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    train_loop(model, dataloder, 50, device)
    