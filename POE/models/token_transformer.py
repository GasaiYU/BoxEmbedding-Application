import torch
import torch.nn as nn

import math
import numpy as np

PAD = 32
BATCH_SIZE = 1

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
                                                                dropout=dropout_p,
                                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src, src_pad_mask=None):
        # Src size must be (batch_size, src sequence length, dim_model)
        src = self.embedding(src)
        src = self.positional_encoder(src)

        # to convert to shape (sequence length, batch_size, dim_model)
        # src = src.permute(1, 0, 2)
        # encoder. Out size: (sequence length, batch_size, dim_model)
        out = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
        
        return out

    @staticmethod
    def create_pad_mask(matrix, pad_token=PAD):
        return (matrix == pad_token)
    
    
class TokenBoxEmbeddingModel(nn.Module):
    def __init__(self, transformer_encoder, box_embedding_model, device, token_len, embed_dim):
        super().__init__()
        self.transformer_encoder = transformer_encoder.to(device)
        self.box_embedding_model = box_embedding_model.to(device)
        self.linear = nn.Linear(in_features=(token_len)*embed_dim, out_features=embed_dim)
        self.linear2 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.relu = nn.LeakyReLU()
        # self.transformer_encoder = nn.Embedding(35, 4)
        pass
    
    def forward(self, x, features, label, epoch, i):
        # begin = torch.ones([x.shape[0], 1], dtype=torch.int64).to(x.device) * 21
        # x = torch.cat([begin, x], dim=1)
        # print(x)
        src_padding_mask = TransformerEncoder.create_pad_mask(x.clone().detach())
        # print(src_padding_mask)
        encoder_out = self.transformer_encoder(x, src_pad_mask=src_padding_mask)
 
        # src_padding_mask = TransformerEncoder.create_pad_mask(x.clone().detach())
        # encoder_out = self.transformer_encoder(x)
        
        batch_size = encoder_out.shape[0]
        encoder_out = encoder_out.reshape(batch_size, -1)
        box_embedding_vector = self.linear(encoder_out)

 
        # change (S, N, E) -> (N, S, E)
        # encoder_out = encoder_out.permute(1,0,2)
        # print(encoder_out)
        # box_embedding_vector = (encoder_out * (1 - src_padding_mask.int()).unsqueeze(-1)).sum(1) / (1 - src_padding_mask.int()).sum(-1, keepdim=True)
        # box_embedding_vector = self.linear2(box_embedding_vector)

        # box_embedding_vector = encoder_out.max(1)[0]
        # box_embedding_vector = encoder_out.mean(1)
        # box_embedding_vector = (encoder_out * (1 - src_padding_mask.int()).unsqueeze(-1)).max(1)[0]
        # box_embedding_vector = self.linear2(box_embedding_vector)
        # box_embedding_vector = encoder_out[:, 0, :]
            
        x1, x2 = TokenBoxEmbeddingModel.split_dim(box_embedding_vector)

        if epoch > 400: 
            self.box_embedding_model.visual_shape_color_embedding(features[0][0], features[1][0], x1[0], x2[0], epoch, i, label[0])
            
        # self.box_embedding_model.vis_all()
        pos_probs = []
        neg_probs = []

        for feature in features:    
            pos_prob, neg_prob = self.box_embedding_model((x1, x2, feature))
            pos_probs.append(pos_prob)
            neg_probs.append(neg_prob)
        pos_probs = torch.stack(pos_probs, dim=1)
        pos_probs = torch.mean(pos_probs, dim=1)
        neg_probs = torch.stack(neg_probs, dim=1)
        neg_probs = torch.mean(neg_probs, dim=1)
        mi_loss = self.box_embedding_model.get_mi_loss()

        return pos_probs, neg_probs, mi_loss       
    
    def get_embedding(self, x):
        src_padding_mask = TransformerEncoder.create_pad_mask(x.clone().detach())
        encoder_out = self.transformer_encoder(x, src_pad_mask=src_padding_mask)
        batch_size = encoder_out.shape[0]
        encoder_out = encoder_out.reshape(batch_size, -1)
        box_embedding_vector = self.linear(encoder_out)
        x1, x2 = TokenBoxEmbeddingModel.split_dim(box_embedding_vector)
        return self.box_embedding_model.get_x_embeddings(x1, x2)
    
    @staticmethod
    def split_dim(x):
        if len(x.shape) == 1:
            x.unsqueeze(0)
        dim1 = x.shape[1] // 2
        return x[:, :dim1], x[:, dim1:]

        
    
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, num_tokens, dim_model):
        super().__init__()
        
        self.model_name = "Simple Transformer Encoder"
        self.dim_model = dim_model
        # Layers
        self.embedding = nn.Embedding(num_tokens, dim_model)

    def forward(self, src, src_pad_mask=None):
        out = self.embedding(src)
        # Src size must be (batch_size, src sequence length, dim_model)
        return out
    

class SimpleTokenBoxEmbeddingModel(nn.Module):
    def __init__(self, transformer_encoder, box_embedding_model, device, token_len, embed_dim):
        super().__init__()
        self.transformer_encoder = transformer_encoder.to(device)
        self.box_embedding_model = box_embedding_model.to(device)
        self.linear = nn.Linear(in_features=token_len*embed_dim, out_features=embed_dim)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x, color_idx, shape_idx, label, idx, epoch, i):
        src_padding_mask = TransformerEncoder.create_pad_mask(x.clone().detach())
        encoder_out = self.transformer_encoder(x, src_pad_mask=src_padding_mask)
        
        batch_size = encoder_out.shape[0]
        encoder_out = encoder_out.reshape(batch_size, -1)
        box_embedding_vector = self.linear(encoder_out)
        # src_padding_mask = 1.0 - src_padding_mask.float()
        # src_padding_mask = src_padding_mask.unsqueeze(2)
        # box_embedding_vector = torch.sum(src_padding_mask * encoder_out, dim=1) / torch.sum(src_padding_mask)

        x1, x2 = TokenBoxEmbeddingModel.split_dim(box_embedding_vector)

        if label[0][0] == 1 and epoch > 100: 
            self.box_embedding_model.visual_shape_color_embedding(color_idx[0], shape_idx[0], x1[0], x2[0], epoch, i, label[0][0])
    
        pos_prob_color, neg_prob_color = self.box_embedding_model((x1, x2, color_idx))
        pos_prob_shape, neg_prob_shape = self.box_embedding_model((x1, x2, shape_idx))


        return pos_prob_color, neg_prob_color, pos_prob_shape, neg_prob_shape  
        