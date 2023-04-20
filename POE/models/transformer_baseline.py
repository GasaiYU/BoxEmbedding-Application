import torch
import torch.nn as nn
import math

PAD =  32
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
       
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class BaseLineTransformer(nn.Module):
    def __init__(self, num_tokens, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embed_dim)
        
        self.transformer = nn.Transformer(d_model=embed_dim, num_encoder_layers=2, num_decoder_layers=2, \
                                          dim_feedforward=512, batch_first=True)
        
        self.postitional_encoder = PositionalEncoding(embed_dim, dropout=0)
        
        
    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(sz=tgt.shape[-1])
        src_key_paading_mask = BaseLineTransformer.get_key_padding_mask(src)
        tgt_key_padding_mask = BaseLineTransformer.get_key_padding_mask(tgt)

        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        src = self.postitional_encoder(src)
        tgt = self.postitional_encoder(tgt)
        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_paading_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        
        return out
    
    @staticmethod
    def get_key_padding_mask(tokens):
        return tokens == PAD
        


        
        
        