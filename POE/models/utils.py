import torch

import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def log1mexp(input_a):
    if input_a < 0.6931:
        return torch.log(-torch.expm1(-input_a))
    else:
        return torch.log1p(-torch.exp(-input_a))
    
    
def visualization(embedding1, embedding2, embedding3, embedding4, scale_fator=10):
    embedding1 = embedding1.cpu()
    embedding2 = embedding2.cpu()
    embedding3 = embedding3.cpu()
    embedding4 = embedding4.cpu()
    min_idx = torch.argmax(torch.abs(embedding1[0] - embedding3[0]))
    max_idx = torch.argmax(torch.abs(embedding2[0] - embedding4[0]))
    x1_min = embedding1[0][min_idx] * scale_fator
    y1_min = embedding1[0][min_idx] * scale_fator
    x1_max = embedding2[0][max_idx] * scale_fator
    y1_max = embedding2[0][max_idx] * scale_fator
    
    h = x1_max - x1_min
    w = y1_max - y1_min
    
    fig = plt.figure(figsize=(5, 5))
        
    rect = plt.Rectangle((x1_min, y1_min), h, w, color='b')
    plt.gcf().gca().add_artist(rect)
    
    x2_min = embedding3[0][min_idx] * scale_fator
    y2_min = embedding3[0][min_idx] * scale_fator
    x2_max = embedding4[0][max_idx] * scale_fator
    y2_max = embedding4[0][max_idx] * scale_fator
    
    h = x2_max - x2_min
    w = y2_max - y2_min

    rect = plt.Rectangle((x2_min, y2_min), h, w, color='r')
    plt.gcf().gca().add_artist(rect)
    
    plt.axis('equal')
    plt.xlim(-1, 10)
    plt.savefig(f'test.png')
    # exit(0)
    pass
     