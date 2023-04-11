import torch

import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def log1mexp(input_a):
    if input_a < 0.6931:
        return torch.log(-torch.expm1(-input_a))
    else:
        return torch.log1p(-torch.exp(-input_a))
    
    
def visualization(embedding1, embedding2, embedding3, embedding4, scale_fator=1):
    embedding1 = embedding1.cpu()
    embedding2 = embedding2.cpu()
    embedding3 = embedding3.cpu()
    embedding4 = embedding4.cpu()
    min_idx = torch.argmax(torch.abs(embedding1[0] - embedding3[0]))
    max_idx = torch.argmax(torch.abs(embedding2[0] - embedding4[0]))
    x1_min = embedding1[0][0] * scale_fator
    y1_min = embedding1[0][1] * scale_fator
    x1_max = embedding2[0][0] * scale_fator
    y1_max = embedding2[0][1] * scale_fator
    
    h = x1_max - x1_min
    w = y1_max - y1_min
    
    fig = plt.figure(figsize=(5, 5))
        
    rect = plt.Rectangle((x1_min, y1_min), h, w, color='b')
    plt.gcf().gca().add_artist(rect)
    
    x2_min = embedding3[0][0] * scale_fator
    y2_min = embedding3[0][0] * scale_fator
    x2_max = embedding4[0][1] * scale_fator
    y2_max = embedding4[0][1] * scale_fator
    
    h = x2_max - x2_min
    w = y2_max - y2_min
    # breakpoint()
    rect = plt.Rectangle((x2_min, y2_min), h, w, color='r')
    plt.gcf().gca().add_artist(rect)
    
    plt.axis('equal')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.savefig(f'test.png')
    exit(0)
    pass
     
def vis_all(min_embeddings, delta_embeddings):
    colors = ['r', 'b', 'g', 'm', 'k']
    for i in range(3, 8):
        min_embedding = min_embeddings[i]
        delta_embedding = delta_embeddings[i]
        min_embedding = min_embedding.cpu()
        delta_embedding = delta_embedding.cpu()
        x_min = min_embedding[0]
        y_min = min_embedding[1]
        x_delta = delta_embedding[0]
        y_delta = delta_embedding[1] 
        
        rect = plt.Rectangle((x_min, y_min), x_delta, y_delta, color=colors[i-3])
        plt.gcf().gca().add_artist(rect)     
        
    plt.axis('equal')
    plt.xlim(-10, 15)
    plt.ylim(-10, 15)
    plt.savefig(f'./image_res/1.png')
    exit(0)
    pass

def shape_color_embed(color_embeddings, shape_embeddings, x_embeddings, epoch, i, label):
    # breakpoint()
    shape_min_embedding, shape_delta_embedding = shape_embeddings
    color_min_embedding, color_delta_embedding = color_embeddings
    x_min_embedding, x_delta_embedding =  x_embeddings
    
    fig = plt.figure(figsize=(5, 5))
    shape_min_embedding = shape_min_embedding.clone().detach().cpu()
    shape_delta_embedding = shape_delta_embedding.clone().detach().cpu()
    rect = plt.Rectangle((shape_min_embedding[0], shape_min_embedding[1]), shape_delta_embedding[0], \
                         shape_delta_embedding[1], color='r')
    plt.gcf().gca().add_artist(rect)
    
    
    color_min_embedding = color_min_embedding.clone().detach().cpu()
    color_delta_embedding = color_delta_embedding.clone().detach().cpu()
    rect = plt.Rectangle((color_min_embedding[0], color_min_embedding[1]), color_delta_embedding[0], \
                         color_delta_embedding[1], color='g')
    plt.gcf().gca().add_artist(rect)
    
    x_min_embedding = x_min_embedding.clone().detach().cpu()
    x_delta_embedding = x_delta_embedding.clone().detach().cpu()
    # breakpoint()
    rect = plt.Rectangle((x_min_embedding[0], x_min_embedding[1]), x_delta_embedding[0], \
                         x_delta_embedding[1], color='b')
    plt.gcf().gca().add_artist(rect)
    
    plt.axis('equal')
    plt.xlim(-10, 15)
    plt.ylim(-10, 15)
    plt.savefig(f'./image_res/k.png')
    plt.close()
    # breakpoint()
    # exit(0)
    pass