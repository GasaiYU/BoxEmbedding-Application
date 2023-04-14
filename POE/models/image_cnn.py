import torch
import torch.nn as nn


class ImageCNN(nn.Module):
    def __init__(self, num_layers, embed_dim, in_c=3, out_c=16):
        super().__init__()
        layers = [nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=(1, 1))]
        for i in range(num_layers):
            if i != 0:
                layers.append(nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=(1, 1)))
            layers.append(nn.BatchNorm2d(num_features=out_c))
            layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=out_c, out_features=embed_dim)
        
    def forward(self, x):
        x = self.net(x)
        x_mean = x.mean(dim=(2,3))
        out = self.fc(x_mean)
        return out
    
class ImageCNNWithBoxEmbedding(nn.Module):
    def  __init__(self, cnn_model, box_embedding_model, device):
        super().__init__()
        self.cnn_model = cnn_model.to(device)
        self.box_embedding_model = box_embedding_model.to(device)
        
        
    def forward(self, img, color_idx, shape_idx, label, epoch, i):
        # (N, C, H, W) -> (N, C)
        features = self.cnn_model(img)
        x1, x2 = ImageCNNWithBoxEmbedding.split_dim(features)
        if False:
            self.box_embedding_model.visual_shape_color_embedding(color_idx[0], shape_idx[0], x1[0], x2[0], epoch, i, label[0][0])
            
        pos_prob_color, neg_prob_color = self.box_embedding_model((x1, x2, color_idx))
        pos_prob_shape, neg_prob_shape = self.box_embedding_model((x1, x2, shape_idx))
        mi_loss = self.box_embedding_model.get_mi_loss()
        
        return pos_prob_color, neg_prob_color, pos_prob_shape, neg_prob_shape, mi_loss       
    
    @staticmethod
    def split_dim(x):
        if len(x.shape) == 1:
            x.unsqueeze(0)
        dim1 = x.shape[1] // 2
        return x[:, :dim1], x[:, dim1:]