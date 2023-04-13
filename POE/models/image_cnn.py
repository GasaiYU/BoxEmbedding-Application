import torch
import torch.nn as nn


class ImageCNN(nn.Module):
    def __init__(self, num_layers, in_c=3, out_c=16):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=(1, 1)))
            layers.append(nn.BatchNorm2d(num_features=out_c))
            layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.net(x)
        
        return x
        pass