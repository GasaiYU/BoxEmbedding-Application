import torch
import numpy as np

from torch.optim import Adam

import sys
sys.path.append('..')

from models.model import torch_model
from models.token_transformer import TokenBoxEmbeddingModel, TransformerEncoder
from dataset.TokenDataset import TokenDatesetTrain, TokenDatesetTest

from torch.utils.data import DataLoader

TOKEN_PATH = '../config/token_config/token.txt'
TOKEN_INFO_PATH = '../config/token_config/token_info.txt'

