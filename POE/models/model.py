import torch
import torch.nn as nn
import yaml

config_file = '/lustre/I/gaomingju/bachelor/order-embeddings-wordnet/POE/config/model_config.yaml'

class torch_model(nn.Module):
    def __init__(self, data):
        self.data = data
        
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        self.measure = cfg.get('measure', 'uniform')
        self.init_embedding_scale = cfg.get('init_embedding_scale', 'random')
        
        self.min_embed, self.delta_embed, self.rel_embed = self.init_word_embedding(data)
        
    
    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have different init value. """
        if self.measure == 'exp' and not self.term:
            min_lower_scale, min_higher_scale = 0.0, 0.001
            delta_lower_scale, delta_higher_scale = 10.0, 10.5
        elif self.measure == 'uniform' and not self.term:
            min_lower_scale, min_higher_scale = 1e-4, 1e-2
            delta_lower_scale, delta_higher_scale = 0.9, 0.999
        elif self.term and self.measure == 'uniform':
            min_lower_scale, min_higher_scale = 1.0, 1.1
            delta_lower_scale, delta_higher_scale = 5.0, 5.1
        else:
            raise ValueError("Expected either exp or uniform but received", self.measure)
        return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale
            
    
    def init_word_embedding(self, data):
        min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = self.init_embedding_scale
        if self.init_embedding_scale == 'random':
            # random init word embedding
            min_embed = torch.Variable(
                torch.rand([])
            )
            pass
    
    def get_word_embedding(self, t1_idx, t2_idx):
        """read word embedding from embedding table, get unit cube embeddings"""
        pass
        