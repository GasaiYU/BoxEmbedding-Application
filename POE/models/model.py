import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from torch.autograd import Variable
import models.unit_cube as unit_cube

from box_embeddings.modules.intersection import hard_intersection, gumbel_intersection
from box_embeddings.parameterizations.box_tensor import *

from box_embeddings.modules.volume.volume import Volume

config_file = '/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/config/model_config.yaml'

class torch_model(nn.Module):
    def __init__(self, vocab_size, device, config=config_file):
        super(torch_model, self).__init__()
        self.config = config
        self.device = device
        
        with open(config) as f:
            cfg = yaml.safe_load(f)
        self.measure = cfg.get('measure', 'uniform')
        self.init_embedding = cfg.get('init_embedding', 'random')
        self.embed_dim = int(cfg.get('embed_dim', 100))
        self.batch_size = int(cfg.get('batch_size', 16))
        
        self.vocab_size = vocab_size
        self.min_lower_scale, self.min_higher_scale, self.delta_lower_scale, self.delta_higher_scale = self.init_embedding_scale()
        
        self.init_word_embedding()


    def forward(self, x):
        t1x, t2x = x
        self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed = self.get_word_embedding(t1x, t2x)

        """calculate box stats, join, meet and overlap condition"""
        self.join_min, self.join_max, self.meet_min, self.meet_max, self.disjoint = unit_cube.calc_join_and_meet(
            self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed)
        
        self.pos_disjoint_loss_func = unit_cube.lambda_batch_log_upper_bound
        self.pos_overlap_loss_func = unit_cube.lambda_batch_log_prob
        
        self.train_pos_prob = self.get_train_pos_prob()
        
        self.neg_disjoint_loss_func = unit_cube.lambda_zero
        self.neg_overlap_loss_func = unit_cube.lambda_batch_log_1minus_prob
        
        self.train_neg_prob = self.get_train_neg_prob()
        
        # Some changes
        # self.pos = []
        # for i in range(self.t1_min_embed.shape[0]):
        #     x_min = self.t1_min_embed[i].tolist()
        #     x_max = self.t1_max_embed[i].tolist()
        #     data_x = BoxTensor(torch.tensor([x_min, x_max]), requires_grad=True)
            
        #     y_min = self.t2_min_embed[i].tolist()
        #     y_max = self.t2_max_embed[i].tolist()
        #     data_y = BoxTensor(torch.tensor([y_min, y_max]), requires_grad=True)
            
        #     box_vol = Volume(volume_temperature=0.1, intersection_temperature=0.0001)
        #     self.pos.append(box_vol(gumbel_intersection(data_x, data_y)))

        # self.pos = torch.stack(self.pos, dim=0)
        # self.pos = Variable(self.pos, requires_grad=True)
        return self.train_pos_prob, self.train_neg_prob
        
        
    def init_embedding_scale(self):
        """For different measures, min and delta have diffeent init value. """
        if self.measure == 'exp':
            min_lower_scale, min_higher_scale = 0.0, 0.001
            delta_lower_scale, delta_higher_scale = 10.0, 10.5
        elif self.measure == 'uniform':
            min_lower_scale, min_higher_scale = 1e-4, 1e-2
            delta_lower_scale, delta_higher_scale = 0.9, 0.999
        else:
            raise ValueError("Expected either exp or uniform but received", self.measure)
        return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale
            
    
    def init_word_embedding(self):
        if self.init_embedding == 'random':
            # random init word embedding
            self.min_embed = nn.Embedding(self.vocab_size, self.embed_dim)
            self.delta_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        else:
            raise NotImplementedError("Not Implemented Embedding Method.")
        
    
    
    def get_word_embedding(self, t1_idx, t2_idx):
        """read word embedding from embedding table, get unit cube embeddings"""
        min_embed_mean = (self.min_lower_scale + self.min_higher_scale) / 2
        min_embed_var = self.min_higher_scale - min_embed_mean
        delta_embed_mean = (self.delta_lower_scale + self.delta_higher_scale) / 2
        delta_embed_var = self.delta_higher_scale - delta_embed_mean
        
        t1_min_embed = self.min_embed(torch.tensor(t1_idx).clone().detach().to(self.device)) * min_embed_var + min_embed_mean
        t1_delta_embed = self.delta_embed(torch.tensor(t1_idx).clone().detach().to(self.device)) * delta_embed_var + delta_embed_mean
        t2_min_embed = self.min_embed(torch.tensor(t2_idx).clone().detach().to(self.device)) * min_embed_var + min_embed_mean
        t2_delta_embed = self.delta_embed(torch.tensor(t2_idx).clone().detach().to(self.device)) * delta_embed_var + delta_embed_mean

        t1_max_embed = t1_min_embed + t1_delta_embed
        t2_max_embed = t2_min_embed + t2_delta_embed
        
        return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed


    def get_train_pos_prob(self):
        join_min = self.join_min.unsqueeze(1)
        join_max = self.join_max.unsqueeze(1)
        meet_min = self.meet_min.unsqueeze(1)
        meet_max = self.meet_max.unsqueeze(1)
        t1_min_embed = self.t1_min_embed.unsqueeze(1)
        t1_max_embed = self.t1_max_embed.unsqueeze(1)
        t2_min_embed = self.t2_min_embed.unsqueeze(1)
        t2_max_embed = self.t2_max_embed.unsqueeze(1)
        
        train_pos_prob = []
        for i in range(self.disjoint.shape[0]):
            if self.disjoint[i]:
                train_pos_prob.append(self.pos_disjoint_loss_func(
                    join_min[i], join_max[i], meet_min[i], meet_max[i],
                    t1_min_embed[i], t1_max_embed[i], t2_min_embed[i], t2_max_embed[i]
                ))
            else:
                train_pos_prob.append(self.pos_overlap_loss_func(
                    join_min[i], join_max[i], meet_min[i], meet_max[i],
                    t1_min_embed[i], t1_max_embed[i], t2_min_embed[i], t2_max_embed[i]
                ))
        res = torch.cat(train_pos_prob, dim=0)
        return res


    def get_train_neg_prob(self):
        join_min = self.join_min.unsqueeze(1)
        join_max = self.join_max.unsqueeze(1)
        meet_min = self.meet_min.unsqueeze(1)
        meet_max = self.meet_max.unsqueeze(1)
        t1_min_embed = self.t1_min_embed.unsqueeze(1)
        t1_max_embed = self.t1_max_embed.unsqueeze(1)
        t2_min_embed = self.t2_min_embed.unsqueeze(1)
        t2_max_embed = self.t2_max_embed.unsqueeze(1)
        
        train_neg_prob = []
        
        for i in range(self.disjoint.shape[0]):
            if self.disjoint[i]:
                train_neg_prob.append(self.neg_disjoint_loss_func(
                    join_min[i], join_max[i], meet_min[i], meet_max[i],
                    t1_min_embed[i], t1_max_embed[i], t2_min_embed[i], t2_max_embed[i]
                ))
            else:
                train_neg_prob.append(self.neg_overlap_loss_func(
                    join_min[i], join_max[i], meet_min[i], meet_max[i],
                    t1_min_embed[i], t1_max_embed[i], t2_min_embed[i], t2_max_embed[i]
                ))
        res = torch.cat(train_neg_prob, dim=0)
        return res
