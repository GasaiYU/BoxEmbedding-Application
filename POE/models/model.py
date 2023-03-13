import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from torch.autograd import Variable
import unit_cube

config_file = '/lustre/I/gaomingju/bachelor/order-embeddings-wordnet/POE/config/model_config.yaml'

class torch_model(nn.Module):
    def __init__(self, t1x, t2x, label, vocab_size, config=config_file):
        
        self.t1x = t1x
        self.t2x = t2x
        self.label = label
        self.config = config
        
        with open(config) as f:
            cfg = yaml.safe_load(f)
        self.measure = cfg.get('measure', 'uniform')
        self.init_embedding_scale = cfg.get('init_embedding_scale', 'random')
        self.embed_dim = int(cfg.get('embed_dim', 100))
        self.batch_size = int(cfg.get('batch_size', 16))
        
        # Need fixing
        self.vocab_size = vocab_size
        
        self.min_embed, self.delta_embed = self.init_word_embedding()
        self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed = self.get_word_embedding(self.t1x, self.t2x)
        
        """calculate box stats, join, meet and overlap condition"""
        self.join_min, self.join_max, self.meet_min, self.meet_max, self.disjoint = unit_cube.calc_join_and_meet(
            self.t1_min_embed, self.t1_max_embed, self.t2_min_embed, self.t2_max_embed)
        
        self.pos_disjoint_loss_func = unit_cube.lambda_batch_log_upper_bound
        self.pos_overlap_loss_func = unit_cube.lambda_batch_log_prob
        
        self.train_pos_prob = self.get_train_pos_prob()
        
        self.neg_disjoint_loss_func = unit_cube.lambda_zero
        self.neg_overlap_loss_func = unit_cube.lambda_batch_log_1minus_prob
        
        self.train_neg_prob = self.get_train_neg_prob()
        
        """loss function"""
        self.pos = torch.mul(self.train_pos_prob, self.label)
        self.neg = torch.mul(self.train_neg_prob, (1 - self.label))
        
        self.cond_loss = torch.sum(self.pos) / (self.batch_size / 2) + \
                         torch.sum(self.neg) / (self.batch_size / 2)
        self.regularization = 1e-4 * torch.sum(torch.abs(1 - self.min_embed - self.delta_embed)) / self.vocab_size
        
        self.loss = self.cond_loss + self.regularization
        
        
        
    
    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have diffeent init value. """
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
            
    
    def init_word_embedding(self):
        min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = self.init_embedding_scale
        if self.init_embedding_scale == 'random':
            # random init word embedding
            min_embed = torch.rand([self.vocab_size, self.embed_dim], requires_grad=True) * \
                        (min_higher_scale - min_lower_scale) + min_lower_scale
            min_embed = Variable(min_embed, requires_grad=True)
            
            delta_embed = torch.rand([self.vocab_size, self.embed_dim], requires_grad=True) * \
                          (delta_higher_scale - delta_lower_scale) + delta_lower_scale
            delta_embed = Variable(delta_embed, requires_grad=True)
        else:
            raise NotImplementedError("Not Implemented Embedding Method.")
        
        return min_embed, delta_embed
    
    
    def get_word_embedding(self, t1_idx, t2_idx):
        """read word embedding from embedding table, get unit cube embeddings"""
        t1_min_embed = torch.squeeze(F.embedding(t1_idx, self.min_embed), [1])
        t1_delta_embed = torch.squeeze(F.embedding(t1_idx, self.delta_embed), [1])
        t2_min_embed = torch.squeeze(F.embedding(t2_idx, self.min_embed), [1])
        t2_delta_embed = torch.squeeze(F.embedding(t2_idx, self.delta_embed), [1])

        t1_max_embed = t1_min_embed + t1_delta_embed
        t2_max_embed = t2_min_embed + t2_delta_embed
        
        return t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed


    def get_train_pos_prob(self):
        train_pos_prob = []
        for i in range(len(self.disjoint.shape[0])):
            if self.disjoint[i]:
                train_pos_prob.append(self.pos_disjoint_loss_func(
                    self.join_min[i], self.join_max[i], self.meet_min[i], self.meet_max[i],
                    self.t1_min_embed[i], self.t1_max_embed[i], self.t2_min_embed[i], self.t2_max_embed[i]
                ))
            else:
                train_pos_prob.append(self.pos_overlap_loss_func(
                    self.join_min[i], self.join_max[i], self.meet_min[i], self.meet_max[i],
                    self.t1_min_embed[i], self.t1_max_embed[i], self.t2_min_embed[i], self.t2_max_embed[i]
                ))
        res = torch.cat(train_pos_prob, dim=0)
        return res


    def get_train_neg_prob(self):
        train_neg_prob = []
        for i in range(len(self.disjoint.shape[0])):
            if self.disjoint[i]:
                train_neg_prob.append(self.neg_disjoint_loss_func(
                    self.join_min[i], self.join_max[i], self.meet_min[i], self.meet_max[i],
                    self.t1_min_embed[i], self.t1_max_embed[i], self.t2_min_embed[i], self.t2_max_embed[i]
                ))
            else:
                train_neg_prob.append(self.neg_overlap_loss_func(
                    self.join_min[i], self.join_max[i], self.meet_min[i], self.meet_max[i],
                    self.t1_min_embed[i], self.t1_max_embed[i], self.t2_min_embed[i], self.t2_max_embed[i]
                ))
        res = torch.cat(train_neg_prob, dim=0)
        return res