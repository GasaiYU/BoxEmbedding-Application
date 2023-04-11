import torch.nn as nn
import torch

from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection

import models.utils as utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def calc_join_and_meet(t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    """
    # two box embeddings are t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed
    Returns:
        join box, min box, and disjoint condition:
    """
    # join is min value of (a, c), max value of (b, d)
    join_min = torch.min(t1_min_embed, t2_min_embed) # batchsize * embed_size
    join_max = torch.max(t1_max_embed, t2_max_embed) # batchsize * embed_size
    
    # find meet is calculate the max value of (a,c), min value of (b,d)
    meet_min = torch.max(t1_min_embed, t2_min_embed) # batchsize * embed_size
    meet_max = torch.min(t1_max_embed, t2_max_embed) # batchsize * embed_size
    
    # The overlap cube's max value have to be bigger than min value in every dimension to form a valid cube
    # if it's not, then two concepts are disjoint, return none
    cond = torch.le(meet_max, meet_min).float() # batchsize * embed_size
    cond = torch.sum(cond, dim=1).bool() # batchsize. If disjoint, cond > 0; else, cond = 0
    
    return join_min, join_max, meet_min, meet_max, cond


"""Some Helper functions."""
def batch_log_prob(min_embed, max_embed):
    # min_embed: batchsize * embed_size
    # max_embed: batchsize * embed_size
    # log_prob: batch_size
    # numerically stable log probability of a cube probability
    log_prob = torch.sum(torch.log((max_embed - min_embed) + 1e-8), dim=1)
    return log_prob

def smooth_prob(origin_prob):
    lambda_value = 1e-6
    prob1 = torch.log(torch.tensor(1 - lambda_value).detach()) + origin_prob # (batch_size)
    prob2 = torch.stack([prob1, torch.zeros_like(prob1) + torch.log(torch.tensor(lambda_value).detach()) \
                         + torch.log(torch.tensor(0.5).detach())], dim=1) # (batch_size, 2)
    prob3 = torch.logsumexp(prob2, dim=1) # (batch_size)
    return prob3
    
def batch_log_uppper_bound_helper(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    # join_min: batchsize * embed_size
    # join_max: batchsize * embed_size
    # log_prob: batch_size
    join_log_prob = batch_log_prob(join_min, join_max)
    join_log_prob_new = torch.logsumexp(torch.stack( \
        [torch.zeros(join_log_prob.shape[0]).fill_(torch.log(torch.tensor(0.1))).to(device), join_log_prob], dim=1), dim=1)
    x_log_prob = batch_log_prob(t1_min_embed, t1_max_embed)
    y_log_prob = batch_log_prob(t2_min_embed, t2_max_embed)
    log_xy = torch.logsumexp(torch.stack([x_log_prob, y_log_prob], dim=1), dim=1)
    log_upper_bound = join_log_prob_new + utils.log1mexp(join_log_prob_new - log_xy)
    return log_upper_bound

    
"""Some functions to calculate the probability."""

"""Overlap pos prob loss"""
def lambda_batch_log_prob(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    # this function return the negative conditional log probability of positive examplse if they have overlap.
    # we want to minimize the return value -log(p(a|b))
    joint_log = batch_log_prob(meet_min, meet_max)
    domi_log = batch_log_prob(t1_min_embed, t1_max_embed)  # batch_size
    cond_log = joint_log - domi_log # batch_size
    smooth_log_prob = smooth_prob(cond_log) # batch_size
    neg_smooth_log_prob = -smooth_log_prob # batch_size
    loss = torch.sigmoid(neg_smooth_log_prob) - 0.5
    return loss


"""Disjoint pos prob loss"""
    # this function return the upper bound log(p(a join b) + 0.01 - p(a) - p(b)) of positive examplse if they are disjoint
def lambda_batch_log_upper_bound(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    # minus the log probability of the condionaled term
    # we want to minimize the return value too
    # joint_log = batch_log_uppper_bound_helper(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
    # domi_log = batch_log_prob(t1_min_embed, t1_max_embed)
    # cond_log = joint_log - domi_log  # (batch_size)

    joint_log_prob = batch_log_prob(join_min, join_max)
    joint_prob = torch.exp(joint_log_prob)
    a_log_prob = batch_log_prob(t1_min_embed, t1_max_embed)
    a_prob = torch.exp(a_log_prob)
    b_log_prob = batch_log_prob(t2_min_embed, t2_max_embed)
    b_prob = torch.exp(b_log_prob)
    
    cond_log = torch.log(joint_prob - a_prob - b_prob + torch.tensor(0.01))
    cond_loss = torch.sigmoid(cond_log) - 0.5
    return cond_loss
    

"""Disjoint neg prob loss"""
def lambda_zero(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    joint_log_prob = batch_log_prob(join_min, join_max)
    joint_prob = torch.exp(joint_log_prob)
    a_log_prob = batch_log_prob(t1_min_embed, t1_max_embed)
    a_prob = torch.exp(a_log_prob)
    b_log_prob = batch_log_prob(t2_min_embed, t2_max_embed)
    b_prob = torch.exp(b_log_prob)
    
    cond_log = -torch.log(joint_prob - a_prob - b_prob + torch.tensor(0.01))
    cond_loss = torch.sigmoid(cond_log)
    return cond_loss

"""Overlap neg prob loss"""
def lambda_batch_log_1minus_prob(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
    meet_log = batch_log_prob(meet_min, meet_max)
    domi_log = batch_log_prob(t1_min_embed, t1_max_embed)
    cond_log = meet_log - domi_log
    # neg_smooth_log_prob = -smooth_prob(cond_log)

    # onemp_neg_smooth_log_prob = -utils.log1mexp(neg_smooth_log_prob)
    neg_smooth_log_prob = smooth_prob(cond_log)
    loss = torch.sigmoid(neg_smooth_log_prob)
    return loss
    
    