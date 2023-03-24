import torch

def log1mexp(input_a):
    if input_a < 0.6931:
        return torch.log(-torch.expm1(-input_a))
    else:
        return torch.log1p(-torch.exp(-input_a))