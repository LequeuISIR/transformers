import torch
from torch.nn import functional as F


def softpick(x, dim=-1, eps=1e-8):
    # softpick function: relu(exp(x)-1) / sum(abs(exp(x)-1))
    # numerically stable version
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps) # epsilon is only useful if all inputs are EXACTLY 0. we might not even need it

