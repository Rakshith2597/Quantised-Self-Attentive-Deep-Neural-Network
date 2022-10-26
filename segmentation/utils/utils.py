import torch
import numpy as np

def dice_coefficient(pred, target):
    epsilon = 1e-15
    pred = torch.flatten(pred,start_dim=1)
    target = torch.flatten(target,start_dim=1)
    intersection = torch.sum(pred*target,dim=1)
    union = torch.sum(pred + target,dim=1) + epsilon - intersection
    score = intersection/union
    return score
