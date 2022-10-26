import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1e-5):
        
        y_pred = F.softmax(y_pred,dim=1)[:, 1] # only penumbra probabilities
        y_pred = y_pred.flatten()
        y_true = y_true[:,1].flatten()
        intersection = (y_pred * y_true).sum()
        dice = (2.*intersection + smooth)/(y_pred.sum() + y_true.sum() + smooth)
        return 1 - dice
