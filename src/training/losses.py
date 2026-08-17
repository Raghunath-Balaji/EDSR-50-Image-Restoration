import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

def charbonnier_loss(pred, target, eps=1e-6):
    return torch.mean(torch.sqrt((pred - target)**2 + eps))

class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CombinedLoss, self).__init__()
        # SENIOR ADVICE: Reverse the ratio. 85% pixels, 15% SSIM.
        self.alpha = 0.85
        self.beta = 0.15

    def forward(self, target, pred):
        loss_charb = charbonnier_loss(pred, target)
        loss_ssim = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
        return (self.alpha * loss_charb) + (self.beta * loss_ssim)

def get_combined_loss(alpha, beta, gamma):
    return CombinedLoss(alpha, beta, gamma)
