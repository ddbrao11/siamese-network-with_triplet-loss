import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """Triplet loss to separate positive and negative samples"""
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)