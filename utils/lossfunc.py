import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        
    def forward(self, inputs, targets):
        if self.weights is not None:
            self.weights = self.weights.to(inputs.device)
        loss = F.cross_entropy(inputs, targets, weight=self.weights)
        return loss