import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        return self.mlp(x)