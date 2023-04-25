import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        return self.linear(x)
        
