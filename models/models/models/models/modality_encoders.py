import torch
import torch.nn as nn

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.activation(x)
        
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        
        return x
