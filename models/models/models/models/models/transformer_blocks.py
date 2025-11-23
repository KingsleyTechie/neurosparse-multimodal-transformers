import torch
import torch.nn as nn
import torch.nn.functional as F
from .spiking_gating import NeuroSparseAttention

class NeuroSparseTransformerBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, sparsity_ratio=0.3, ff_dim=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.attention = NeuroSparseAttention(hidden_dim, num_heads, sparsity_ratio)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, visual_tokens, audio_tokens, imu_tokens, membrane_potentials=None):
        attn_output, selection_mask, new_membranes = self.attention(
            self.norm1(visual_tokens),
            self.norm1(audio_tokens), 
            self.norm1(imu_tokens),
            membrane_potentials
        )
        
        all_input_tokens = torch.cat([visual_tokens, audio_tokens, imu_tokens], dim=1)
        
        batch_size = visual_tokens.shape[0]
        total_tokens = all_input_tokens.shape[1]
        
        output_tokens = all_input_tokens.clone()
        
        for i in range(batch_size):
            selected_indices = selection_mask[i].nonzero().squeeze()
            if selected_indices.numel() > 0:
                num_selected = min(selected_indices.numel(), attn_output.shape[1])
                output_tokens[i, selected_indices[:num_selected]] = attn_output[i, :num_selected]
        
        ff_output = self.ffn(self.norm2(output_tokens))
        output_tokens = output_tokens + self.dropout(ff_output)
        
        visual_len = visual_tokens.shape[1]
        audio_len = audio_tokens.shape[1]
        
        visual_output = output_tokens[:, :visual_len]
        audio_output = output_tokens[:, visual_len:visual_len+audio_len]
        imu_output = output_tokens[:, visual_len+audio_len:]
        
        return visual_output, audio_output, imu_output, new_membranes
