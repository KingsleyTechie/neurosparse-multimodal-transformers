import torch
import torch.nn as nn
from .modality_encoders import ModalityEncoder
from .transformer_blocks import NeuroSparseTransformerBlock

class NeuroSparseTransformer(nn.Module):
    def __init__(self, 
                 visual_input_dim=3072,
                 audio_input_dim=40,
                 imu_input_dim=6,
                 hidden_dim=256,
                 num_layers=4,
                 num_heads=8,
                 sparsity_ratio=0.3,
                 num_classes=5,
                 dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sparsity_ratio = sparsity_ratio
        
        self.visual_encoder = ModalityEncoder(visual_input_dim, hidden_dim)
        self.audio_encoder = ModalityEncoder(audio_input_dim, hidden_dim)
        self.imu_encoder = ModalityEncoder(imu_input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            NeuroSparseTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                sparsity_ratio=sparsity_ratio,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.modality_weights = nn.Parameter(torch.ones(3))
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, visual_seq, audio_seq, imu_seq):
        batch_size, seq_len = visual_seq.shape[0], visual_seq.shape[1]
        
        if visual_seq.dim() == 5:
            visual_seq = visual_seq.view(batch_size, seq_len, -1)
        
        visual_tokens = self.visual_encoder(visual_seq)
        audio_tokens = self.audio_encoder(audio_seq)
        imu_tokens = self.imu_encoder(imu_seq)
        
        membrane_potentials = None
        
        for layer in self.layers:
            visual_tokens, audio_tokens, imu_tokens, membrane_potentials = layer(
                visual_tokens, audio_tokens, imu_tokens, membrane_potentials
            )
        
        weights = torch.softmax(self.modality_weights, dim=0)
        
        visual_repr = visual_tokens.mean(dim=1)
        audio_repr = audio_tokens.mean(dim=1)
        imu_repr = imu_tokens.mean(dim=1)
        
        combined_repr = torch.cat([
            weights[0] * visual_repr,
            weights[1] * audio_repr, 
            weights[2] * imu_repr
        ], dim=1)
        
        logits = self.classifier(combined_repr)
        return logits
