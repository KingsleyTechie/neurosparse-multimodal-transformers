import torch
import torch.nn as nn
from models.modality_encoders import ModalityEncoder

class DenseMultimodalTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=4, num_classes=5):
        super().__init__()
        self.visual_encoder = ModalityEncoder(3072, hidden_dim)
        self.audio_encoder = ModalityEncoder(40, hidden_dim)
        self.imu_encoder = ModalityEncoder(6, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, visual_seq, audio_seq, imu_seq):
        batch_size, seq_len = visual_seq.shape[0], visual_seq.shape[1]
        
        if visual_seq.dim() == 5:
            visual_seq = visual_seq.view(batch_size, seq_len, -1)
            
        visual_tokens = self.visual_encoder(visual_seq)
        audio_tokens = self.audio_encoder(audio_seq)
        imu_tokens = self.imu_encoder(imu_seq)
        
        all_tokens = torch.cat([visual_tokens, audio_tokens, imu_tokens], dim=1)
        encoded = self.transformer(all_tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

class LSTMMultimodal(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=5):
        super().__init__()
        self.visual_proj = nn.Linear(3072, hidden_dim)
        self.audio_proj = nn.Linear(40, hidden_dim)
        self.imu_proj = nn.Linear(6, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim * 3, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, visual_seq, audio_seq, imu_seq):
        batch_size, seq_len = visual_seq.shape[0], visual_seq.shape[1]
        
        if visual_seq.dim() == 5:
            visual_seq = visual_seq.view(batch_size, seq_len, -1)
            
        visual_proj = self.visual_proj(visual_seq)
        audio_proj = self.audio_proj(audio_seq)
        imu_proj = self.imu_proj(imu_seq)
        
        combined = torch.cat([visual_proj, audio_proj, imu_proj], dim=-1)
        lstm_out, (hidden, cell) = self.lstm(combined)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.classifier(last_hidden)
