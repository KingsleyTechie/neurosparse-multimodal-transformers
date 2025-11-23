import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.neurosparse import NeuroSparseTransformer
from models.spiking_gating import SpikingGatingNetwork

def test_spiking_gating():
    print("Testing Spiking Gating Network...")
    
    batch_size, seq_len, hidden_dim = 2, 50, 256
    visual_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    audio_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    imu_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    
    gating_net = SpikingGatingNetwork(hidden_dim=hidden_dim, sparsity_ratio=0.3)
    selected_tokens, selection_mask, membranes = gating_net(visual_tokens, audio_tokens, imu_tokens)
    
    print(f"Input tokens: {visual_tokens.shape}, {audio_tokens.shape}, {imu_tokens.shape}")
    print(f"Selected tokens: {selected_tokens.shape}")
    print(f"Sparsity achieved: {selection_mask.float().mean().item():.3f}")
    print("Spiking Gating Network test passed!")

def test_neurosparse_attention():
    print("Testing NeuroSparse Attention...")
    
    batch_size, seq_len, hidden_dim = 2, 50, 256
    visual_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    audio_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    imu_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    
    attention = NeuroSparseAttention(hidden_dim=hidden_dim, num_heads=8, sparsity_ratio=0.3)
    output, selection_mask, membranes = attention(visual_tokens, audio_tokens, imu_tokens)
    
    print(f"Attention output shape: {output.shape}")
    print(f"Selection mask shape: {selection_mask.shape}")
    print("NeuroSparse Attention test passed!")

if __name__ == "__main__":
    test_spiking_gating()
    test_neurosparse_attention()
