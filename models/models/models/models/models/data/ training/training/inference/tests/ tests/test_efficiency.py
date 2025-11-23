import torch
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.neurosparse import NeuroSparseTransformer
from training.benchmarks import DenseMultimodalTransformer

def compute_flops(model, seq_length=50):
    hidden_dim = model.hidden_dim
    num_heads = 8
    head_dim = hidden_dim // num_heads
    num_layers = model.num_layers
    sparsity_ratio = getattr(model, 'sparsity_ratio', 1.0)
    
    total_tokens = seq_length * 3
    
    if hasattr(model, 'sparsity_ratio'):
        sparse_tokens = int(total_tokens * sparsity_ratio)
        sparse_qk_flops = sparse_tokens * sparse_tokens * head_dim * num_heads
        sparse_av_flops = sparse_tokens * sparse_tokens * head_dim * num_heads
        sparse_attention_flops = sparse_qk_flops + sparse_av_flops
        sparse_ffn_flops = sparse_tokens * hidden_dim * 512 * 2
        sparse_per_layer = sparse_attention_flops + sparse_ffn_flops
        total_flops = sparse_per_layer * num_layers
    else:
        dense_qk_flops = total_tokens * total_tokens * head_dim * num_heads
        dense_av_flops = total_tokens * total_tokens * head_dim * num_heads
        dense_attention_flops = dense_qk_flops + dense_av_flops
        dense_ffn_flops = total_tokens * hidden_dim * 512 * 2
        dense_per_layer = dense_attention_flops + dense_ffn_flops
        total_flops = dense_per_layer * num_layers
    
    return total_flops

def test_efficiency():
    print("Testing Model Efficiency...")
    
    neurosparse_model = NeuroSparseTransformer()
    dense_model = DenseMultimodalTransformer()
    
    neurosparse_flops = compute_flops(neurosparse_model)
    dense_flops = compute_flops(dense_model)
    
    reduction = ((dense_flops - neurosparse_flops) / dense_flops * 100)
    
    print(f"NeuroSparse FLOPs: {neurosparse_flops:,}")
    print(f"Dense Transformer FLOPs: {dense_flops:,}")
    print(f"FLOPs reduction: {reduction:.1f}%")
    
    batch_size, seq_len = 1, 50
    visual_test = torch.randn(batch_size, seq_len, 3, 32, 32)
    audio_test = torch.randn(batch_size, seq_len, 40)
    imu_test = torch.randn(batch_size, seq_len, 6)
    
    start_time = time.time()
    with torch.no_grad():
        neurosparse_output = neurosparse_model(visual_test, audio_test, imu_test)
    neurosparse_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        dense_output = dense_model(visual_test, audio_test, imu_test)
    dense_time = time.time() - start_time
    
    print(f"NeuroSparse inference time: {neurosparse_time:.4f}s")
    print(f"Dense inference time: {dense_time:.4f}s")
    print(f"Speedup: {dense_time/neurosparse_time:.2f}x")

if __name__ == "__main__":
    test_efficiency()
