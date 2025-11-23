import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingGatingNetwork(nn.Module):
    def __init__(self, hidden_dim=256, sparsity_ratio=0.3, leak_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sparsity_ratio = sparsity_ratio
        self.leak_rate = leak_rate
        
        self.membrane_decay = 0.95
        self.firing_threshold = 1.0
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.visual_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.audio_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.imu_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, visual_tokens, audio_tokens, imu_tokens, membrane_potentials=None):
        batch_size, visual_len, _ = visual_tokens.shape
        _, audio_len, _ = audio_tokens.shape
        _, imu_len, _ = imu_tokens.shape
        
        visual_tokens = visual_tokens + self.visual_embed
        audio_tokens = audio_tokens + self.audio_embed  
        imu_tokens = imu_tokens + self.imu_embed
        
        all_tokens = torch.cat([visual_tokens, audio_tokens, imu_tokens], dim=1)
        total_tokens = all_tokens.shape[1]
        
        if membrane_potentials is None:
            membrane_potentials = torch.zeros(batch_size, total_tokens, device=all_tokens.device)
        
        importance_scores = self.gate_network(all_tokens).squeeze(-1)
        
        membrane_potentials = self.membrane_decay * membrane_potentials + importance_scores
        membrane_potentials = membrane_potentials * (1 - self.leak_rate)
        
        firing_mask = membrane_potentials >= self.firing_threshold
        
        non_firing_mask = ~firing_mask
        non_firing_potentials = membrane_potentials * non_firing_mask.float()
        
        num_firing = firing_mask.sum(dim=1)
        target_sparse = int(self.sparsity_ratio * total_tokens)
        additional_needed = torch.clamp(target_sparse - num_firing, min=0)
        
        selection_mask = firing_mask.clone()
        for i in range(batch_size):
            if additional_needed[i] > 0:
                non_firing_idx = non_firing_mask[i].nonzero().squeeze()
                if non_firing_idx.numel() > 0:
                    if non_firing_idx.numel() <= additional_needed[i]:
                        topk_indices = non_firing_idx
                    else:
                        non_firing_vals = non_firing_potentials[i, non_firing_idx]
                        topk_values, topk_indices = torch.topk(non_firing_vals, 
                                                             additional_needed[i].item())
                        topk_indices = non_firing_idx[topk_indices]
                    
                    selection_mask[i, topk_indices] = True
        
        new_membranes = membrane_potentials * (~selection_mask).float()
        
        selected_tokens = []
        for i in range(batch_size):
            batch_tokens = all_tokens[i]
            batch_mask = selection_mask[i]
            selected_batch_tokens = batch_tokens[batch_mask]
            selected_tokens.append(selected_batch_tokens)
        
        max_selected = max([t.shape[0] for t in selected_tokens])
        padded_tokens = torch.zeros(batch_size, max_selected, self.hidden_dim, 
                                  device=all_tokens.device)
        selection_padding_mask = torch.ones(batch_size, max_selected, 
                                          device=all_tokens.device, dtype=torch.bool)
        
        for i, tokens in enumerate(selected_tokens):
            num_tokens = tokens.shape[0]
            padded_tokens[i, :num_tokens] = tokens
            selection_padding_mask[i, :num_tokens] = False
        
        return padded_tokens, selection_mask, new_membranes

class NeuroSparseAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, sparsity_ratio=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.gating_network = SpikingGatingNetwork(hidden_dim, sparsity_ratio)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, visual_tokens, audio_tokens, imu_tokens, membrane_potentials=None):
        batch_size = visual_tokens.shape[0]
        
        selected_tokens, selection_mask, new_membranes = self.gating_network(
            visual_tokens, audio_tokens, imu_tokens, membrane_potentials
        )
        
        q = self.q_proj(selected_tokens).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(selected_tokens).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(selected_tokens).view(batch_size, -1, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_output = torch.matmul(F.softmax(attn_weights, dim=-1), v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        
        output = self.out_proj(attn_output)
        
        return output, selection_mask, new_membranes
