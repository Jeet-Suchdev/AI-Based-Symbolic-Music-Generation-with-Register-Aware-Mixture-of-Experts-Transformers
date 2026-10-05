import torch
from torch import nn
from .router import TopKRouter
from .expert import ExpertFFN

class MoEBlock(nn.Module):
    def __init__(self, d_model, expert_hidden, num_experts=3, routing="top2", dropout=0.1):
        super().__init__()
        k = 1 if routing == "top1" else 2
        self.router = TopKRouter(d_model, num_experts, k)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, expert_hidden, dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        logits, probs, top_probs, top_idx = self.router(x)
        b,t,d = x.shape
        flat = x.reshape(-1,d)
        out = torch.zeros_like(flat)
        flat_probs = top_probs.reshape(-1, top_probs.size(-1))
        flat_idx = top_idx.reshape(-1, top_idx.size(-1))
        for expert_id, expert in enumerate(self.experts):
            for k in range(flat_idx.size(1)):
                mask = flat_idx[:,k] == expert_id
                if mask.any():
                    y = expert(flat[mask])
                    out[mask] += y * flat_probs[mask,k].unsqueeze(-1)
        return out.view(b,t,d), {
            "router_probs": probs,
            "top_indices": top_idx,
            "router_logits": logits,
        }
