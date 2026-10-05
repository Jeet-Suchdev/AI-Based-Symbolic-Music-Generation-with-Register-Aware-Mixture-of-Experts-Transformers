import torch
from torch import nn

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts=3, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, self.k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return logits, probs, top_probs, top_idx
