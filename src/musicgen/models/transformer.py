import torch
from torch import nn
from .attention import CausalSelfAttention
from .expert import ExpertFFN
from .moe import MoEBlock
from .cross_register import CrossRegisterFusion

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_moe=False, use_cross_register=False):
        super().__init__()
        d = cfg["model"]["d_model"]
        h = cfg["model"]["n_heads"]
        ff = cfg["model"]["ffn_dim"]
        drop = cfg["model"]["dropout"]
        self.norm1 = RMSNorm(d) if cfg["model"].get("normalization") == "rmsnorm" else nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h, drop, cfg["model"]["max_seq_len"])
        self.norm2 = RMSNorm(d) if cfg["model"].get("normalization") == "rmsnorm" else nn.LayerNorm(d)
        if use_moe:
            self.ff = MoEBlock(d, cfg["moe"]["expert_ffn_dim"], cfg["moe"]["num_experts"], cfg["moe"]["routing"], drop)
        else:
            self.ff = ExpertFFN(d, ff, drop)
        self.use_moe = use_moe
        self.cross = CrossRegisterFusion(d, cfg["cross_register"]["num_heads"], cfg["cross_register"]["dropout"]) if use_cross_register else None

    def forward(self, x, register_ids, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        y = self.ff(self.norm2(x))
        aux = {}
        if self.use_moe:
            y, aux = y
        x = x + y
        if self.cross is not None:
            x = self.cross(x, register_ids)
        return x, aux
