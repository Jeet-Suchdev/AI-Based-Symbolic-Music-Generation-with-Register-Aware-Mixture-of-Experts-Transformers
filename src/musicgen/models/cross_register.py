import torch
from torch import nn

class CrossRegisterFusion(nn.Module):
    """
    Lightweight cross-register interaction.
    Register-specific pooled summaries are exchanged through attention-like
    query/key/value projections, then broadcast back to token states.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, register_ids):
        b,t,d = x.shape
        summaries = []
        for reg in (1,2,3):
            mask = register_ids.eq(reg)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
            summaries.append(pooled)
        reg_tokens = torch.stack(summaries, dim=1)
        mixed, _ = self.attn(reg_tokens, reg_tokens, reg_tokens, need_weights=False)
        mixed = self.norm(reg_tokens + mixed)
        token_reg = register_ids.clamp(1,3) - 1
        gathered = mixed.gather(1, token_reg.unsqueeze(-1).expand(-1,-1,d))
        return x + self.proj(gathered)
