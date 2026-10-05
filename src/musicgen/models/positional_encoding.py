import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def rotate(self, q, k):
        # q/k: [B,H,T,D]
        t = q.size(-2)
        cos = self.cos[:t].to(q.device, q.dtype)[None, None, :, :]
        sin = self.sin[:t].to(q.device, q.dtype)[None, None, :, :]
        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., ::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out
        return rotate(q), rotate(k)
