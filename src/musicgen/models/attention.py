import torch
from torch import nn
import torch.nn.functional as F
from .positional_encoding import RotaryEmbedding

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, attention_mask=None):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q, k = self.rope.rotate(q, k)
        # PyTorch SDPA uses optimized kernels where supported.
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
        else:
            mask = None
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(mask is None),
        )
        y = y.transpose(1,2).contiguous().view(b,t,c)
        return self.out(y)
