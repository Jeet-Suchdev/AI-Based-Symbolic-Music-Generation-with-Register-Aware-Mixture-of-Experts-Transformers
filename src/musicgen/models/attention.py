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

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )

        self.out = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.dropout = dropout

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, x, attention_mask=None):
        b, t, c = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            b,
            t,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rope.rotate(q, k)

        # ---------------------------------------------------------
        # Explicit additive causal mask.
        #
        # Each position can attend only to itself and earlier
        # positions. Future positions receive -inf and therefore
        # zero attention probability after softmax.
        # ---------------------------------------------------------

        mask = torch.zeros(
            (t, t),
            device=x.device,
            dtype=q.dtype,
        )

        mask = mask.masked_fill(
            torch.triu(
                torch.ones(
                    (t, t),
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            float("-inf"),
        )

        # ---------------------------------------------------------
        # Apply padding mask to keys if supplied.
        # ---------------------------------------------------------

        if attention_mask is not None:
            padding_mask = attention_mask.bool()

            mask = mask.unsqueeze(0).unsqueeze(0)

            mask = mask.masked_fill(
                ~padding_mask[:, None, None, :],
                float("-inf"),
            )
        else:
            mask = mask.unsqueeze(0).unsqueeze(0)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=False,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(b, t, c)
        )

        return self.out(y)
