from __future__ import annotations
import torch
from torch import nn
from .embeddings import MusicEmbeddings
from .transformer import TransformerBlock, RMSNorm

class MusicTransformer(nn.Module):
    def __init__(self, cfg, vocab_size, pad_id=0):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        m = cfg["model"]
        self.emb = MusicEmbeddings(
            vocab_size,
            m["d_model"],
            m.get("num_registers", 4),
            m.get("num_instruments", 129),
            m.get("register_embedding", True),
            m.get("instrument_embedding", True),
        )
        arch = m["architecture"]
        use_moe = arch in {"moe", "cross_register_moe"} and cfg["moe"]["enabled"]
        use_cross = arch == "cross_register_moe" and cfg["cross_register"]["enabled"]
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, use_moe=use_moe, use_cross_register=use_cross)
            for _ in range(m["n_layers"])
        ])
        self.norm = RMSNorm(m["d_model"]) if m.get("normalization") == "rmsnorm" else nn.LayerNorm(m["d_model"])
        self.lm_head = nn.Linear(m["d_model"], vocab_size, bias=False)
        self.lm_head.weight = self.emb.token.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is not self.lm_head:
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, input_ids, register_ids=None, instrument_ids=None, attention_mask=None):
        x = self.emb(input_ids, register_ids, instrument_ids)
        moe_stats = []
        for block in self.blocks:
            x, aux = block(x, register_ids, attention_mask)
            if aux:
                moe_stats.append(aux)
        logits = self.lm_head(self.norm(x))
        return {"logits": logits, "moe_stats": moe_stats}

    @torch.no_grad()
    def generate(self, input_ids, register_ids, instrument_ids, max_new_tokens=256, temperature=1.0, top_k=50, top_p=0.95, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            max_len = self.cfg["model"]["max_seq_len"]
            ids = input_ids[:, -max_len:]
            regs = register_ids[:, -max_len:]
            inst = instrument_ids[:, -max_len:]
            out = self(ids, regs, inst)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_probs[remove] = 0
                probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
            next_id = torch.multinomial(probs, 1)
            next_reg = torch.zeros_like(next_id)
            next_inst = instrument_ids[:, -1:]
            input_ids = torch.cat([input_ids, next_id], 1)
            register_ids = torch.cat([register_ids, next_reg], 1)
            instrument_ids = torch.cat([instrument_ids, next_inst], 1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return input_ids
