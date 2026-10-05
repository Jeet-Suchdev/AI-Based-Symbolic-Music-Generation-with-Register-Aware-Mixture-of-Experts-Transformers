import torch
import torch.nn.functional as F

def language_model_loss(logits, labels, pad_id=0):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )

def moe_load_balance_loss(router_probs, top_indices):
    # router_probs: [B,T,E], top_indices: [B,T,K]
    p_mean = router_probs.mean(dim=(0,1))
    flat_idx = top_indices.reshape(-1)
    counts = torch.bincount(flat_idx, minlength=router_probs.size(-1)).float()
    f = counts / counts.sum().clamp_min(1.0)
    # Switch-style auxiliary loss: E * sum(f_i * p_i)
    return router_probs.size(-1) * torch.sum(f * p_mean)

def routing_entropy(router_probs):
    p = router_probs.clamp_min(1e-9)
    return float((-(p * p.log()).sum(-1)).mean().detach().cpu())
