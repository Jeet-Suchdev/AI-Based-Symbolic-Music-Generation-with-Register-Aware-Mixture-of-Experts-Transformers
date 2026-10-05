import torch

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.95):
    logits = logits / max(temperature, 1e-5)
    if top_k:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_probs[remove] = 0
        probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
    return torch.multinomial(probs, 1)
