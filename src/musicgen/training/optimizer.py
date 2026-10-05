import torch

def build_optimizer(model, cfg):
    t = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(t["learning_rate"]),
        betas=tuple(t.get("betas", [0.9, 0.95])),
        weight_decay=float(t.get("weight_decay", 0.1)),
    )
