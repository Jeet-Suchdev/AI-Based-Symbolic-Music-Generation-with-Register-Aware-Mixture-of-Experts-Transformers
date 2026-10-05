import math

def build_scheduler(optimizer, cfg):
    warmup = int(cfg["scheduler"]["warmup_steps"])
    total = int(cfg["training"]["max_steps"])
    min_lr = float(cfg["scheduler"]["min_lr"])
    base_lr = float(cfg["training"]["learning_rate"])

    def fn(step):
        if step < warmup:
            return max(1e-8, step / max(1, warmup))
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine
    return __import__("torch").optim.lr_scheduler.LambdaLR(optimizer, fn)
