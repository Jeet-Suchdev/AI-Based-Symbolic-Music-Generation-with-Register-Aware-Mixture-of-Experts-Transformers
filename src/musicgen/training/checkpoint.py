from __future__ import annotations
from pathlib import Path
import os
import torch
from ..seed import get_rng_state, set_rng_state

def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, cfg, tokenizer_path=None):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": cfg,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "rng_state": get_rng_state(),
    }
    _atomic_save(state, Path(path))

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    set_rng_state(ckpt.get("rng_state"))
    return ckpt

def rotate_checkpoints(directory, keep_last_n):
    directory = Path(directory)
    paths = sorted(directory.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[keep_last_n:]:
        p.unlink(missing_ok=True)
