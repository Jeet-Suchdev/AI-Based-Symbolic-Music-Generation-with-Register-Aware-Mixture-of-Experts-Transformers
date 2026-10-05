from __future__ import annotations
from pathlib import Path
import torch

def save_shard(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)

def load_shard(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def list_shards(directory):
    return sorted(Path(directory).glob("shard_*.pt"))
