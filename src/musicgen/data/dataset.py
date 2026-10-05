from __future__ import annotations
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from .shards import list_shards, load_shard

class TokenShardDataset(Dataset):
    def __init__(self, directory, max_seq_len=512):
        self.paths = list_shards(directory)
        self.max_seq_len = max_seq_len
        self.samples = []
        for p in self.paths:
            self.samples.extend(load_shard(p))
        if not self.samples:
            raise RuntimeError(f"No tokenized samples found in {directory}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ids = s["ids"]
        regs = s["registers"]
        inst = s.get("instruments", [128] * len(ids))
        if len(ids) > self.max_seq_len:
            start = random.randint(0, len(ids) - self.max_seq_len)
            ids, regs, inst = ids[start:start+self.max_seq_len], regs[start:start+self.max_seq_len], inst[start:start+self.max_seq_len]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        r = torch.tensor(regs[:-1], dtype=torch.long)
        i = torch.tensor(inst[:-1], dtype=torch.long)
        return {"input_ids": x, "labels": y, "register_ids": r, "instrument_ids": i}

class TinySyntheticDataset(Dataset):
    """Deterministic tiny dataset used to validate the complete training stack."""
    def __init__(self, vocab_size=128, seq_len=64, n=32):
        self.data = []
        for j in range(n):
            ids = [1] + [2 + ((j + k) % max(4, vocab_size-4)) for k in range(seq_len-1)] + [3]
            regs = [0 if k == 0 else 1 + (k % 3) for k in range(seq_len)]
            inst = [128] * seq_len
            self.data.append({
                "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                "labels": torch.tensor(ids[1:], dtype=torch.long),
                "register_ids": torch.tensor(regs[:-1], dtype=torch.long),
                "instrument_ids": torch.tensor(inst[:-1], dtype=torch.long),
            })

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_batch(batch, pad_id=0):
    max_len = max(x["input_ids"].numel() for x in batch)
    keys = ["input_ids", "labels", "register_ids", "instrument_ids"]
    out = {}
    for key in keys:
        vals = []
        fill = pad_id if key in ("input_ids", "labels") else 0
        for x in batch:
            t = x[key]
            vals.append(torch.cat([t, torch.full((max_len-len(t),), fill, dtype=torch.long)]))
        out[key] = torch.stack(vals)
    out["attention_mask"] = out["input_ids"].ne(pad_id)
    return out
