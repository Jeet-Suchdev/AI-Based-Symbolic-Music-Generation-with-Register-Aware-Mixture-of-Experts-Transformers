from collections import Counter
import json
from pathlib import Path

def summarize_samples(samples):
    lengths = [len(s["ids"]) for s in samples]
    regs = Counter()
    for s in samples:
        regs.update(s.get("registers", []))
    return {
        "num_sequences": len(samples),
        "mean_length": sum(lengths)/max(1, len(lengths)),
        "max_length": max(lengths, default=0),
        "min_length": min(lengths, default=0),
        "register_counts": dict(regs),
    }

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
