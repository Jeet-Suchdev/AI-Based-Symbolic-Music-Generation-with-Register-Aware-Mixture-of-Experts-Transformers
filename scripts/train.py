from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from musicgen.config import load_config, apply_overrides
from musicgen.seed import seed_everything
from musicgen.device import get_device, device_report
from musicgen.logging_utils import setup_logging
from musicgen.data.dataset import TokenShardDataset, TinySyntheticDataset, collate_batch
from musicgen.data.tokenizer import MusicTokenizer
from musicgen.models.music_transformer import MusicTransformer
from musicgen.training.optimizer import build_optimizer
from musicgen.training.scheduler import build_scheduler
from musicgen.training.trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    ap.add_argument("--tiny-overfit", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(int(cfg["seed"]))
    logger = setup_logging(cfg["logging"]["log_dir"])
    device = get_device()
    logger.info("Device report: %s", device_report(device))

    tokenizer_path = Path(cfg["data"]["train_path"]).parent / "tokenizer" / "tokenizer.json"
    if args.tiny_overfit:
        vocab_size = 256
        pad_id = 0
        train_ds = TinySyntheticDataset(vocab_size=vocab_size, seq_len=min(64, cfg["model"]["max_seq_len"]), n=32)
        val_ds = TinySyntheticDataset(vocab_size=vocab_size, seq_len=min(64, cfg["model"]["max_seq_len"]), n=8)
    else:
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}. Run build_dataset.py first.")
        tok = MusicTokenizer.load(tokenizer_path)
        vocab_size, pad_id = tok.vocab_size, tok.pad_id
        train_ds = TokenShardDataset(cfg["data"]["train_path"], cfg["data"]["max_seq_len"])
        val_ds = TokenShardDataset(cfg["data"]["val_path"], cfg["data"]["max_seq_len"])

    collate = lambda b: collate_batch(b, pad_id=pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]), pin_memory=bool(cfg["data"]["pin_memory"]),
        persistent_workers=bool(cfg["data"]["persistent_workers"]), collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]), pin_memory=bool(cfg["data"]["pin_memory"]),
        persistent_workers=False, collate_fn=collate
    )

    model = MusicTransformer(cfg, vocab_size, pad_id).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda" and cfg["training"]["precision"] in ("auto","fp16")))
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    trainer = Trainer(model, optimizer, scheduler, scaler, train_loader, val_loader, cfg, device, tokenizer_path, logger)

    if args.resume is not None:
        ckpt_dir = Path(cfg["checkpoint"]["directory"])
        path = ckpt_dir / "latest.pt" if args.resume == "auto" else Path(args.resume)
        if path.exists():
            trainer.resume(path)
        else:
            logger.info("No checkpoint at %s; starting from scratch.", path)

    trainer.fit()

if __name__ == "__main__":
    main()
