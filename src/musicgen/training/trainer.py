from __future__ import annotations
from pathlib import Path
import time
import torch
from tqdm import tqdm
from .losses import language_model_loss, moe_load_balance_loss, routing_entropy
from .checkpoint import save_checkpoint, load_checkpoint, rotate_checkpoints
from .metrics import perplexity

class Trainer:
    def __init__(self, model, optimizer, scheduler, scaler, train_loader, val_loader, cfg, device, tokenizer_path=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.tokenizer_path = tokenizer_path
        self.logger = logger
        self.ckpt_dir = Path(cfg["checkpoint"]["directory"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def resume(self, path):
        ckpt = load_checkpoint(path, self.model, self.optimizer, self.scheduler, self.scaler, map_location=self.device)
        self.global_step = int(ckpt["global_step"])
        self.epoch = int(ckpt.get("epoch", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.logger.info("Resumed from %s at step=%d", path, self.global_step)

    def _batch_loss(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k,v in batch.items()}
        out = self.model(
            batch["input_ids"],
            batch["register_ids"],
            batch["instrument_ids"],
            batch["attention_mask"],
        )
        lm = language_model_loss(out["logits"], batch["labels"], self.model.pad_id)
        balance = torch.zeros((), device=self.device)
        entropies = []
        if out["moe_stats"]:
            for stat in out["moe_stats"]:
                balance = balance + moe_load_balance_loss(stat["router_probs"], stat["top_indices"])
                entropies.append(routing_entropy(stat["router_probs"]))
            balance = balance / len(out["moe_stats"])
        total = lm + float(self.cfg["moe"]["load_balance_weight"]) * balance
        return total, lm.detach(), balance.detach(), (sum(entropies)/len(entropies) if entropies else 0.0)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total = 0.0
        n = 0
        max_batches = int(self.cfg["training"].get("max_val_batches", 50))
        precision = self._amp_dtype()
        for batch in self.val_loader:
            with torch.autocast(device_type=self.device.type, dtype=precision, enabled=precision is not None):
                loss, _, _, _ = self._batch_loss(batch)
            total += float(loss)
            n += 1
            if n >= max_batches:
                break
        self.model.train()
        return total / max(1, n)

    def _amp_dtype(self):
        if self.device.type != "cuda":
            return None
        p = self.cfg["training"].get("precision", "auto")
        if p == "fp32":
            return None
        if p == "bf16" or (p == "auto" and torch.cuda.is_bf16_supported()):
            return torch.bfloat16
        return torch.float16

    def _save(self, name):
        save_checkpoint(
            self.ckpt_dir / name,
            self.model, self.optimizer, self.scheduler, self.scaler,
            self.epoch, self.global_step, self.best_val_loss,
            self.cfg, self.tokenizer_path
        )

    def fit(self):
        self.model.train()
        accum = max(1, int(self.cfg["training"]["gradient_accumulation_steps"]))
        max_steps = int(self.cfg["training"]["max_steps"])
        grad_clip = float(self.cfg["training"]["gradient_clip"])
        log_every = int(self.cfg["training"]["log_every_steps"])
        validate_every = int(self.cfg["training"]["validate_every_steps"])
        save_every = int(self.cfg["checkpoint"]["save_every_steps"])
        precision = self._amp_dtype()
        use_scaler = precision == torch.float16 and self.device.type == "cuda"
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(self.train_loader)
        running = 0.0
        micro_steps = 0
        pbar = tqdm(total=max_steps, initial=self.global_step, desc="optimizer steps")
        start = time.time()
        while self.global_step < max_steps:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            accumulation_loss = 0.0
            last_batch = None

            for _ in range(accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    self.epoch += 1
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                last_batch = batch
                with torch.autocast(device_type=self.device.type, dtype=precision, enabled=precision is not None):
                    loss, lm, balance, entropy = self._batch_loss(batch)
                    scaled_loss = loss / accum
                if use_scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                accumulation_loss += float(loss.detach())
                micro_steps += 1

            if use_scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            if use_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            running += accumulation_loss / accum
            pbar.update(1)

            if self.global_step % log_every == 0:
                avg = running / log_every
                running = 0.0
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = max(1e-6, time.time() - start)
                tok = last_batch["input_ids"].numel() * micro_steps / elapsed
                micro_steps = 0
                self.logger.info(
                    "step=%d loss=%.4f ppl=%.2f lm=%.4f balance=%.4f entropy=%.4f lr=%.3e tok/s=%.0f",
                    self.global_step, avg, perplexity(avg), float(lm), float(balance), entropy, lr, tok
                )

            if self.global_step % validate_every == 0:
                val = self.validate()
                self.logger.info("validation step=%d loss=%.4f ppl=%.2f", self.global_step, val, perplexity(val))
                if val < self.best_val_loss:
                    self.best_val_loss = val
                    if self.cfg["checkpoint"].get("save_best", True):
                        self._save("best.pt")

            if self.global_step % save_every == 0:
                self._save(f"step_{self.global_step}.pt")
                self._save("latest.pt")
                rotate_checkpoints(self.ckpt_dir, int(self.cfg["checkpoint"].get("keep_last_n", 3)))

        pbar.close()
        self._save("latest.pt")
        self.logger.info("Training complete at optimizer step=%d", self.global_step)
