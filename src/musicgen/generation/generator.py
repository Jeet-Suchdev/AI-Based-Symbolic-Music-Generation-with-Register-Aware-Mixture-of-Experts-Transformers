from pathlib import Path
import torch
from ..data.tokenizer import MusicTokenizer

def generate_from_checkpoint(model, tokenizer: MusicTokenizer, checkpoint_path, output_path, device, max_tokens=1024, temperature=0.9, top_k=50, top_p=0.95):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    ids = torch.tensor([[bos]], device=device)
    regs = torch.zeros_like(ids)
    inst = torch.full_like(ids, 128)
    generated = model.generate(ids, regs, inst, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=eos)
    return tokenizer.decode(generated[0].tolist(), output_path)
