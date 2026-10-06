import argparse
from pathlib import Path
import torch
from musicgen.config import load_config
from musicgen.device import get_device
from musicgen.data.tokenizer import MusicTokenizer
from musicgen.models.music_transformer import MusicTransformer
from musicgen.generation.generator import generate_from_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--style",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="Style: 0=Baroque, 1=Classical, 2=Romantic, 3=Modern",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tokenizer_path = ckpt.get("tokenizer_path")
    if not tokenizer_path:
        tokenizer_path = Path(cfg["data"]["train_path"]).parent / "tokenizer" / "tokenizer.json"
    tokenizer = MusicTokenizer.load(tokenizer_path)
    model = MusicTransformer(cfg, tokenizer.vocab_size, tokenizer.pad_id)
    device = get_device()
    output = generate_from_checkpoint(
        model, tokenizer, args.checkpoint, args.output, device,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.style,
    )
    print(f"Generated: {output}")

if __name__ == "__main__":
    main()
