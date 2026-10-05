from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end symbolic music training pipeline.")
    ap.add_argument("--config", default="configs/cross_register_moe.yaml")
    ap.add_argument("--dataset", default="maestro")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--drive-root", default="")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    root = Path(args.workdir).resolve()
    generated = root / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    # In Colab, put the persistent dataset cache and checkpoints on Drive. This
    # means a fresh runtime after disconnect does not need to redownload or
    # retokenize the corpus. Training itself still runs from the active VM.
    persistent_root = Path(args.drive_root).expanduser().resolve() if args.drive_root else root
    persistent_root.mkdir(parents=True, exist_ok=True)
    raw = persistent_root / "data/raw"
    processed = persistent_root / "data/processed"
    tokenized = persistent_root / "data/tokenized"
    checkpoint_dir = persistent_root / "checkpoints"
    log_dir = persistent_root / "logs"
    for d in (raw, processed, tokenized, checkpoint_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    if not args.skip_tests:
        run([python, "-m", "pytest", "-q"])

    run([python, "scripts/download_dataset.py", "--dataset", args.dataset,
         "--output", str(raw), "--cache", str(persistent_root / "data/cache")]
        + (["--force"] if args.force_data else []))

    processed_marker = processed / "metadata.json"
    if args.force_data or not processed_marker.exists():
        run([python, "scripts/preprocess.py", "--input", str(raw), "--output", str(processed)])
    else:
        print("\nProcessed dataset already exists; skipping cleaning.")

    stats_file = tokenized / "dataset_statistics.json"
    if args.force_data or not stats_file.exists():
        run([python, "scripts/build_dataset.py", "--input", str(processed),
             "--output", str(tokenized), "--config", args.config, "--raw-input", str(raw)])
    else:
        print("\nTokenized dataset already exists; skipping tokenization/sharding.")

    train_overrides = [
        f"data.train_path={tokenized / 'train'}",
        f"data.val_path={tokenized / 'val'}",
        f"data.test_path={tokenized / 'test'}",
        f"checkpoint.directory={checkpoint_dir}",
        f"logging.log_dir={log_dir}",
    ]
    run([python, "scripts/train.py", "--config", args.config, "--resume", *train_overrides])

    best = checkpoint_dir / "best.pt"
    latest = checkpoint_dir / "latest.pt"
    checkpoint = best if best.exists() else latest
    if not checkpoint.exists():
        raise FileNotFoundError("Training finished without producing a checkpoint.")

    if not args.skip_generation:
        output = generated / "sample.mid"
        run([python, "scripts/generate.py", "--checkpoint", str(checkpoint),
             "--output", str(output), "--max-tokens", str(args.max_tokens),
             "--temperature", str(args.temperature), "--top-k", str(args.top_k),
             "--top-p", str(args.top_p)])
        run([python, "scripts/evaluate.py", "--generated", str(generated),
             "--output", str(root / "evaluation" / "generated_metrics.json")])

    summary = {
        "dataset": args.dataset,
        "checkpoint": str(checkpoint),
        "generated": None if args.skip_generation else str(generated / "sample.mid"),
        "status": "complete",
    }
    (root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nPIPELINE COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
