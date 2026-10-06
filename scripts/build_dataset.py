from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
from musicgen.config import load_config
from musicgen.data.midi_processor import create_tokenizer
from musicgen.data.tokenizer import MusicTokenizer
from musicgen.data.shards import save_shard
from musicgen.data.statistics import summarize_samples, save_json


def chunk_sample(sample, max_len):
    ids, regs, inst = sample["ids"], sample["registers"], sample["instruments"]
    if len(ids) < 4:
        return []
    out = []
    step = max(1, max_len - 64)
    for start in range(0, len(ids), step):
        end = min(len(ids), start + max_len)
        if end - start >= 4:
            out.append({"ids": ids[start:end], "registers": regs[start:end], "instruments": inst[start:end]})
        if end == len(ids):
            break
    return out


def load_official_split(raw_input: Path, processed_input: Path) -> dict[str, str]:
    """Map MIDI relative paths to official train/validation/test labels when metadata exists."""
    candidates = list(raw_input.glob("*.csv"))
    if not candidates:
        return {}
    csv_path = candidates[0]
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            midi = row.get("midi_filename", "")
            split = row.get("split", "")
            if not midi or split not in {"train", "validation", "test"}:
                continue
            mapping[Path(midi).as_posix()] = {"validation": "val", "train": "train", "test": "test"}[split]
    return mapping


STYLE_NAMES = {
    0: "baroque",
    1: "classical",
    2: "romantic",
    3: "modern",
}

def assign_style(composer: str) -> int:
    """Assign a composer-period style proxy label.

    IMPORTANT:
    These are composer-based historical style proxies, not
    ground-truth genre labels from MAESTRO.
    """
    c = str(composer).lower()

    if any(x in c for x in [
        "bach",
        "handel",
        "scarlatti",
        "vivaldi",
        "couperin",
        "rameau",
        "telemann",
    ]):
        return 0

    if any(x in c for x in [
        "mozart",
        "haydn",
        "beethoven",
        "clementi",
    ]):
        return 1

    if any(x in c for x in [
        "chopin",
        "liszt",
        "schumann",
        "brahms",
        "mendelssohn",
        "tchaikovsky",
        "rachmaninoff",
        "scriabin",
        "grieg",
        "dvorak",
        "smetana",
        "saint-saens",
        "schubert",
    ]):
        return 2

    return 3


def load_style_metadata(raw_input: Path) -> dict[str, int]:
    """Map MIDI relative paths to composer-period style IDs."""
    candidates = list(raw_input.glob("*.csv"))

    if not candidates:
        return {}

    csv_path = candidates[0]
    mapping = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            midi = row.get("midi_filename", "")
            composer = row.get("canonical_composer", "")

            if not midi:
                continue

            mapping[Path(midi).as_posix()] = assign_style(composer)

    return mapping


def resolve_style(
    path: Path,
    processed_input: Path,
    style_metadata: dict[str, int],
) -> int:
    """Resolve style ID for a MIDI file."""
    rel = path.relative_to(processed_input).as_posix()

    if rel in style_metadata:
        return style_metadata[rel]

    matches = [
        v for k, v in style_metadata.items()
        if rel.endswith(k)
    ]

    return matches[0] if matches else 3


def resolve_split(path: Path, processed_input: Path, official: dict[str, str]) -> str | None:
    rel = path.relative_to(processed_input).as_posix()
    if rel in official:
        return official[rel]
    # Metadata may use a dataset root prefix; match suffix as a fallback.
    matches = [v for k, v in official.items() if rel.endswith(k)]
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw-input", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)
    random.seed(args.seed)
    input_dir, out = Path(args.input), Path(args.output)
    raw_input = Path(args.raw_input) if args.raw_input else input_dir
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = MusicTokenizer(create_tokenizer())
    tok_dir = out / "tokenizer"
    tokenizer.save(tok_dir / "tokenizer.json")

    paths = sorted([*input_dir.rglob("*.mid"), *input_dir.rglob("*.midi")])
    official = load_official_split(raw_input, input_dir)
    style_metadata = load_style_metadata(raw_input)

    assigned = {"train": [], "val": [], "test": []}
    unassigned = []
    for p in paths:
        split = resolve_split(p, input_dir, official)
        (assigned[split] if split else unassigned).append(p)

    # For datasets without official splits, use deterministic file-level splitting.
    if unassigned:
        rng = random.Random(args.seed)
        rng.shuffle(unassigned)
        n = len(unassigned)
        n_test = max(1, int(n * args.test_ratio)) if n else 0
        n_val = max(1, int(n * args.val_ratio)) if n else 0
        assigned["test"].extend(unassigned[:n_test])
        assigned["val"].extend(unassigned[n_test:n_test+n_val])
        assigned["train"].extend(unassigned[n_test+n_val:])

    split_manifest = {k: [str(p) for p in v] for k, v in assigned.items()}
    save_json(split_manifest, out / "split_manifest.json")

    stats = {}
    for split, split_paths in assigned.items():
        samples = []
        for p in split_paths:
            try:
                s = tokenizer.tokenize_midi(p)

                style_id = resolve_style(
                    p,
                    input_dir,
                    style_metadata,
                )

                chunks = chunk_sample(
                    s,
                    int(cfg["data"]["max_seq_len"]),
                )

                for chunk in chunks:
                    chunk["style_id"] = style_id

                samples.extend(chunks)
            except Exception as exc:
                print(f"Skipping {p}: {exc}")
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for old in split_dir.glob("shard_*.pt"):
            old.unlink()
        for i in range(0, len(samples), 256):
            save_shard(samples[i:i+256], split_dir / f"shard_{i//256:05d}.pt")
        stats[split] = summarize_samples(samples)
        stats[split]["source_files"] = len(split_paths)
        print(split, stats[split])
    save_json(stats, out / "dataset_statistics.json")

if __name__ == "__main__":
    main()
