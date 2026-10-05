from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "maestro": {
        "version": "v3.0.0",
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "sha256": "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c",
        "metadata_csv": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv",
        "metadata_json": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json",
        "license": "CC BY-NC-SA 4.0",
    }
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0"))
        done = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(destination)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and cache a supported symbolic MIDI dataset.")
    ap.add_argument("--dataset", default="maestro", choices=sorted(DATASETS))
    ap.add_argument("--output", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    raw = Path(args.output)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    archive = cache / f"{args.dataset}-{spec['version']}-midi.zip"
    manifest = raw / "dataset_manifest.json"

    if not archive.exists() or args.force:
        download(spec["url"], archive)

    digest = sha256(archive)
    if digest != spec["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {archive}: expected {spec['sha256']}, got {digest}")

    extract_marker = raw / ".dataset_extracted"
    if args.force and extract_marker.exists():
        extract_marker.unlink()

    if not extract_marker.exists():
        print(f"Extracting {archive} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)
        extract_marker.write_text(spec["version"], encoding="utf-8")
    else:
        print("Dataset archive already extracted; skipping extraction.")

    # Keep metadata next to the raw MIDI files so the dataset builder can use
    # the official composition-level split when one is supplied.
    for key, filename in (("metadata_csv", f"{args.dataset}-{spec['version']}.csv"),
                          ("metadata_json", f"{args.dataset}-{spec['version']}.json")):
        destination = raw / filename
        if not destination.exists() or args.force:
            download(spec[key], destination)

    manifest.write_text(json.dumps({
        "dataset": args.dataset,
        "version": spec["version"],
        "archive_sha256": digest,
        "license": spec["license"],
        "source": spec["url"],
    }, indent=2), encoding="utf-8")

    midi_count = sum(1 for p in raw.rglob("*") if p.suffix.lower() in {".mid", ".midi"})
    print(f"Dataset ready: {midi_count} MIDI files under {raw}")


if __name__ == "__main__":
    main()
