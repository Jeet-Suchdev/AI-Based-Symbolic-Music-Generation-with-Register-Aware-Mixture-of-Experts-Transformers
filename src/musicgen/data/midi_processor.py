from __future__ import annotations
from pathlib import Path
import hashlib
import json
import logging
from miditok import REMI, TokenizerConfig
from symusic import Score

def create_tokenizer():
    config = TokenizerConfig(
        pitch_range=(0, 128),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=True,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=True,
        num_tempos=32,
        tempo_range=(40, 250),
        remove_duplicated_notes=True,
    )
    return REMI(config)

def validate_midi(path: Path, min_notes: int = 8, min_ticks: int = 240) -> dict:
    try:
        score = Score(path)
        notes = 0
        programs = set()
        pitch_min, pitch_max = 127, 0
        for track in score.tracks:
            try:
                programs.add(int(track.program))
            except Exception:
                pass
            for note in track.notes:
                notes += 1
                pitch_min = min(pitch_min, int(note.pitch))
                pitch_max = max(pitch_max, int(note.pitch))
        if notes < min_notes:
            return {"valid": False, "reason": f"too_few_notes:{notes}"}
        if int(score.end()) < min_ticks:
            return {"valid": False, "reason": "too_short"}
        return {
            "valid": True,
            "notes": notes,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "programs": sorted(programs),
            "ticks": int(score.end()),
        }
    except Exception as exc:
        return {"valid": False, "reason": f"{type(exc).__name__}:{exc}"}

def preprocess_directory(input_dir: str, output_dir: str, logger=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger("musicgen")
    paths = sorted([*input_dir.rglob("*.mid"), *input_dir.rglob("*.midi")])
    seen = set()
    rows = []
    for src in paths:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if digest in seen:
            rows.append({"path": str(src), "status": "duplicate"})
            continue
        seen.add(digest)
        info = validate_midi(src)
        if not info["valid"]:
            rows.append({"path": str(src), "status": "invalid", "reason": info["reason"]})
            continue
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        rows.append({"path": str(src), "status": "valid", **{k:v for k,v in info.items() if k != "valid"}})
    (output_dir / "metadata.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Found %d MIDI files; valid=%d", len(paths), sum(r["status"]=="valid" for r in rows))
    return rows
