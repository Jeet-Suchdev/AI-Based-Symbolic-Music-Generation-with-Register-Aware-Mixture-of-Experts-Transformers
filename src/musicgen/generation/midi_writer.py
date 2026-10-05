from pathlib import Path

def validate_midi_path(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 0
