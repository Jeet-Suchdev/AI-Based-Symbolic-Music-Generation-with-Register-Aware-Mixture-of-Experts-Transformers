from pathlib import Path
from .music_metrics import midi_statistics
from .structural_metrics import token_repetition_ratio

def evaluate_midi_directory(directory):
    results = {}
    for p in sorted(Path(directory).glob("*.mid")):
        try:
            results[p.name] = midi_statistics(p)
        except Exception as exc:
            results[p.name] = {"error": str(exc)}
    return results
