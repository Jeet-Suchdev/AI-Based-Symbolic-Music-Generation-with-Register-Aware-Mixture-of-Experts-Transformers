from pathlib import Path
import json
import math
from collections import Counter
from miditok import REMI
from symusic import Score

def midi_statistics(path):
    score = Score(path)
    pitches, velocities, durations = [], [], []
    for track in score.tracks:
        for note in track.notes:
            pitches.append(int(note.pitch))
            velocities.append(int(note.velocity))
            durations.append(int(note.duration))
    if not pitches:
        return {"notes": 0}
    pitch_classes = Counter(p % 12 for p in pitches)
    return {
        "notes": len(pitches),
        "pitch_mean": sum(pitches)/len(pitches),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "velocity_mean": sum(velocities)/len(velocities),
        "duration_mean_ticks": sum(durations)/len(durations),
        "pitch_class_entropy": -sum(
            (c/len(pitches))*math.log(c/len(pitches)+1e-12)
            for c in pitch_classes.values()
        ),
    }

def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
