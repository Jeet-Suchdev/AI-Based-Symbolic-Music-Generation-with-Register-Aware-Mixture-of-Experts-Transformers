import argparse, json
from pathlib import Path
from musicgen.evaluation.evaluator import evaluate_midi_directory

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--output", default="generated/evaluation.json")
    args = ap.parse_args()
    result = evaluate_midi_directory(args.generated)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
