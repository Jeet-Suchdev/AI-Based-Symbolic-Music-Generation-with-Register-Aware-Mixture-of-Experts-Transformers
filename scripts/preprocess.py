import argparse
from musicgen.data.midi_processor import preprocess_directory
from musicgen.logging_utils import setup_logging

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    logger = setup_logging("logs")
    preprocess_directory(args.input, args.output, logger)

if __name__ == "__main__":
    main()
