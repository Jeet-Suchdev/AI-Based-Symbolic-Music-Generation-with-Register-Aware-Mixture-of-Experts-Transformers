from __future__ import annotations
from pathlib import Path
import json
import torch
from miditok import REMI
from .register import token_string_to_register, Register

class MusicTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls):
        from .midi_processor import create_tokenizer
        return cls(create_tokenizer())

    @classmethod
    def load(cls, path: str | Path):
        return cls(REMI(params=Path(path)))

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def pad_id(self):
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_id(self):
        return int(self.tokenizer["BOS_None"])

    @property
    def eos_id(self):
        return int(self.tokenizer["EOS_None"])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)

    def tokenize_midi(self, midi_path: str | Path):
        seq = self.tokenizer(Path(midi_path))
        if isinstance(seq, list):
            ids = []
            for s in seq:
                ids.extend(s.ids)
        else:
            ids = seq.ids
        ids = [int(x) for x in ids]
        tokens = self.tokenizer.ids_to_tokens(ids)
        registers = [
            int(token_string_to_register(str(tok)))
            for tok in tokens
        ]
        # Program tokens can carry instrument information. Keep a simple current-program track.
        instruments = []
        current_program = 128
        for tok in tokens:
            if str(tok).startswith("Program_"):
                try:
                    current_program = int(str(tok).rsplit("_", 1)[1])
                except ValueError:
                    current_program = 128
            instruments.append(current_program)
        return {
            "ids": ids,
            "registers": registers,
            "instruments": instruments,
        }

    def decode(self, ids, output_path: str | Path):
        ids = [int(x) for x in ids]
        # Remove padding and stop after EOS.
        cleaned = []
        for x in ids:
            if x == self.pad_id:
                continue
            cleaned.append(x)
            if x == self.eos_id:
                break
        try:
            seq = self.tokenizer.decode(cleaned)
        except AttributeError:
            from miditok import TokSequence
            seq = TokSequence(ids=cleaned)
            self.tokenizer.decode_token_ids(seq)
        midi = self.tokenizer(seq)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.dump_midi(output_path)
        return output_path
