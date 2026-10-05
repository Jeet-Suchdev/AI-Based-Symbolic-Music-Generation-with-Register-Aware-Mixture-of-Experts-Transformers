import torch
from torch import nn

class MusicEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, num_registers=4, num_instruments=129, use_register=True, use_instrument=True):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.register = nn.Embedding(num_registers, d_model) if use_register else None
        self.instrument = nn.Embedding(num_instruments, d_model) if use_instrument else None

    def forward(self, input_ids, register_ids=None, instrument_ids=None):
        x = self.token(input_ids)
        if self.register is not None and register_ids is not None:
            x = x + self.register(register_ids.clamp(0, self.register.num_embeddings - 1))
        if self.instrument is not None and instrument_ids is not None:
            x = x + self.instrument(instrument_ids.clamp(0, self.instrument.num_embeddings - 1))
        return x
