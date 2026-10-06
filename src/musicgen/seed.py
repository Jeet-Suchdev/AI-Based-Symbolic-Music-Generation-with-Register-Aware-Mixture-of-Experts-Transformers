import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def _as_rng_byte_tensor(value):
    """Convert a saved RNG state into the ByteTensor expected by PyTorch."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8).contiguous()
    return torch.tensor(value, dtype=torch.uint8)

def set_rng_state(state):
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch.set_rng_state(
        _as_rng_byte_tensor(state["torch"])
    )

    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            _as_rng_byte_tensor(s)
            for s in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
