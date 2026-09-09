from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, object]:
    """Seed supported RNGs and optionally enforce deterministic PyTorch operations.

    This makes repeated runs *more* reproducible on a fixed software/hardware stack.
    It does not claim bitwise reproducibility across platforms or library releases.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    state: dict[str, object] = {
        "seed": seed,
        "python_hash_seed": str(seed),
        "numpy_seeded": True,
        "torch_available": False,
        "torch_deterministic": False,
    }

    try:
        import torch
    except ImportError:
        return state

    state["torch_available"] = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        state["torch_deterministic"] = True
    return state
