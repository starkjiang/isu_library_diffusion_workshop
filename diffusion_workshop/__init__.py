"""Helper package for the two-day "Generative AI with Diffusion Models" workshop.

The notebooks keep the *ideas* (U-Net blocks, the diffusion math, guidance, CLIP)
in plain sight. This package holds the plumbing (data loading, plotting) plus a
reference implementation of everything the students build, so that later
notebooks never depend on an earlier exercise having been finished.
"""
import os

__version__ = "1.0.0"

#: Set WORKSHOP_SMOKE=1 to run every notebook on tiny synthetic data in seconds.
#: Used by tools/smoke_test.py and handy for instructors checking an install.
SMOKE = os.environ.get("WORKSHOP_SMOKE", "0") == "1"


def pick(full, smoke):
    """Return `full` normally and `smoke` when WORKSHOP_SMOKE=1."""
    return smoke if SMOKE else full


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed=0):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
