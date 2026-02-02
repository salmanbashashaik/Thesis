# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
ldm3d/utils_seed.py

Reproducibility helpers.
"""

# Import dependencies used by this module.
from __future__ import annotations

import random
import numpy as np
import torch

# Function: `set_seed` implements a reusable processing step.
def set_seed(seed: int = 42) -> None:
    print(f"[SETUP] Using seed = {seed}")
    # Seed Python, NumPy, and PyTorch RNGs for reproducible experiments.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=False allows non-deterministic CuDNN kernels; benchmark=True
    # lets CuDNN pick fastest kernels for fixed input shapes.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
