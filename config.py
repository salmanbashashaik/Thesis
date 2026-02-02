# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
ldm3d/config.py

Global configuration/constants for the project.
Keep only stable, cross-module constants here.
Runtime hyperparams still come from argparse in main.py.
"""

# Import dependencies used by this module.
from __future__ import annotations

import os
import torch

# ----------------------------
# GLOBALS / CONFIG
# ----------------------------
# Default training volume size (you currently use 112³)
IMAGE_SIZE: int = int(os.environ.get("LDM3D_IMAGE_SIZE", "112"))

# Number of MRI modalities/channels (t1, t2, flair)
IMAGE_CHANNELS: int = int(os.environ.get("LDM3D_IMAGE_CHANNELS", "3"))

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
