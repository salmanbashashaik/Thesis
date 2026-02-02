#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
ldm3d/ema.py

Exponential Moving Average (EMA) helper for model parameters.

What EMA is (in one breath):
  EMA keeps a slow-moving "smoothed" copy of model weights over training steps:
      shadow = decay * shadow + (1 - decay) * current_params
  Using EMA weights at sampling/inference often improves stability and perceptual quality
  for diffusion/gan-ish training, because it filters out high-frequency optimizer noise.

Fix implemented here:
  - Makes EMA device/dtype-safe when loading shadows from CPU checkpoints.
  - Ensures shadow tensors match the device/dtype of the model params before update/copy.

Why device/dtype safety matters:
  - It's common to save EMA on CPU (or load with map_location="cpu") for portability.
  - Later, we can train/sample on GPU (or mixed precision), so shadows must be moved/cast.
  - If not, we'll get runtime errors (device mismatch) or subtle precision issues.
"""

# Import dependencies used by this module.
from __future__ import annotations

import torch
import torch.nn as nn


# Class definition: `EMA` encapsulates related model behavior.
class EMA:
    """
    Exponential Moving Average tracker for a model's trainable parameters.

    Attributes:
      decay:
        Smoothing factor in [0,1). Higher -> slower updates -> more smoothing.
        Typical values: 0.999, 0.9999 for diffusion models.

      shadow:
        Dictionary mapping parameter names -> EMA-smoothed tensors.
        Keys match model.named_parameters() names.
    """

    # Function: `__init__` implements a reusable processing step.
    def __init__(self, model: nn.Module, decay: float = 0.999):
        # Force decay into float so downstream math is consistent
        self.decay = float(decay)

        # Shadow weights live here (same structure as state_dict for trainable params)
        self.shadow: dict[str, torch.Tensor] = {}

        # Initialize shadows from the model's current parameters

        self.backup: dict[str, torch.Tensor] = {}
        self._init(model)

    # Function: `_init` implements a reusable processing step.
    def _init(self, model: nn.Module) -> None:
        """
        Populate self.shadow with clones of the model's trainable parameters.

        Notes:
          - We only track parameters that require gradients (requires_grad=True).
          - detach().clone() ensures:
              * detach: not part of autograd graph
              * clone: independent storage (won't change if model params change)
        """
        # Control-flow branch for conditional or iterative execution.
        for name, p in model.named_parameters():
            # Control-flow branch for conditional or iterative execution.
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    # Function: `store` implements a reusable processing step.
    def store(self, model: nn.Module) -> None:
        """
        Save current model parameters so we can restore them after sampling.
        """
        self.backup = {}
        # Control-flow branch for conditional or iterative execution.
        for name, p in model.named_parameters():
            # Control-flow branch for conditional or iterative execution.
            if p.requires_grad:
                self.backup[name] = p.detach().clone()

    @torch.no_grad()
    # Function: `restore` implements a reusable processing step.
    def restore(self, model: nn.Module) -> None:
        """
        Restore parameters saved by store().
        """
        # Control-flow branch for conditional or iterative execution.
        if not getattr(self, "backup", None):
            # Return the computed value to the caller.
            return
        # Control-flow branch for conditional or iterative execution.
        for name, p in model.named_parameters():
            # Control-flow branch for conditional or iterative execution.
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    @torch.no_grad()
    # Function: `_ensure_shadow_matches` implements a reusable processing step.
    def _ensure_shadow_matches(self, model: nn.Module) -> None:
        """
        Ensure every tracked shadow tensor matches the corresponding model parameter's:
          - device (CPU vs GPU)
          - dtype  (float32 vs float16/bfloat16/etc.)

        When this is needed:
          - After loading EMA shadows from a checkpoint saved on CPU
          - When switching devices (CPU->GPU) or precision modes

        Behavior:
          - If a parameter name is missing in shadow, initialize it from model.
            (This is rare, but can happen if architecture changed or new params were added.)
          - If a stored shadow isn't a tensor, convert it into one.
          - If device/dtype differ, move/cast shadow to match the parameter.
        """
        # Control-flow branch for conditional or iterative execution.
        for name, p in model.named_parameters():
            # EMA typically tracks only trainable parameters
            # Control-flow branch for conditional or iterative execution.
            if not p.requires_grad:
                continue

            # Control-flow branch for conditional or iterative execution.
            if name not in self.shadow:
                # If a param is new (rare), initialize it
                self.shadow[name] = p.detach().clone()
                continue

            s = self.shadow[name]

            # Ensure tensor type (defensive programming for weird checkpoint formats)
            # Control-flow branch for conditional or iterative execution.
            if not torch.is_tensor(s):
                s = torch.tensor(s)

            # If mismatch, align shadow with model param
            # Control-flow branch for conditional or iterative execution.
            if s.device != p.device or s.dtype != p.dtype:
                s = s.to(device=p.device, dtype=p.dtype)

            self.shadow[name] = s

    @torch.no_grad()
    # Function: `update` implements a reusable processing step.
    def update(self, model: nn.Module) -> None:
        """
        Update EMA shadows using the current model parameters.

        Formula:
          shadow[name] = decay * shadow[name] + (1 - decay) * param[name]

        Implementation detail:
          - Uses in-place ops (mul_ and add_) for performance and to avoid extra allocations.
          - p.detach() is used to ensure we don't create autograd history.
        """
        # First guarantee shadow tensors are aligned with current model params
        self._ensure_shadow_matches(model)

        # Control-flow branch for conditional or iterative execution.
        for name, p in model.named_parameters():
            # Control-flow branch for conditional or iterative execution.
            if not p.requires_grad:
                continue

            # In-place EMA update:
            # shadow = decay*shadow + (1-decay)*p
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    # Function: `copy_to` implements a reusable processing step.
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA-smoothed parameters into the model (commonly used before sampling).

        Typical usage in diffusion sampling:
          - Save current weights (optional)
          - ema.copy_to(unet)
          - run sampling

        Notes:
          - Uses p.data.copy_(...) to directly overwrite parameter storage.
          - Only copies into parameters that require gradients and exist in shadow.
        """
        self._ensure_shadow_matches(model)

        # Control-flow branch for conditional or iterative execution.
        for name, p in model.named_parameters():
            # Control-flow branch for conditional or iterative execution.
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])
