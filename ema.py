#!/usr/bin/env python3
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

from __future__ import annotations

import torch
import torch.nn as nn


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

    def __init__(self, model: nn.Module, decay: float = 0.999):
        # Force decay into float so downstream math is consistent
        self.decay = float(decay)

        # Shadow weights live here (same structure as state_dict for trainable params)
        self.shadow: dict[str, torch.Tensor] = {}

        # Initialize shadows from the model's current parameters
        self._init(model)

    def _init(self, model: nn.Module) -> None:
        """
        Populate self.shadow with clones of the model's trainable parameters.

        Notes:
          - We only track parameters that require gradients (requires_grad=True).
          - detach().clone() ensures:
              * detach: not part of autograd graph
              * clone: independent storage (won't change if model params change)
        """
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
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
        for name, p in model.named_parameters():
            # EMA typically tracks only trainable parameters
            if not p.requires_grad:
                continue

            if name not in self.shadow:
                # If a param is new (rare), initialize it
                self.shadow[name] = p.detach().clone()
                continue

            s = self.shadow[name]

            # Ensure tensor type (defensive programming for weird checkpoint formats)
            if not torch.is_tensor(s):
                s = torch.tensor(s)

            # If mismatch, align shadow with model param
            if s.device != p.device or s.dtype != p.dtype:
                s = s.to(device=p.device, dtype=p.dtype)

            self.shadow[name] = s

    @torch.no_grad()
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

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # In-place EMA update:
            # shadow = decay*shadow + (1-decay)*p
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
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

        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])
