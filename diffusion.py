#!/usr/bin/env python3
"""
ldm3d/diffusion.py

DDPM utilities operating in latent space:
- beta schedule
- forward diffusion q_sample
- reverse sampling p_sample_loop (supports mask + optional control, with CFG)

This file implements the *diffusion process* in latent space, not pixel/voxel space.

Key idea:
  We diffuse and denoise latent tensors z (shape [B, zC, D, H, W]) produced by the VAE encoder.
  This makes diffusion cheaper and often more stable than operating on full-resolution volumes.

Notation (standard DDPM):
  - z0: clean latent (from VAE encoder)
  - zt: noisy latent after t steps
  - eps ~ N(0, I): Gaussian noise
  - UNet predicts eps (noise) given zt and timestep t (and conditioning signals)

Conditioning supported:
  - mask:   [B, 1, D, H, W]  (e.g., tumor region at latent resolution)
  - control:[B, Cc, D, H, W] (e.g., derived features like [mask, edge, dist])
  The UNet is expected to accept these arguments and incorporate them internally.

Classifier-Free Guidance (CFG):
  - Run the model twice: conditional and "unconditional" (by zeroing conditioning inputs)
  - Combine predictions to steer samples toward the condition.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ldm3d.config import DEVICE


# ----------------------------
# BETA SCHEDULE
# ----------------------------
def make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """
    Create a simple *linear* beta schedule.

    Inputs:
      T: number of diffusion steps (timesteps)
      beta_start: initial diffusion noise variance
      beta_end: final diffusion noise variance

    Output:
      betas: [T] float32 tensor, linearly interpolated from beta_start -> beta_end

    Notes:
      - betas control how much noise is added at each forward step.
      - Many schedules exist (cosine, quadratic, etc.); linear is a common baseline.
    """
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


# ----------------------------
# DDPM (LATENT)
# ----------------------------
class LatentDDPM:
    """
    A minimal DDPM implementation that operates on latent tensors.

    Responsibilities:
      - Precompute diffusion buffers (betas, alphas, cumulative products, posterior variance)
      - Forward diffusion sampling (q_sample): z0 -> zt
      - Reverse denoising sampling loop (p_sample_loop): noise -> sample latent

    Important implementation detail:
      - Buffers are placed on DEVICE at init time for convenience/performance.
      - During calls, buffers are re-cast/moved to match z's device/dtype when needed.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        # Store number of timesteps
        self.T = int(T)

        # keep all diffusion buffers on DEVICE
        # betas[t] controls noise injected at step t in the forward process
        betas = make_beta_schedule(self.T, beta_start, beta_end).to(DEVICE)
        self.betas = betas

        # alphas[t] = 1 - beta[t]
        # Often interpreted as "signal retention" per step.
        self.alphas = 1.0 - betas

        # alphas_cumprod[t] = Π_{s=0..t} alpha[s]
        # This is the cumulative signal retention up to timestep t.
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute square roots used in closed-form forward diffusion:
        # z_t = sqrt(acp_t) * z_0 + sqrt(1-acp_t) * eps
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_om_acp = torch.sqrt(1.0 - self.alphas_cumprod)

        # acp_prev[t] = alphas_cumprod[t-1] with acp_prev[0]=1
        # Used in reverse posterior q(z_{t-1} | z_t, z_0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=DEVICE), self.alphas_cumprod[:-1]],
            dim=0,
        )

        # Posterior variance for q(z_{t-1} | z_t, z_0):
        # var_t = beta_t * (1 - acp_{t-1}) / (1 - acp_t)
        # Clamp for numerical safety.
        self.posterior_variance = (
            betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod + 1e-8)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    # ----------------------------
    # FORWARD PROCESS
    # ----------------------------
    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Sample from the *forward diffusion* distribution q(z_t | z_0).

        Inputs:
          z0:    [B, zC, D, H, W] clean latent tensor
          t:     [B] integer timesteps (long); can be on any device (will be moved)
          noise: optional eps noise tensor of same shape as z0;
                 if None, drawn from N(0, I)

        Returns:
          zt:    noisy latents at timestep t
          noise: the noise used (handy for training targets)

        Formula:
          z_t = sqrt(alphas_cumprod[t]) * z_0 + sqrt(1 - alphas_cumprod[t]) * eps
        """
        if noise is None:
            # Default: standard Gaussian noise
            noise = torch.randn_like(z0)

        # ensure t is on same device
        # (indexing buffers requires t to be on same device as z0 for subsequent ops)
        t = t.to(device=z0.device, dtype=torch.long)

        # Gather per-sample scalars and reshape to broadcast over [zC,D,H,W]
        a = self.sqrt_acp[t].view(-1, 1, 1, 1, 1).to(device=z0.device, dtype=z0.dtype)
        b = self.sqrt_om_acp[t].view(-1, 1, 1, 1, 1).to(device=z0.device, dtype=z0.dtype)

        # Produce z_t and return noise used
        return a * z0 + b * noise, noise

    # ----------------------------
    # REVERSE SAMPLING
    # ----------------------------
    @torch.no_grad()
    def p_sample_loop(
        self,
        unet: nn.Module,
        shape: Tuple[int, ...],
        *,
        use_ema=None,
        seed: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Run the full reverse diffusion loop to sample latents from pure noise.

        Inputs:
          unet:
            Denoiser network that predicts eps (noise) given current z and timestep t.
            Expected signature: unet(z, t, mask=..., control=...)

          shape:
            Desired latent shape: (B, zC, D, H, W)

          use_ema:
            Optional EMA helper whose weights can be temporarily copied into `unet`
            during sampling (common practice for higher-quality samples).

          seed:
            Optional integer seed to make the *sampling trajectory* deterministic.
            Implemented via torch.Generator tied to the sampling device.

          mask:
            Optional conditioning tensor: (B, 1, D, H, W) at latent resolution.

          control:
            Optional conditioning tensor: (B, Cc, D, H, W) at latent resolution.

          guidance_scale:
            Classifier-Free Guidance (CFG) scale.
              - 1.0 (or <=1.0) means no guidance (single forward pass per step)
              - >1.0 means do conditional + unconditional passes and combine

        Output:
          z: final denoised latents (approximate samples from the learned latent distribution)

        Internal mechanics:
          - Initialize z ~ N(0, I)
          - For t = T-1 down to 0:
              1) predict eps with UNet (optionally CFG)
              2) estimate x0 (clean latent) from current z and eps
              3) compute posterior mean of q(z_{t-1} | z_t, x0)
              4) sample z_{t-1} by adding noise scaled by posterior variance (except at t=0)
        """
        # decide device from model / configured DEVICE
        dev = DEVICE

        # If a seed is provided, create a device-specific generator so randn() is reproducible
        if seed is not None:
            g = torch.Generator(device=dev)
            g.manual_seed(int(seed))
        else:
            g = None

        # Start from pure Gaussian noise in latent space
        z = torch.randn(shape, device=dev, generator=g)

        # move conditioning to device/dtype of z (keeps things consistent)
        # non_blocking=True helps when tensors come from pinned host memory
        if mask is not None:
            mask = mask.to(device=z.device, dtype=z.dtype, non_blocking=True)
        if control is not None:
            control = control.to(device=z.device, dtype=z.dtype, non_blocking=True)

        # apply EMA weights temporarily if provided
        backup = None
        if use_ema is not None:
            # NOTE: cloning full state_dict is expensive but safe and simple.
            # This creates a full copy so we can restore the original weights after sampling.
            backup = {k: v.detach().clone() for k, v in unet.state_dict().items()}
            use_ema.copy_to(unet)

        try:
            # Reverse-time loop: T-1 -> 0
            for ti in reversed(range(self.T)):
                # Create a timestep tensor of shape [B] for the current step
                t = torch.full(
                    (shape[0],),
                    ti,
                    device=z.device,
                    dtype=torch.long,
                )

                # ---- Predict eps (noise) with optional CFG
                if guidance_scale is not None and float(guidance_scale) > 1.0:
                    # Conditional prediction: uses provided mask/control
                    eps_cond = unet(z, t, mask=mask, control=control)

                    # Unconditional prediction:
                    # Here "unconditional" is approximated by *zeroing* mask/control.
                    # (This assumes the model learned to treat zero conditioning as "no condition".)
                    mask0 = torch.zeros_like(mask) if mask is not None else None
                    ctrl0 = torch.zeros_like(control) if control is not None else None
                    eps_uncond = unet(z, t, mask=mask0, control=ctrl0)

                    # CFG combination:
                    # eps = eps_uncond + gs * (eps_cond - eps_uncond)
                    # gs>1 pushes samples toward conditional behavior.
                    gs = float(guidance_scale)
                    eps = eps_uncond + gs * (eps_cond - eps_uncond)
                else:
                    # No guidance: single model call per step
                    eps = unet(z, t, mask=mask, control=control)

                # fetch scalar schedule values and put on correct device/dtype
                # These are scalars (per timestep) but we cast them to match z for math stability.
                beta_t = self.betas[ti].to(device=z.device, dtype=z.dtype)
                alpha_t = self.alphas[ti].to(device=z.device, dtype=z.dtype)
                acp_t = self.alphas_cumprod[ti].to(device=z.device, dtype=z.dtype)
                acp_prev = self.alphas_cumprod_prev[ti].to(device=z.device, dtype=z.dtype)

                sqrt_acp_t = torch.sqrt(acp_t)
                sqrt_om_acp_t = torch.sqrt(1.0 - acp_t)

                # predict x0 (clean latent) from current z_t and predicted eps:
                # x0 = (z_t - sqrt(1-acp_t)*eps) / sqrt(acp_t)
                x0_pred = (z - sqrt_om_acp_t * eps) / (sqrt_acp_t + 1e-8)

                # Clamp predicted x0 for stability (prevents exploding values early in training)
                x0_pred = x0_pred.clamp(-3.0, 3.0)

                # posterior mean for q(z_{t-1} | z_t, x0)
                # mean = c1*x0 + c2*z_t
                # where:
                #   c1 = sqrt(acp_prev) * beta_t / (1 - acp_t)
                #   c2 = sqrt(alpha_t) * (1 - acp_prev) / (1 - acp_t)
                c1 = torch.sqrt(acp_prev) * beta_t / (1.0 - acp_t + 1e-8)
                c2 = torch.sqrt(alpha_t) * (1.0 - acp_prev) / (1.0 - acp_t + 1e-8)
                mean = c1 * x0_pred + c2 * z

                if ti > 0:
                    # For t>0, sample z_{t-1} with posterior variance:
                    # z_{t-1} = mean + sqrt(var) * noise
                    var = self.posterior_variance[ti].to(device=z.device, dtype=z.dtype)

                    # Draw fresh noise (optionally from seeded generator)
                    if g is None:
                        noise = torch.randn_like(z)
                    else:
                        noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=g)

                    z = mean + torch.sqrt(var) * noise
                else:
                    # At t=0, output the mean (no additional noise)
                    z = mean

        finally:
            # Always restore original model weights if EMA was applied
            if backup is not None:
                unet.load_state_dict(backup)

        return z
