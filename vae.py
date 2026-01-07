#!/usr/bin/env python3
"""
ldm3d/vae.py

3D VAE for Latent Diffusion (ANATOMY-PRESERVING VERSION)

High-level role in the pipeline:
  1) Encoder compresses a full 3D MRI volume (112^3, 3 channels: T1/T2/FLAIR)
     into a lower-resolution latent representation (28^3, z_channels=8 by default).
  2) Decoder reconstructs the MRI volume from that latent.
  3) Diffusion operates in latent space (28^3) for speed + stability.
     This VAE defines what "latent space" even means.

Key design choices in THIS VAE:
  - Downsampling is x4 total (112 -> 56 -> 28), not x8.
    That preserves more anatomical detail (sulci, tissue boundaries).
  - Capacity is increased via "base" channels (default 64) and refinement blocks at 28^3.
  - Uses InstanceNorm3d (often stable in medical imaging with small batch sizes).

Important shape conventions:
  - Input x:     [B, 3, 112, 112, 112]  (values typically in [-1, 1])
  - Latent mu:   [B, zC, 28, 28, 28]
  - Latent z:    [B, zC, 28, 28, 28]
  - Recon xhat:  [B, 3, 112, 112, 112]  (squashed to [-1, 1] via Tanh)

Notes about naming:
  - This file defines VAEEncoder3D/VAEDecoder3D, and the wrapper VAE3D that exposes
    encode/decode/forward, plus helper losses used by our training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# POSTERIOR
# ============================================================
class DiagonalGaussian(nn.Module):
    """
    Reparameterization for spatial mu/logvar feature maps.

    In a VAE, the encoder does not output a single point in latent space.
    It outputs a distribution q(z|x) = N(mu, diag(sigma^2)).

    This module samples z from that distribution using the reparameterization trick:
      z = mu + sigma * eps
      where eps ~ N(0, I)

    Why this matters:
      - Enables backprop through sampling (eps is the randomness, not the parameters).
      - Encourages a smooth latent space (helpful for diffusion).
    """

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clamp log-variance to keep std in a numerically safe range.
        # Prevents exploding std (huge noise) or collapsing std (near-zero).
        logvar = torch.clamp(logvar, -20.0, 10.0)

        # Convert log-variance to standard deviation:
        #   var = exp(logvar)  =>  std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)

        # Sample eps with same shape as std.
        eps = torch.randn_like(std)

        # Reparameterization trick: sample z from q(z|x).
        return mu + std * eps


# ============================================================
# ENCODER (x4 downsample: 112 -> 28)
# ============================================================
class VAEEncoder3D(nn.Module):
    """
    Encoder network.

    Input:
      x: [B, 3, 112, 112, 112]

    Output:
      mu:     [B, zC, 28, 28, 28]
      logvar: [B, zC, 28, 28, 28]

    Architectural intent:
      - Two strided convs do the only downsampling:
          112 -> 56 -> 28
      - After that, two 3x3x3 conv "refinement" blocks keep the 28^3 resolution,
        increasing representational power without further spatial loss.

    Why keep a spatial latent (28^3) instead of a vector?
      - Spatial latents preserve locality and structure (super important for anatomy).
      - Diffusion in spatial latents behaves more like image diffusion (but 3D).
    """

    def __init__(self, z_channels: int = 8, base: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            # Strided convs reduce spatial size while growing channels.
            # 112 -> 56
            nn.Conv3d(3, base, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.InstanceNorm3d(base, affine=True),

            # 56 -> 28
            nn.Conv3d(base, base * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.InstanceNorm3d(base * 2, affine=True),

            # refinement blocks (NO further downsampling)
            # These keep resolution at 28 while increasing capacity.
            nn.Conv3d(base * 2, base * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.InstanceNorm3d(base * 4, affine=True),

            nn.Conv3d(base * 4, base * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.InstanceNorm3d(base * 4, affine=True),
        )

        # Project to per-voxel mean/log-variance of the latent distribution.
        # 1x1x1 conv = channel mixing at each voxel, no spatial change.
        self.to_mu = nn.Conv3d(base * 4, z_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(base * 4, z_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Shared trunk produces high-level features for both mu/logvar heads.
        h = self.net(x)
        return self.to_mu(h), self.to_logvar(h)


# ============================================================
# DECODER (x4 upsample: 28 -> 112)
# ============================================================
class VAEDecoder3D(nn.Module):
    """
    Decoder network.

    Input:
      z: [B, zC, 28, 28, 28]

    Output:
      xhat: [B, 3, 112, 112, 112]  (tanh -> [-1, 1])

    Architectural intent:
      - "pre" block lifts latent channels into a wider feature space (base*4).
      - Two upsampling stages:
          28 -> 56 -> 112
        implemented as Upsample + Conv3d (often more stable than ConvTranspose3d).
      - Final convs map features back to 3 modalities, and Tanh constrains range.

    Why trilinear upsample + conv?
      - Avoids checkerboard artifacts that can happen with deconvolutions.
      - Plays nicely with medical volumes (smoothness + stability).
    """

    def __init__(self, z_channels: int = 8, base: int = 64):
        super().__init__()

        self.pre = nn.Sequential(
            # Initial conv stack lifts latent channels to decoder width.
            nn.Conv3d(z_channels, base * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.InstanceNorm3d(base * 4, affine=True),

            nn.Conv3d(base * 4, base * 4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.InstanceNorm3d(base * 4, affine=True),
        )

        def up(in_c: int, out_c: int):
            """
            One upsampling stage:
              - spatial x2 via trilinear interpolation
              - conv to learn features at the new resolution
              - nonlinearity + normalization
            """
            return nn.Sequential(
                # Trilinear upsample + conv is stable for 3D volumes.
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.InstanceNorm3d(out_c, affine=True),
            )

        self.up = nn.Sequential(
            # Two upsampling stages recover 112^3 output size.
            up(base * 4, base * 2),  # 28 -> 56
            up(base * 2, base),      # 56 -> 112
        )

        self.out = nn.Sequential(
            # Final convs map to 3 channels and squash to [-1, 1].
            nn.Conv3d(base, base, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(base, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Lift latent to feature space, upsample twice, then project to image channels.
        h = self.pre(z)
        h = self.up(h)
        return self.out(h)


# ============================================================
# FULL VAE
# ============================================================
class VAE3D(nn.Module):
    """
    Wrapper module that exposes the standard VAE interface.

    Components:
      - enc: maps x -> (mu, logvar)
      - posterior: samples z from (mu, logvar)
      - dec: maps z -> xhat

    Downstream expectations in the project:
      - train.py calls vae(x) which returns (xhat, z, mu, logvar)
      - diffusion training often uses mu as "z0" (deterministic) OR samples a noisy z0
        depending on use_posterior_noise.
    """
    def __init__(self, z_channels: int = 8, base: int = 64):
        super().__init__()
        self.enc = VAEEncoder3D(z_channels=z_channels, base=base)
        self.dec = VAEDecoder3D(z_channels=z_channels, base=base)
        self.posterior = DiagonalGaussian()

    def encode(self, x: torch.Tensor):
        # Encode into distribution parameters, then sample a latent z.
        mu, logvar = self.enc(x)
        z = self.posterior(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Decode latent back to 3-channel MRI volume.
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        # Full VAE pass: recon + sampled latent + distribution params.
        z, mu, logvar = self.encode(x)
        xhat = self.decode(z)
        return xhat, z, mu, logvar


# ============================================================
# LOSSES / HELPERS
# ============================================================
def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence term for a diagonal Gaussian posterior.

    Computes:
      KL( N(mu, sigma^2) || N(0,1) )

    Why this exists:
      - Encourages latents to stay near a unit Gaussian prior.
      - Prevents the encoder from putting arbitrary information anywhere in latent space.
      - Makes the latent distribution nicer for diffusion (which usually assumes Gaussian-like priors).

    Note:
      - This implementation averages over batch and all spatial locations/channels.
    """
    # KL divergence between N(mu, sigma) and N(0,1), averaged over batch+spatial.
    logvar = torch.clamp(logvar, -20.0, 10.0)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.mean()


def gradient_loss_3d(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Structure-preserving gradient loss.

    What it penalizes:
      - Differences in local spatial gradients between reconstruction and input.

    Why it helps medical anatomy:
      - MRI anatomy is often about edges and boundaries (sulci, ventricles, tumor margins).
      - Pure pixel losses (L1/L2) can blur.
      - Gradient matching encourages sharper structural fidelity.

    Output:
      - A scalar loss (sum of L1 differences of gradients along x/y/z axes).
    """
    def grad(t):
        # Finite differences along each spatial axis.
        dx = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        dy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        dz = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        return dx, dy, dz

    gx_h, gy_h, gz_h = grad(x_hat)
    gx, gy, gz = grad(x)

    return (
        F.l1_loss(gx_h, gx) +
        F.l1_loss(gy_h, gy) +
        F.l1_loss(gz_h, gz)
    )


@torch.no_grad()
def get_latent_z0(
    vae: VAE3D,
    x: torch.Tensor,
    use_posterior_noise: bool = False,
) -> torch.Tensor:
    """
    Returns the latent "z0" used by diffusion training/sampling.

    Two modes:
      - mu-only (deterministic):
          returns mu
        Pros: stable targets, less noise injected into diffusion training.
      - posterior-sampled (stochastic):
          returns mu + std*eps
        Pros: makes diffusion see the same kind of latent variability as VAE sampling,
              sometimes improves robustness / realism.

    Important detail:
      - Uses vae.enc(x) directly to avoid decoding overhead.
      - Decorated with @torch.no_grad() because diffusion training typically treats VAE as frozen.

    Output:
      z0: [B, zC, 28, 28, 28]
    """
    vae.eval()
    # Use encoder directly to avoid decoding overhead.
    mu, logvar = vae.enc(x)
    if use_posterior_noise:
        # Sample a noisy latent to match training stochasticity.
        logvar = torch.clamp(logvar, -20.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    return mu
