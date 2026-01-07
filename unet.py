#!/usr/bin/env python3
"""
ldm3d/unet.py

UNet backbone for latent diffusion (LATENT-SIZE AGNOSTIC, WORKS FOR 28^3):

Changes vs the 14^3 version:
 Down path is 28 -> 14 -> 7 (instead of 14 -> 7 -> 3)
Up path is 7 -> 14 -> 28
 ControlNet produces residuals at (S, S/2, S/4) = (28, 14, 7)
What this module does (in plain English):
  - Takes a noisy latent volume z_t and a timestep t.
  - Optionally takes a tumor mask and/or multi-channel "control" tensor.
  - Predicts the diffusion noise ε (epsilon) at that timestep so DDPM can denoise.
  - Uses a UNet encoder/decoder with skip connections plus:
      (1) FiLM-style conditioning from the tumor mask
      (2) ControlNet-style residual injections at multiple scales

Key shapes:
  - z:       [B, zC, S, S, S]      (latent channels zC, e.g., 8)
  - mask:    [B, 1,  S, S, S]      (binary/soft mask)
  - control: [B, Cc, S, S, S]      (e.g., 3 channels: mask/edge/dist)
  - output:  [B, zC, S, S, S]      (predicted noise epsilon)

Design intent:
  - Latent-size agnostic: uses pooling/upsample (scale_factor=2) rather than hard-coded
    sizes so it works for S=28 (28->14->7) and can adapt if S changes.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# TIMESTEP EMBEDDING
# ----------------------------
def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings (Transformer-style) for diffusion.

    Inputs:
      timesteps: [B] (typically int64/long), diffusion step indices 0..T-1
      dim: embedding dimension (e.g., 256)

    Output:
      emb: [B, dim]

    Why this exists:
      Diffusion models need to know *which* timestep they are denoising.
      Sin/cos embeddings provide a smooth, multi-frequency encoding of time.

    Implementation details:
      - half dims for cos, half for sin
      - frequencies follow log-spaced scale like in Transformers
      - if dim is odd, pad one extra zero column for consistent shape
    """
    # Classic transformer-style sin/cos embedding for diffusion timesteps.
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# ----------------------------
# RESBLOCK
# ----------------------------
class ResBlock3D(nn.Module):
    """
    A simple 3D residual block with timestep conditioning.

    Structure:
      GN -> SiLU -> Conv3d -> + timestep bias -> GN -> SiLU -> Conv3d -> + skip

    Timestep conditioning:
      - t_emb is projected to out_c and added as a bias after the first conv.
      - This is a FiLM-like trick: the block can behave differently per timestep.

    Notes:
      - GroupNorm is used (more stable than BatchNorm for small batch sizes, common in 3D).
      - Groups are chosen to divide channels cleanly to avoid GroupNorm errors.
    """
    def __init__(self, in_c: int, out_c: int, t_dim: int):
        super().__init__()
        # Choose GroupNorm groups that divide channels cleanly.
        g1 = 8 if in_c % 8 == 0 else 4 if in_c % 4 == 0 else 1
        g2 = 8 if out_c % 8 == 0 else 4 if out_c % 4 == 0 else 1

        self.norm1 = nn.GroupNorm(g1, in_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, 1, 1)

        # Projects timestep embedding into a per-channel bias term (out_c).
        self.tproj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_c))

        self.norm2 = nn.GroupNorm(g2, out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, 1)

        # If in/out channels differ, learn a 1x1 conv skip; otherwise identity.
        self.skip = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          x:     [B, in_c, D, H, W]
          t_emb: [B, t_dim]

        Output:
          h:     [B, out_c, D, H, W]
        """
        # First conv, then add timestep bias (FiLM-like).
        h = self.conv1(F.silu(self.norm1(x)))
        t = self.tproj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        # Second conv and residual skip.
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ----------------------------
# FiLM CONDITIONING
# ----------------------------
class MaskFiLM(nn.Module):
    """
    Very lightweight FiLM-style conditioning using a global scalar derived from mask/control.

    Idea:
      - Reduce mask_like to a single scalar per sample: mean over spatial dims.
      - Feed that scalar through a tiny MLP to produce:
          gamma, beta  (both per feature channel)
      - Modulate features:
          h * (1 + 0.1*gamma) + 0.1*beta

    Why so minimal?
      - 3D models are heavy; full cross-attention is expensive.
      - This adds a cheap global conditioning signal that nudges feature statistics.

    Why scale by 0.1?
      - Keeps modulation small so the network doesn't destabilize early training.
      - Lets the UNet learn baseline denoising first, then gradually use conditioning.
    """
    def __init__(self, feat_c: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_c * 2),
        )

    def forward(self, h: torch.Tensor, mask_like: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          h:         [B, C, D, H, W] feature tensor to modulate
          mask_like: [B, 1 or Cc, D, H, W] mask/control tensor

        Output:
          Modulated feature tensor same shape as h.
        """
        # Reduce mask/control to a global scalar signal per batch item.
        m = mask_like.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)  # [B,1]
        # Predict gamma/beta and apply a small modulation to features.
        gamma_beta = self.net(m)  # [B,2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(-1, h.size(1), 1, 1, 1)
        beta = beta.view(-1, h.size(1), 1, 1, 1)
        return h * (1.0 + 0.1 * gamma) + 0.1 * beta


# ----------------------------
# ZERO MODULE (ControlNet trick)
# ----------------------------
class ZeroConv3d(nn.Module):
    """
    A 1x1x1 conv with weights/bias initialized to zero.

    This is the ControlNet "zero module" trick:
      - At initialization, the residual output is exactly zero,
        so the main UNet behaves as if ControlNet isn't there.
      - During training, the residuals can gradually learn meaningful corrections.

    This greatly improves stability when adding ControlNet conditioning to an existing backbone.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 1)
        # Zero-init makes residuals start at zero (ControlNet stability).
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ----------------------------
# CONTROLNET (multi-scale residuals)
# ----------------------------
class ControlNet3D(nn.Module):
    """
    A small side-network that processes the control tensor and produces additive residuals
    at multiple spatial scales matching UNet feature maps.

    Inputs:
      control: [B, Cc, S, S, S]  (same spatial size as latent z)

    Outputs:
      r0: [B, base,     S,   S,   S  ]  (full res)
      r1: [B, base*2,   S/2, S/2, S/2]  (half res)
      r2: [B, base*4,   S/4, S/4, S/4]  (quarter res)

    For S=28:
      r0: 28^3
      r1: 14^3
      r2:  7^3

    Why multi-scale residuals:
      - High-res residuals help with localization (where the tumor is).
      - Low-res residuals help with global structure and context.
    """

    def __init__(self, control_channels: int, base: int, t_dim: int):
        super().__init__()

        # Shallow encoder to produce multi-scale residuals.
        self.in_conv = nn.Conv3d(control_channels, base, 3, 1, 1)

        self.c0 = ResBlock3D(base, base, t_dim)          # S
        self.pool1 = nn.AvgPool3d(2, 2)                  # S -> S/2
        self.c1 = ResBlock3D(base, base * 2, t_dim)      # S/2

        self.pool2 = nn.AvgPool3d(2, 2)                  # S/2 -> S/4
        self.c2 = ResBlock3D(base * 2, base * 4, t_dim)  # S/4

        # ZeroConv projections ensure residuals start as zeros.
        self.zero0 = ZeroConv3d(base, base)
        self.zero1 = ZeroConv3d(base * 2, base * 2)
        self.zero2 = ZeroConv3d(base * 4, base * 4)

    def forward(self, control: torch.Tensor, t_emb: torch.Tensor):
        """
        Inputs:
          control: [B, Cc, S, S, S]
          t_emb:   [B, t_dim]

        Outputs:
          r0, r1, r2: residual tensors to add to UNet activations at each scale
        """
        # Process control at full resolution, then downsample twice.
        hc0 = self.in_conv(control)
        hc0 = self.c0(hc0, t_emb)               # S

        hc1 = self.pool1(hc0)
        hc1 = self.c1(hc1, t_emb)               # S/2

        hc2 = self.pool2(hc1)
        hc2 = self.c2(hc2, t_emb)               # S/4

        # Project to residuals that are added into the UNet activations.
        r0 = self.zero0(hc0)
        r1 = self.zero1(hc1)
        r2 = self.zero2(hc2)
        return r0, r1, r2


# ----------------------------
# UNET (LATENT SPACE) + CONTROLNET
# ----------------------------
class UNet3DLatentCond(nn.Module):
    """
    Main denoiser network for latent DDPM.

    Inputs:
      - z: noisy latent z_t
      - t: timestep indices
      - mask: single-channel conditioning mask (optional; defaults to zeros)
      - control: multi-channel conditioning tensor (optional; defaults to mask)

    Conditioning mechanisms:
      (A) Mask concatenation:
          The mask is concatenated with z at the input as an extra channel.
      (B) MaskFiLM:
          Applies a small per-channel affine modulation using a global mask summary.
      (C) ControlNet residuals:
          Side network produces residual feature maps at each UNet scale and adds them in.

    Output:
      - Predicted noise epsilon in latent space: [B, z_channels, S, S, S]
    """
    def __init__(
        self,
        z_channels: int = 8,
        cond_channels: int = 1,
        control_channels: int = 1,
        base: int = 64,
        t_dim: int = 256,
        use_controlnet: bool = True,
    ):
        super().__init__()
        self.t_dim = t_dim
        self.use_controlnet = use_controlnet

        # Input includes latent z and a single-channel conditioning mask.
        # z_channels + cond_channels is the channel count for the first conv.
        self.in_conv = nn.Conv3d(z_channels + cond_channels, base, 3, 1, 1)

        # Expand/reproject timestep embedding before feeding into ResBlocks.
        # This gives the network a richer nonlinear mapping of time -> conditioning.
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # FiLM blocks at two resolutions (full and half) to inject mask signal cheaply.
        self.film0 = MaskFiLM(base)
        self.film1 = MaskFiLM(base * 2)

        if self.use_controlnet:
            # ControlNet produces residuals aligned with UNet scales.
            self.controlnet = ControlNet3D(control_channels=control_channels, base=base, t_dim=t_dim)
        else:
            self.controlnet = None

        # Down path: S -> S/2 -> S/4
        self.down0 = ResBlock3D(base, base, t_dim)            # S
        self.pool1 = nn.AvgPool3d(2, 2)                       # S -> S/2

        self.down1 = ResBlock3D(base, base * 2, t_dim)        # S/2
        self.pool2 = nn.AvgPool3d(2, 2)                       # S/2 -> S/4

        self.down2 = ResBlock3D(base * 2, base * 4, t_dim)    # S/4

        # Mid / bottleneck at lowest resolution: extra capacity where receptive field is global.
        self.mid1 = ResBlock3D(base * 4, base * 4, t_dim)
        self.mid2 = ResBlock3D(base * 4, base * 4, t_dim)

        # Up path: S/4 -> S/2 -> S
        # Each up stage:
        #   - upsample
        #   - conv to reduce channels
        #   - concat skip connection
        #   - ResBlock to fuse
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(base * 4, base * 2, 3, 1, 1),
        )
        self.up1_res = ResBlock3D(base * 4, base * 2, t_dim)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(base * 2, base, 3, 1, 1),
        )
        self.up2_res = ResBlock3D(base * 2, base, t_dim)

        # Final projection to z_channels (predict epsilon).
        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv3d(base, z_channels, 3, 1, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict noise epsilon.

        Inputs:
          z:      [B, zC, S, S, S]
          t:      [B] diffusion step indices
          mask:   [B, 1,  S, S, S] (optional)
          control:[B, Cc, S, S, S] (optional)

        Output:
          eps:    [B, zC, S, S, S]
        """
        B = z.size(0)

        # Ensure mask always exists so input channel count stays consistent.
        if mask is None:
            # Default to a zero mask so conditioning channel always exists.
            mask = torch.zeros((B, 1, *z.shape[2:]), device=z.device, dtype=z.dtype)
        else:
            mask = mask.to(device=z.device, dtype=z.dtype, non_blocking=True)

        # Embed timestep then project to the UNet conditioning space.
        # sinusoidal embedding -> time_mlp -> t_emb used by all ResBlocks
        t_emb = sinusoidal_timestep_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)

        # ControlNet residuals are optional (only if enabled).
        r0 = r1 = r2 = None
        if self.use_controlnet:
            # If no explicit control given, fall back to using mask as control.
            if control is None:
                control = mask
            control = control.to(device=z.device, dtype=z.dtype, non_blocking=True)

            # Ensure control spatial size matches latent z.
            # This is the "latent-size agnostic" protection.
            if control.shape[2:] != z.shape[2:]:
                control = F.interpolate(control, size=z.shape[2:], mode="nearest")

            # Produce multi-scale residuals aligned with UNet resolutions.
            r0, r1, r2 = self.controlnet(control, t_emb)

        # Concatenate latent and mask as UNet input.
        # This is a direct, local conditioning signal (per-voxel).
        zcat = torch.cat([z, mask], dim=1)

        # ----------------------------
        # Down path (encoder)
        # ----------------------------
        h0 = self.in_conv(zcat)

        # Apply FiLM at full resolution using the mask.
        h0 = self.film0(h0, mask)

        if r0 is not None:
            # Add ControlNet residual at full resolution.
            h0 = h0 + r0

        # First residual block at full resolution.
        h0 = self.down0(h0, t_emb)

        # Downsample to half resolution.
        h1 = self.pool1(h0)
        h1 = self.down1(h1, t_emb)

        # Recompute mask at this scale for FiLM conditioning.
        # (Mask must match feature spatial size.)
        mask_h1 = F.interpolate(mask, size=h1.shape[2:], mode="nearest")
        h1 = self.film1(h1, mask_h1)

        if r1 is not None:
            # Safety: ensure ControlNet residual has matching spatial size.
            if r1.shape[2:] != h1.shape[2:]:
                r1 = F.interpolate(r1, size=h1.shape[2:], mode="nearest")
            h1 = h1 + r1

        # Downsample again to quarter resolution.
        h2 = self.pool2(h1)
        h2 = self.down2(h2, t_emb)

        if r2 is not None:
            if r2.shape[2:] != h2.shape[2:]:
                r2 = F.interpolate(r2, size=h2.shape[2:], mode="nearest")
            h2 = h2 + r2

        # ----------------------------
        # Bottleneck (lowest res)
        # ----------------------------
        # Two ResBlocks at lowest scale: large receptive field, global context.
        hm = self.mid1(h2, t_emb)
        hm = self.mid2(hm, t_emb)

        # ----------------------------
        # Up path (decoder) + skip connections
        # ----------------------------
        # Upsample to half-res and fuse with skip h1.
        hu1 = self.up1(hm)                 # -> S/2
        hu1 = torch.cat([hu1, h1], dim=1)  # skip connection
        hu1 = self.up1_res(hu1, t_emb)

        # Upsample to full-res and fuse with skip h0.
        hu2 = self.up2(hu1)                # -> S
        hu2 = torch.cat([hu2, h0], dim=1)  # skip connection
        hu2 = self.up2_res(hu2, t_emb)

        # Final projection: predict noise residual in latent space.
        return self.out(hu2)
