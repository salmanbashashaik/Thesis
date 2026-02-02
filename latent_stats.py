# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
ldm3d/latent_stats.py

Latent-space statistics utilities:
- estimate_latent_stats
- normalize / denormalize (robust reshape + device/dtype safe)
- load cached stats if available
- recompute_and_save_latent_stats (one-shot convenience)

Why this file exists:
  Latent diffusion is dramatically easier to train if latents are normalized.
  We compute per-channel mean/std for the VAE latent tensor z and use them to:
    - normalize:   z_norm = (z - mean) / std
    - denormalize: z      = z_norm * std + mean

The BIG gotcha:
  Latent stats must match the *exact* VAE weights used to produce z.
  If we want change the VAE (train more, load a different checkpoint, etc.), recompute stats.
"""

# Import dependencies used by this module.
from __future__ import annotations
from typing import Dict, Optional

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ldm3d.config import DEVICE
from ldm3d.vae import VAE3D
from ldm3d.data import VolFolder


# ----------------------------
# ESTIMATION
# ----------------------------
@torch.no_grad()
# Function: `estimate_latent_stats` implements a reusable processing step.
def estimate_latent_stats(
    dl,
    vae: VAE3D,
    max_batches: int = 400,
    use_posterior_noise: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-channel mean/std of the VAE latent tensor z over a dataset.

    Inputs:
      dl:
        A DataLoader that yields (x, mask, sid) tuples (or compatible).
        Only x is used here.
      vae:
        The VAE model whose encoder defines the latent distribution.
      max_batches:
        Stop after this many batches to keep computation bounded.
        (Useful when the dataset is large.)
      use_posterior_noise:
        Controls whether z is taken as:
          - mu only               (deterministic latents)
          - mu + sigma*eps        (stochastic posterior sampling)
        This should match how we plan to produce latents during diffusion training.
        If diffusion trains on noisy posterior samples, stats should be computed on those.

    Output:
      dict with:
        - "mean": [zC] tensor (CPU)
        - "std" : [zC] tensor (CPU)

    Implementation notes:
      - We accumulate sum and sum of squares per channel over all voxels in the sampled batches.
      - We compute variance via E[x^2] - (E[x])^2.
      - We clamp variance to avoid sqrt of tiny/negative values from numeric round-off.
      - Returned tensors are moved to CPU for checkpoint portability.
    """
    # Make sure encoder runs in eval mode (disables dropout, etc.)
    vae.eval()

    # Running accumulators (per latent channel)
    sum_c = None
    sumsq_c = None

    # Total number of voxels accumulated per channel (same for each channel)
    n = 0

    # Control-flow branch for conditional or iterative execution.
    for i, (x, _, _) in enumerate(dl):
        # Control-flow branch for conditional or iterative execution.
        if i >= max_batches:
            break

        # Push input to the configured device so encoder runs on GPU if available
        x = x.to(DEVICE)

        # Encoder produces parameters for q(z|x): mean (mu) and log-variance (logvar)
        mu, logvar = vae.enc(x)

        # Choose latent definition depending on training design
        # Control-flow branch for conditional or iterative execution.
        if use_posterior_noise:
            # Clamp logvar to prevent extreme std values from destabilizing stats
            logvar = torch.clamp(logvar, -20.0, 10.0)
            std = torch.exp(0.5 * logvar)

            # Sample z from posterior: z = mu + std * eps
            z = mu + std * torch.randn_like(std)
        else:
            # Deterministic latents: use the mean only
            z = mu

        # Flatten all voxels into one axis while keeping channel axis:
        # z: [B,zC,D,H,W] -> zc: [zC, N]
        zc = z.permute(1, 0, 2, 3, 4).contiguous().view(z.shape[1], -1)

        # Initialize accumulators on first batch, then accumulate sums
        # Control-flow branch for conditional or iterative execution.
        if sum_c is None:
            sum_c = zc.sum(dim=1)              # [zC]
            sumsq_c = (zc ** 2).sum(dim=1)     # [zC]
        else:
            sum_c += zc.sum(dim=1)
            sumsq_c += (zc ** 2).sum(dim=1)

        # N = number of voxels aggregated per channel in this batch
        n += zc.shape[1]

    # Compute mean and std per channel
    mean = sum_c / max(1, n)

    # Var = E[x^2] - (E[x])^2
    var = (sumsq_c / max(1, n)) - mean ** 2

    # Clamp to avoid negative variance due to numerical errors
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)

    # Print for debugging: helps catch pathological stats (std ~ 0 or huge)
    print("[LDM] latent stats:")
    print("      mean:", mean.detach().cpu().numpy())
    print("      std :", std.detach().cpu().numpy())

    # IMPORTANT: store as [zC] on CPU for portability across devices
    # Return the computed value to the caller.
    return {
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "use_posterior_noise": bool(use_posterior_noise),
    }



# ----------------------------
# RECOMPUTE + SAVE (CONVENIENCE)
# ----------------------------
@torch.no_grad()
# Function: `recompute_and_save_latent_stats` implements a reusable processing step.
def recompute_and_save_latent_stats(
    outdir: str,
    gbm_root: str,
    vae: VAE3D,
    batches: int = 400,
    batch_size: int = 1,
    num_workers: int = 2,
    use_posterior_noise: bool = False,
    stats_filename: str = "latent_stats.pt",
    pin_memory: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper that recomputes latent stats from scratch and writes them to disk.

    What it does:
      1) Builds a VolFolder dataset from gbm_root (source domain).
      2) Wraps it in a DataLoader with the provided loader settings.
      3) Calls estimate_latent_stats(...) using the provided VAE.
      4) Saves the resulting dict to outdir/stats_filename.
      5) Returns the stats dict (on CPU).

    Why this helper is useful:
      - Lets us recompute stats without having to manually construct a dataloader
        at the call site.
      - Keeps "how to compute stats" centralized and consistent.

    IMPORTANT:
      Call this only AFTER the VAE weights are finalized (loaded/trained),
      otherwise the diffusion normalization will be mismatched.
    """
    # Ensure output directory exists so saving does not fail
    os.makedirs(outdir, exist_ok=True)

    # Build dataset/dataloader from root
    ds = VolFolder(gbm_root)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,  # shuffle gives a representative sample earlier
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # Estimate stats on the constructed loader
    stats = estimate_latent_stats(
        dl,
        vae,
        max_batches=batches,
        use_posterior_noise=use_posterior_noise,
    )

    # Save to disk for reuse across runs
    save_path = Path(outdir) / stats_filename
    torch.save(stats, save_path)
    print(f"[SAVE] Saved latent stats -> {save_path}")

    # Return the computed value to the caller.
    return stats


# ----------------------------
# SHAPE/DEVICE SAFE HELPERS
# ----------------------------
# Function: `_as_chan_vec` implements a reusable processing step.
def _as_chan_vec(v: torch.Tensor, zc: int) -> torch.Tensor:
    """
    Convert an input tensor `v` into a per-channel vector of shape [zC].

    This exists because latent stats may be stored/loaded in different shapes depending on:
      - how they were computed
      - how they were saved
      - whether someone accidentally saved a broadcastable shape (e.g., [1,zC,1,1,1])

    Accepted shapes include:
      - [zC]
      - [1, zC, 1, 1, 1]
      - [zC, 1, 1, 1]
      - [B, D, H, W, zC] (channel-last)
      - [B, zC, D, H, W] (channel-first)
      - Any accidental spatial stats (we reduce them to per-channel by mean)

    If `v` cannot be coerced into [zC], a RuntimeError is raised.
    """
    # Control-flow branch for conditional or iterative execution.
    if not torch.is_tensor(v):
        raise TypeError("latent stats must be torch.Tensors")

    # Already correct
    # Control-flow branch for conditional or iterative execution.
    if v.ndim == 1 and v.numel() == zc:
        # Return the computed value to the caller.
        return v

    # Control-flow branch for conditional or iterative execution.
    if v.ndim == 5:
        # channel-first: [B?, zC, D, H, W]
        # Control-flow branch for conditional or iterative execution.
        if v.shape[1] == zc:
            # If it's already broadcast-shape like [1,zC,1,1,1], collapse cleanly
            # Control-flow branch for conditional or iterative execution.
            if v.shape[2:] == (1, 1, 1):
                # Return the computed value to the caller.
                return v[0, :, 0, 0, 0]
            # Otherwise reduce over batch+spatial dims to get [zC]
            # Return the computed value to the caller.
            return v.mean(dim=(0, 2, 3, 4))  # -> [zC]

        # channel-last: [B, D, H, W, zC]
        # Control-flow branch for conditional or iterative execution.
        if v.shape[-1] == zc:
            # Broadcast-shape like [1,1,1,1,zC]
            # Control-flow branch for conditional or iterative execution.
            if v.shape[0] == 1 and v.shape[1:4] == (1, 1, 1):
                # Return the computed value to the caller.
                return v[0, 0, 0, 0, :]
            # Reduce over batch+spatial dims
            # Return the computed value to the caller.
            return v.mean(dim=(0, 1, 2, 3))

    # Control-flow branch for conditional or iterative execution.
    if v.ndim == 4 and v.shape[0] == zc:
        # channel-first without batch: [zC, D, H, W]
        # Control-flow branch for conditional or iterative execution.
        if v.shape[1:] == (1, 1, 1):
            # Return the computed value to the caller.
            return v[:, 0, 0, 0]
        # Return the computed value to the caller.
        return v.mean(dim=(1, 2, 3))

    # last resort: flatten and see if it matches exactly zC
    flat = v.reshape(-1)
    # Control-flow branch for conditional or iterative execution.
    if flat.numel() == zc:
        # Return the computed value to the caller.
        return flat

    raise RuntimeError(f"latent stat has incompatible shape {tuple(v.shape)} for zC={zc}")


# Function: `_reshape_lat_stats` implements a reusable processing step.
def _reshape_lat_stats(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Reshape a latent-stat vector into a broadcastable tensor matching z.

    Input:
      v: latent stat tensor in any accepted shape (see _as_chan_vec)
      z: latent tensor [B, zC, D, H, W]

    Output:
      v_reshaped: [1, zC, 1, 1, 1] on z.device with z.dtype

    Why:
      - Normalization needs per-channel mean/std applied across all spatial locations.
      - This broadcast shape works with (z - mean) / std cleanly.
    """
    zc = z.shape[1]
    v = _as_chan_vec(v, zc).to(device=z.device, dtype=z.dtype)
    # Return the computed value to the caller.
    return v.view(1, zc, 1, 1, 1)


# Function: `normalize_latents` implements a reusable processing step.
def normalize_latents(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize latent tensor z using per-channel mean/std.

    z_norm = (z - mean) / (std + eps)

    Notes:
      - eps prevents division-by-zero (or huge scaling if std is extremely tiny).
      - mean/std are reshaped to broadcast over [B,zC,D,H,W].
    """
    m = _reshape_lat_stats(mean, z)
    s = _reshape_lat_stats(std, z)
    # Return the computed value to the caller.
    return (z - m) / (s + 1e-6)


# Function: `denormalize_latents` implements a reusable processing step.
def denormalize_latents(z_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Invert normalization to recover original latent scale.

    z = z_norm * (std + eps) + mean
    """
    m = _reshape_lat_stats(mean, z_norm)
    s = _reshape_lat_stats(std, z_norm)
    # Return the computed value to the caller.
    return z_norm * (s + 1e-6) + m


# ----------------------------
# LOAD CACHED
# ----------------------------
# Function: `maybe_load_latent_stats` implements a reusable processing step.
def maybe_load_latent_stats(outdir: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Try to load cached latent stats from {outdir}/latent_stats.pt.

    Returns:
      - dict(mean=..., std=...) if successful and well-formed
      - None otherwise

    Defensive behavior:
      - Loads on CPU for portability.
      - Ensures mean/std are tensors even if checkpoint stored them as lists/arrays.
      - Silently returns None if file is missing/corrupt or structure is unexpected.

    This is designed to make training scripts concise:
      stats = maybe_load_latent_stats(outdir)
      if stats is None: stats = estimate_latent_stats(...); save(...)
    """
    stats_path = f"{outdir}/latent_stats.pt"
    # Control-flow branch for conditional or iterative execution.
    try:
        d = torch.load(stats_path, map_location="cpu")
        # Control-flow branch for conditional or iterative execution.
        if isinstance(d, dict) and "mean" in d and "std" in d:
            # Ensure tensors
            # Control-flow branch for conditional or iterative execution.
            if not torch.is_tensor(d["mean"]):
                d["mean"] = torch.tensor(d["mean"])
            # Control-flow branch for conditional or iterative execution.
            if not torch.is_tensor(d["std"]):
                d["std"] = torch.tensor(d["std"])
            # Control-flow branch for conditional or iterative execution.
            if "use_posterior_noise" not in d:
                d["use_posterior_noise"] = None
            print(f"[LOAD] Loaded latent stats: {stats_path}")
            # Return the computed value to the caller.
            return d
    # Control-flow branch for conditional or iterative execution.
    except Exception:
        # Intentionally quiet: caller decides whether to recompute
        pass
    # Return the computed value to the caller.
    return None
