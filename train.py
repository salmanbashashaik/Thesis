#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
ldm3d/train.py

Training + sampling loops:
- VAE training
- Latent Diffusion training (with ControlNet conditioning)
- Snapshot sampling
- Final sampling dump

IMPROVEMENTS INCLUDED (for better tumor realism -> higher CNN accuracy on synth):
Tumor-weighted diffusion loss in latent space (downsampled + optional dilation)
   - Reduces tumor wash-out, reduces FN slices in synth eval

High-level pipeline (what this file orchestrates):
  1) Train VAE (src then optionally tgt) to get a good latent space for 3D MRI volumes.
  2) Freeze the VAE; train a latent DDPM (UNet denoiser) to predict noise in latent space.
  3) Condition the denoiser using tumor masks (+ derived control channels: edge + soft-dist).
  4) Periodically snapshot reconstructions/samples for qualitative inspection.
  5) At the end, dump a fixed number of final samples for evaluation.

Conventions:
  - x is MRI volume tensor: [B, 3, D, H, W] (channels: T1,T2,FLAIR)
  - mask is binary tumor mask: [B, 1, D, H, W] values {0,1}
  - latents z are: [B, zC, d, h, w] where d=h=w=args.latent_size (e.g., 28)

Important conceptual distinction used throughout:
  - mask_gt: "ground truth" mask used ONLY for loss weighting (never augmented, never dropped)
  - mask_cond: conditioning mask (CAN be augmented, blurred, or dropped for CFG-style robustness)
"""

# Import dependencies used by this module.
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ldm3d.config import DEVICE
from ldm3d.io_nifti import save_nifti
from ldm3d.data import augment_mask, blur3d
from ldm3d.vae import kl_loss, get_latent_z0
from ldm3d.latent_stats import normalize_latents
import hashlib
import numpy as np
import torch

# ============================================================
# SMALL HELPERS
# ============================================================
# Function: `edge_aware_loss_3d` implements a reusable processing step.
def edge_aware_loss_3d(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Edge-aware loss using Sobel operators for tumor boundary preservation.
    
    This is critical for medical imaging where tumor boundaries matter
    for downstream segmentation tasks.
    """
    # Function: `sobel_3d` implements a reusable processing step.
    def sobel_3d(t):
        # Simplified 3D edge detection using differences
        dx = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        dy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        dz = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        # Return the computed value to the caller.
        return dx, dy, dz
    
    dx_h, dy_h, dz_h = sobel_3d(x_hat)
    dx, dy, dz = sobel_3d(x)
    
    edge_loss = (
        F.l1_loss(dx_h, dx) +
        F.l1_loss(dy_h, dy) +
        F.l1_loss(dz_h, dz)
    )
    
    # Return the computed value to the caller.
    return edge_loss

# Function: `high_frequency_loss_3d` implements a reusable processing step.
def high_frequency_loss_3d(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    High-frequency compensation loss using Laplacian pyramid.
    Preserves fine texture details often lost in latent compression.
    """
    # Compute high-frequency component via Laplacian
    # Function: `get_hf` implements a reusable processing step.
    def get_hf(img):
        # Simple approximation: difference from Gaussian blur
        blurred = F.avg_pool3d(
            F.pad(img, (1, 1, 1, 1, 1, 1), mode='replicate'),
            kernel_size=3,
            stride=1,
        )
        # Return the computed value to the caller.
        return img - blurred
    
    hf_hat = get_hf(x_hat)
    hf_gt = get_hf(x)
    
    # Return the computed value to the caller.
    return F.l1_loss(hf_hat, hf_gt)

# Function: `mask_hash_latent` implements a reusable processing step.
def mask_hash_latent(maskL: torch.Tensor) -> str:
    """
    maskL: [1,1,L,L,L] float {0,1} (or close) on GPU/CPU
    Returns a collision-resistant hash string.
    """
    # Import dependencies used by this module.
    import hashlib

    m = maskL.detach()
    m = (m > 0.5).to(torch.uint8)
    b = m.cpu().numpy().tobytes()
    # Return the computed value to the caller.
    return hashlib.sha1(b).hexdigest()  # plenty strong for this


# Function: `dilate_mask_3d` implements a reusable processing step.
def dilate_mask_3d(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Binary dilation in 3D using max-pooling.

    Inputs:
      mask: [B,1,D,H,W] float tensor in {0,1}
      k: kernel size for dilation (odd recommended: 3/5/7). If <=1, no-op.

    Output:
      Dilated mask in {0,1} with the same shape as input.

    Why max-pooling works:
      - For binary masks, max_pool3d computes whether any voxel in the neighborhood is 1.
      - This is equivalent to morphological dilation with a cubic structuring element.

    Why we want dilation here:
      - Tumor regions are small relative to the full volume.
      - Loss weighting strictly inside the tumor can under-emphasize borders and nearby context.
      - Dilation slightly expands the weighted region so diffusion learns to preserve tumor context.
    """
    # Use max-pooling as a cheap binary dilation.
    # Control-flow branch for conditional or iterative execution.
    if k <= 1:
        # Return the computed value to the caller.
        return (mask > 0.5).float()
    # Control-flow branch for conditional or iterative execution.
    if k % 2 == 0:
        k = k + 1  # force odd
    pad = k // 2
    # Return the computed value to the caller.
    return (F.max_pool3d(mask, kernel_size=k, stride=1, padding=pad) > 0.5).float()


# Function: `_prep_latent_stats_for_z` implements a reusable processing step.
def _prep_latent_stats_for_z(
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make mean/std broadcast cleanly over z: [B, zC, D, H, W].

    Problem this solves:
      - latent stats sometimes get saved with weird shapes (channel-last, spatial stats, etc.)
      - if mean/std don't broadcast correctly, we get errors like:
            "The size of tensor a (14) must match the size of tensor b (8) ..."
      - or, worse, silent broadcasting to the wrong dimensions.

    Strategy:
      1) Validate mean/std are tensors.
      2) Move them to z.device and z.dtype to avoid device/dtype mismatch bugs.
      3) Convert each to a clean per-channel vector [zC] via `to_chan_vec`.
         This function accepts many shapes and reduces spatial dims if needed.
      4) Reshape to broadcastable form: [1, zC, 1, 1, 1].

    Returns:
      mean_b, std_b: both shaped [1, zC, 1, 1, 1] on same device/dtype as z.
    """
    # Control-flow branch for conditional or iterative execution.
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise TypeError("lat_mean/lat_std must be torch.Tensors")

    zc = z.shape[1]

    # Move stats to z device/dtype for safe arithmetic.
    mean = mean.to(device=z.device, dtype=z.dtype)
    std = std.to(device=z.device, dtype=z.dtype)

    # Function: `to_chan_vec` implements a reusable processing step.
    def to_chan_vec(x: torch.Tensor) -> torch.Tensor:
        """
        Coerce x into a per-channel vector [zC].

        Supported patterns:
          - [zC]
          - 5D channel-first: [B?, zC, D, H, W]  (reduce over batch/spatial if needed)
          - 5D channel-last : [B?, D, H, W, zC]  (reduce over batch/spatial if needed)
          - 4D channel-first: [zC, D, H, W]      (reduce over spatial if needed)
          - Any tensor with exactly zC elements (flatten fallback)

        If none match, raise an error because stats are incompatible with the model's z_channels.
        """
        # Accept [zC]
        # Control-flow branch for conditional or iterative execution.
        if x.ndim == 1:
            # Control-flow branch for conditional or iterative execution.
            if x.numel() != zc:
                raise RuntimeError(f"latent stat vector numel={x.numel()} != zC={zc}")
            # Return the computed value to the caller.
            return x

        # Accept 5D tensors, channel-first or channel-last, optionally with spatial dims
        # Control-flow branch for conditional or iterative execution.
        if x.ndim == 5:
            # channel-first: [B?, zC, D, H, W]
            # Control-flow branch for conditional or iterative execution.
            if x.shape[1] == zc:
                # Control-flow branch for conditional or iterative execution.
                if x.shape[2:] != (1, 1, 1):
                    # Reduce spatial dims to get a single vector per channel.
                    x = x.mean(dim=(0, 2, 3, 4))  # -> [zC]
                else:
                    x = x[0, :, 0, 0, 0]
                # Return the computed value to the caller.
                return x

            # channel-last: [B?, D, H, W, zC]
            # Control-flow branch for conditional or iterative execution.
            if x.shape[-1] == zc:
                # Control-flow branch for conditional or iterative execution.
                if x.shape[:-1] != (1, 1, 1, 1):
                    # Reduce spatial dims to get a single vector per channel.
                    x = x.mean(dim=(0, 1, 2, 3))  # -> [zC]
                else:
                    x = x[0, 0, 0, 0, :]
                # Return the computed value to the caller.
                return x

        # Accept 4D channel-first with spatial dims: [zC, D, H, W]
        # Control-flow branch for conditional or iterative execution.
        if x.ndim == 4 and x.shape[0] == zc:
            # Control-flow branch for conditional or iterative execution.
            if x.shape[1:] != (1, 1, 1):
                # Reduce spatial dims to get a single vector per channel.
                x = x.mean(dim=(1, 2, 3))  # -> [zC]
            else:
                x = x[:, 0, 0, 0]
            # Return the computed value to the caller.
            return x

        # Last resort: flatten if exactly zC elements
        x_flat = x.view(-1)
        # Control-flow branch for conditional or iterative execution.
        if x_flat.numel() == zc:
            # Return the computed value to the caller.
            return x_flat

        raise RuntimeError(
            f"latent stat has incompatible shape {tuple(x.shape)} for zC={zc}"
        )

    # Reduce/convert mean/std into clean channel vectors
    mean_c = to_chan_vec(mean)  # [zC]
    std_c = to_chan_vec(std)    # [zC]

    # Reshape for broadcasting over [B,zC,D,H,W]
    mean_b = mean_c.view(1, zc, 1, 1, 1)
    std_b = std_c.view(1, zc, 1, 1, 1)
    # Return the computed value to the caller.
    return mean_b, std_b


# Function: `denormalize_latents_safe` implements a reusable processing step.
def denormalize_latents_safe(
    z_norm: torch.Tensor,
    lat_mean: torch.Tensor,
    lat_std: torch.Tensor,
) -> torch.Tensor:
    """
    Invert latent normalization safely:
        z = z_norm * std + mean

    Uses _prep_latent_stats_for_z to guarantee mean/std broadcast properly and match device/dtype.
    """
    # Normalize-safe inverse: z = z_norm * std + mean with broadcasted stats.
    m, s = _prep_latent_stats_for_z(z_norm, lat_mean, lat_std)
    # Return the computed value to the caller.
    return z_norm * (s + 1e-6) + m


# ============================================================
# CONTROL HELPERS
# ============================================================

# Function: `mask_to_edge_3d` implements a reusable processing step.
def mask_to_edge_3d(mask: torch.Tensor) -> torch.Tensor:
    """
    Create a thin boundary/edge map from a binary mask.

    Input:
      mask: [B,1,D,H,W] in {0,1}

    Output:
      edge: [B,1,D,H,W] in [0,1], where 1s highlight mask boundaries.

    How it works:
      - Compute an "eroded" version of the mask.
      - Subtract eroded mask from original mask to keep only boundary voxels.

    Why this is useful for ControlNet:
      - Edges provide crisp localization cues and can help the model preserve structure.
      - Especially helpful if raw mask is blurred or partially dropped for conditioning robustness.
    """
    # Erode via max-pool on inverse, then subtract to get a thin edge band.
    inv = 1.0 - mask
    eroded = 1.0 - F.max_pool3d(inv, kernel_size=3, stride=1, padding=1)
    edge = (mask - eroded).clamp(0.0, 1.0)
    # Return the computed value to the caller.
    return edge


# Function: `mask_to_soft_dist` implements a reusable processing step.
def mask_to_soft_dist(mask: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """
    Create a cheap smooth "distance-like" transform from a binary mask.

    Input:
      mask: [B,1,D,H,W] in {0,1}
      steps: number of smoothing iterations

    Output:
      dist: [B,1,D,H,W] in [0,1]
        Values are higher farther away from the mask (approximate inverse distance proxy).

    Method:
      - Start from x = (1 - mask): background=1, mask=0
      - Repeatedly apply avg_pool3d to blur/smooth.
      - Normalize per-sample to [0,1].

    Why not a true Euclidean distance transform?
      - True EDT is more expensive / not built into core torch for 3D.
      - This approximation is cheap, differentiable, and often good enough as a control signal.
    """
    # Repeated avg-pooling creates a smooth inverse-distance proxy.
    x = (1.0 - mask).float()
    # Control-flow branch for conditional or iterative execution.
    for _ in range(steps):
        x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)

    # Normalize to [0,1] per sample for stability across different mask sizes
    x = x - x.amin(dim=(2, 3, 4), keepdim=True)
    x = x / (x.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
    # Return the computed value to the caller.
    return x


@torch.no_grad()
# Function: `_vae_snapshot` implements a reusable processing step.
def _vae_snapshot(stage_name: str, ep: int, dl, vae, outdir: str):
    """
    Save a VAE reconstruction snapshot (real vs recon) so we can visually verify:
      - anatomy is preserved
      - intensities look sane (not all black/white)
      - reconstruction is improving over epochs

    What gets saved:
      - For the first sample in the first batch:
          * t1_real / t2_real / flair_real
          * t1_recon / t2_recon / flair_recon
          * per-modality absolute error volumes (abs(recon - real))

    Where:
      outdir/vae_{stage}_e{ep}_snap/{SubjectID}/...

    Also:
      - Saves a VAE checkpoint at the same epoch for later reuse/debugging.
    """
    vae.eval()

    snap_root = Path(outdir) / f"vae_{stage_name}_e{ep}_snap"
    snap_root.mkdir(parents=True, exist_ok=True)

    # take 1 batch
    # Control-flow branch for conditional or iterative execution.
    for x, _, sids in dl:
        x = x.to(DEVICE)
        sid = str(sids[0]) if len(sids) > 0 else f"sample_ep{ep}"

        # Forward through VAE to get reconstruction
        xhat, _, _, _ = vae(x)

        # Write each modality separately for quick inspection in viewers.
        # NOTE: save_nifti expects [D,H,W]
        sdir = snap_root / sid
        sdir.mkdir(parents=True, exist_ok=True)

        # Input
        save_nifti(x[0, 0], sdir / "t1_real.nii.gz", verbose=False)
        save_nifti(x[0, 1], sdir / "t2_real.nii.gz", verbose=False)
        save_nifti(x[0, 2], sdir / "flair_real.nii.gz", verbose=False)

        # Recon
        save_nifti(xhat[0, 0], sdir / "t1_recon.nii.gz", verbose=False)
        save_nifti(xhat[0, 1], sdir / "t2_recon.nii.gz", verbose=False)
        save_nifti(xhat[0, 2], sdir / "flair_recon.nii.gz", verbose=False)

        # Optional: absolute error (useful to see where detail is lost)
        err = (xhat - x).abs()
        save_nifti(err[0, 0], sdir / "t1_abs_err.nii.gz", verbose=False)
        save_nifti(err[0, 1], sdir / "t2_abs_err.nii.gz", verbose=False)
        save_nifti(err[0, 2], sdir / "flair_abs_err.nii.gz", verbose=False)

        break

    # save checkpoint too (lets yusou resume or reuse this exact epoch later)
    torch.save(vae.state_dict(), Path(outdir) / f"vae_{stage_name}_ep{ep}.pt")

    print(f"[VAE SNAPSHOT] Saved recon snapshot -> {snap_root}")
    vae.train()

# Function: `train_vae_stage` implements a reusable processing step.
def train_vae_stage(
    stage: str,
    dl,
    vae,
    opt,
    epochs: int,
    outdir: str,
    *,
    kl_w: float = 0.01,  # REDUCED from 1e-4 to 0.01 for medical imaging
    rec_w: float = 1.0,
    edge_w: float = 0.1,  # NEW: edge loss weight
    hf_w: float = 0.05,   # NEW: high-frequency loss weight
    kl_warmup_frac: float = 0.1,
    grad_clip: float = 1.0,
):
    """
    Improved VAE training with:
    - Edge-aware loss for boundary preservation
    - High-frequency loss for texture
    - Proper gradient clipping
    - KL warmup
    """
    print(f"[VAE {stage}] Training for {epochs} epochs")
    print(f"  kl_w={kl_w:.2e}, rec_w={rec_w:.2e}, edge_w={edge_w:.2e}, hf_w={hf_w:.2e}")
    
    vae.train()
    
    # Control-flow branch for conditional or iterative execution.
    for ep in range(epochs):
        ep_loss = 0.0
        ep_rec = 0.0
        ep_kl = 0.0
        ep_edge = 0.0
        ep_hf = 0.0
        count = 0
        
        # KL warmup
        # Control-flow branch for conditional or iterative execution.
        if kl_warmup_frac > 0:
            warmup_steps = int(kl_warmup_frac * epochs)
            kl_mult = min(1.0, (ep + 1) / max(1, warmup_steps))
        else:
            kl_mult = 1.0
        
        # Control-flow branch for conditional or iterative execution.
        for x, _, _ in dl:
            x = x.to(DEVICE)
            
            # Forward pass
            xhat, z, mu, logvar = vae(x)
            
            # Reconstruction loss (L1 + L2 blend often works well)
            rec_loss = F.l1_loss(xhat, x) + 0.5 * F.mse_loss(xhat, x)
            
            # KL divergence
            kl = kl_loss(mu, logvar)
            
            # Edge-aware loss
            edge_loss = edge_aware_loss_3d(xhat, x)
            
            # High-frequency loss
            hf_loss = high_frequency_loss_3d(xhat, x)
            
            # Combined loss
            loss = (
                rec_w * rec_loss +
                kl_w * kl_mult * kl +
                edge_w * edge_loss +
                hf_w * hf_loss
            )
            
            # Backprop with gradient clipping
            opt.zero_grad()
            loss.backward()
            # Control-flow branch for conditional or iterative execution.
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            opt.step()
            
            # Track metrics
            ep_loss += loss.item()
            ep_rec += rec_loss.item()
            ep_kl += kl.item()
            ep_edge += edge_loss.item()
            ep_hf += hf_loss.item()
            count += 1
        
        # Log epoch stats
        # Control-flow branch for conditional or iterative execution.
        if count > 0:
            avg_loss = ep_loss / count
            avg_rec = ep_rec / count
            avg_kl = ep_kl / count
            avg_edge = ep_edge / count
            avg_hf = ep_hf / count
            
            print(
                f"[VAE {stage}] Ep {ep+1}/{epochs} | "
                f"Loss={avg_loss:.4f} | Rec={avg_rec:.4f} | "
                f"KL={avg_kl:.6f} | Edge={avg_edge:.4f} | HF={avg_hf:.4f} | "
                f"KL_mult={kl_mult:.3f}"
            )
        
        # Save checkpoint every 50 epochs
        # Control-flow branch for conditional or iterative execution.
        if (ep + 1) % 50 == 0:
            ckpt_path = Path(outdir) / f"vae_{stage}_ep{ep+1:04d}.pt"
            torch.save(vae.state_dict(), ckpt_path)
            print(f"[SAVE] {ckpt_path}")

        # Control-flow branch for conditional or iterative execution.
        if ep % 50 == 0 or ep == epochs or ep==10:
            _vae_snapshot(stage, ep, dl, vae, outdir)


# ============================================================
# IMPROVED DIFFUSION TRAINING
# ============================================================

# Function: `train_ldm_stage` implements a reusable processing step.
def train_ldm_stage(
    stage: str,
    dl,
    vae,
    unet,
    ema,
    ddpm,  # This should be the improved LatentDDPM
    opt,
    epochs: int,
    outdir: str,
    lat_mean,
    lat_std,
    args,
):
    """
    Improved latent diffusion training with:
    - Min-SNR-γ loss weighting (already in improved ddpm)
    - Gradient clipping
    - Warmup scheduler
    - Better tumor-weighted loss
    """
    print(f"[LDM {stage}] Training for {epochs} epochs")
    print(f"  guidance_scale={getattr(args, 'guidance_scale', 2.0)}")
    print(f"  min_snr_gamma={ddpm.min_snr_gamma}")
    print(f"  prediction_type={ddpm.prediction_type}")
    
    vae.eval()
    unet.train()
    
    total_steps = 0
    warmup_steps = getattr(args, 'warmup_steps', 500)
    base_lr = getattr(args, 'lr_unet', 2e-4)
    
    # Learning rate warmup scheduler
    # Function: `get_lr_mult` implements a reusable processing step.
    def get_lr_mult(step):
        # Control-flow branch for conditional or iterative execution.
        if step < warmup_steps:
            # Return the computed value to the caller.
            return step / max(1, warmup_steps)
        # Return the computed value to the caller.
        return 1.0
    
    # Control-flow branch for conditional or iterative execution.
    for ep in range(epochs):
        ep_loss = 0.0
        count = 0
        
        # Control-flow branch for conditional or iterative execution.
        for x, mask_gt, _ in dl:
            x = x.to(DEVICE)
            mask_gt = mask_gt.to(DEVICE)
            
            # Get latent z0
            # Control-flow branch for conditional or iterative execution.
            with torch.no_grad():
                z0 = get_latent_z0(
                    vae, x,
                    use_posterior_noise=getattr(args, 'use_posterior_noise', False)
                )
                z0_norm = normalize_latents(z0, lat_mean, lat_std)
            
            # Random timesteps
            B = z0_norm.shape[0]
            t = torch.randint(0, ddpm.T, (B,), device=DEVICE, dtype=torch.long)
            
            # Forward diffusion
            noise = torch.randn_like(z0_norm)
            zt, _ = ddpm.q_sample(z0_norm, t, noise)
            
            # Prepare conditioning
            maskL = F.interpolate(
                mask_gt,
                size=z0_norm.shape[2:],
                mode='nearest',
            )
            
            # Build control tensor (mask + edge + distance)
            controlL = build_control_tensor(maskL, args)
            
            # Conditional dropout for CFG training
            # Control-flow branch for conditional or iterative execution.
            if random.random() < getattr(args, 'cond_drop_p', 0.1):
                maskL = torch.zeros_like(maskL)
                controlL = torch.zeros_like(controlL)
            else:
                # Optional mask augmentation
                # Control-flow branch for conditional or iterative execution.
                if random.random() < getattr(args, 'mask_aug_p', 0.7):
                    maskL = augment_mask(maskL, p=1.0)
            
            # Model prediction
            pred = unet(zt, t, mask=maskL, control=controlL)
            
            # Compute target based on prediction type
            # Control-flow branch for conditional or iterative execution.
            if ddpm.prediction_type == "v_prediction":
                # Import dependencies used by this module.
                from diffusion_improved import get_velocity
                acp_t = ddpm.alphas_cumprod[t].view(-1, 1, 1, 1, 1).to(zt.device, zt.dtype)
                target = get_velocity(z0_norm, noise, acp_t)
            else:
                target = noise
            
            # Tumor-weighted loss mask
            # Control-flow branch for conditional or iterative execution.
            if getattr(args, 'tumor_loss_alpha', 1.0) > 0:
                # from ldm3d.train import dilate_mask_3d
                k = getattr(args, 'tumor_dilate_k', 5)
                mask_weight = dilate_mask_3d(maskL, k=k)
                alpha = getattr(args, 'tumor_loss_alpha', 1.0)
                mask_weight = 1.0 + alpha * mask_weight
            else:
                mask_weight = None
            
            # Compute loss (min-SNR weighting is inside ddpm.compute_loss)
            loss = ddpm.compute_loss(pred, target, t, mask_weight)
            
            # Backprop with gradient clipping
            opt.zero_grad()
            loss.backward()
            
            grad_clip = getattr(args, 'grad_clip', 1.0)
            # Control-flow branch for conditional or iterative execution.
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
            
            # Apply learning rate warmup
            lr_mult = get_lr_mult(total_steps)
            # Control-flow branch for conditional or iterative execution.
            for param_group in opt.param_groups:
                param_group['lr'] = base_lr * lr_mult
            
            opt.step()
            
            # Update EMA
            ema.update(unet)
            
            # Track metrics
            ep_loss += loss.item()
            count += 1
            total_steps += 1
        
        # Log epoch stats
        # Control-flow branch for conditional or iterative execution.
        if count > 0:
            avg_loss = ep_loss / count
            current_lr = opt.param_groups[0]['lr']
            print(
                f"[LDM {stage}] Ep {ep+1}/{epochs} | "
                f"Loss={avg_loss:.6f} | LR={current_lr:.2e}"
            )
        
        # Save checkpoint every 50 epochs
        # Control-flow branch for conditional or iterative execution.
        if (ep + 1) % 50 == 0:
            unet_path = Path(outdir) / f"unet_{stage}_ep{ep+1:04d}.pt"
            ema_path = Path(outdir) / f"ema_{stage}_ep{ep+1:04d}.pt"
            torch.save(unet.state_dict(), unet_path)
            torch.save(ema.shadow, ema_path)
            print(f"[SAVE] {unet_path}")
            print(f"[SAVE] {ema_path}")

        # Control-flow branch for conditional or iterative execution.
        if ep % 50 == 0 or ep == epochs or ep==10:
            _ldm_snapshot(
                stage, ep, dl, vae, unet, ema, ddpm,
                lat_mean, lat_std, outdir, args
            )

# Function: `build_control_tensor` implements a reusable processing step.
def build_control_tensor(maskL: torch.Tensor, args) -> torch.Tensor:
    """
    Build control tensor [B, 3, D, H, W] from mask.
    Channels: [mask, edge, distance_transform]
    """
    B = maskL.shape[0]
    D, H, W = maskL.shape[2:]
    
    # Channel 0: mask itself
    mask_chan = maskL
    
    # Channel 1: edge detection (simple gradient magnitude)
    # Function: `get_edges` implements a reusable processing step.
    def get_edges(m):
        dx = m[:, :, 1:, :, :] - m[:, :, :-1, :, :]
        dy = m[:, :, :, 1:, :] - m[:, :, :, :-1, :]
        dz = m[:, :, :, :, 1:] - m[:, :, :, :, :-1]
        
        # Pad back to original size
        dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dz = F.pad(dz, (0, 1, 0, 0, 0, 0))
        
        edge_mag = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
        # Return the computed value to the caller.
        return edge_mag
    
    edge_chan = get_edges(maskL)
    
    # Channel 2: soft distance transform (optional, can be expensive)
    # For speed, use a simple approximation: blur the mask
    dist_chan = blur3d(maskL, k=5)
    
    # Stack channels
    control = torch.cat([mask_chan, edge_chan, dist_chan], dim=1)
    
    # Return the computed value to the caller.
    return control


# ============================================================
# DYNAMIC CFG SAMPLING
# ============================================================

# Function: `dynamic_cfg_scale` implements a reusable processing step.
def dynamic_cfg_scale(step: int, total_steps: int, min_scale: float = 1.0, max_scale: float = 3.0) -> float:
    """
    Adaptive CFG scheduling for better quality-diversity balance.
    
    Start with low guidance for diversity, increase for quality refinement.
    """
    progress = step / max(1, total_steps)
    
    # Cosine schedule: low -> high -> low
    scale = min_scale + (max_scale - min_scale) * (1 - math.cos(math.pi * progress)) / 2
    
    # Return the computed value to the caller.
    return float(scale)


# Function: `_ldm_snapshot` implements a reusable processing step.
def _ldm_snapshot(
    stage_name, ep, dl, vae, unet, ema, ddpm,
    lat_mean, lat_std, outdir, args
):
    """
    Periodically sample a few synthetic volumes during training and save to disk.

    What this does:
      - Builds conditioning (mask/control) from real masks in the dataloader.
      - Runs ddpm.p_sample_loop(...) to generate normalized latents z_norm.
      - Denormalizes latents back to original VAE scale.
      - Decodes with VAE decoder to obtain synthetic MRI volumes.
      - Saves T1/T2/FLAIR NIfTI outputs.

    Why snapshots matter:
      - Diffusion training can silently fail (mode collapse, over-smoothing, black outputs).
      - Visual inspection every ~50 epochs catches issues early.
    """
    unet.eval()
    snap_root = Path(outdir) / f"{stage_name}_e{ep}_samples"
    snap_root.mkdir(parents=True, exist_ok=True)

    # Ensure latent stats are on the right device for denormalization
    lat_mean = lat_mean.to(DEVICE)
    lat_std = lat_std.to(DEVICE)

    saved = 0
    # Control-flow branch for conditional or iterative execution.
    with torch.no_grad():
        # Control-flow branch for conditional or iterative execution.
        for _, mask, sids in dl:
            mask = mask.to(DEVICE)
            mask = (mask > 0.5).float()

            # Save only a few subjects to keep snapshots lightweight
            # Control-flow branch for conditional or iterative execution.
            for b, sid in enumerate(list(sids)[:4]):
                # Build control channels from the downsampled mask.
                mask14 = F.interpolate(
                    mask[b:b + 1],
                    size=(args.latent_size,) * 3,
                    mode="nearest",
                )
                mask14 = mask14.clamp(0, 1)
                edge14 = mask_to_edge_3d(mask14)
                dist14 = mask_to_soft_dist(mask14)
                control14 = torch.cat([mask14, edge14, dist14], dim=1)

                # Sample normalized latents via reverse diffusion.
                # guidance_scale=1.0 keeps snapshots conservative/less likely to artifact.
                z_norm = ddpm.p_sample_loop(
                    unet,
                    shape=(1, args.z_channels, args.latent_size,
                           args.latent_size, args.latent_size),
                    use_ema=ema,
                    seed=args.sample_seed if args.sample_seed >= 0 else None,
                    mask=mask14,
                    control=control14,
                    guidance_scale=1.0,  # keep snapshot sane
                )

                # De-normalize latents and decode to image space.
                z = denormalize_latents_safe(z_norm, lat_mean, lat_std)
                x = vae.decode(z)

                # Write outputs per subject
                sdir = snap_root / sid
                sdir.mkdir(exist_ok=True)

                save_nifti(x[0, 0], sdir / "t1_synth.nii.gz", verbose=False)
                save_nifti(x[0, 1], sdir / "t2_synth.nii.gz", verbose=False)
                save_nifti(x[0, 2], sdir / "flair_synth.nii.gz", verbose=False)

                saved += 1
            break

    # Save checkpoints corresponding to this snapshot epoch
    torch.save(unet.state_dict(), Path(outdir) / f"unet_{stage_name}_ep{ep}.pt")
    torch.save(ema.shadow, Path(outdir) / f"ema_{stage_name}_ep{ep}.pt")

    print(f"[SNAPSHOT] Saved {saved} samples -> {snap_root}")
    unet.train()


# ============================================================
# FINAL SAMPLING
# ============================================================

@torch.no_grad()
# Function: `final_sampling_dump` implements a reusable processing step.
def final_sampling_dump(
    *,
    gbm_dl,
    pdgm_dl,
    vae,
    unet,
    ema,
    ddpm,
    outdir: str,
    lat_mean,
    lat_std,
    args,
    wipe_existing: bool = False,
    save_debug_conditioning: bool = True,
):
    """
    Robust final sampler that ALWAYS produces args.final_dump_n outputs per domain,
    even when the dataloader has fewer subjects (e.g., few-shot = 16).

    Correctly sets mask_reused using MASK IDENTITY (not sid string).

    IMPORTANT FIX:
      - If N > #subjects and k_per_subject == 1, you still want repeats.
      - We now track per-sid repetition across epochs:
          rep_idx = how many times this sid_img has already produced an output.
        For PDGM:
          rep_idx == 0 => use own GT mask (mask_reused=False)
          rep_idx  > 0 => use other mask from pool (mask_reused=True if different)

    meta.json includes:
      sid_img, sid_mask, img_mask_id, cond_mask_id, mask_reused, rep, global_idx, domain
    """
    # Import dependencies used by this module.
    import json
    import random
    from pathlib import Path

    vae.eval()
    unet.eval()

    # Keep latent stats on model device
    lat_mean = lat_mean.to(DEVICE)
    lat_std = lat_std.to(DEVICE)

    dump_root = Path(outdir) / "final_samples"
    # Control-flow branch for conditional or iterative execution.
    if wipe_existing and dump_root.exists():
        # Import dependencies used by this module.
        import shutil
        shutil.rmtree(dump_root)
    dump_root.mkdir(parents=True, exist_ok=True)

    # ControlNet unconditional safety
    cfg_null_cond_effective = bool(getattr(args, "cfg_null_cond", False))
    # Control-flow branch for conditional or iterative execution.
    if cfg_null_cond_effective:
        print("[WARN] cfg_null_cond=True requested, but ControlNet expects control tensor.")
        print("       Using zero-control unconditional pass instead (cfg_null_cond_effective=False).")
        cfg_null_cond_effective = False

    # ----------------------------
    # Safe getters
    # ----------------------------
    # Function: `_int` implements a reusable processing step.
    def _int(x, default):
        # Control-flow branch for conditional or iterative execution.
        try:
            # Return the computed value to the caller.
            return int(x)
        # Control-flow branch for conditional or iterative execution.
        except Exception:
            # Return the computed value to the caller.
            return default

    # Function: `_float` implements a reusable processing step.
    def _float(x, default):
        # Control-flow branch for conditional or iterative execution.
        try:
            # Return the computed value to the caller.
            return float(x)
        # Control-flow branch for conditional or iterative execution.
        except Exception:
            # Return the computed value to the caller.
            return default

    N_default = _int(getattr(args, "final_dump_n", 16), 16)
    L_default = _int(getattr(args, "latent_size", 28), 28)
    zC_default = _int(getattr(args, "z_channels", 8), 8)

    # outputs per image subject before moving on
    K_per_img = _int(getattr(args, "k_per_subject", 1), 1)
    # Control-flow branch for conditional or iterative execution.
    if K_per_img < 1:
        K_per_img = 1

    # ----------------------------
    # Build conditioning helper
    # ----------------------------
    # Function: `_build_cond` implements a reusable processing step.
    def _build_cond(mask_1ch: torch.Tensor):
        """
        mask_1ch: [1,1,D,H,W] on DEVICE float in {0,1}
        returns: maskL [1,1,L,L,L], controlL [1,3,L,L,L]
        """
        L = _int(getattr(args, "latent_size", L_default), L_default)

        maskL = F.interpolate(mask_1ch, size=(L, L, L), mode="nearest").clamp(0, 1)
        edgeL = mask_to_edge_3d(maskL)
        distL = mask_to_soft_dist(maskL)

        ctrl_mask_w = _float(getattr(args, "ctrl_mask_w", 1.0), 1.0)
        ctrl_edge_w = _float(getattr(args, "ctrl_edge_w", 0.3), 0.3)
        ctrl_dist_w = _float(getattr(args, "ctrl_dist_w", 0.7), 0.7)

        controlL = torch.cat(
            [ctrl_mask_w * maskL, ctrl_edge_w * edgeL, ctrl_dist_w * distL],
            dim=1,
        )
        # Return the computed value to the caller.
        return maskL, controlL

    # ----------------------------
    # Per-output seed offset
    # ----------------------------
    # Function: `_sample_seed_for` implements a reusable processing step.
    def _sample_seed_for(global_saved_idx: int):
        base = getattr(args, "sample_seed", -1)
        # Control-flow branch for conditional or iterative execution.
        try:
            base = int(base)
        # Control-flow branch for conditional or iterative execution.
        except Exception:
            base = -1
        # Control-flow branch for conditional or iterative execution.
        if base < 0:
            # Return the computed value to the caller.
            return None
        # Return the computed value to the caller.
        return base + int(global_saved_idx)

    # ----------------------------
    # Build PDGM mask pool (iterate over *all* items in each batch)
    # ----------------------------
    print("[FINAL] Building PDGM mask pool...")
    mask_pool = []   # list[{sid_mask, mask_id, maskL, controlL}]
    by_sid = {}      # sid_mask -> list[{mask_id, maskL, controlL}]
    by_img_sid = {}  # sid_img  -> (mask_id, maskL, controlL) using GT mask from that sid

    seen_any_pdgm_batch = False
    # Control-flow branch for conditional or iterative execution.
    for batch_idx, batch in enumerate(pdgm_dl):
        seen_any_pdgm_batch = True

        # expected: x, mask, sids
        # Control-flow branch for conditional or iterative execution.
        try:
            _, maskB, sidsB = batch
        # Control-flow branch for conditional or iterative execution.
        except Exception as e:
            print(f"[WARN] pdgm mask-pool: bad batch unpack at idx={batch_idx}: {e}")
            continue

        # Control-flow branch for conditional or iterative execution.
        try:
            B = int(maskB.shape[0])
        # Control-flow branch for conditional or iterative execution.
        except Exception:
            B = 1

        # Control-flow branch for conditional or iterative execution.
        for bi in range(B):
            # Control-flow branch for conditional or iterative execution.
            try:
                mask = maskB[bi : bi + 1]  # [1,1,D,H,W]
            # Control-flow branch for conditional or iterative execution.
            except Exception as e:
                print(f"[WARN] pdgm mask-pool: cannot slice mask at idx={batch_idx}, bi={bi}: {e}")
                continue

            # sid string robust
            sid_mask = None
            # Control-flow branch for conditional or iterative execution.
            try:
                # Control-flow branch for conditional or iterative execution.
                if sidsB is None:
                    sid_mask = f"unknownmask_b{batch_idx:04d}_{bi:02d}"
                else:
                    # Control-flow branch for conditional or iterative execution.
                    if isinstance(sidsB, (list, tuple)):
                        sid_mask = str(sidsB[bi])
                    else:
                        sid_mask = str(sidsB[bi].item()) if hasattr(sidsB[bi], "item") else str(sidsB[bi])
            # Control-flow branch for conditional or iterative execution.
            except Exception:
                sid_mask = f"unknownmask_b{batch_idx:04d}_{bi:02d}"

            # Control-flow branch for conditional or iterative execution.
            try:
                mask_bin = (mask.to(DEVICE) > 0.5).float()
                maskL, controlL = _build_cond(mask_bin)
                mask_id = mask_hash_latent(maskL)
            # Control-flow branch for conditional or iterative execution.
            except Exception as e:
                print(f"[WARN] pdgm mask-pool: failed building cond for {sid_mask}: {e}")
                continue

            item = {"sid_mask": sid_mask, "mask_id": mask_id, "maskL": maskL, "controlL": controlL}
            mask_pool.append(item)

            by_sid.setdefault(sid_mask, []).append({"mask_id": mask_id, "maskL": maskL, "controlL": controlL})
            # Control-flow branch for conditional or iterative execution.
            if sid_mask not in by_img_sid:
                by_img_sid[sid_mask] = (mask_id, maskL, controlL)

    # Control-flow branch for conditional or iterative execution.
    if not seen_any_pdgm_batch or len(mask_pool) == 0:
        raise RuntimeError("PDGM mask pool is empty. Check pdgm_dl / few-shot list / masks on disk.")

    print(f"[FINAL] Mask pool size = {len(mask_pool)}")

    # Function: `_pick_other_mask` implements a reusable processing step.
    def _pick_other_mask(img_mask_id: str):
        """
        Returns a conditioning mask that is different from img_mask_id whenever possible.
        """
        # Control-flow branch for conditional or iterative execution.
        if len(mask_pool) == 1:
            it = mask_pool[0]
            # Return the computed value to the caller.
            return it["sid_mask"], it["mask_id"], it["maskL"], it["controlL"]

        # Control-flow branch for conditional or iterative execution.
        for _ in range(500):
            it = random.choice(mask_pool)
            # Control-flow branch for conditional or iterative execution.
            if it["mask_id"] != img_mask_id:
                # Return the computed value to the caller.
                return it["sid_mask"], it["mask_id"], it["maskL"], it["controlL"]

        # fallback: pool might all be identical (rare but possible)
        it = random.choice(mask_pool)
        # Return the computed value to the caller.
        return it["sid_mask"], it["mask_id"], it["maskL"], it["controlL"]

    # ----------------------------
    # Main per-domain dump
    # ----------------------------
    # Function: `_dump_domain` implements a reusable processing step.
    def _dump_domain(domain_name: str, dl):
        domain_root = dump_root / domain_name
        domain_root.mkdir(parents=True, exist_ok=True)

        N = _int(getattr(args, "final_dump_n", N_default), N_default)
        L = _int(getattr(args, "latent_size", L_default), L_default)
        zC = _int(getattr(args, "z_channels", zC_default), zC_default)

        saved = 0
        seen_any_batch = False
        stale_guard_iters = 0
        STALE_GUARD_MAX = max(1000, N * 80)

        # ✅ FIX: track repeats across epochs (sid_img -> count produced so far)
        sid_out_counts = {}

        # Control-flow branch for conditional or iterative execution.
        while saved < N:
            progressed_this_epoch = False

            # Control-flow branch for conditional or iterative execution.
            for batch_idx, batch in enumerate(dl):
                seen_any_batch = True

                # expected: x, mask_img_gt, sids
                # Control-flow branch for conditional or iterative execution.
                try:
                    x, mask_img_gtB, sidsB = batch
                # Control-flow branch for conditional or iterative execution.
                except Exception as e:
                    print(f"[WARN] {domain_name}: bad batch unpack at idx={batch_idx}: {e}")
                    stale_guard_iters += 1
                    # Control-flow branch for conditional or iterative execution.
                    if stale_guard_iters > STALE_GUARD_MAX:
                        print(f"[ERROR] {domain_name}: aborting (too many bad batches).")
                        break
                    continue

                # Control-flow branch for conditional or iterative execution.
                try:
                    B = int(mask_img_gtB.shape[0])
                # Control-flow branch for conditional or iterative execution.
                except Exception:
                    B = 1

                # Control-flow branch for conditional or iterative execution.
                for bi in range(B):
                    # Control-flow branch for conditional or iterative execution.
                    if saved >= N:
                        break

                    # sid_img robust
                    # Control-flow branch for conditional or iterative execution.
                    try:
                        # Control-flow branch for conditional or iterative execution.
                        if sidsB is None:
                            sid_img = f"unknownimg_b{batch_idx:04d}_{bi:02d}"
                        else:
                            # Control-flow branch for conditional or iterative execution.
                            if isinstance(sidsB, (list, tuple)):
                                sid_img = str(sidsB[bi])
                            else:
                                sid_img = str(sidsB[bi].item()) if hasattr(sidsB[bi], "item") else str(sidsB[bi])
                    # Control-flow branch for conditional or iterative execution.
                    except Exception:
                        sid_img = f"unknownimg_b{batch_idx:04d}_{bi:02d}"

                    # image GT mask -> identity + optional debug save
                    mask_img_ok = None
                    img_mask_id = "no_img_mask"
                    # Control-flow branch for conditional or iterative execution.
                    try:
                        m = mask_img_gtB[bi : bi + 1].to(DEVICE)
                        mask_img_ok = (m > 0.5).float()
                        img_maskL = F.interpolate(mask_img_ok, size=(L, L, L), mode="nearest").clamp(0, 1)
                        img_mask_id = mask_hash_latent(img_maskL)
                    # Control-flow branch for conditional or iterative execution.
                    except Exception:
                        mask_img_ok = None
                        img_mask_id = "no_img_mask"

                    # Generate K outputs for this image subject
                    # Control-flow branch for conditional or iterative execution.
                    for local_rep in range(K_per_img):
                        # Control-flow branch for conditional or iterative execution.
                        if saved >= N:
                            break

                        # ✅ rep index across whole dump for this sid
                        rep_idx = int(sid_out_counts.get(sid_img, 0))

                        # --- choose conditioning mask ---
                        # PDGM:
                        #   rep_idx == 0: own GT mask if available
                        #   rep_idx  > 0: other mask from pool (try to be different)
                        # Control-flow branch for conditional or iterative execution.
                        if domain_name == "pdgm" and rep_idx == 0 and sid_img in by_img_sid:
                            sid_mask = sid_img
                            cond_mask_id, maskL, controlL = by_img_sid[sid_img]
                        else:
                            sid_mask, cond_mask_id, maskL, controlL = _pick_other_mask(img_mask_id)

                        # ✅ mask_reused means: "conditioning mask is NOT the GT mask for this image"
                        mask_reused = (cond_mask_id != img_mask_id)

                        out_sid = f"s{saved:05d}"
                        sdir = domain_root / out_sid
                        # Control-flow branch for conditional or iterative execution.
                        try:
                            sdir.mkdir(parents=True, exist_ok=True)
                        # Control-flow branch for conditional or iterative execution.
                        except Exception as e:
                            print(f"[WARN] {domain_name}: could not create dir {sdir}: {e}")
                            stale_guard_iters += 1
                            # Control-flow branch for conditional or iterative execution.
                            if stale_guard_iters > STALE_GUARD_MAX:
                                print(f"[ERROR] {domain_name}: aborting (dir creation failing).")
                                break
                            continue

                        # --- write metadata ---
                        # Control-flow branch for conditional or iterative execution.
                        try:
                            meta = {
                                "sid_img": sid_img,
                                "sid_mask": sid_mask,
                                "img_mask_id": img_mask_id,
                                "cond_mask_id": cond_mask_id,
                                "mask_reused": bool(mask_reused),
                                "rep": int(rep_idx),          # ✅ global-per-sid repetition index
                                "global_idx": int(saved),
                                "domain": domain_name,
                            }
                            # Control-flow branch for conditional or iterative execution.
                            with open(sdir / "meta.json", "w") as f:
                                json.dump(meta, f, indent=2)
                        # Control-flow branch for conditional or iterative execution.
                        except Exception as e:
                            print(f"[WARN] {domain_name}: failed writing meta.json for {out_sid}: {e}")

                        # --- save debug masks ---
                        # Control-flow branch for conditional or iterative execution.
                        if save_debug_conditioning:
                            # Control-flow branch for conditional or iterative execution.
                            try:
                                save_nifti(maskL[0, 0], sdir / "mask14_cond.nii.gz", verbose=False)
                            # Control-flow branch for conditional or iterative execution.
                            except Exception as e:
                                print(f"[WARN] {domain_name}: failed saving mask14_cond for {out_sid}: {e}")

                            # Control-flow branch for conditional or iterative execution.
                            if mask_img_ok is not None:
                                # Control-flow branch for conditional or iterative execution.
                                try:
                                    save_nifti(mask_img_ok[0, 0], sdir / "mask_img_gt.nii.gz", verbose=False)
                                # Control-flow branch for conditional or iterative execution.
                                except Exception:
                                    pass

                        # --- sample ---
                        # Control-flow branch for conditional or iterative execution.
                        try:
                            seed = _sample_seed_for(saved)

                            z_norm = ddpm.p_sample_loop(
                                unet,
                                shape=(1, zC, L, L, L),
                                use_ema=ema if getattr(args, "use_ema_for_sampling", False) else None,
                                seed=seed,
                                mask=maskL,
                                control=controlL,
                                guidance_scale=_float(getattr(args, "guidance_scale", 1.0), 1.0),
                                noise_mult=_float(getattr(args, "noise_mult", 1.0), 1.0),
                                noise_end_frac=_float(getattr(args, "noise_end_frac", 0.0), 0.0),
                                cfg_null_cond=cfg_null_cond_effective,
                                x0_clip=_float(getattr(args, "x0_clip", 3.0), 3.0),
                            )
                        # Control-flow branch for conditional or iterative execution.
                        except Exception as e:
                            print(f"[WARN] {domain_name}: sampling failed for {out_sid}: {e}")
                            stale_guard_iters += 1
                            # Control-flow branch for conditional or iterative execution.
                            if stale_guard_iters > STALE_GUARD_MAX:
                                print(f"[ERROR] {domain_name}: aborting (too many sampling failures).")
                                break
                            continue

                        # --- decode + save ---
                        # Control-flow branch for conditional or iterative execution.
                        try:
                            z = denormalize_latents_safe(z_norm, lat_mean, lat_std)
                            xhat = vae.decode(z)

                            save_nifti(xhat[0, 0], sdir / "t1_synth.nii.gz", verbose=False)
                            save_nifti(xhat[0, 1], sdir / "t2_synth.nii.gz", verbose=False)
                            save_nifti(xhat[0, 2], sdir / "flair_synth.nii.gz", verbose=False)
                        # Control-flow branch for conditional or iterative execution.
                        except Exception as e:
                            print(f"[WARN] {domain_name}: decode/save failed for {out_sid}: {e}")
                            stale_guard_iters += 1
                            # Control-flow branch for conditional or iterative execution.
                            if stale_guard_iters > STALE_GUARD_MAX:
                                print(f"[ERROR] {domain_name}: aborting (too many decode/save failures).")
                                break
                            continue

                        # ✅ increment counters after a successful save
                        sid_out_counts[sid_img] = rep_idx + 1

                        saved += 1
                        progressed_this_epoch = True
                        stale_guard_iters = 0

                    # Control-flow branch for conditional or iterative execution.
                    if saved >= N:
                        break

                # Control-flow branch for conditional or iterative execution.
                if saved >= N:
                    break

            # Control-flow branch for conditional or iterative execution.
            if saved >= N:
                break

            # Control-flow branch for conditional or iterative execution.
            if not seen_any_batch:
                print(f"[ERROR] {domain_name}: dataloader produced zero batches. Cannot sample.")
                break

            # Control-flow branch for conditional or iterative execution.
            if not progressed_this_epoch:
                print(f"[ERROR] {domain_name}: no progress in an epoch (all failures). Aborting to avoid infinite loop.")
                break

        print(f"[FINAL] Dumped {saved} / {N} samples for {domain_name}")

    print("--- FINAL SAMPLING DUMP (ROBUST + CORRECT MASK REUSE FLAGS) ---")
    _dump_domain("gbm", gbm_dl)
    _dump_domain("pdgm", pdgm_dl)
    print("[FINAL] Done.")