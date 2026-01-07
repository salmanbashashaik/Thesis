#!/usr/bin/env python3
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

# ============================================================
# SMALL HELPERS
# ============================================================

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
    if k <= 1:
        return (mask > 0.5).float()
    if k % 2 == 0:
        k = k + 1  # force odd
    pad = k // 2
    return (F.max_pool3d(mask, kernel_size=k, stride=1, padding=pad) > 0.5).float()


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
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise TypeError("lat_mean/lat_std must be torch.Tensors")

    zc = z.shape[1]

    # Move stats to z device/dtype for safe arithmetic.
    mean = mean.to(device=z.device, dtype=z.dtype)
    std = std.to(device=z.device, dtype=z.dtype)

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
        if x.ndim == 1:
            if x.numel() != zc:
                raise RuntimeError(f"latent stat vector numel={x.numel()} != zC={zc}")
            return x

        # Accept 5D tensors, channel-first or channel-last, optionally with spatial dims
        if x.ndim == 5:
            # channel-first: [B?, zC, D, H, W]
            if x.shape[1] == zc:
                if x.shape[2:] != (1, 1, 1):
                    # Reduce spatial dims to get a single vector per channel.
                    x = x.mean(dim=(0, 2, 3, 4))  # -> [zC]
                else:
                    x = x[0, :, 0, 0, 0]
                return x

            # channel-last: [B?, D, H, W, zC]
            if x.shape[-1] == zc:
                if x.shape[:-1] != (1, 1, 1, 1):
                    # Reduce spatial dims to get a single vector per channel.
                    x = x.mean(dim=(0, 1, 2, 3))  # -> [zC]
                else:
                    x = x[0, 0, 0, 0, :]
                return x

        # Accept 4D channel-first with spatial dims: [zC, D, H, W]
        if x.ndim == 4 and x.shape[0] == zc:
            if x.shape[1:] != (1, 1, 1):
                # Reduce spatial dims to get a single vector per channel.
                x = x.mean(dim=(1, 2, 3))  # -> [zC]
            else:
                x = x[:, 0, 0, 0]
            return x

        # Last resort: flatten if exactly zC elements
        x_flat = x.view(-1)
        if x_flat.numel() == zc:
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
    return mean_b, std_b


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
    return z_norm * (s + 1e-6) + m


# ============================================================
# CONTROL HELPERS
# ============================================================

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
    return edge


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
    for _ in range(steps):
        x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)

    # Normalize to [0,1] per sample for stability across different mask sizes
    x = x - x.amin(dim=(2, 3, 4), keepdim=True)
    x = x / (x.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
    return x


@torch.no_grad()
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


def train_vae_stage(
    stage_name: str,
    dl,
    vae,
    opt,
    epochs: int,
    outdir: str,
    *,
    kl_w: float = 1e-4,
    rec_w: float = 1.0,
    kl_warmup_frac: float = 0.35,   # 35% of training to ramp KL
    l2_w: float = 0.0,              # set to 0.05 or 0.1 if we  want
):
    """
    Train the VAE for a given stage (source or target).

    Inputs:
      stage_name:
        Label used for logging + snapshot folder naming (e.g., "src" or "tgt")
      dl:
        DataLoader returning (x, mask, sid). Mask is unused for VAE.
      vae:
        VAE model (encoder+decoder)
      opt:
        Optimizer for VAE parameters
      epochs:
        Number of epochs to train
      outdir:
        Output directory for snapshots/checkpoints

    Loss:
      - Reconstruction: L1 + optional L2
      - KL divergence: encourages latent distribution to match N(0,I)
      - KL warmup: ramps KL weight gradually to avoid posterior collapse early

    Notes:
      - clip_grad_norm_ prevents gradient explosions, especially early on.
      - rec_list logs L1 only to keep logs comparable across experiments.
    """
    if epochs <= 0:
        return

    print(f"--- Starting VAE {stage_name} Training ({epochs} eps) ---")
    vae.train()

    # Warmup schedule for KL weight to avoid posterior collapse early.
    warmup_epochs = max(1, int(round(epochs * kl_warmup_frac)))

    for ep in range(1, epochs + 1):
        t0 = time.time()
        rec_list, kl_list = [], []

        # KL anneal: 0 -> kl_w over warmup, then stay at kl_w
        if ep <= warmup_epochs:
            kl_w_ep = kl_w * (ep / warmup_epochs)
        else:
            kl_w_ep = kl_w

        for x, _, _ in dl:
            x = x.to(DEVICE)

            # Forward pass through VAE gives reconstruction plus posterior params
            xhat, _, mu, logvar = vae(x)

            # Reconstruction term: L1 plus optional L2.
            l1 = F.l1_loss(xhat, x)
            l2 = F.mse_loss(xhat, x) if l2_w > 0.0 else 0.0
            l_rec = l1 + (l2_w * l2)

            # KL divergence of posterior from N(0,I)
            l_kl = kl_loss(mu, logvar)

            # Total loss combines recon + KL with warmup.
            loss = rec_w * l_rec + kl_w_ep * l_kl

            opt.zero_grad()
            loss.backward()

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            opt.step()

            # Track per-batch metrics for epoch averages
            rec_list.append(float(l1.item()))  # keep our rec logging comparable
            kl_list.append(float(l_kl.item()))

        print(
            f"[VAE:{stage_name}][Ep {ep}] "
            f"rec(L1)={np.mean(rec_list):.4f} "
            f"kl={np.mean(kl_list):.6f} "
            f"kl_w={kl_w_ep:.2e} "
            f"time={time.time() - t0:.1f}s"
        )

        # Periodic snapshot so we can visually inspect recon quality
        if ep % 50 == 0 or ep == epochs:
            _vae_snapshot(stage_name, ep, dl, vae, outdir)

# ============================================================
# LDM TRAINING (CONTROLNET ENABLED)
# ============================================================
def train_ldm_stage(
    stage_name: str,
    dl,
    vae,
    unet,
    ema,
    ddpm,
    opt,
    epochs: int,
    outdir: str,
    lat_mean,
    lat_std,
    args,
):
    """
    Train the latent diffusion model (UNet denoiser) for a given stage.

    Key training idea:
      - Use VAE encoder to get z0 latents from x.
      - Normalize z0 using per-channel mean/std.
      - Sample a timestep t and diffuse to zt via ddpm.q_sample.
      - Train UNet to predict the noise eps used to create zt.

    Conditioning:
      - Use downsampled tumor masks as conditioning (mask14).
      - Build additional control channels (edge14, dist14) and concatenate:
          control14 = [mask14, edge14, dist14] -> [B,3,latent,latent,latent]
      - Apply optional conditioning dropout to enable CFG-like behavior.

    Tumor-weighted loss (optional):
      - If tumor_alpha > 0, weight the MSE inside tumor regions more heavily.
      - This combats "tumor wash-out" where diffusion learns the easy background first
        and neglects small tumor features.
    """
    if epochs <= 0:
        return

    print(f"--- Starting LDM {stage_name} Training ({epochs} eps) ---")

    # Freeze VAE during diffusion training.
    # We want the latent space to stay fixed while the UNet learns to denoise within it.
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    unet.train()

    # Gradient accumulation:
    # - Source domain likely has many samples -> can accumulate more for stable updates.
    # - Target few-shot stage is tiny -> use smaller/no accumulation to update more frequently.
    accum_steps = 1 if stage_name == "ldm_tgt" else 4
    print(f"[INFO] Gradient accumulation: {accum_steps}")

    # Keep stats on GPU (training uses them repeatedly)
    # This avoids repeatedly moving tensors device inside the inner training loop.
    lat_mean = lat_mean.to(DEVICE)
    lat_std = lat_std.to(DEVICE)

    # Optional tumor-weighted loss settings.
    # tumor-weight knobs (safe defaults if args doesn't have them)
    tumor_alpha = float(getattr(args, "tumor_loss_alpha", 0.0))   # try 2.0–5.0
    tumor_dilate_k = int(getattr(args, "tumor_dilate_k", 3))      # 3 or 5
    tumor_edge_mix = float(getattr(args, "tumor_edge_mix", 0.0))  # 0.0..1.0 (optional)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_list = []
        opt.zero_grad()

        for i, (x, mask, _) in enumerate(dl):
            x = x.to(DEVICE)

            # ----------------------------
            # Latents
            # ----------------------------
            with torch.no_grad():
                # Encode to z0 and normalize with dataset stats.
                # get_latent_z0 controls whether z0 is mu-only or posterior sampled
                z0 = get_latent_z0(
                    vae,
                    x,
                    use_posterior_noise=args.use_posterior_noise,
                )
                z0 = normalize_latents(z0, lat_mean, lat_std)

            bsz = z0.size(0)

            # Sample random diffusion timestep for each sample in the batch
            t = torch.randint(0, ddpm.T, (bsz,), device=DEVICE)

            # --------------------------------------------------
            # MASKS (split GT vs conditioning!)
            # --------------------------------------------------
            mask = mask.to(DEVICE)

            # Stable GT mask for LOSS weighting (never augmented, never dropped)
            mask_gt = (mask > 0.5).float()      # [B,1,D,H,W]

            # Conditioning mask (can be augmented)
            mask_cond = mask_gt.clone()
            mask_cond = augment_mask(mask_cond, p=getattr(args, "mask_aug_p", 0.7))

            # Downsample conditioning mask to latent resolution.
            # NOTE: mode="nearest" preserves binary-ish labels.
            mask14 = F.interpolate(
                mask_cond,
                size=(args.latent_size,) * 3,
                mode="nearest",
            ).clamp(0, 1)

            # Optional blur:
            # Smoothing the mask can help reduce overly sharp conditioning that the model overfits to.
            # But too much blur makes conditioning mushy; hence a smaller default p.
            blur_p = float(getattr(args, "mask_blur_p", 0.3))   # was 0.7 (too high)
            if random.random() < blur_p:
                mask14 = blur3d(mask14, k=random.choice([3, 5])).clamp(0, 1)

            # Build edge/dist BEFORE cond-drop (so control doesn't become junk)
            edge14 = mask_to_edge_3d(mask14)
            dist14 = mask_to_soft_dist(mask14, steps=6)

            # Cond-drop should only affect the *conditioning channels*, not GT loss mask
            # This simulates unconditional training examples for CFG at sampling time.
            if random.random() < float(getattr(args, "cond_drop_p", 0.1)):
                mask14 = torch.zeros_like(mask14)
                edge14 = torch.zeros_like(edge14)
                dist14 = torch.zeros_like(dist14)

            # Concatenate control channels: [mask, edge, dist] -> [B,3,latent,latent,latent]
            control14 = torch.cat([mask14, edge14, dist14], dim=1)  # [B,3,14,14,14]

            # --------------------------------------------------
            # DIFFUSION STEP
            # --------------------------------------------------
            # Forward diffuse z0 -> zt and keep the sampled noise target.
            zt, noise = ddpm.q_sample(z0, t)

            # UNet predicts the noise eps given (zt, t, conditioning)
            pred = unet(
                zt,
                t,
                mask=mask14,
                control=control14,
            )

            # --------------------------------------------------
            # (OPTIONAL) TUMOR-WEIGHTED DIFFUSION LOSS
            # --------------------------------------------------
            # Base diffusion objective: MSE between predicted noise and true noise.
            mse = (pred - noise) ** 2  # [B, zC, D, H, W]

            if tumor_alpha > 0.0:
                # IMPORTANT: weight using mask_gt (NOT augmented/dropped)
                # Downsample mask_gt to match latent spatial dims for weighting.
                mask_z = F.interpolate(mask_gt, size=mse.shape[-3:], mode="nearest")  # [B,1,D,H,W]
                mask_z = (mask_z > 0.5).float()

                # Optional dilation expands the weighted tumor region.
                if tumor_dilate_k and tumor_dilate_k > 1:
                    mask_z = dilate_mask_3d(mask_z, k=tumor_dilate_k)

                # Optional edge mix: emphasize boundary band along with interior.
                if tumor_edge_mix > 0.0:
                    edge_z = mask_to_edge_3d(mask_z)
                    mask_z = (1.0 - tumor_edge_mix) * mask_z + tumor_edge_mix * edge_z

                # Weight map: 1 outside tumor, 1+tumor_alpha inside tumor (broadcast over channels)
                w = 1.0 + tumor_alpha * mask_z
                mse = mse * w

                # Debug print once in a while to confirm tumor fraction isn't crazy small/large
                if (i % 250 == 0) and (ep == 1):
                    frac = float((mask_z > 0.0).float().mean().item())
                    print(
                        f"[DBG] tumor_loss_alpha={tumor_alpha} dilate_k={tumor_dilate_k} "
                        f"edge_mix={tumor_edge_mix} latent_tumor_frac~{frac:.4f}"
                    )

            # Gradient accumulation: scale loss down so the effective gradient is averaged
            loss = mse.mean() / accum_steps
            loss.backward()

            # Apply gradient accumulation to stabilize training.
            # Only step optimizer every accum_steps batches.
            if (i + 1) % accum_steps == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

                # Update EMA after each optimizer step (EMA tracks trained weights)
                ema.update(unet)

            # Store scaled-back loss for logging
            loss_list.append(loss.item() * accum_steps)

        print(
            f"[LDM:{stage_name}][Ep {ep}] "
            f"mse={np.mean(loss_list):.6f} "
            f"time={time.time() - t0:.1f}s"
        )

        # Periodic sample snapshot to visually inspect diffusion outputs
        if ep % 50 == 0 or ep == epochs:
            _ldm_snapshot(
                stage_name, ep, dl, vae, unet, ema, ddpm,
                lat_mean, lat_std, outdir, args
            )


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
    with torch.no_grad():
        for _, mask, sids in dl:
            mask = mask.to(DEVICE)
            mask = (mask > 0.5).float()

            # Save only a few subjects to keep snapshots lightweight
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
):
    """
    Generate and save the final set of synthetic samples used for evaluation.

    This is the "official" dump used by downstream evaluation scripts:
      - FID/SSIM comparisons
      - CNN classifier evaluation on synthesized volumes
      - Qualitative figures for the thesis/paper

    Behavior:
      - Creates outdir/final_samples/{gbm|pdgm}/{SubjectID}/
      - For each dataset (gbm and pdgm):
          * Take subjects from the dataloader (one at a time)
          * Build conditioning at latent resolution (mask14 + control14)
          * Sample latents with guidance_scale=args.guidance_scale
          * Decode via VAE and save nifti volumes per modality
          * Stop after args.final_dump_n samples
    """
    print("--- FINAL SAMPLING DUMP ---")

    vae.eval()
    unet.eval()

    # Keep latent stats on the same device as the model.
    # This avoids device mismatch and is cheaper than moving every iteration.
    lat_mean = lat_mean.to(DEVICE)
    lat_std = lat_std.to(DEVICE)

    dump_root = Path(outdir) / "final_samples"
    dump_root.mkdir(parents=True, exist_ok=True)

    for name, dl in [("gbm", gbm_dl), ("pdgm", pdgm_dl)]:
        saved = 0

        for _, mask, sids in dl:
            # Force batch size 1 for consistent directory naming and easier evaluation matching
            mask = mask[:1]
            sids = [sids[0]]

            mask = mask.to(DEVICE)
            mask = (mask > 0.5).float()

            # Build conditioning control at latent resolution.
            mask14 = F.interpolate(
                mask,
                size=(args.latent_size,) * 3,
                mode="nearest",
            )
            mask14 = mask14.clamp(0, 1)
            edge14 = mask_to_edge_3d(mask14)
            dist14 = mask_to_soft_dist(mask14)
            control14 = torch.cat([mask14, edge14, dist14], dim=1)

            sid = sids[0]

            # Sample normalized latents with CFG (if guidance_scale > 1)
            z_norm = ddpm.p_sample_loop(
                unet,
                shape=(1, args.z_channels,
                       args.latent_size, args.latent_size, args.latent_size),
                use_ema=ema,
                seed=args.sample_seed if args.sample_seed >= 0 else None,
                mask=mask14,
                control=control14,
                guidance_scale=args.guidance_scale,
            )

            # Decode and save final synth samples.
            z = denormalize_latents_safe(z_norm, lat_mean, lat_std)
            x = vae.decode(z)

            sdir = dump_root / name / sid
            sdir.mkdir(parents=True, exist_ok=True)

            save_nifti(x[0, 0], sdir / "t1_synth.nii.gz", verbose=False)
            save_nifti(x[0, 1], sdir / "t2_synth.nii.gz", verbose=False)
            save_nifti(x[0, 2], sdir / "flair_synth.nii.gz", verbose=False)

            saved += 1
            if saved >= args.final_dump_n:
                break

        print(f"[FINAL] Dumped {saved} samples for {name}")
