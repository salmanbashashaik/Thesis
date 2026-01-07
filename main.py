#!/usr/bin/env python3
"""
ldm3d/main.py

Entry point for training a 3D Latent Diffusion Model (LDM) for MRI synthesis.

High-level pipeline (what this script orchestrates):
  1) Load source-domain GBM volumes and target-domain PDGM volumes.
  2) Train (or load) a 3D VAE to compress 3D MRI volumes into a latent space.
  3) Compute per-channel latent mean/std for stable diffusion training (latent normalization).
  4) Train a UNet-based DDPM in latent space on GBM (source), then optionally adapt to PDGM (target).
  5) Dump final qualitative samples and save final checkpoints.

Important concept:
  - Diffusion is trained on normalized latents, so latent stats (mean/std) must match the VAE
    that produced those latents. If the VAE weights change, latent stats should be recomputed.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ldm3d.config import DEVICE
from ldm3d.utils_seed import set_seed
from ldm3d.io_nifti import validate_input_data
from ldm3d.data import VolFolder, load_subject_list

from ldm3d.vae import VAE3D
from ldm3d.unet import UNet3DLatentCond
from ldm3d.ema import EMA
from ldm3d.diffusion import LatentDDPM

# Latent stats utilities:
# - maybe_load_latent_stats: read cached stats from outdir/latent_stats.pt if available
# - estimate_latent_stats: compute mean/std from a dataloader + current VAE encoder
# - recompute_and_save_latent_stats: convenience wrapper that builds its own GBM dataloader
from ldm3d.latent_stats import maybe_load_latent_stats, estimate_latent_stats, recompute_and_save_latent_stats

# Training orchestration:
# - train_vae_stage: VAE training loop for a given domain ("src"/"tgt")
# - train_ldm_stage: latent diffusion training loop for a given domain ("ldm_src"/"ldm_tgt")
# - final_sampling_dump: produces final samples from trained diffusion model
from ldm3d.train import train_vae_stage, train_ldm_stage, final_sampling_dump


def build_argparser() -> argparse.ArgumentParser:
    """
    Define all CLI arguments for data paths, training hyperparameters, model sizes,
    diffusion schedule, and utility flags.

    Note: This script uses argparse.Namespace directly throughout main().
    """
    ap = argparse.ArgumentParser()

    # ----------------------------
    # Data paths
    # ----------------------------
    ap.add_argument("--gbm_root", default="/home/j98my/Pre-Processing/prep/gbm_all")
    ap.add_argument("--pdgm_root", default="/home/j98my/Pre-Processing/prep/pdgm_target")
    ap.add_argument("--fewshot", default="/home/j98my/Pre-Processing/prep/pdgm_fewshot.txt")
    ap.add_argument("--outdir", default="/home/j98my/models/runs/ldm_3d_diffuse_glioma")

    # ----------------------------
    # Optional checkpoints (resume / start from pretrained)
    # ----------------------------
    ap.add_argument("--vae_ckpt", default="", help="Optional path to a pretrained VAE .pt")
    ap.add_argument("--unet_ckpt", default="", help="Optional path to a pretrained UNet .pt")
    ap.add_argument("--ema_ckpt", default="", help="Optional path to an EMA shadow .pt")

    # ----------------------------
    # Training lengths (epochs)
    # ----------------------------
    # VAE stage:
    # - src trains on GBM, to learn strong anatomical reconstruction from data-rich domain
    # - tgt optionally fine-tunes on PDGM to better match target domain intensity/style
    ap.add_argument("--epochs_vae_src", type=int, default=200)
    ap.add_argument("--epochs_vae_tgt", type=int, default=50)

    # LDM stage:
    # - ldm_src trains diffusion on GBM latents
    # - ldm_tgt optionally adapts diffusion to PDGM latents
    ap.add_argument("--epochs_ldm_src", type=int, default=300)
    ap.add_argument("--epochs_ldm_tgt", type=int, default=100)

    # ----------------------------
    # Optimization
    # ----------------------------
    ap.add_argument("--lr_vae", type=float, default=1e-4)
    ap.add_argument("--lr_unet", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # ----------------------------
    # VAE loss weights
    # ----------------------------
    # KL regularization weight (latent prior pressure)
    ap.add_argument("--kl_w", type=float, default=1e-4)
    # Reconstruction weight (pixel/voxel fidelity)
    ap.add_argument("--rec_w", type=float, default=1.0)

    # ----------------------------
    # Model sizes (VAE + UNet)
    # ----------------------------
    # z_channels: number of channels in latent tensor (controls compression capacity)
    ap.add_argument("--z_channels", type=int, default=8)
    ap.add_argument("--vae_base", type=int, default=32)
    ap.add_argument("--unet_base", type=int, default=64)

    # t_dim: embedding dimension for diffusion timestep embedding
    ap.add_argument("--t_dim", type=int, default=256)

    # ----------------------------
    # Diffusion schedule
    # ----------------------------
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)

    # EMA decay for smoothing UNet weights during training
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # ----------------------------
    # Latent spatial size (informational / used by some architectures)
    # ----------------------------
    ap.add_argument("--latent_size", type=int, default=28)

    # ----------------------------
    # ControlNet scaling
    # ----------------------------
    # control_scale: global knob for how strongly control tensors influence the UNet
    ap.add_argument("--control_scale", type=float, default=1.0,
                    help="Scale applied to control tensor inside training/sampling.")

    # ----------------------------
    # Latent stats estimation
    # ----------------------------
    # Latent mean/std is estimated from GBM latents produced by current VAE encoder.
    ap.add_argument("--latent_stat_batches", type=int, default=400,
                    help="How many batches to estimate latent mean/std.")

    # ----------------------------
    # Final sampling
    # ----------------------------
    ap.add_argument("--final_dump_n", type=int, default=16)

    # ----------------------------
    # Cached stats control
    # ----------------------------
    ap.add_argument(
        "--force_recompute_latent_stats",
        action="store_true",
        help="Ignore cached latent_stats.pt and recompute mean/std from data.",
    )

    # ----------------------------
    # Training options / regularization knobs
    # ----------------------------
    # If set, diffusion trains on sampled posterior latents (mu + sigma*eps) instead of mu only.
    # This adds stochasticity and can better match sampling-time latents (depending on our VAE usage).
    ap.add_argument("--use_posterior_noise", action="store_true",
                    help="If set: diffusion trains on (mu + std*eps) instead of mu-only.")

    # Reproducible sampling: if >=0, fixes random seed used for sampling.
    ap.add_argument("--sample_seed", type=int, default=-1,
                    help="If >=0, fixes sampling randomness for reproducibility.")

    # Conditional dropout probability for classifier-free guidance-style training
    ap.add_argument("--cond_drop_p", type=float, default=0.1)

    # Guidance scale for sampling (higher => stronger conditioning, but too high can distort)
    ap.add_argument("--guidance_scale", type=float, default=4.0)

    # Probability of applying mask augmentations (used inside train.py)
    ap.add_argument("--mask_aug_p", type=float, default=0.7)

    # Extra weight inside tumor region for diffusion loss (focus model on lesion fidelity)
    ap.add_argument("--tumor_loss_alpha", type=float, default=1.0,
                help="Extra weight inside tumor region for diffusion loss. 0 disables.")

    # Dilate tumor mask in latent space (odd kernel sizes: 3/5/7, etc.)
    ap.add_argument("--tumor_dilate_k", type=int, default=5,
                help="Kernel size for tumor mask dilation in latent space (odd int, e.g., 3/5/7).")

    # Fraction of VAE training over which KL is warmed up (ramps from 0 -> kl_w)
    ap.add_argument("--kl_warmup_frac", type=float, default=0.35)

    # Optional L2 weight decay inside VAE (separate from AdamW weight_decay)
    ap.add_argument("--vae_l2_w", type=float, default=0.0)

    # Probability of applying mask blur augmentation (used inside train.py)
    ap.add_argument("--mask_blur_p", type=float, default=0.3)

    return ap


def maybe_load_ckpt(model: torch.nn.Module, ckpt_path: str, name: str) -> None:
    """
    Load a standard PyTorch state_dict checkpoint into `model` if `ckpt_path` is provided.

    - strict=True ensures architecture matches exactly (safer, but will error if mismatch).
    - map_location="cpu" makes loading independent of GPU availability.
    """
    if not ckpt_path:
        return
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"{name} ckpt not found: {ckpt_path}")
    sd = torch.load(str(p), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    print(f"[LOAD] Loaded {name} ckpt: {ckpt_path}")


def maybe_load_ema(ema: EMA, ema_path: str, model: torch.nn.Module) -> None:
    """
    Load EMA shadow weights and ensure they are placed on the same device/dtype
    as the current model parameters.

    Why this matters:
      - EMA shadows are often saved on CPU.
      - During training/sampling, applying EMA requires tensors to match device/dtype.
    """
    if not ema_path:
        return

    p = Path(ema_path)
    if not p.exists():
        raise FileNotFoundError(f"EMA ckpt not found: {ema_path}")

    shadow = torch.load(str(p), map_location="cpu")

    # Assign the raw shadow dict first (keys: parameter names, values: tensors)
    ema.shadow = shadow

    # Move each shadow tensor to match the corresponding model parameter's device/dtype
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in ema.shadow:
            ema.shadow[name] = ema.shadow[name].to(
                device=param.device,
                dtype=param.dtype,
            )

    print(f"[LOAD] Loaded EMA shadow (moved to {next(model.parameters()).device}): {ema_path}")


def main(args: argparse.Namespace) -> None:
    """
    Orchestrate the full experiment:
      - data loading
      - model initialization
      - checkpoint loading
      - VAE training
      - latent stats estimation (cached/recomputed)
      - diffusion training (src then optional tgt)
      - final sampling dump
      - checkpoint saves
    """
    # Ensure deterministic-ish behavior where possible (PyTorch nondeterminism can still exist on GPU)
    set_seed(args.seed)

    # Make sure output directory exists for saving checkpoints, stats, and sample dumps
    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------
    # 1) Load Source Domain (GBM)
    # ----------------------------
    print("[INIT] Loading Source (GBM)...")
    gbm_ds = VolFolder(args.gbm_root)
    gbm_dl = DataLoader(
        gbm_ds,
        batch_size=args.batch_size,
        shuffle=True,  # shuffle training data for better SGD behavior
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Run an immediate sanity check on data pipeline (saves example volumes, checks normalization, etc.)
    validate_input_data(gbm_dl, args.outdir)

    # ----------------------------
    # 2) Load Target Domain (PDGM few-shot)
    # ----------------------------
    print("[INIT] Loading Target (Diffuse glioma / PDGM)...")
    pdgm_subs = load_subject_list(args.fewshot)

    # If a fewshot list exists, restrict to those subjects; otherwise load all under pdgm_root.
    pdgm_ds = VolFolder(args.pdgm_root, subjects=pdgm_subs if (args.fewshot and pdgm_subs) else None)
    pdgm_dl = DataLoader(
        pdgm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ----------------------------
    # 3) Build Models (VAE + UNet + EMA + DDPM schedule)
    # ----------------------------
    # VAE3D:
    # - encoder produces (mu, logvar) for each latent location
    # - decoder reconstructs the 3D volume from z
    vae = VAE3D(z_channels=args.z_channels, base=args.vae_base).to(DEVICE)

    # NOTE: This block attempts to compute or load latent stats *before* VAE ckpt loading and VAE training.
    # In general, latent stats must correspond to the finalized VAE weights used during diffusion.
    # This script later computes latent stats again after VAE training (the "---- latent stats" block).
    
    #
    # after loading VAE (and before sampling):
    if args.recompute_latent_stats:
        stats = recompute_and_save_latent_stats(
            outdir=args.outdir,
            gbm_root=args.gbm_root,
            vae=vae,
            batches=args.latent_stat_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_posterior_noise=args.use_posterior_noise,
        )
    else:
        stats = torch.load(args.latent_stats, map_location="cpu") if args.latent_stats else \
                torch.load(Path(args.outdir) / "latent_stats.pt", map_location="cpu")

    # Move stats tensors to DEVICE for convenience (training code may expect GPU tensors)
    lat_mean = stats["mean"].to(DEVICE)
    lat_std  = stats["std"].to(DEVICE)

    # UNet3DLatentCond:
    # - operates in latent space (channels = z_channels)
    # - supports conditioning (cond_channels=1) and control tensors (control_channels=3)
    # IMPORTANT: control_channels=3 because train.py builds [mask, edge, dist]
    unet = UNet3DLatentCond(
        z_channels=args.z_channels,
        cond_channels=1,
        control_channels=3,
        base=args.unet_base,
        t_dim=args.t_dim,
        use_controlnet=True,
    ).to(DEVICE)

    # EMA:
    # - tracks a smoothed copy of UNet parameters over training steps
    # - often improves sampling quality / stability at inference time
    ema = EMA(unet, decay=args.ema_decay)

    # DDPM schedule:
    # - defines betas/alphas over T timesteps for forward diffusion and reverse denoising
    ddpm = LatentDDPM(T=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)

    # ----------------------------
    # 4) Optional checkpoint loading
    # ----------------------------
    # Loading order here:
    # - VAE weights
    # - UNet weights
    # - EMA shadow weights (must match UNet parameter names)
    maybe_load_ckpt(vae, args.vae_ckpt, "VAE")
    maybe_load_ckpt(unet, args.unet_ckpt, "UNet")
    maybe_load_ema(ema, args.ema_ckpt, unet)

    # ----------------------------
    # 5) Optimizers
    # ----------------------------
    # AdamW for both VAE and UNet (weight_decay provides decoupled regularization)
    opt_vae = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr_vae,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    opt_unet = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr_unet,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # ----------------------------
    # 6) VAE Training Stages
    # ----------------------------
    # Stage "src": train VAE on GBM for stable reconstruction and robust anatomical latent space.
    train_vae_stage("src", gbm_dl, vae, opt_vae, args.epochs_vae_src, args.outdir,
                    kl_w=args.kl_w, rec_w=args.rec_w, kl_warmup_frac=args.kl_warmup_frac,
                    l2_w=args.vae_l2_w,)

    # Stage "tgt": optional fine-tuning on PDGM few-shot for target-domain adaptation.
    if args.epochs_vae_tgt > 0:
        train_vae_stage("tgt", pdgm_dl, vae, opt_vae, args.epochs_vae_tgt, args.outdir,
                        kl_w=args.kl_w, rec_w=args.rec_w, kl_warmup_frac=args.kl_warmup_frac,
                        l2_w=args.vae_l2_w,)

    # Save VAE checkpoint only if we actually trained it (epochs > 0)
    if args.epochs_vae_src > 0 or args.epochs_vae_tgt > 0:
        torch.save(vae.state_dict(), Path(args.outdir) / "vae_final.pt")

    # ----------------------------
    # 7) Latent Stats (mean/std) for normalization
    # ----------------------------
    # This is the "real" latent stats block used for diffusion training below.
    # It tries to load cached stats unless forcing recompute; otherwise estimates from GBM via the current VAE.
    stats = None if args.force_recompute_latent_stats else maybe_load_latent_stats(args.outdir)
    if stats is None:
        if args.force_recompute_latent_stats:
            print("[LDM] Forcing latent stat recomputation.")
        stats = estimate_latent_stats(
            gbm_dl,
            vae,
            max_batches=getattr(args, "latent_stat_batches", 200),
            use_posterior_noise=args.use_posterior_noise,
        )
        torch.save(stats, Path(args.outdir) / "latent_stats.pt")
        print(f"[SAVE] Saved latent stats -> {Path(args.outdir) / 'latent_stats.pt'}")

    # Note: these are left on CPU here (estimate_latent_stats returns CPU tensors).
    # Training code may move them as needed.
    lat_mean = stats["mean"]
    lat_std = stats["std"]

    # ----------------------------
    # 8) Diffusion Training Stages (LDM)
    # ----------------------------
    # Stage "ldm_src": train diffusion on GBM latents (data-rich).
    train_ldm_stage(
        "ldm_src", gbm_dl, vae, unet, ema, ddpm, opt_unet,
        args.epochs_ldm_src, args.outdir, lat_mean, lat_std, args
    )

    # Stage "ldm_tgt": optional adaptation on PDGM few-shot latents (data-scarce).
    if args.epochs_ldm_tgt > 0:
        train_ldm_stage(
            "ldm_tgt", pdgm_dl, vae, unet, ema, ddpm, opt_unet,
            args.epochs_ldm_tgt, args.outdir, lat_mean, lat_std, args
        )

    # ----------------------------
    # 9) Final Sampling Dump
    # ----------------------------
    # Produces qualitative samples for both domains (depending on how final_sampling_dump is implemented).
    # Typically uses EMA weights for sampling (more stable).
    final_sampling_dump(
        gbm_dl=gbm_dl,
        pdgm_dl=pdgm_dl,
        vae=vae,
        unet=unet,
        ema=ema,
        ddpm=ddpm,
        outdir=args.outdir,
        lat_mean=lat_mean,
        lat_std=lat_std,
        args=args,
    )

    # ----------------------------
    # 10) Save Final Checkpoints
    # ----------------------------
    # Save final UNet weights and EMA shadow weights for later sampling/evaluation.
    torch.save(unet.state_dict(), Path(args.outdir) / "unet_final.pt")
    torch.save(ema.shadow, Path(args.outdir) / "ema_final.pt")
    print("--- DONE ---")


if __name__ == "__main__":
    # Standard CLI entry:
    #   - parse args
    #   - run main training pipeline
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
