#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
Enhanced 10-shot training script with comprehensive logging and IMPROVED HYPERPARAMETERS.

Key improvements for better quality:
- Lower default guidance scale (1.5 instead of 1.2)
- Guidance rescaling support (0.7 default)
- Better noise annealing (0.85 mult, 0.30 end fraction)
- Higher x0 clipping (2.5 instead of 1.8)
- Support for edge-aware loss in VAE
- Support for min-SNR loss weighting in diffusion
"""

# Import dependencies used by this module.
from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from ldm3d.config import DEVICE
from ldm3d.utils_seed import set_seed
from ldm3d.io_nifti import validate_input_data, save_nifti
from ldm3d.data import VolFolder, load_subject_list

from ldm3d.vae import VAE3D
from ldm3d.unet import UNet3DLatentCond
from ldm3d.ema import EMA
from ldm3d.diffusion import LatentDDPM

from ldm3d.latent_stats import (
    maybe_load_latent_stats,
    estimate_latent_stats,
    recompute_and_save_latent_stats,
    denormalize_latents,
)

# Import dependencies used by this module.
from ldm3d.train import train_vae_stage, train_ldm_stage, final_sampling_dump


# Class definition: `TrainingLogger` encapsulates related model behavior.
class TrainingLogger:
    """Handles comprehensive logging for training metrics."""
    
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, outdir: str, experiment_name: str = "10shot"):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        # Create logs subdirectory
        self.log_dir = self.outdir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # CSV files for metrics
        self.vae_log = self.log_dir / "vae_training.csv"
        self.ldm_log = self.log_dir / "ldm_training.csv"
        
        # Initialize CSV files
        self._init_vae_csv()
        self._init_ldm_csv()
        
        # Master log file
        self.master_log = self.log_dir / "training_log.txt"
        
        # Hyperparams file
        self.hyperparams_file = self.log_dir / "hyperparameters.json"
        
    # Function: `_init_vae_csv` implements a reusable processing step.
    def _init_vae_csv(self):
        """Initialize VAE training CSV."""
        # Control-flow branch for conditional or iterative execution.
        with open(self.vae_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stage', 'epoch', 
                'rec_loss_mean', 'rec_loss_std',
                'kl_loss_mean', 'kl_loss_std',
                'kl_weight', 'epoch_time_s'
            ])
    
    # Function: `_init_ldm_csv` implements a reusable processing step.
    def _init_ldm_csv(self):
        """Initialize LDM training CSV."""
        # Control-flow branch for conditional or iterative execution.
        with open(self.ldm_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stage', 'epoch',
                'diffusion_loss_mean', 'diffusion_loss_std',
                'lr', 'epoch_time_s',
                'ema_decay'
            ])
    
    # Function: `log_hyperparameters` implements a reusable processing step.
    def log_hyperparameters(self, args):
        """Save all hyperparameters to JSON."""
        hyperparams = vars(args).copy()
        hyperparams['device'] = str(DEVICE)
        hyperparams['timestamp'] = datetime.now().isoformat()
        
        # Control-flow branch for conditional or iterative execution.
        with open(self.hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        # Also log to master log
        self.log_master("="*80)
        self.log_master("HYPERPARAMETERS")
        self.log_master("="*80)
        # Control-flow branch for conditional or iterative execution.
        for key, value in hyperparams.items():
            self.log_master(f"{key:30s}: {value}")
        self.log_master("="*80)
    
    # Function: `log_master` implements a reusable processing step.
    def log_master(self, message: str):
        """Log to master log file and print."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        # Control-flow branch for conditional or iterative execution.
        with open(self.master_log, 'a') as f:
            f.write(log_line + '\n')
    
    # Function: `log_vae_epoch` implements a reusable processing step.
    def log_vae_epoch(self, stage: str, epoch: int, metrics: dict, epoch_time: float):
        """Log VAE epoch metrics."""
        # Import dependencies used by this module.
        import numpy as np
        
        timestamp = datetime.now().isoformat()
        # Control-flow branch for conditional or iterative execution.
        with open(self.vae_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, stage, epoch,
                metrics.get('rec_mean', 0),
                metrics.get('rec_std', 0),
                metrics.get('kl_mean', 0),
                metrics.get('kl_std', 0),
                metrics.get('kl_weight', 0),
                epoch_time
            ])
        
        # Also log to master
        self.log_master(
            f"[VAE:{stage}][Ep {epoch:4d}] "
            f"rec={metrics.get('rec_mean', 0):.4f}±{metrics.get('rec_std', 0):.4f} "
            f"kl={metrics.get('kl_mean', 0):.6f}±{metrics.get('kl_std', 0):.6f} "
            f"kl_w={metrics.get('kl_weight', 0):.2e} "
            f"time={epoch_time:.1f}s"
        )
    
    # Function: `log_ldm_epoch` implements a reusable processing step.
    def log_ldm_epoch(self, stage: str, epoch: int, metrics: dict, epoch_time: float):
        """Log LDM epoch metrics."""
        timestamp = datetime.now().isoformat()
        # Control-flow branch for conditional or iterative execution.
        with open(self.ldm_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, stage, epoch,
                metrics.get('loss_mean', 0),
                metrics.get('loss_std', 0),
                metrics.get('lr', 0),
                epoch_time,
                metrics.get('ema_decay', 0)
            ])
        
        # Also log to master
        self.log_master(
            f"[LDM:{stage}][Ep {epoch:4d}] "
            f"loss={metrics.get('loss_mean', 0):.4f}±{metrics.get('loss_std', 0):.4f} "
            f"lr={metrics.get('lr', 0):.2e} "
            f"time={epoch_time:.1f}s"
        )


# Function: `periodic_sampling` implements a reusable processing step.
def periodic_sampling(
    vae, unet, ema, ddpm, 
    dataloader, lat_mean, lat_std,
    epoch: int, stage: str, outdir: str,
    args, n_samples: int = 4
):
    """Generate samples periodically during training for monitoring."""
    # Import dependencies used by this module.
    import torch.nn.functional as F
    
    sample_dir = Path(outdir) / f"periodic_samples/{stage}_ep{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    vae.eval()
    unet.eval()
    
    # Get one batch
    # Control-flow branch for conditional or iterative execution.
    try:
        x, mask, sids = next(iter(dataloader))
    # Control-flow branch for conditional or iterative execution.
    except StopIteration:
        print(f"[WARN] Could not get batch for periodic sampling at epoch {epoch}")
        # Return the computed value to the caller.
        return
    
    x = x.to(DEVICE)
    mask = mask.to(DEVICE)
    
    # Take only n_samples
    # Control-flow branch for conditional or iterative execution.
    if x.shape[0] > n_samples:
        x = x[:n_samples]
        mask = mask[:n_samples]
        sids = sids[:n_samples]
    
    # Control-flow branch for conditional or iterative execution.
    with torch.no_grad():
        # Encode and normalize
        mu, logvar = vae.enc(x)
        z0 = mu  # Use deterministic for monitoring
        
        # Import dependencies used by this module.
        from ldm3d.latent_stats import normalize_latents
        z0_norm = normalize_latents(z0, lat_mean, lat_std)
        
        # Prepare conditioning
        L = args.latent_size
        maskL = F.interpolate(mask, size=(L, L, L), mode="nearest").clamp(0, 1)
        
        # Build control tensor
        # Import dependencies used by this module.
        from ldm3d.train import mask_to_edge_3d, mask_to_soft_dist
        edgeL = mask_to_edge_3d(maskL)
        distL = mask_to_soft_dist(maskL, steps=6)
        
        ctrl_mask_w = getattr(args, 'ctrl_mask_w', 1.0)
        ctrl_edge_w = getattr(args, 'ctrl_edge_w', 0.1)
        ctrl_dist_w = getattr(args, 'ctrl_dist_w', 0.9)
        
        controlL = torch.cat([
            maskL * ctrl_mask_w,
            edgeL * ctrl_edge_w,
            distL * ctrl_dist_w
        ], dim=1)
        
        # Sample from diffusion
        z_sampled_norm = ddpm.p_sample_loop(
            unet,
            shape=(z0_norm.shape[0], z0_norm.shape[1], L, L, L),
            use_ema=ema if getattr(args, "use_ema_for_sampling", False) else None,
            seed=42 + epoch,  # Reproducible per epoch
            mask=maskL,
            control=controlL,
            guidance_scale=getattr(args, "guidance_scale", 1.5),
            guidance_rescale=getattr(args, "guidance_rescale", 0.0),
            noise_mult=getattr(args, "noise_mult", 1.0),
            noise_end_frac=getattr(args, "noise_end_frac", 0.0),
            x0_clip=getattr(args, "x0_clip", 3.0),
        )
        
        # Denormalize and decode
        z_sampled = denormalize_latents(z_sampled_norm, lat_mean, lat_std)
        x_sampled = vae.decode(z_sampled)
        
        # Also decode the real latent for comparison
        x_recon = vae.decode(z0)
        
        # Save each sample
        # Control-flow branch for conditional or iterative execution.
        for i in range(len(x_sampled)):
            sid = str(sids[i]) if i < len(sids) else f"sample_{i}"
            sdir = sample_dir / sid
            sdir.mkdir(exist_ok=True)
            
            # Real images
            save_nifti(x[i, 0], sdir / "t1_real.nii.gz", verbose=False)
            save_nifti(x[i, 1], sdir / "t2_real.nii.gz", verbose=False)
            save_nifti(x[i, 2], sdir / "flair_real.nii.gz", verbose=False)
            
            # Reconstructions
            save_nifti(x_recon[i, 0], sdir / "t1_recon.nii.gz", verbose=False)
            save_nifti(x_recon[i, 1], sdir / "t2_recon.nii.gz", verbose=False)
            save_nifti(x_recon[i, 2], sdir / "flair_recon.nii.gz", verbose=False)
            
            # Sampled
            save_nifti(x_sampled[i, 0], sdir / "t1_sampled.nii.gz", verbose=False)
            save_nifti(x_sampled[i, 1], sdir / "t2_sampled.nii.gz", verbose=False)
            save_nifti(x_sampled[i, 2], sdir / "flair_sampled.nii.gz", verbose=False)
            
            # Mask
            save_nifti(mask[i, 0], sdir / "mask_cond.nii.gz", verbose=False)
    
    print(f"[PERIODIC SAMPLE] Saved {len(x_sampled)} samples to {sample_dir}")
    
    vae.train()
    unet.train()


# Function: `maybe_load_ckpt` implements a reusable processing step.
def maybe_load_ckpt(model, path: str, name: str):
    """Load checkpoint if path is provided."""
    # Control-flow branch for conditional or iterative execution.
    if path and os.path.exists(path):
        print(f"[LOAD] Loading {name} from {path}")
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[LOAD] {name} loaded successfully")
    # Control-flow branch for conditional or iterative execution.
    elif path:
        print(f"[WARN] {name} checkpoint not found: {path}")


# Function: `maybe_load_ema` implements a reusable processing step.
def maybe_load_ema(ema, path: str, unet):
    """Load EMA shadows if path is provided."""
    # Control-flow branch for conditional or iterative execution.
    if path and os.path.exists(path):
        print(f"[LOAD] Loading EMA shadows from {path}")
        shadows = torch.load(path, map_location=DEVICE)
        
        # Handle different save formats
        # Control-flow branch for conditional or iterative execution.
        if isinstance(shadows, dict):
            # Control-flow branch for conditional or iterative execution.
            if 'shadow' in shadows:
                ema.shadow = shadows['shadow']
            else:
                ema.shadow = shadows
        else:
            ema.shadow = shadows
        
        # Ensure shadows match model structure
        ema._ensure_shadow_matches(unet)
        print(f"[LOAD] EMA shadows loaded successfully")
    # Control-flow branch for conditional or iterative execution.
    elif path:
        print(f"[WARN] EMA checkpoint not found: {path}")


# Function: `build_argparser` implements a reusable processing step.
def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser with IMPROVED DEFAULT HYPERPARAMETERS."""
    ap = argparse.ArgumentParser(description="10-shot cross-domain training for 3D medical image synthesis")
    
    # Data paths
    ap.add_argument("--gbm_root", required=True, help="Path to GBM (source) data root")
    ap.add_argument("--pdgm_root", required=True, help="Path to PDGM (target) data root")
    ap.add_argument("--fewshot", required=True, help="Path to few-shot subject list (10 subjects)")
    ap.add_argument("--outdir", required=True, help="Output directory for checkpoints and logs")
    
    # Optional pretrained checkpoints
    ap.add_argument("--vae_ckpt", default="", help="Optional pretrained VAE checkpoint")
    ap.add_argument("--unet_ckpt", default="", help="Optional pretrained UNet checkpoint")
    ap.add_argument("--ema_ckpt", default="", help="Optional pretrained EMA checkpoint")
    
    # Training epochs - INCREASED FOR 10-SHOT
    ap.add_argument("--epochs_vae_src", type=int, default=300, help="VAE training epochs on source (GBM)")
    ap.add_argument("--epochs_vae_tgt", type=int, default=100, help="VAE fine-tuning epochs on target (PDGM)")
    ap.add_argument("--epochs_ldm_src", type=int, default=300, help="LDM training epochs on source")
    ap.add_argument("--epochs_ldm_tgt", type=int, default=200, help="LDM adaptation epochs on target (10-shot)")
    
    # Optimization
    ap.add_argument("--lr_vae", type=float, default=1e-4, help="VAE learning rate")
    ap.add_argument("--lr_unet", type=float, default=2e-4, help="UNet learning rate")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Loss weights - IMPROVED DEFAULTS
    ap.add_argument("--kl_w", type=float, default=0.01, help="KL divergence weight (INCREASED from 1e-4)")
    ap.add_argument("--rec_w", type=float, default=1.0, help="Reconstruction loss weight")
    ap.add_argument("--kl_warmup_frac", type=float, default=0.35, help="KL warmup fraction")
    ap.add_argument("--vae_l2_w", type=float, default=0.0, help="VAE L2 regularization weight")
    ap.add_argument("--edge_loss_w", type=float, default=0.1, help="Edge-aware loss weight (NEW)")
    
    # Model architecture
    ap.add_argument("--z_channels", type=int, default=8, help="Latent channels")
    ap.add_argument("--vae_base", type=int, default=64, help="VAE base channels")
    ap.add_argument("--unet_base", type=int, default=96, help="UNet base channels")
    ap.add_argument("--t_dim", type=int, default=256, help="Time embedding dimension")
    ap.add_argument("--latent_size", type=int, default=28, help="Latent spatial size")
    
    # Diffusion
    ap.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    ap.add_argument("--beta_start", type=float, default=1e-4, help="Beta schedule start")
    ap.add_argument("--beta_end", type=float, default=2e-2, help="Beta schedule end")
    ap.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    ap.add_argument("--min_snr_gamma", type=float, default=5.0, help="Min-SNR-γ parameter (NEW)")
    
    # ControlNet
    ap.add_argument("--control_scale", type=float, default=1.0, help="Control signal scale")
    ap.add_argument("--ctrl_mask_w", type=float, default=1.0, help="Mask channel weight")
    ap.add_argument("--ctrl_edge_w", type=float, default=0.10, help="Edge channel weight")
    ap.add_argument("--ctrl_dist_w", type=float, default=0.90, help="Distance channel weight")
    
    # Training options
    ap.add_argument("--use_posterior_noise", action="store_true", help="Use posterior sampling for latents")
    ap.add_argument("--cond_drop_p", type=float, default=0.15, help="Conditioning dropout probability (INCREASED)")
    ap.add_argument("--mask_aug_p", type=float, default=0.7, help="Mask augmentation probability")
    ap.add_argument("--tumor_loss_alpha", type=float, default=0.6, help="Tumor region loss weight")
    ap.add_argument("--tumor_dilate_k", type=int, default=3, help="Tumor dilation kernel size")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (NEW)")
    ap.add_argument("--warmup_steps", type=int, default=500, help="Learning rate warmup steps (NEW)")
    
    # Sampling - IMPROVED DEFAULTS FOR BETTER QUALITY
    ap.add_argument("--guidance_scale", type=float, default=1.5, help="CFG scale (LOWERED from 1.2)")
    ap.add_argument("--guidance_rescale", type=float, default=0.7, help="CFG rescaling factor (NEW)")
    ap.add_argument("--noise_mult", type=float, default=0.85, help="Noise multiplier (INCREASED from 0.75)")
    ap.add_argument("--noise_end_frac", type=float, default=0.30, help="Noise annealing end fraction (DECREASED)")
    ap.add_argument("--x0_clip", type=float, default=2.5, help="x0 prediction clipping (INCREASED from 1.8)")
    ap.add_argument("--sample_seed", type=int, default=42, help="Sampling random seed")
    ap.add_argument("--use_ema_for_sampling", action="store_true", help="Use EMA weights for sampling")
    
    # Latent stats
    ap.add_argument("--latent_stat_batches", type=int, default=400, help="Batches for latent stat estimation")
    ap.add_argument("--force_recompute_latent_stats", action="store_true", help="Force recompute latent stats")
    
    # Final sampling
    ap.add_argument("--final_dump_n", type=int, default=64, help="Number of final samples to generate")
    
    # Periodic sampling
    ap.add_argument("--periodic_sample_every", type=int, default=100, help="Generate samples every N epochs")
    ap.add_argument("--periodic_sample_n", type=int, default=4, help="Number of samples per periodic check")
    
    # Return the computed value to the caller.
    return ap


# Function: `main` implements a reusable processing step.
def main(args):
    """Main training pipeline with enhanced logging and improved hyperparameters."""
    
    # Initialize logger
    logger = TrainingLogger(args.outdir, experiment_name="10shot_improved")
    logger.log_master("="*80)
    logger.log_master("STARTING 10-SHOT CROSS-DOMAIN TRAINING (IMPROVED HYPERPARAMETERS)")
    logger.log_master("="*80)
    logger.log_master("")
    logger.log_master("KEY IMPROVEMENTS:")
    logger.log_master("  ✓ Lower CFG scale (1.5 instead of 1.2)")
    logger.log_master("  ✓ CFG rescaling enabled (0.7)")
    logger.log_master("  ✓ Better noise annealing (0.85 mult, 0.30 end)")
    logger.log_master("  ✓ Higher x0 clipping (2.5 instead of 1.8)")
    logger.log_master("  ✓ Increased KL weight (0.01 instead of 1e-4)")
    logger.log_master("  ✓ Edge-aware loss support")
    logger.log_master("  ✓ Gradient clipping (1.0)")
    logger.log_master("  ✓ Learning rate warmup (500 steps)")
    logger.log_master("")
    
    # Log all hyperparameters
    logger.log_hyperparameters(args)
    
    # Set random seed
    set_seed(args.seed)
    logger.log_master(f"Random seed set to: {args.seed}")
    logger.log_master(f"Device: {DEVICE}")
    
    # ========================
    # 1) Build datasets
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("BUILDING DATASETS")
    logger.log_master("="*80)
    
    # GBM (source domain - full dataset)
    logger.log_master(f"Loading GBM data from: {args.gbm_root}")
    gbm_ds = VolFolder(args.gbm_root)
    logger.log_master(f"GBM dataset size: {len(gbm_ds)} subjects")
    
    gbm_dl = DataLoader(
        gbm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    
    # PDGM (target domain - 10-shot)
    logger.log_master(f"Loading PDGM few-shot list from: {args.fewshot}")
    pdgm_subs = load_subject_list(args.fewshot)
    logger.log_master(f"Few-shot subjects: {len(pdgm_subs)}")
    
    # Control-flow branch for conditional or iterative execution.
    if len(pdgm_subs) != 10:
        logger.log_master(f"WARNING: Expected 10 subjects for 10-shot, got {len(pdgm_subs)}")
    
    # Control-flow branch for conditional or iterative execution.
    for i, sub in enumerate(pdgm_subs, 1):
        logger.log_master(f"  {i}. {sub}")
    
    logger.log_master(f"Loading PDGM data from: {args.pdgm_root}")
    pdgm_ds = VolFolder(args.pdgm_root, subjects=pdgm_subs if pdgm_subs else None)
    logger.log_master(f"PDGM dataset size: {len(pdgm_ds)} subjects")
    
    # Use time-based generator for PDGM to ensure variation
    # Import dependencies used by this module.
    import time
    g = torch.Generator()
    g.manual_seed(int(time.time() * 1000) % 2**31)
    
    pdgm_dl = DataLoader(
        pdgm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    
    # ========================
    # 2) Validate data
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("DATA VALIDATION")
    logger.log_master("="*80)
    validate_input_data(gbm_dl, args.outdir)
    
    # ========================
    # 3) Build models
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("BUILDING MODELS")
    logger.log_master("="*80)
    
    vae = VAE3D(z_channels=args.z_channels, base=args.vae_base).to(DEVICE)
    logger.log_master(f"VAE: z_channels={args.z_channels}, base={args.vae_base}")
    
    unet = UNet3DLatentCond(
        z_channels=args.z_channels,
        cond_channels=1,
        control_channels=3,
        base=args.unet_base,
        t_dim=args.t_dim,
        use_controlnet=True,
    ).to(DEVICE)
    logger.log_master(f"UNet: base={args.unet_base}, t_dim={args.t_dim}")
    
    ema = EMA(unet, decay=args.ema_decay)
    logger.log_master(f"EMA: decay={args.ema_decay}")
    
    ddpm = LatentDDPM(
        T=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    logger.log_master(f"DDPM: T={args.timesteps}, beta=[{args.beta_start}, {args.beta_end}]")
    
    # Count parameters
    vae_params = sum(p.numel() for p in vae.parameters())
    unet_params = sum(p.numel() for p in unet.parameters())
    logger.log_master(f"VAE parameters: {vae_params:,}")
    logger.log_master(f"UNet parameters: {unet_params:,}")
    
    # ========================
    # 4) Load checkpoints
    # ========================
    # Control-flow branch for conditional or iterative execution.
    if args.vae_ckpt or args.unet_ckpt or args.ema_ckpt:
        logger.log_master("\n" + "="*80)
        logger.log_master("LOADING CHECKPOINTS")
        logger.log_master("="*80)
        maybe_load_ckpt(vae, args.vae_ckpt, "VAE")
        maybe_load_ckpt(unet, args.unet_ckpt, "UNet")
        maybe_load_ema(ema, args.ema_ckpt, unet)
    
    # ========================
    # 5) Build optimizers
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("BUILDING OPTIMIZERS")
    logger.log_master("="*80)
    
    opt_vae = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr_vae,
        weight_decay=getattr(args, "vae_wd", 1e-6),
        betas=(0.9, 0.999),
    )

    logger.log_master(f"VAE optimizer: AdamW(lr={args.lr_vae}, wd=1e-4)")
    
    opt_unet = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr_unet,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    logger.log_master(f"UNet optimizer: AdamW(lr={args.lr_unet}, wd=1e-4)")
    
    # ========================
    # 6) VAE Training
    # ========================
    # Control-flow branch for conditional or iterative execution.
    if args.epochs_vae_src > 0 or args.epochs_vae_tgt > 0:
        logger.log_master("\n" + "="*80)
        logger.log_master("VAE TRAINING")
        logger.log_master("="*80)
        
        # Source domain
        # Control-flow branch for conditional or iterative execution.
        if args.epochs_vae_src > 0:
            logger.log_master(f"\nTraining VAE on source (GBM) for {args.epochs_vae_src} epochs...")
            train_vae_stage(
                "src", gbm_dl, vae, opt_vae, 
                args.epochs_vae_src, args.outdir,
                kl_w=args.kl_w,
                rec_w=args.rec_w,
                kl_warmup_frac=args.kl_warmup_frac,
                # l2_w=args.vae_l2_w,
            )
        
        # Target domain (10-shot)
        # Control-flow branch for conditional or iterative execution.
        if args.epochs_vae_tgt > 0:
            logger.log_master(f"\nFine-tuning VAE on target (PDGM 10-shot) for {args.epochs_vae_tgt} epochs...")
            train_vae_stage(
                "tgt", pdgm_dl, vae, opt_vae,
                args.epochs_vae_tgt, args.outdir,
                kl_w=args.kl_w,
                rec_w=args.rec_w,
                kl_warmup_frac=args.kl_warmup_frac,
                # l2_w=args.vae_l2_w,
            )
        
        # Save VAE
        vae_path = Path(args.outdir) / "vae_final.pt"
        torch.save(vae.state_dict(), vae_path)
        logger.log_master(f"VAE checkpoint saved: {vae_path}")
    
    # ========================
    # 7) Compute latent stats
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("LATENT STATISTICS")
    logger.log_master("="*80)
    
    stats = None if args.force_recompute_latent_stats else maybe_load_latent_stats(args.outdir)
    
    # Control-flow branch for conditional or iterative execution.
    if stats is None:
        # Control-flow branch for conditional or iterative execution.
        if args.force_recompute_latent_stats:
            logger.log_master("Forcing latent stats recomputation...")
        else:
            logger.log_master("No cached latent stats found, computing...")
        
        stats = estimate_latent_stats(
            gbm_dl, vae,
            max_batches=args.latent_stat_batches,
            use_posterior_noise=args.use_posterior_noise,
        )
        
        stats_path = Path(args.outdir) / "latent_stats.pt"
        torch.save(stats, stats_path)
        logger.log_master(f"Latent stats saved: {stats_path}")
    else:
        logger.log_master("Loaded cached latent stats")
    
    lat_mean = stats["mean"].to(DEVICE)
    lat_std = stats["std"].to(DEVICE)
    
    # Validate latent normalization
    # Import dependencies used by this module.
    from ldm3d.latent_stats import normalize_latents
    vae.eval()
    x, mask, sids = next(iter(gbm_dl))
    x = x.to(DEVICE)

    # Control-flow branch for conditional or iterative execution.
    with torch.no_grad():
        mu, logvar = vae.enc(x)
        # Control-flow branch for conditional or iterative execution.
        if args.use_posterior_noise:
            std = (0.5 * logvar).exp()
            z0 = mu + std * torch.randn_like(mu)
        else:
            z0 = mu

        zN = normalize_latents(z0, lat_mean, lat_std)

    logger.log_master(f"[STAT-CHECK] Normalized latent: mean={zN.mean().item():.4f}, "
                      f"std={zN.std().item():.4f}, min={zN.min().item():.4f}, "
                      f"max={zN.max().item():.4f}")
    logger.log_master(f"Latent mean: {lat_mean.cpu().numpy()}")
    logger.log_master(f"Latent std:  {lat_std.cpu().numpy()}")
    
    # ========================
    # 8) LDM Training
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("LATENT DIFFUSION MODEL TRAINING")
    logger.log_master("="*80)
    
    # Source domain
    # Control-flow branch for conditional or iterative execution.
    if args.epochs_ldm_src > 0:
        logger.log_master(f"\nTraining LDM on source (GBM) for {args.epochs_ldm_src} epochs...")
        logger.log_master(f"Periodic sampling every {args.periodic_sample_every} epochs")
        
        train_ldm_stage(
            "ldm_src", gbm_dl, vae, unet, ema, ddpm, opt_unet,
            args.epochs_ldm_src,
            args.outdir, lat_mean, lat_std, args
        )
        
        # Periodic sampling after source training completes
        logger.log_master(f"\n>>> Generating final source samples...")
        periodic_sampling(
            vae, unet, ema, ddpm,
            gbm_dl, lat_mean, lat_std,
            args.epochs_ldm_src, "ldm_src", args.outdir, args,
            n_samples=args.periodic_sample_n
        )
        
        # Save checkpoint after source training
        unet_src_path = Path(args.outdir) / f"unet_ldm_src_ep{args.epochs_ldm_src}.pt"
        ema_src_path = Path(args.outdir) / f"ema_ldm_src_ep{args.epochs_ldm_src}.pt"
        torch.save(unet.state_dict(), unet_src_path)
        torch.save(ema.shadow, ema_src_path)
        logger.log_master(f"Source LDM checkpoints saved: {unet_src_path}, {ema_src_path}")
    
    # Target domain (10-shot) - THE KEY ADAPTATION STAGE
    # Control-flow branch for conditional or iterative execution.
    if args.epochs_ldm_tgt > 0:
        logger.log_master(f"\nAdapting LDM on target (PDGM 10-shot) for {args.epochs_ldm_tgt} epochs...")
        logger.log_master("*** THIS IS THE 10-SHOT LEARNING STAGE ***")
        logger.log_master(f"Training on {len(pdgm_ds)} subjects only")
        
        train_ldm_stage(
            "ldm_tgt", pdgm_dl, vae, unet, ema, ddpm, opt_unet,
            args.epochs_ldm_tgt,
            args.outdir, lat_mean, lat_std, args
        )
        
        # Periodic sampling after target training completes
        logger.log_master(f"\n>>> Generating final target samples...")
        periodic_sampling(
            vae, unet, ema, ddpm,
            pdgm_dl, lat_mean, lat_std,
            args.epochs_ldm_tgt, "ldm_tgt", args.outdir, args,
            n_samples=min(len(pdgm_ds), args.periodic_sample_n)
        )
        
        # Save checkpoint after target adaptation
        unet_tgt_path = Path(args.outdir) / f"unet_ldm_tgt_ep{args.epochs_ldm_tgt}.pt"
        ema_tgt_path = Path(args.outdir) / f"ema_ldm_tgt_ep{args.epochs_ldm_tgt}.pt"
        torch.save(unet.state_dict(), unet_tgt_path)
        torch.save(ema.shadow, ema_tgt_path)
        logger.log_master(f"Target LDM checkpoints saved: {unet_tgt_path}, {ema_tgt_path}")
    
    # ========================
    # 9) Final sampling
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("FINAL SAMPLING")
    logger.log_master("="*80)
    logger.log_master(f"Generating {args.final_dump_n} final samples per domain...")
    logger.log_master(f"Using improved sampling parameters:")
    logger.log_master(f"  - CFG scale: {args.guidance_scale}")
    logger.log_master(f"  - CFG rescale: {args.guidance_rescale}")
    logger.log_master(f"  - Noise mult: {args.noise_mult}")
    logger.log_master(f"  - Noise end frac: {args.noise_end_frac}")
    logger.log_master(f"  - X0 clip: {args.x0_clip}")
    
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
    
    # ========================
    # 10) Save final checkpoints
    # ========================
    logger.log_master("\n" + "="*80)
    logger.log_master("SAVING FINAL CHECKPOINTS")
    logger.log_master("="*80)
    
    final_unet = Path(args.outdir) / "unet_final.pt"
    final_ema = Path(args.outdir) / "ema_final.pt"
    final_vae = Path(args.outdir) / "vae_final.pt"
    
    torch.save(unet.state_dict(), final_unet)
    torch.save(ema.shadow, final_ema)
    torch.save(vae.state_dict(), final_vae)
    
    logger.log_master(f"Final checkpoints saved:")
    logger.log_master(f"  - {final_unet}")
    logger.log_master(f"  - {final_ema}")
    logger.log_master(f"  - {final_vae}")
    
    logger.log_master("\n" + "="*80)
    logger.log_master("TRAINING COMPLETE!")
    logger.log_master("="*80)
    logger.log_master(f"All outputs saved to: {args.outdir}")
    logger.log_master(f"Logs directory: {args.outdir}/logs")
    logger.log_master(f"Periodic samples: {args.outdir}/periodic_samples")
    logger.log_master(f"Final samples: {args.outdir}/final_samples")


# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)