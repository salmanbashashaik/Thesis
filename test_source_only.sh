#!/bin/bash
# =============================================================================
# SOURCE-ONLY SAMPLING EXPERIMENT
#
# Goal:
#   Generate synthetic GBM and PDGM MRI volumes using a diffusion model that
#   was trained ONLY on the source domain (GBM), without any target (PDGM)
#   fine-tuning of the diffusion UNet.
#
# Motivation / Hypothesis:
#   Few-shot fine-tuning on PDGM may cause mode collapse and reduced diversity.
#   By freezing diffusion dynamics at the source-trained stage, we test whether
#   representation-level adaptation (VAE + latent stats) alone is sufficient.
#
# Key properties of this experiment:
#   - Diffusion UNet: source-trained only (epoch 300, pre-target)
#   - EMA weights: source-trained only
#   - VAE + latent stats: loaded from prior run (few-shot–aware representation)
#   - Conditioning: explicit anatomical control via tumor masks (ControlNet)
#
# This isolates diffusion dynamics from few-shot fine-tuning effects.
# =============================================================================

set -e  # Exit immediately if any command fails (fail-fast behavior)

# -----------------------------------------------------------------------------
# Paths to previous experiment artifacts and output location
# -----------------------------------------------------------------------------

# Directory containing trained checkpoints and latent statistics
PREV_RUN=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_IMPROVED

# Output directory for this source-only sampling experiment
OUTDIR=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_SOURCE_ONLY

# -----------------------------------------------------------------------------
# Runtime logging
# -----------------------------------------------------------------------------

echo "============================================================"
echo "TEST: SOURCE MODEL ONLY (NO TARGET FINE-TUNING)"
echo "============================================================"
echo ""
echo "Using source-trained diffusion model to generate GBM + PDGM samples"
echo "Purpose: evaluate whether target fine-tuning degrades diversity"
echo ""

# -----------------------------------------------------------------------------
# Prepare output directory structure
# -----------------------------------------------------------------------------

# Separate folders for GBM and PDGM synthetic outputs
mkdir -p ${OUTDIR}/final_samples/gbm
mkdir -p ${OUTDIR}/final_samples/pdgm

# Ensure imports resolve correctly
cd /home/j98my/models

# -----------------------------------------------------------------------------
# Python sampling pipeline (inline for reproducibility)
# -----------------------------------------------------------------------------

python3 -u -c "
# ============================================================================
# Python stage: Source-only diffusion sampling with anatomical conditioning
# ============================================================================

import sys
sys.path.insert(0, '/home/j98my/models')

import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import json

# Core model components
from ldm3d.config import DEVICE
from ldm3d.vae import VAE3D
from ldm3d.unet import UNet3DLatentCond
from ldm3d.ema import EMA
from ldm3d.diffusion import LatentDDPM

# Data + utilities
from ldm3d.data import VolFolder, load_subject_list
from ldm3d.io_nifti import save_nifti
from ldm3d.latent_stats import denormalize_latents
from ldm3d.train import mask_to_edge_3d, mask_to_soft_dist

print('[LOAD] Initializing SOURCE-ONLY models...')

# --------------------------------------------------------------------------
# Load VAE (latent encoder/decoder)
#
# NOTE:
#   This VAE was trained in the prior run and may include few-shot adaptation.
#   In this experiment, we intentionally keep the VAE fixed to isolate the
#   effect of diffusion fine-tuning.
# --------------------------------------------------------------------------

vae = VAE3D(z_channels=8, base=64).to(DEVICE)
vae.load_state_dict(torch.load('${PREV_RUN}/vae_final.pt', map_location=DEVICE))
vae.eval()

# --------------------------------------------------------------------------
# Load SOURCE diffusion UNet (pre-target fine-tuning)
#
# This UNet defines the diffusion score function and was trained ONLY on GBM.
# --------------------------------------------------------------------------

unet = UNet3DLatentCond(
    z_channels=8,        # Latent channels produced by the VAE
    cond_channels=1,     # Binary tumor mask conditioning
    control_channels=3,  # Mask, edge map, distance transform (ControlNet)
    base=96,             # UNet base channel width
    t_dim=256,           # Timestep embedding dimensionality
    use_controlnet=True  # Enable ControlNet-style conditioning
).to(DEVICE)

unet.load_state_dict(
    torch.load('${PREV_RUN}/unet_ldm_src_ep300.pt', map_location=DEVICE)
)
unet.eval()

# --------------------------------------------------------------------------
# Load EMA weights corresponding to the source-trained UNet
#
# EMA provides a temporally smoothed version of the model parameters, which
# typically improves sampling stability and perceptual quality.
# --------------------------------------------------------------------------

ema = EMA(unet, decay=0.9999)
ema.shadow = torch.load(
    '${PREV_RUN}/ema_ldm_src_ep300.pt', map_location=DEVICE
)

# Ensure EMA tensors match parameter device/dtype
for name, param in unet.named_parameters():
    if name in ema.shadow:
        ema.shadow[name] = ema.shadow[name].to(
            device=param.device, dtype=param.dtype
        )

# --------------------------------------------------------------------------
# Load latent normalization statistics
#
# These statistics map normalized diffusion latents back into the VAE latent
# space prior to decoding.
# --------------------------------------------------------------------------

stats = torch.load('${PREV_RUN}/latent_stats.pt', map_location=DEVICE)
lat_mean = stats['mean'].to(DEVICE)
lat_std  = stats['std'].to(DEVICE)

# --------------------------------------------------------------------------
# Initialize DDPM sampler operating in latent space
# --------------------------------------------------------------------------

ddpm = LatentDDPM(
    T=1000,
    beta_start=1e-4,
    beta_end=2e-2
)

# --------------------------------------------------------------------------
# Load datasets
#
# GBM: full source dataset
# PDGM: restricted to few-shot subject list
# --------------------------------------------------------------------------

print('[LOAD] Loading datasets...')

gbm_ds = VolFolder('/home/j98my/Pre-Processing/prep/gbm_all_aligned')
gbm_dl = DataLoader(gbm_ds, batch_size=1, shuffle=True, num_workers=2)

pdgm_subs = load_subject_list(
    '/home/j98my/Pre-Processing/prep/pdgm_fewshot.txt'
)
pdgm_ds = VolFolder(
    '/home/j98my/Pre-Processing/prep/pdgm_target_aligned',
    subjects=pdgm_subs
)
pdgm_dl = DataLoader(pdgm_ds, batch_size=1, shuffle=True, num_workers=2)

# --------------------------------------------------------------------------
# Sampling hyperparameters
# --------------------------------------------------------------------------

GUIDANCE = 3.0            # Classifier-free guidance strength
GUIDANCE_RESCALE = 0.7    # CFG rescaling to reduce overexposure
X0_CLIP = 3.0             # Latent clipping for stability
NOISE_MULT = 1.0
NOISE_END_FRAC = 0.0

N_SAMPLES = 64             # Samples per domain
L = 28                     # Latent spatial resolution
zC = 8                     # Latent channels

print(f'[CONFIG] guidance={GUIDANCE}, x0_clip={X0_CLIP}')
print('[CONFIG] Diffusion UNet: SOURCE ONLY (epoch 300)')

# --------------------------------------------------------------------------
# Sampling function for a single domain
# --------------------------------------------------------------------------

def sample_domain(dl, domain_name, outdir):
    print(f'[SAMPLE] Generating {N_SAMPLES} samples for {domain_name}...')
    
    domain_dir = Path(outdir) / 'final_samples' / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    dl_iter = iter(dl)
    
    while saved < N_SAMPLES:
        try:
            x, mask, sids = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            x, mask, sids = next(dl_iter)
        
        mask = mask.to(DEVICE)

        # Resolve subject ID for metadata
        try:
            if isinstance(sids, (list, tuple)):
                sid = str(sids[0])
            elif hasattr(sids, 'item'):
                sid = str(sids.item())
            else:
                sid = str(sids)
        except Exception:
            sid = f'unknown_{saved}'

        # ------------------------------------------------------------------
        # Build ControlNet conditioning signals
        #
        # maskL : downsampled binary tumor mask
        # edgeL : edge map derived from mask
        # distL : soft distance transform (shape context)
        # ------------------------------------------------------------------

        maskL = F.interpolate(mask, size=(L, L, L), mode='nearest').clamp(0, 1)
        edgeL = mask_to_edge_3d(maskL)
        distL = mask_to_soft_dist(maskL, steps=6)

        controlL = torch.cat([
            maskL * 1.0,
            edgeL * 0.1,
            distL * 0.9
        ], dim=1)

        # ------------------------------------------------------------------
        # Diffusion sampling (latent space)
        # ------------------------------------------------------------------

        with torch.no_grad():
            z_norm = ddpm.p_sample_loop(
                unet,
                shape=(1, zC, L, L, L),
                use_ema=ema,
                seed=42 + saved,
                mask=maskL,
                control=controlL,
                guidance_scale=GUIDANCE,
                guidance_rescale=GUIDANCE_RESCALE,
                noise_mult=NOISE_MULT,
                noise_end_frac=NOISE_END_FRAC,
                x0_clip=X0_CLIP,
            )

            # Convert normalized latent back to VAE latent space
            z = denormalize_latents(z_norm, lat_mean, lat_std)

            # Decode latent into 3-channel MRI volume
            xhat = vae.decode(z)

        # ------------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------------

        sdir = domain_dir / f's{saved:05d}'
        sdir.mkdir(exist_ok=True)

        save_nifti(xhat[0, 0], sdir / 't1_synth.nii.gz', verbose=False)
        save_nifti(xhat[0, 1], sdir / 't2_synth.nii.gz', verbose=False)
        save_nifti(xhat[0, 2], sdir / 'flair_synth.nii.gz', verbose=False)

        meta = {
            'sid_img': sid,
            'sid_mask': sid,
            'model': 'source_ep300',
            'guidance': GUIDANCE
        }

        with open(sdir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        saved += 1
        if saved % 10 == 0:
            print(f'  [{domain_name}] {saved}/{N_SAMPLES}')

    print(f'[DONE] {domain_name}: {saved} samples')

# --------------------------------------------------------------------------
# Run sampling for both domains
# --------------------------------------------------------------------------

sample_domain(gbm_dl, 'gbm', '${OUTDIR}')
sample_domain(pdgm_dl, 'pdgm', '${OUTDIR}')

print('[COMPLETE] Source-only sampling finished.')
"

# -----------------------------------------------------------------------------
# Post-run instructions
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "SOURCE-ONLY SAMPLING COMPLETE"
echo "============================================================"
echo "Samples written to: ${OUTDIR}/final_samples/"
echo ""
echo "Run evaluation:"
echo "  cd /home/j98my/Evaluation"
echo "  ./eval*sh --synth_gbm_root ${OUTDIR}/final_samples/gbm \\"
echo "    --synth_pdgm_root ${OUTDIR}/final_samples/pdgm \\"
echo "    --original_gbm_root /home/j98my/Pre-Processing/prep/gbm_all_aligned \\"
echo "    --original_pdgm_root /home/j98my/Pre-Processing/prep/pdgm_target_aligned \\"
echo "    --cnn_model /home/j98my/models/runs/alexlite_dg_real_longrun/new_run/best.pt \\"
echo "    --output_dir /home/j98my/Evaluation/results/source_only \\"
echo "    --use_resized"
echo ""
