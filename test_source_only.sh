#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This shell script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

#
# TEST 1: Use SOURCE MODEL ONLY (no target fine-tuning)
# 
# Hypothesis: The source model might actually perform better on PDGM
# since fine-tuning causes mode collapse. Source model has more diversity.
#
set -e

# Set a configuration variable used later in the script.
PREV_RUN=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_IMPROVED
# Set a configuration variable used later in the script.
OUTDIR=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_SOURCE_ONLY

# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo "TEST: SOURCE MODEL ONLY (NO TARGET FINE-TUNING)"
# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "Using source-trained model to generate PDGM samples"
# Print a status message for runtime visibility.
echo "This tests if fine-tuning is actually hurting performance"
# Print a status message for runtime visibility.
echo ""

# Create required output directories if they do not exist.
mkdir -p ${OUTDIR}/final_samples/gbm
# Create required output directories if they do not exist.
mkdir -p ${OUTDIR}/final_samples/pdgm

# Switch into the expected working directory.
cd /home/j98my/models

# Run the Python stage for this pipeline step.
python3 -u -c "
import sys
sys.path.insert(0, '/home/j98my/models')

import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import json

from ldm3d.config import DEVICE
from ldm3d.vae import VAE3D
from ldm3d.unet import UNet3DLatentCond
from ldm3d.ema import EMA
from ldm3d.diffusion import LatentDDPM
from ldm3d.data import VolFolder, load_subject_list
from ldm3d.io_nifti import save_nifti
from ldm3d.latent_stats import denormalize_latents
from ldm3d.train import mask_to_edge_3d, mask_to_soft_dist

print('[LOAD] Loading SOURCE models (no target fine-tuning)...')

# Load VAE
vae = VAE3D(z_channels=8, base=64).to(DEVICE)
vae.load_state_dict(torch.load('${PREV_RUN}/vae_final.pt', map_location=DEVICE))
vae.eval()

# Load SOURCE UNet (ep300, before any target training)
unet = UNet3DLatentCond(
    # Set a configuration variable used later in the script.
    z_channels=8,
    # Set a configuration variable used later in the script.
    cond_channels=1,
    # Set a configuration variable used later in the script.
    control_channels=3,
    # Set a configuration variable used later in the script.
    base=96,
    # Set a configuration variable used later in the script.
    t_dim=256,
    # Set a configuration variable used later in the script.
    use_controlnet=True,
).to(DEVICE)
unet.load_state_dict(torch.load('${PREV_RUN}/unet_ldm_src_ep300.pt', map_location=DEVICE))
unet.eval()

# Load SOURCE EMA
ema = EMA(unet, decay=0.9999)
ema.shadow = torch.load('${PREV_RUN}/ema_ldm_src_ep300.pt', map_location=DEVICE)
# Shell control-flow statement managing script execution.
for name, param in unet.named_parameters():
    # Shell control-flow statement managing script execution.
    if name in ema.shadow:
        ema.shadow[name] = ema.shadow[name].to(device=param.device, dtype=param.dtype)

# Load latent stats
stats = torch.load('${PREV_RUN}/latent_stats.pt', map_location=DEVICE)
lat_mean = stats['mean'].to(DEVICE)
lat_std = stats['std'].to(DEVICE)

# DDPM
ddpm = LatentDDPM(T=1000, beta_start=1e-4, beta_end=2e-2)

# Load data
print('[LOAD] Loading datasets...')
gbm_ds = VolFolder('/home/j98my/Pre-Processing/prep/gbm_all_aligned')
gbm_dl = DataLoader(gbm_ds, batch_size=1, shuffle=True, num_workers=2)

pdgm_subs = load_subject_list('/home/j98my/Pre-Processing/prep/pdgm_fewshot.txt')
pdgm_ds = VolFolder('/home/j98my/Pre-Processing/prep/pdgm_target_aligned', subjects=pdgm_subs)
pdgm_dl = DataLoader(pdgm_ds, batch_size=1, shuffle=True, num_workers=2)

# Sampling parameters
GUIDANCE = 3.0
X0_CLIP = 3.0
NOISE_MULT = 1.0
NOISE_END_FRAC = 0.0
GUIDANCE_RESCALE = 0.7
N_SAMPLES = 64
L = 28
zC = 8

print(f'[CONFIG] guidance={GUIDANCE}, x0_clip={X0_CLIP}')
print(f'[CONFIG] Using SOURCE model (ep300) - NO target fine-tuning')

def sample_domain(dl, domain_name, outdir):
    print(f'[SAMPLE] Generating {N_SAMPLES} samples for {domain_name}...')
    
    domain_dir = Path(outdir) / 'final_samples' / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    dl_iter = iter(dl)
    
    # Shell control-flow statement managing script execution.
    while saved < N_SAMPLES:
        try:
            x, mask, sids = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            x, mask, sids = next(dl_iter)
        
        x = x.to(DEVICE)
        mask = mask.to(DEVICE)
        
        try:
            # Shell control-flow statement managing script execution.
            if isinstance(sids, (list, tuple)):
                sid = str(sids[0])
            # Shell control-flow statement managing script execution.
            elif hasattr(sids, 'item'):
                sid = str(sids.item())
            # Shell control-flow statement managing script execution.
            else:
                sid = str(sids)
        except:
            sid = f'unknown_{saved}'
        
        # Prepare conditioning
        maskL = F.interpolate(mask, size=(L, L, L), mode='nearest').clamp(0, 1)
        edgeL = mask_to_edge_3d(maskL)
        distL = mask_to_soft_dist(maskL, steps=6)
        
        controlL = torch.cat([
            maskL * 1.0,
            edgeL * 0.1,
            distL * 0.9
        ], dim=1)
        
        with torch.no_grad():
            z_norm = ddpm.p_sample_loop(
                unet,
                # Set a configuration variable used later in the script.
                shape=(1, zC, L, L, L),
                # Set a configuration variable used later in the script.
                use_ema=ema,
                # Set a configuration variable used later in the script.
                seed=42 + saved,
                # Set a configuration variable used later in the script.
                mask=maskL,
                # Set a configuration variable used later in the script.
                control=controlL,
                # Set a configuration variable used later in the script.
                guidance_scale=GUIDANCE,
                # Set a configuration variable used later in the script.
                guidance_rescale=GUIDANCE_RESCALE,
                # Set a configuration variable used later in the script.
                noise_mult=NOISE_MULT,
                # Set a configuration variable used later in the script.
                noise_end_frac=NOISE_END_FRAC,
                # Set a configuration variable used later in the script.
                x0_clip=X0_CLIP,
            )
            
            z = denormalize_latents(z_norm, lat_mean, lat_std)
            xhat = vae.decode(z)
        
        # Save
        sdir = domain_dir / f's{saved:05d}'
        sdir.mkdir(exist_ok=True)
        
        save_nifti(xhat[0, 0], sdir / 't1_synth.nii.gz', verbose=False)
        save_nifti(xhat[0, 1], sdir / 't2_synth.nii.gz', verbose=False)
        save_nifti(xhat[0, 2], sdir / 'flair_synth.nii.gz', verbose=False)
        
        meta = {'sid_img': sid, 'sid_mask': sid, 'model': 'source_ep300', 'guidance': GUIDANCE}
        with open(sdir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        saved += 1
        # Shell control-flow statement managing script execution.
        if saved % 10 == 0:
            print(f'  [{domain_name}] {saved}/{N_SAMPLES}')
    
    print(f'[DONE] {domain_name}: {saved} samples')

sample_domain(gbm_dl, 'gbm', '${OUTDIR}')
sample_domain(pdgm_dl, 'pdgm', '${OUTDIR}')

print('[COMPLETE] Source-only sampling done!')
"

# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo "SOURCE-ONLY SAMPLING COMPLETE"
# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo "Samples: ${OUTDIR}/final_samples/"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "Run evaluation:"
# Print a status message for runtime visibility.
echo "  cd /home/j98my/Evaluation"
# Print a status message for runtime visibility.
echo "  ./eval*sh --synth_gbm_root ${OUTDIR}/final_samples/gbm \\"
# Print a status message for runtime visibility.
echo "    --synth_pdgm_root ${OUTDIR}/final_samples/pdgm \\"
# Print a status message for runtime visibility.
echo "    --original_gbm_root /home/j98my/Pre-Processing/prep/gbm_all_aligned \\"
# Print a status message for runtime visibility.
echo "    --original_pdgm_root /home/j98my/Pre-Processing/prep/pdgm_target_aligned \\"
# Print a status message for runtime visibility.
echo "    --cnn_model /home/j98my/models/runs/alexlite_dg_real_longrun/new_run/best.pt \\"
# Print a status message for runtime visibility.
echo "    --output_dir /home/j98my/Evaluation/results/source_only \\"
# Print a status message for runtime visibility.
echo "    --use_resized"
# Print a status message for runtime visibility.
echo ""