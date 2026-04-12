#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# so it works even if launched from elsewhere
sys.path.insert(0, "/home/j98my/models")

from ldm3d.config import DEVICE
from ldm3d.vae import VAE3D
from ldm3d.unet import UNet3DLatentCond
from ldm3d.ema import EMA
from ldm3d.diffusion import LatentDDPM
from ldm3d.data import VolFolder, load_subject_list
from ldm3d.io_nifti import save_nifti
from ldm3d.latent_stats import denormalize_latents
from ldm3d.train import mask_to_edge_3d, mask_to_soft_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fewshot", required=True)
    ap.add_argument("--pdgm_root", required=True)

    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--guidance_rescale", type=float, default=0.7)
    ap.add_argument("--x0_clip", type=float, default=3.0)

    ap.add_argument("--L", type=int, default=28)
    ap.add_argument("--zC", type=int, default=8)
    ap.add_argument("--vae_base", type=int, default=64)
    ap.add_argument("--unet_base", type=int, default=96)

    ap.add_argument("--ctrl_mask_w", type=float, default=1.0)
    ap.add_argument("--ctrl_edge_w", type=float, default=0.1)
    ap.add_argument("--ctrl_dist_w", type=float, default=0.9)

    ap.add_argument("--noise_mult", type=float, default=1.0)
    ap.add_argument("--noise_end_frac", type=float, default=0.0)

    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    out_final = outdir / "final_samples" / "pdgm"
    out_final.mkdir(parents=True, exist_ok=True)

    print("[16-SHOT] Loading models...")
    vae = VAE3D(z_channels=args.zC, base=args.vae_base).to(DEVICE)
    vae.load_state_dict(torch.load(model_dir / "vae_final.pt", map_location=DEVICE))
    vae.eval()

    unet = UNet3DLatentCond(
        z_channels=args.zC, cond_channels=1, control_channels=3,
        base=args.unet_base, t_dim=256, use_controlnet=True
    ).to(DEVICE)
    unet.load_state_dict(torch.load(model_dir / "unet_ldm_src_ep300.pt", map_location=DEVICE))
    unet.eval()

    ema = EMA(unet, decay=0.999)
    ema.shadow = torch.load(model_dir / "ema_ldm_src_ep300.pt", map_location=DEVICE)
    for name, param in unet.named_parameters():
        if name in ema.shadow:
            ema.shadow[name] = ema.shadow[name].to(device=param.device, dtype=param.dtype)

    stats = torch.load(model_dir / "latent_stats.pt", map_location=DEVICE)
    lat_mean, lat_std = stats["mean"].to(DEVICE), stats["std"].to(DEVICE)

    ddpm = LatentDDPM(T=1000, beta_start=1e-4, beta_end=2e-2)

    pdgm_subs = load_subject_list(args.fewshot)
    ds = VolFolder(args.pdgm_root, subjects=pdgm_subs)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    saved = 0
    for x, mask, sids in dl:
        if saved >= args.n:
            break

        mask = mask.to(DEVICE)
        sid = str(sids[0]) if isinstance(sids, (list, tuple)) else str(sids)

        L = args.L
        maskL = F.interpolate(mask, size=(L, L, L), mode="nearest").clamp(0, 1)
        edgeL = mask_to_edge_3d(maskL)
        distL = mask_to_soft_dist(maskL, steps=6)

        controlL = torch.cat(
            [maskL * args.ctrl_mask_w, edgeL * args.ctrl_edge_w, distL * args.ctrl_dist_w],
            dim=1
        )

        with torch.no_grad():
            z_norm = ddpm.p_sample_loop(
                unet,
                shape=(1, args.zC, L, L, L),
                use_ema=ema,
                seed=args.seed + saved,
                mask=maskL,
                control=controlL,
                guidance_scale=args.guidance,
                guidance_rescale=args.guidance_rescale,
                noise_mult=args.noise_mult,
                noise_end_frac=args.noise_end_frac,
                x0_clip=args.x0_clip,
            )
            z = denormalize_latents(z_norm, lat_mean, lat_std)
            xhat = vae.decode(z)

        sdir = out_final / f"s{saved:05d}"
        sdir.mkdir(parents=True, exist_ok=True)

        save_nifti(xhat[0, 0], sdir / "t1_synth.nii.gz", verbose=False)
        save_nifti(xhat[0, 1], sdir / "t2_synth.nii.gz", verbose=False)
        save_nifti(xhat[0, 2], sdir / "flair_synth.nii.gz", verbose=False)

        # meta.json for GT mapping
        (sdir / "meta.json").write_text(
            f'{{"model":"16shot","sid_img":"{sid}","sid_mask":"{sid}"}}'
        )

        print(f"[16-SHOT] Saved {sid} -> {sdir}")
        saved += 1

    print("[16-SHOT DONE]")


if __name__ == "__main__":
    main()
