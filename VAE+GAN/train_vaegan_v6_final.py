#!/usr/bin/env python3
"""
train_vaegan_v6_final.py — VAE-GAN with Organized Snapshots
----------------------------------------------------
- Immediate Sanity Check.
- Adaptive Discriminator Pausing.
- Snapshots saved as: /runs/src_e50_samples/{SubjectID}/{modality}_synth.nii.gz
"""

import os
import random
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

# --- CONFIG ---
IMAGE_SIZE = 112 
IMAGE_CHANNELS = 3
LATENT_DIM = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    print(f"[SETUP] Using seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --- 1. ROBUST DATA LOADING ---
def load_vol_safe(path):
    """Loads volume, handles NaNs, and normalizes strictly to [-1, 1]."""
    try:
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)
        arr = np.nan_to_num(arr) # Fix NaNs
        
        if arr.ndim == 3: arr = arr[None] 
        
        # Robust Scaling (Percentile based)
        mask = arr > 0
        if mask.sum() > 0:
            mn = np.percentile(arr[mask], 1.0)
            mx = np.percentile(arr[mask], 99.0) # Cut top 0.5% outliers
            arr = np.clip(arr, mn, mx)
            
            if mx - mn > 1e-6:
                arr = (arr - mn) / (mx - mn) # [0, 1]
                arr = arr * 2.0 - 1.0        # [-1, 1]
            else:
                arr = np.zeros_like(arr) - 1.0
        else:
            arr = np.zeros_like(arr) - 1.0 # Background
            
        return arr
    except Exception as e:
        print(f"[ERR] Failed to load {path}: {e}")
        return np.zeros((3, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) - 1.0

class VolFolder(Dataset):
    def __init__(self, root, subjects=None):
        self.root = root
        try:
            full_subs = sorted(next(os.walk(root))[1])
        except StopIteration:
            raise ValueError(f"Root dir {root} is empty or invalid.")
        
        self.subs = subjects if subjects else full_subs
        if not self.subs: raise ValueError(f"No subjects found in {root}")
        print(f"[DATA] Found {len(self.subs)} subjects in {root}")

    def __len__(self): return len(self.subs)
    def __getitem__(self, i):
        s = self.subs[i]
        d = os.path.join(self.root, s)
        try:
            imgs = []
            # LOAD ORDER: T1, T2, FLAIR
            for m in ["t1", "t2", "flair"]:
                p = os.path.join(d, f"{m}.nii.gz")
                if not os.path.exists(p):
                    # Fallback: search for filenames containing modality
                    cands = [f for f in os.listdir(d) if m in f and f.endswith('.nii.gz')]
                    if cands:
                        p = os.path.join(d, cands[0])
                    else:
                        raise FileNotFoundError(f"Missing {m} in {s}")
                imgs.append(load_vol_safe(p))  # each: [1, D, H, W] (or similar)

            # [3, D, H, W] in whatever spatial ordering nib gives
            x = np.concatenate(imgs, axis=0).astype(np.float32)

            # To torch and RESIZE to [3, 112, 112, 112]
            x = torch.from_numpy(x)  # [C, D, H, W]
            if x.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE):
                # DataLoader expects [C, D, H, W] but F.interpolate wants [N, C, D, H, W]
                x = x.unsqueeze(0)  # [1, C, D, H, W]
                x = F.interpolate(
                    x,
                    size=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE),
                    mode="trilinear",
                    align_corners=False,
                )
                x = x.squeeze(0)  # back to [C, 112, 112, 112]

            return x, s

        except Exception as e:
            print(f"[WARN] Error loading {s}: {e}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)) - 1.0, s


# --- 2. SAFETY & VISUALIZATION UTILS ---
def save_nifti(tensor, path, verbose=True):
    """
    Saves tensor to NIfTI.
    CRITICAL: Un-normalizes [-1, 1] -> [0, 1] for visibility.
    """
    arr = tensor.detach().cpu().numpy()
    
    # Un-normalize
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    
    # Check emptiness
    if verbose:
        mn, mx, mean = arr.min(), arr.max(), arr.mean()
        if mean < 0.001:
            print(f"[CRITICAL WARN] SAVED IMAGE IS BLACK! {path}")
    
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

def validate_input_data(dataloader, outdir):
    """Checks the first batch immediately."""
    print("--- RUNNING IMMEDIATE DATA SANITY CHECK ---")
    try:
        x, sid = next(iter(dataloader))
        p = os.path.join(outdir, f"sanity_check_input_{sid[0]}.nii.gz")
        save_nifti(x[0, 0], p) # Save T1
        print("--- SANITY CHECK COMPLETE: CHECK FILE NOW ---")
    except Exception as e:
        print(f"[CRITICAL FAILURE] Data Loader crashed: {e}")
        exit(1)

# --- 3. MODELS (VAE + Spectral Norm D) ---
LATENT_DIM = 256   # set this near the top of the file

class VAEEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 112 -> 56
            nn.Conv3d(3, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm3d(16, affine=True),

            # 56 -> 28
            nn.Conv3d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm3d(32, affine=True),

            # 28 -> 14
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm3d(64, affine=True),

            # 14 -> 7
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm3d(128, affine=True),
        )

        # final feature map: [B, 128, 7, 7, 7]
        self.flat_dim = 128 * 7 * 7 * 7  # 128 * 343 = 43904
        self.fc_mu     = nn.Linear(self.flat_dim, LATENT_DIM)
        self.fc_logvar = nn.Linear(self.flat_dim, LATENT_DIM)

    def forward(self, x):
        h = self.net(x)              # [B, 128, 7, 7, 7]
        h = h.view(h.size(0), -1)    # [B, 43904]
        # print("[ENC SHAPE]", h.shape)  # keep for debugging if you like
        return self.fc_mu(h), self.fc_logvar(h)


    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        # print("[ENC SHAPE]", h.shape)
        return self.fc_mu(h), self.fc_logvar(h)

class VAEDecoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 128 * 7 * 7 * 7)
        self.unflat = nn.Unflatten(1, (128, 7, 7, 7))

        def up(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_c, out_c, 3, 1, 1),
                nn.InstanceNorm3d(out_c, affine=True),
                nn.LeakyReLU(0.2, True),
            )

        # 7 -> 14 -> 28 -> 56 -> 112
        self.net = nn.Sequential(
            up(128, 64),   # 7  -> 14
            up(64, 32),    # 14 -> 28
            up(32, 32),    # 28 -> 56
            up(32, 32),    # 56 -> 112

            nn.Conv3d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, target_shape=None):
        h = self.unflat(self.fc(z))  # [B, 128, 7, 7, 7]
        h = self.net(h)              # [B, 3, 112, 112, 112]
        return h



class VAE3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = VAEEncoder3D()
        self.dec = VAEDecoder3D()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z, x.shape[2:]), mu, logvar

class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3d(in_c, out_c, 4, 2, 1)),
                nn.LeakyReLU(0.2, True)
            )
        self.net = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.Conv3d(256, 1, 4, 1, 0)
        )
    def forward(self, x): return self.net(x)

# --- 4. TRAINING LOGIC ---
def train_stage(stage_name, dataloader, vae, D, opt_vae, opt_d, epochs, outdir, use_gan = False):
    """Unified training loop."""
    if not dataloader or epochs <= 0: return

    base_L_REC = 20.0
    base_L_KL  = 1e-4
    base_L_ADV = 0.05
    warmup_epochs_kl  =50      # ramp KL over first 25 epochs
    warmup_epochs_gan = 100      # ramp GAN over first 25 epochs

    print(f"--- Starting {stage_name} Training ({epochs} eps) ---")
    
    for epoch in range(1, epochs + 1):
        kl_factor  = min(1.0, epoch / warmup_epochs_kl)
        gan_factor = min(1.0, epoch / warmup_epochs_gan)

        L_REC = base_L_REC
        L_KL  = base_L_KL  * kl_factor
        L_ADV = base_L_ADV * gan_factor


        vae.train(); D.train()
        loss_d_list = []
        loss_rec_list = []
        d_status = "ACTIVE"

        for i, (real, _) in enumerate(dataloader):
            real = real.to(DEVICE)
            
            # --- A. TRAIN DISCRIMINATOR ---
            if use_gan:

                with torch.no_grad():
                    fake, _, _ = vae(real)
                
                d_real = D(real)
                d_fake = D(fake)
                loss_d = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
                
                # ADAPTIVE PAUSE
                if loss_d.item() > 0.5:
                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()
                    d_status = "UPDATED"
                else:
                    d_status = "PAUSED (Too Strong)"
                    
                loss_d_list.append(loss_d.item())
            else:
                d_status = "DISABLED"
                loss_d_list.append(0.0)

            # --- B. TRAIN GENERATOR ---
            fake, mu, logvar = vae(real)

            if use_gan:
                d_fake_new = D(fake)
                l_adv = -d_fake_new.mean()
            else:
                l_adv = torch.tensor(0.0, device = real.device)
            l_rec = F.l1_loss(fake, real)
            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / real.size(0)
            
            loss_g = (L_REC * l_rec) + (L_ADV * l_adv) + (L_KL * l_kl)
            
            opt_vae.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0) 
            opt_vae.step()
            
            loss_rec_list.append(l_rec.item())

        # Logs
        avg_d = np.mean(loss_d_list)
        avg_rec = np.mean(loss_rec_list)
        print(f"[{stage_name}][Ep {epoch}] D: {avg_d:.4f} ({d_status}) | Rec: {avg_rec:.4f}")

        # --- SNAPSHOT SAVING (Organized Folders) ---
        if epoch % 50 == 0 or epoch == epochs:
            print(f"--- Saving Batch Snapshot {stage_name} Ep {epoch} ---")
            vae.eval()
            
            # Create base folder: /runs/src_e50_samples
            snap_root = Path(outdir) / f"{stage_name}_e{epoch}_samples"
            snap_root.mkdir(parents=True, exist_ok=True)
            
            # Save 8 random samples (iterate loader to get fresh batch)
            dl_iter = iter(dataloader)
            samples_saved = 0
            
            with torch.no_grad():
                while samples_saved < 8:
                    try:
                        x_batch, sids = next(dl_iter)
                    except StopIteration:
                        break # End of data
                    
                    x_batch = x_batch.to(DEVICE)
                    recon, _, _ = vae(x_batch)
                    
                    # Iterate items in batch (usually size 1, so this runs once per load)
                    for b in range(x_batch.size(0)):
                        sid = sids[b]
                        
                        # Create subject folder
                        s_dir = snap_root / sid
                        s_dir.mkdir(exist_ok=True)
                        
                        # Save Split Channels (T1=0, T2=1, FLAIR=2)
                        save_nifti(recon[b, 0], s_dir / "t1_synth.nii.gz", verbose=False)
                        save_nifti(recon[b, 1], s_dir / "t2_synth.nii.gz", verbose=False)
                        save_nifti(recon[b, 2], s_dir / "flair_synth.nii.gz", verbose=False)
                        
                        samples_saved += 1
                        if samples_saved >= 8: break
            
            print(f"[SNAPSHOT] Saved {samples_saved} subjects to {snap_root}")
            torch.save(vae.state_dict(), Path(outdir) / f"vae_{stage_name}_ep{epoch}.pt")

def main(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. DATASETS
    print("[INIT] Loading Source (GBM)...")
    gbm_ds = VolFolder(args.gbm_root)
    gbm_dl = DataLoader(gbm_ds, batch_size=1, shuffle=True, num_workers=2)
    
    # Sanity Check
    validate_input_data(gbm_dl, args.outdir)

    print("[INIT] Loading Target (PDGM)...")
    pdgm_subs = []
    if args.fewshot and os.path.exists(args.fewshot):
        pdgm_subs = [l.strip() for l in open(args.fewshot) if l.strip()]
    
    pdgm_ds = VolFolder(args.pdgm_root, subjects=pdgm_subs if pdgm_subs else None)
    pdgm_dl = DataLoader(pdgm_ds, batch_size=1, shuffle=True, num_workers=2)

    # 2. MODELS
    vae = VAE3D().to(DEVICE)
    D = Discriminator3D().to(DEVICE)
    
    opt_vae = torch.optim.Adam(vae.parameters(), lr=args.lr_vae, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))

    # 3. RUN TRAINING
    train_stage("src", gbm_dl, vae, D, opt_vae, opt_d, args.epochs_src, args.outdir, use_gan = True)
    train_stage("tgt", pdgm_dl, vae, D, opt_vae, opt_d, args.epochs_tgt, args.outdir, use_gan = True)

    print("--- Training Finished Successfully ---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--gbm_root', default='/home/j98my/Pre-Processing/prep/gbm_all')
    ap.add_argument('--pdgm_root', default='/home/j98my/Pre-Processing/prep/pdgm_target')
    ap.add_argument('--fewshot', default='/home/j98my/Pre-Processing/prep/pdgm_fewshot.txt')
    ap.add_argument('--epochs_src', type=int, default=300)
    ap.add_argument('--epochs_tgt', type=int, default=300)
    ap.add_argument('--outdir', default='/home/j98my/models/runs/vaegan_v6_final')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr_vae', type=float, default=1e-4)
    ap.add_argument('--lr_gan', type=float, default=1e-5)
    args = ap.parse_args()
    
    main(args)