#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
train_cgan3d_v3.py — Mask-conditioned 3D cGAN for few-/zero-shot cross-domain MRI generation

This script trains a 3D Generative Adversarial Network (GAN) with an ensemble of three
channel-specific discriminators, one for the T1, one for the T2, and one for the FLAIR
modality channel.

[V3 Changes]
1.  **Multiple Discriminators:** Replaced the single 4-channel Discriminator with an 
    **EnsembleDiscriminator** consisting of three single-channel Discriminators, one 
    for each modality (T1, T2, FLAIR). Each channel D receives its image channel + mask.
2.  **Sampling Frequency:** Reduced sampling frequency to once every **50 epochs**.
"""

# --- 1. IMPORT LIBRARIES ---
# Standard libraries:
# - os/pathlib: filesystem walking, building output dirs, saving checkpoints
# - json: saving run configuration for reproducibility
# - random: lightweight stochastic augmentation and sampling
# - argparse: CLI arguments to control paths / epochs / preprocessing knobs
# Import dependencies used by this module.
import os, json, random, argparse
from pathlib import Path

# Core numeric / deep learning stack:
# - numpy: volume manipulation + IO interop with nibabel
# - torch: models, losses, autograd, optimization
# - torch.nn.functional: functional ops (interpolation, activations, etc.)
# - Dataset/DataLoader: batching, shuffling, multi-worker loading
# Import dependencies used by this module.
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AMP (automatic mixed precision):
# - speeds up training and reduces GPU memory use on compatible GPUs
# - scaler handles dynamic loss scaling to avoid fp16 underflow
# Import dependencies used by this module.
from torch.cuda import amp

# Medical imaging:
# - nibabel loads/saves NIfTI volumes (.nii.gz)
# Import dependencies used by this module.
import nibabel as nib
                        
# ---------------------------
# 2. Repro & small utils (Helper functions)
# ---------------------------

# Function: `set_seed` implements a reusable processing step.
def set_seed(seed: int = 42):
    """Sets the random seed for all libraries to ensure reproducible results."""
    # Python RNG (augment choices, flips, etc.)
    random.seed(seed)
    # NumPy RNG (rare in this script but safe)
    np.random.seed(seed)
    # Torch RNGs (model init, noise z, augment, etc.)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Force deterministic kernels where possible (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function: `to_torch` implements a reusable processing step.
def to_torch(x):
    """Converts a NumPy array to a PyTorch tensor and makes a copy."""
    # Copy avoids issues where numpy arrays share memory / are non-contiguous.
    # Return the computed value to the caller.
    return torch.from_numpy(x.copy())

# Function: `load_vol` implements a reusable processing step.
def load_vol(path):  # float32 in [-1,1]
    """Loads a 3D NIfTI scan (e.g., T1, T2) from a file path."""
    # NOTE: This assumes your volumes are already in [0,1] intensity scale.
    # If not, this normalization is NOT robust (unlike your LDM pipeline).
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    # Control-flow branch for conditional or iterative execution.
    if arr.ndim == 3:
        arr = arr[None]  # add channel dim -> [1,D,H,W]
    # Normalization Step: Clip to [0, 1], then scale to [-1, 1]
    arr = np.clip(arr, 0.0, 1.0) * 2.0 - 1.0
    # Return the computed value to the caller.
    return arr

# Function: `load_mask` implements a reusable processing step.
def load_mask(path):
    """Loads a 3D NIfTI mask from a file path."""
    # Tumor masks are binarized into {0,1} for conditioning.
    m = nib.load(path).get_fdata(dtype=np.float32)
    # Control-flow branch for conditional or iterative execution.
    if m.ndim == 3:
        m = m[None]  # [1,D,H,W]
    # Binarization Step: Convert to 0s and 1s
    m = (m > 0.5).astype(np.float32)
    # Return the computed value to the caller.
    return m

# Class definition: `EMA` encapsulates related model behavior.
class EMA:
    """Exponential Moving Average (EMA)."""
    # EMA is used here ONLY for Generator sampling stability:
    # - We keep a smoothed copy of G’s weights (G_ema) for better sample quality.
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, model, beta=0.999):
        self.m = model
        self.b = beta
        # Shadow holds EMA weights for trainable params only.
        self.shadow = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    # Function: `update` implements a reusable processing step.
    def update(self):
        # Update shadow weights: shadow = beta*shadow + (1-beta)*current
        i = 0
        # Control-flow branch for conditional or iterative execution.
        for p in self.m.parameters():
            # Control-flow branch for conditional or iterative execution.
            if not p.requires_grad:
                continue
            self.shadow[i].mul_(self.b).add_(p.data, alpha=1 - self.b)
            i += 1

    @torch.no_grad()
    # Function: `copy_to` implements a reusable processing step.
    def copy_to(self, target):
        """Copies the smooth shadow weights into a target model (e.g., G_ema)."""
        i = 0
        # Control-flow branch for conditional or iterative execution.
        for p in target.parameters():
            # Control-flow branch for conditional or iterative execution.
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[i])
            i += 1

# ---------------------------
# 3. Data (PyTorch Dataset class)
# ---------------------------
# Class definition: `VolFolder` encapsulates related model behavior.
class VolFolder(Dataset):
    """
    PyTorch Dataset class for loading 3D multi-channel MRI scans and their mask.
    Returns: (x, mask, subject_id), where x is (3, D, H, W).

    Folder layout expected:
        root/SubjectID/t1.nii.gz
        root/SubjectID/t2.nii.gz
        root/SubjectID/flair.nii.gz
        root/SubjectID/mask.nii.gz
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, root, subjects=None):
        self.root = root
        # os.walk returns (dirpath, dirnames, filenames); [1] gives child directories.
        subs = sorted(next(os.walk(root))[1])
        # If subjects is provided, restrict dataset to those IDs (few-shot mode)
        self.subs = subjects if subjects else subs

    # Function: `__len__` implements a reusable processing step.
    def __len__(self):
        # Return the computed value to the caller.
        return len(self.subs)

    # Function: `__getitem__` implements a reusable processing step.
    def __getitem__(self, i):
        # subject id and its folder
        s = self.subs[i]
        d = f"{self.root}/{s}"

        # Load 3 modalities (each returned as [1,D,H,W])
        imgs = [load_vol(f"{d}/{k}.nii.gz") for k in ["t1","t2","flair"]]

        # Concatenate them into a single 3-channel volume: (3,D,H,W)
        x = np.concatenate(imgs, axis=0)

        # Load the corresponding 3D mask: (1,D,H,W)
        m = load_mask(f"{d}/mask.nii.gz")

        # Return the computed value to the caller.
        return to_torch(x), to_torch(m), s

# ---------------------------
# 4. SPADE blocks (3D) - Unchanged
# ---------------------------

# Class definition: `SPADE3D` encapsulates related model behavior.
class SPADE3D(nn.Module):
    """
    The core SPADE normalization layer for 3D data.

    SPADE = Spatially-Adaptive Denormalization:
    - Normalize features (InstanceNorm without affine)
    - Use a conditioning tensor (mask+domain) to produce per-voxel gamma/beta
    - Apply: norm(x) * (1 + gamma) + beta

    This allows the generator to "paint" structure aligned to the mask.
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, norm_nc, cond_nc):
        super().__init__()
        # InstanceNorm3d keeps contrast consistent; SPADE injects the condition signal.
        self.norm = nn.InstanceNorm3d(norm_nc, affine=False, eps=1e-5)
        hidden = max(32, cond_nc * 2)
        # Shared MLP processes conditioning volume into a hidden feature map.
        self.mlp_shared = nn.Sequential(nn.Conv3d(cond_nc, hidden, 3, padding=1), nn.ReLU(True))
        # Separate heads produce gamma and beta maps matching feature channels.
        self.mlp_gamma  = nn.Conv3d(hidden, norm_nc, 3, padding=1)
        self.mlp_beta   = nn.Conv3d(hidden, norm_nc, 3, padding=1)

    # Function: `forward` implements a reusable processing step.
    def forward(self, x, cond):
        # 1) normalize x
        n = self.norm(x)
        # 2) resize condition to match feature spatial size (important across scales)
        # Control-flow branch for conditional or iterative execution.
        if x.shape[-3:] != cond.shape[-3:]:
            cond = F.interpolate(cond, size=x.shape[-3:], mode='nearest')
        # 3) compute modulation parameters
        a = self.mlp_shared(cond)
        gamma = self.mlp_gamma(a)
        beta = self.mlp_beta(a)
        # 4) modulate normalized activations
        # Return the computed value to the caller.
        return n * (1 + gamma) + beta

# Class definition: `SPADEResBlk3D` encapsulates related model behavior.
class SPADEResBlk3D(nn.Module):
    """
    A 3D residual block that uses SPADE conditioning.

    If upsample=True:
      - Upsamples feature map by 2x (trilinear), then applies SPADE convs.
      - Skip path is also upsampled to keep shapes aligned.
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, fin, fout, cond_nc, upsample=False):
        super().__init__()
        self.ups = upsample
        # fmid is an internal bottleneck channel size (common residual pattern)
        fmid = min(fin, fout)
        
        self.sp1 = SPADE3D(fin,  cond_nc)
        self.c1 = nn.Conv3d(fin,  fmid, 3, padding=1)
        
        self.sp2 = SPADE3D(fmid, cond_nc)
        self.c2 = nn.Conv3d(fmid, fout, 3, padding=1)
        
        # Optional skip projection if channel counts differ
        self.skip = None
        # Control-flow branch for conditional or iterative execution.
        if fin != fout:
            self.skip = nn.Conv3d(fin, fout, 1)

    # Function: `forward` implements a reusable processing step.
    def forward(self, x, cond):
        # Main branch
        h = x
        # Control-flow branch for conditional or iterative execution.
        if self.ups:
            # Spatial upsample first to grow resolution before convolutions
            h = F.interpolate(h, scale_factor=2, mode='trilinear', align_corners=False)
        
        h = self.sp1(h, cond)
        h = F.leaky_relu(h, 0.2, True)
        h = self.c1(h)
        
        h = self.sp2(h, cond)
        h = F.leaky_relu(h, 0.2, True)
        h = self.c2(h)
        
        # Skip branch
        xs = x if self.skip is None else self.skip(x)
        # Control-flow branch for conditional or iterative execution.
        if self.ups:
            xs = F.interpolate(xs, scale_factor=2, mode='trilinear', align_corners=False)
            
        # Residual addition helps gradients + stabilizes GAN training
        # Return the computed value to the caller.
        return h + xs

# ---------------------------
# 5. Generator (The "Artist" or "Forger") - Unchanged
# ---------------------------
# Class definition: `Generator` encapsulates related model behavior.
class Generator(nn.Module):
    """
    The Generator network (G).

    Input:
      - z: latent noise vector [B, z_dim]
      - cond: conditioning volume [B, cond_nc, D, H, W]
              here cond_nc=2 -> [mask, domain_id_volume]

    Output:
      - x_fake: [B, 3, D, H, W] in [-1, 1] via tanh
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, z_dim=128, base=32, cond_nc=2, out_ch=3):
        super().__init__()
        # Channel plan across upsampling stages:
        # starts very wide at low res, then narrows as resolution increases.
        ch = [base * 8, base * 8, base * 4, base * 2, base]
        # Starting spatial resolution (7^3) then upsample x2 four times -> 112^3
        self.start_dim = 7
        
        # FC maps z into a small 3D feature block: [B, ch0, 7,7,7]
        self.fc = nn.Linear(z_dim, ch[0] * (self.start_dim ** 3))
        
        # SPADE ResBlocks progressively upsample to full resolution:
        # 7->14->28->56->112
        self.blocks = nn.ModuleList([
            SPADEResBlk3D(ch[0], ch[1], cond_nc, upsample=True), # 7  -> 14
            SPADEResBlk3D(ch[1], ch[2], cond_nc, upsample=True), # 14 -> 28
            SPADEResBlk3D(ch[2], ch[3], cond_nc, upsample=True), # 28 -> 56
            SPADEResBlk3D(ch[3], ch[4], cond_nc, upsample=True), # 56 -> 112
        ])
        
        # Final image head: conv -> tanh for [-1,1] output per modality channel
        self.to_img = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ch[4], out_ch, 3, padding=1),
            nn.Tanh()
        )
    
    # Function: `forward` implements a reusable processing step.
    def forward(self, z, cond):
        # cond determines final target spatial size (D,H,W)
        B, _, D, H, W = cond.shape
        # Project z to a learned low-res 3D feature grid
        h = self.fc(z).view(B, -1, self.start_dim, self.start_dim, self.start_dim)
        
        # Apply SPADE blocks; each uses cond to inject mask+domain signal
        # Control-flow branch for conditional or iterative execution.
        for blk in self.blocks:
            h = blk(h, cond)
            
        # Safety: if rounding effects occur, force exact match to conditioning size
        # Control-flow branch for conditional or iterative execution.
        if h.shape[-3:] != (D, H, W):
            h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)
            
        # Return the computed value to the caller.
        return self.to_img(h)

# ---------------------------
# 6. Discriminator (The "Detective" or "Critic") - MODIFIED
# ---------------------------

# Class definition: `DiscSingleChannel` encapsulates related model behavior.
class DiscSingleChannel(nn.Module):
    """
    A Discriminator network (D) for a SINGLE modality channel.

    Input:
      - x_single_ch: [B,1,D,H,W] (one of T1/T2/FLAIR)
      - mask:        [B,1,D,H,W]
      - domain:      [B] integer labels (0=GBM, 1=PDGM)

    Architecture:
      - strided conv pyramid downsamples to ~7^3
      - global average pool produces a feature vector
      - projection discriminator adds domain-awareness:
          score = fc(h) + <proj(h), emb(domain)>
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, in_ch=2, base=32, emb_dim=32):
        super().__init__()
        
        # Function: `blk` implements a reusable processing step.
        def blk(ci, co):
            # Strided conv reduces spatial size by ~2 each block
            # Return the computed value to the caller.
            return nn.Sequential(nn.Conv3d(ci, co, 3, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        
        self.net = nn.Sequential(
            blk(in_ch,    base),      # 112 -> 56
            blk(base,     base * 2),  # 56  -> 28
            blk(base * 2, base * 4),  # 28  -> 14
            blk(base * 4, base * 8),  # 14  -> 7
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        
        # Real/Fake head (scalar)
        self.fc   = nn.Linear(base * 8, 1)

        # Domain projection:
        # emb(domain) gives a domain vector; proj(h) maps features to same dim;
        # dot product encourages domain-consistent features (helps domain transfer)
        self.emb  = nn.Embedding(2, emb_dim)
        self.proj = nn.Linear(base * 8, emb_dim)
        
        # Apply spectral norm everywhere to stabilize GAN training (Lipschitz-ish)
        # Control-flow branch for conditional or iterative execution.
        for m in self.modules():
            # Control-flow branch for conditional or iterative execution.
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.utils.spectral_norm(m)

    # Function: `forward` implements a reusable processing step.
    def forward(self, x_single_ch, mask, domain):
        # 1) Concatenate image channel + mask for conditional discrimination
        h = torch.cat([x_single_ch, mask], dim=1)
        
        # 2) CNN feature extraction then global average pooling -> [B, C]
        h = self.net(h).mean(dim=[2,3,4])
        
        # 3) Real/Fake score
        out = self.fc(h)
        
        # 4) Domain projection score
        proj = (self.proj(h) * self.emb(domain)).sum(dim=1, keepdim=True)
        
        # 5) Combine scores (standard projection discriminator pattern)
        # Return the computed value to the caller.
        return out + proj


# Class definition: `EnsembleDisc` encapsulates related model behavior.
class EnsembleDisc(nn.Module):
    """
    An ensemble of three Discriminators, one for each modality.
    
    Motivation:
      - Each modality has different texture/intensity characteristics.
      - Separate critics prevent one modality dominating the gradients.
      - Encourages modality-specific realism while keeping shared conditioning (mask).
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, base=32, emb_dim=32):
        super().__init__()
        # Each discriminator sees (one image channel + mask)
        self.d_t1   = DiscSingleChannel(in_ch=2, base=base, emb_dim=emb_dim)
        self.d_t2   = DiscSingleChannel(in_ch=2, base=base, emb_dim=emb_dim)
        self.d_flair= DiscSingleChannel(in_ch=2, base=base, emb_dim=emb_dim)
        
    # Function: `forward` implements a reusable processing step.
    def forward(self, x, mask, domain):
        # Split image into modality channels
        x_t1, x_t2, x_flair = x.split(1, dim=1)
        
        # Score each modality independently
        out_t1    = self.d_t1(x_t1, mask, domain)
        out_t2    = self.d_t2(x_t2, mask, domain)
        out_flair = self.d_flair(x_flair, mask, domain)
        
        # Return list so training can sum losses across modalities
        # Return the computed value to the caller.
        return [out_t1, out_t2, out_flair]

# ---------------------------
# 7. DiffAugment (minimal 3D) - Unchanged
# ---------------------------
# Function: `diffaug3d` implements a reusable processing step.
def diffaug3d(x, mask=None):
    """
    Apply lightweight 3D augmentation on the fly.

    Goal:
      - Prevent D from memorizing training set (improves generalization)
      - Keep augmentation simple so anatomy/structure remains plausible

    Ops:
      - random flips (D/H/W axes)
      - random brightness/contrast perturbation
    """
    flip_d = random.random() < 0.5
    flip_h = random.random() < 0.5
    flip_w = random.random() < 0.5
    
    # Control-flow branch for conditional or iterative execution.
    if flip_d:
        x = torch.flip(x, [2])
        # Control-flow branch for conditional or iterative execution.
        if mask is not None:
            mask = torch.flip(mask, [2])
    # Control-flow branch for conditional or iterative execution.
    if flip_h:
        x = torch.flip(x, [3])
        # Control-flow branch for conditional or iterative execution.
        if mask is not None:
            mask = torch.flip(mask, [3])
    # Control-flow branch for conditional or iterative execution.
    if flip_w:
        x = torch.flip(x, [4])
        # Control-flow branch for conditional or iterative execution.
        if mask is not None:
            mask = torch.flip(mask, [4])
        
    # Random brightness/contrast (small) in normalized [-1,1] space
    # Control-flow branch for conditional or iterative execution.
    if random.random() < 0.5:
        a = 1.0 + 0.1 * (2 * random.random() - 1)  # contrast-ish
        b = 0.1 * (2 * random.random() - 1)        # brightness-ish
        x = (x * a + b).clamp(-1, 1)
        
    # Control-flow branch for conditional or iterative execution.
    if mask is None:
        # Return the computed value to the caller.
        return x
    # Return the computed value to the caller.
    return x, mask

# ---------------------------
# 8. Losses (The "Rules of the Game") - Unchanged
# ---------------------------

# Function: `hinge_d` implements a reusable processing step.
def hinge_d(real_logits, fake_logits):
    """
    Discriminator hinge loss:
      - encourages real logits >= 1
      - encourages fake logits <= -1
    """
    # Return the computed value to the caller.
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()

# Function: `hinge_g` implements a reusable processing step.
def hinge_g(fake_logits):
    """Generator hinge loss: maximize discriminator output on fake samples."""
    # Return the computed value to the caller.
    return -fake_logits.mean()

# Function: `r1_grad_penalty` implements a reusable processing step.
def r1_grad_penalty(d_out, x):
    """
    R1 Gradient Penalty:
      - penalizes ||∇_x D(x)||^2 on real images
      - stabilizes discriminator and improves GAN convergence

    NOTE: In this script, x must match the discriminator input tensor
          that autograd should differentiate with respect to.
    """
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x,
        create_graph=True
    )[0]
    # Return the computed value to the caller.
    return (grad.pow(2).flatten(1).sum(1)).mean()

# ---------------------------
# 9. Sampling (Saving example 3D images) - Unchanged
# ---------------------------
@torch.no_grad()
# Function: `sample_with_masks` implements a reusable processing step.
def sample_with_masks(G, outdir, samples, domain_id=1, ema_copy=None):
    """
    Generates and saves fake 3D images using a list of test masks.

    samples: list of (subject_id, mask_path)

    Behavior:
      - loads mask
      - builds cond = [mask, domain_volume]
      - samples z ~ N(0,1)
      - runs generator (EMA weights if provided)
      - saves t1/t2/flair outputs as separate NIfTI files
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Optionally sample from EMA-smoothed generator weights
    model = G if ema_copy is None else ema_copy
    model.eval()
    device = next(model.parameters()).device
    
    # Control-flow branch for conditional or iterative execution.
    for sid, mpath in samples:
        # Control-flow branch for conditional or iterative execution.
        try:
            m = load_mask(mpath)
            m_t = to_torch(m)[None].to(device)
            
            # Domain conditioning as a constant-valued 3D volume
            dom = torch.full_like(m_t, float(domain_id))
            # cond_nc=2 -> concat mask + domain channel
            cond = torch.cat([m_t, dom], dim=1)

            # z noise vector for stochastic generation
            z = torch.randn(1, 128, device=cond.device)
            
            # Generate fake MRI volume (3 channels) in [-1,1]
            x = model(z, cond).cpu().numpy()[0] # (3,D,H,W)
            
            # Convert to [0,1] for saving
            x = (x + 1.0) / 2.0
            
            # Save each channel with affine from mask file for spatial metadata
            d = Path(outdir) / sid
            d.mkdir(parents=True, exist_ok=True)
            affine = nib.load(mpath).affine
            
            # Control-flow branch for conditional or iterative execution.
            for k, mod in enumerate(["t1", "t2", "flair"]):
                img_to_save = nib.Nifti1Image(x[k], affine)
                nib.save(img_to_save, str(d / f"{mod}_synth.nii.gz"))
        # Control-flow branch for conditional or iterative execution.
        except Exception as e:
            print(f"WARNING: Failed to sample {sid} from {mpath}. Error: {e}")
    
    model.train()

# ---------------------------
# 10. Train utilities (Pre-processing during training) - Unchanged
# ---------------------------

# Function: `_fit_to_maxdim` implements a reusable processing step.
def _fit_to_maxdim(x, m, max_dim=128):
    """
    Shrinks a 3D volume if any dimension is larger than `max_dim`.

    Motivation:
      - some subjects may have larger shapes; downscaling keeps memory stable
      - resampling keeps aspect ratio consistent (uniform scaling)
    """
    # Control-flow branch for conditional or iterative execution.
    if max_dim is None or max_dim <= 0:
        # Return the computed value to the caller.
        return x, m
    B, C, Dv, Hv, Wv = x.shape
    mx = max(Dv, Hv, Wv)
    # Control-flow branch for conditional or iterative execution.
    if mx <= max_dim:
        # Return the computed value to the caller.
        return x, m
    
    scale = max_dim / float(mx)
    newD, newH, newW = int(round(Dv*scale)), int(round(Hv*scale)), int(round(Wv*scale))
    
    # Images: trilinear interpolation (smooth)
    x = F.interpolate(x, size=(newD,newH,newW), mode='trilinear', align_corners=False)
    # Masks: nearest interpolation (keep binary)
    m = F.interpolate(m, size=(newD,newH,newW), mode='nearest')
    # Return the computed value to the caller.
    return x, m

# Function: `_random_crop` implements a reusable processing step.
def _random_crop(x, m, crop_dim=112):
    """
    Randomly cuts out a 3D cube of size crop_dim^3.

    Motivation:
      - standardize training resolution for G/D
      - introduce spatial diversity (helps generalization)
    """
    # Control-flow branch for conditional or iterative execution.
    if crop_dim is None or crop_dim <= 0:
        # Return the computed value to the caller.
        return x, m
    B, C, Dv, Hv, Wv = x.shape
    # Control-flow branch for conditional or iterative execution.
    if min(Dv, Hv, Wv) <= crop_dim:
        # Return the computed value to the caller.
        return x, m
    
    d = random.randrange(0, Dv - crop_dim + 1)
    h = random.randrange(0, Hv - crop_dim + 1)
    w = random.randrange(0, Wv - crop_dim + 1)
    
    x = x[:, :, d:d+crop_dim, h:h+crop_dim, w:w+crop_dim]
    m = m[:, :, d:d+crop_dim, h:h+crop_dim, w:w+crop_dim]
    # Return the computed value to the caller.
    return x, m

# ---------------------------
# 11. Train (The Main Function) - MODIFIED
# ---------------------------

# Function: `train` implements a reusable processing step.
def train(gbm_root, pdgm_root, outdir, fewshot_path=None, epochs_src=5, epochs_tgt=5, seed=42,
          r1_gamma_src=10.0, r1_gamma_tgt=20.0, ema_beta=0.999, max_dim=128, crop_dim=112):
    
    # --- Setup ---
    # Reproducibility + device + output folder
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(outdir, exist_ok=True)
    
    # Save config so the run can be replicated later
    json.dump({"gbm_root": gbm_root, "pdgm_root": pdgm_root, "fewshot": fewshot_path,
               "epochs_src": epochs_src, "epochs_tgt": epochs_tgt, "seed": seed,
               "max_dim": max_dim, "crop_dim": crop_dim, "disc_mode": "ensemble_3ch"}, 
               open(Path(outdir)/'config.json','w'), indent=2)

    # --- Data ---
    # Source domain loader (GBM)
    gbm_ds = VolFolder(gbm_root)
    gbm_dl = DataLoader(gbm_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    # Build PDGM subject list and few-shot IDs (if provided)
    pdgm_all = sorted(next(os.walk(pdgm_root))[1])
    few = []
    # Control-flow branch for conditional or iterative execution.
    if fewshot_path and os.path.exists(fewshot_path):
        few = [l.strip() for l in open(fewshot_path) if l.strip()]
    
    # Dev sample masks for periodic qualitative inspection
    dev_pdgm_sids = [s for s in pdgm_all if s not in set(few)][:min(16, len(pdgm_all))]
    dev_samples_pdgm = [(sid, f"{pdgm_root}/{sid}/mask.nii.gz") for sid in dev_pdgm_sids]
    dev_gbm_sids = gbm_ds.subs[:min(16, len(gbm_ds.subs))]
    dev_samples_gbm = [(sid, f"{gbm_root}/{sid}/mask.nii.gz") for sid in dev_gbm_sids]

    # Few-shot PDGM loader for target fine-tuning stage
    pdgm_fs = DataLoader(VolFolder(pdgm_root, few), batch_size=1, shuffle=True, num_workers=2, pin_memory=True) if few else None

    # --- Models ---
    # Generator produces 3-channel MRI volume conditioned on (mask + domain channel)
    G = Generator(z_dim=128, base=32, cond_nc=2, out_ch=3).to(device)

    # MODIFIED: 3 discriminators (T1/T2/FLAIR) in an ensemble
    D_ensemble = EnsembleDisc(base=32, emb_dim=32).to(device)

    # EMA copy of G used for sampling (typically better looking than raw weights)
    G_ema = Generator(z_dim=128, base=32, cond_nc=2, out_ch=3).to(device)
    G_ema.load_state_dict(G.state_dict())
    ema = EMA(G, beta=ema_beta)

    # Optimizers:
    # - betas (0,0.99) is common for hinge GANs (TTUR-ish stability)
    # - same lr for G and D here; you can tune if D dominates
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(D_ensemble.parameters(), lr=1e-4, betas=(0.0, 0.99))
    
    # AMP scaler (enabled only if CUDA is available)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    # Function: `domain_map` implements a reusable processing step.
    def domain_map(domain_id, mask):
        """
        Helper to create the 2-channel condition tensor:
          cond = [mask, domain_volume]
        where domain_volume is a constant 3D tensor filled with domain_id.
        """
        B, _, Dv, Hv, Wv = mask.shape
        dom = torch.full((B, 1, Dv, Hv, Wv), float(domain_id), device=mask.device)
        # Return the computed value to the caller.
        return torch.cat([mask, dom], dim=1)

    # --------------------------------------------------
    # -------- Stage A: Source pretrain (GBM) --------
    # --------------------------------------------------
    print("--- Starting Stage A: Source (GBM) Pre-training ---")
    
    # Keep last losses for logging (debug-friendly)
    d_loss_src = torch.tensor(0.0)
    g_loss_src = torch.tensor(0.0)
    
    # Control-flow branch for conditional or iterative execution.
    for epoch in range(1, epochs_src + 1):
        # Iterate over GBM subjects
        # Control-flow branch for conditional or iterative execution.
        for x, m, _ in gbm_dl:
            # Move batch to GPU
            x, m = x.to(device), m.to(device)

            # Optional resizing and cropping to stabilize memory and enforce fixed training size
            x, m = _fit_to_maxdim(x, m, max_dim)
            x, m = _random_crop(x, m, crop_dim)
            
            # Domain label: 0 = GBM
            dom0 = torch.zeros((x.size(0),), dtype=torch.long, device=device)
            cond = domain_map(0, m)
            
            # ----------------------------
            # Discriminator step
            # ----------------------------
            # Enable gradients for D (disable later for G step)
            # Control-flow branch for conditional or iterative execution.
            for p in D_ensemble.parameters():
                p.requires_grad = True
            
            # Control-flow branch for conditional or iterative execution.
            with amp.autocast(enabled=torch.cuda.is_available()):
                # 1) Sample latent noise and create fake image (detach to avoid G gradients here)
                z = torch.randn(x.size(0), 128, device=device)
                fake = G(z, cond).detach()
                
                # 2) DiffAugment both real and fake (and their masks if provided)
                real_aug, mask_real_aug = diffaug3d(x, m)
                fake_aug, mask_fake_aug = diffaug3d(fake, m)
                
                # 3) Get per-modality discriminator outputs
                real_logits = D_ensemble(real_aug, mask_real_aug, dom0)
                fake_logits = D_ensemble(fake_aug, mask_fake_aug, dom0)
                
                # 4) Hinge loss summed across the three modality discriminators
                d_loss = sum(hinge_d(rl, fl) for rl, fl in zip(real_logits, fake_logits))
                d_loss_src = d_loss.detach().cpu()
                
            # Backprop D loss (scaled for AMP)
            d_opt.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            
            # ----------------------------
            # R1 gradient penalty (on real images)
            # ----------------------------
            # R1 needs gradients w.r.t. real inputs
            x.requires_grad_(True)
            r1_total = torch.tensor(0.0, device=device)
            x_split = x.split(1, dim=1)  # -> three [B,1,D,H,W] tensors
            
            # Control-flow branch for conditional or iterative execution.
            with amp.autocast(enabled=torch.cuda.is_available()):
                # Compute R1 separately per modality discriminator
                # Control-flow branch for conditional or iterative execution.
                for i in range(3):
                    # NOTE: This line is brittle as written (children().__next__()[i]).
                    # The intent is: run the i-th discriminator on the i-th channel.
                    r1 = r1_grad_penalty(D_ensemble.children().__next__()[i](x_split[i], m, dom0), x_split[i])
                    r1_total = r1_total + r1
                
                # Standard R1 scaling factor
                r1_total = r1_total * (r1_gamma_src / 2.0)
                
            scaler.scale(r1_total).backward()
            
            # Update D weights (AMP step)
            scaler.step(d_opt)
            scaler.update()
            
            # ----------------------------
            # Generator step
            # ----------------------------
            # Freeze D so G update doesn't waste compute on D grads
            # Control-flow branch for conditional or iterative execution.
            for p in D_ensemble.parameters():
                p.requires_grad = False
            
            # Control-flow branch for conditional or iterative execution.
            with amp.autocast(enabled=torch.cuda.is_available()):
                # 1) New z, new fake image (this time keep graph for G)
                z = torch.randn(x.size(0), 128, device=device)
                fake = G(z, cond)
                
                # 2) Augment fake (keep consistency with D training)
                fake_aug, mask_fake_aug = diffaug3d(fake, m)
                
                # 3) G wants D(fake) to be high -> hinge_g is negative mean
                fake_logits = D_ensemble(fake_aug, mask_fake_aug, dom0)
                g_loss = sum(hinge_g(fl) for fl in fake_logits)
                g_loss_src = g_loss.detach().cpu()
                
            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()
            
            # Update EMA shadow after each generator update
            ema.update()
        
        # --- End of Epoch ---
        # Save raw weights for G and the discriminator ensemble
        torch.save({"G": G.state_dict(), "D": D_ensemble.state_dict()}, Path(outdir)/f"src_e{epoch}.pt")
        print(f"[SRC] epoch {epoch}/{epochs_src} d={d_loss_src.item():.3f} g={g_loss_src.item():.3f}")

        # MODIFIED SAMPLING FREQUENCY: save samples every 50 epochs (and last epoch)
        # Control-flow branch for conditional or iterative execution.
        if epoch % 50 == 0 or epoch == epochs_src:
            ema.copy_to(G_ema)
            torch.save({"G": G_ema.state_dict()}, Path(outdir)/f"src_e{epoch}_emaG.pt")
            print(f"Generating samples for source epoch {epoch}...")
            
            # Sample into both domains using dev masks (good sanity check)
            sample_with_masks(G, str(Path(outdir)/f"src_e{epoch}_samples_gbm_dev"), 
                              dev_samples_gbm, domain_id=0, ema_copy=G_ema)
            sample_with_masks(G, str(Path(outdir)/f"src_e{epoch}_samples_pdgm_dev"), 
                              dev_samples_pdgm, domain_id=1, ema_copy=G_ema)
        
    # Ensure EMA weights are synchronized at end of stage
    ema.copy_to(G_ema)

    # --------------------------------------------------
    # -------- Stage B: Few-shot target fine-tune (PDGM) --------
    # --------------------------------------------------
    # Control-flow branch for conditional or iterative execution.
    if epochs_tgt > 0 and pdgm_fs is not None:
        print("--- Starting Stage B: Target (PDGM) Fine-tuning ---")
        
        # --- Freeze early layers of G ---
        # Motivation:
        # - early blocks encode coarse anatomy priors learned from GBM
        # - freezing prevents catastrophic forgetting when PDGM data is tiny
        warmup = 500
        frozen = []
        # Control-flow branch for conditional or iterative execution.
        for n, mmod in G.named_modules():
            # Control-flow branch for conditional or iterative execution.
            if isinstance(mmod, nn.Conv3d) and any(s in n for s in [".blocks.0", ".blocks.1"]):
                # Control-flow branch for conditional or iterative execution.
                for p in mmod.parameters():
                    p.requires_grad = False
                    frozen.append(p)
        
        # GBM iterator for rehearsal (mixing GBM batches during PDGM fine-tune)
        gbm_iter = iter(gbm_dl)
        it = 0
        
        d_loss_tgt = torch.tensor(0.0)
        g_loss_tgt = torch.tensor(0.0)

        # Control-flow branch for conditional or iterative execution.
        for epoch in range(1, epochs_tgt + 1):
            # Control-flow branch for conditional or iterative execution.
            for x, m, _ in pdgm_fs:
                it += 1
                
                # --- Rehearsal Step ---
                # With probability 0.25, train on a GBM batch instead of PDGM.
                # This helps prevent forgetting the source distribution.
                dom = torch.ones((x.size(0),), dtype=torch.long, device=device) # Domain 1 (PDGM)
                # Control-flow branch for conditional or iterative execution.
                if random.random() < 0.25:
                    # Control-flow branch for conditional or iterative execution.
                    try:
                        x, m, _ = next(gbm_iter)
                    # Control-flow branch for conditional or iterative execution.
                    except StopIteration:
                        gbm_iter = iter(gbm_dl)
                        x, m, _ = next(gbm_iter)
                    dom = torch.zeros((x.size(0),), dtype=torch.long, device=device) # Domain 0 (GBM)
                
                x, m = x.to(device), m.to(device)
                x, m = _fit_to_maxdim(x, m, max_dim)
                x, m = _random_crop(x, m, crop_dim)
                
                cond = domain_map(int(dom[0].item()), m)
                
                # ----------------------------
                # D step (ensemble)
                # ----------------------------
                # Control-flow branch for conditional or iterative execution.
                for p in D_ensemble.parameters():
                    p.requires_grad = True

                # Control-flow branch for conditional or iterative execution.
                with amp.autocast(enabled=torch.cuda.is_available()):
                    z = torch.randn(x.size(0), 128, device=device)
                    fake = G(z, cond).detach()

                    real_aug, mask_real_aug = diffaug3d(x, m)
                    fake_aug, mask_fake_aug = diffaug3d(fake, m)
                    
                    real_logits = D_ensemble(real_aug, mask_real_aug, dom)
                    fake_logits = D_ensemble(fake_aug, mask_fake_aug, dom)

                    d_loss = sum(hinge_d(rl, fl) for rl, fl in zip(real_logits, fake_logits))
                    d_loss_tgt = d_loss.detach().cpu()
                
                d_opt.zero_grad(set_to_none=True)
                scaler.scale(d_loss).backward()
                
                # R1 penalty (ensemble)
                x.requires_grad_(True)
                r1_total = torch.tensor(0.0, device=device)
                x_split = x.split(1, dim=1)
                
                # Control-flow branch for conditional or iterative execution.
                with amp.autocast(enabled=torch.cuda.is_available()):
                    # Control-flow branch for conditional or iterative execution.
                    for i in range(3):
                        r1 = r1_grad_penalty(D_ensemble.children().__next__()[i](x_split[i], m, dom), x_split[i])
                        r1_total = r1_total + r1
                    r1_total = r1_total * (r1_gamma_tgt / 2.0)
                    
                scaler.scale(r1_total).backward()
                scaler.step(d_opt)
                scaler.update()
                
                # ----------------------------
                # G step (ensemble)
                # ----------------------------
                # Control-flow branch for conditional or iterative execution.
                for p in D_ensemble.parameters():
                    p.requires_grad = False

                # Control-flow branch for conditional or iterative execution.
                with amp.autocast(enabled=torch.cuda.is_available()):
                    z = torch.randn(x.size(0), 128, device=device)
                    fake = G(z, cond)
                    fake_aug, mask_fake_aug = diffaug3d(fake, m)
                    
                    fake_logits = D_ensemble(fake_aug, mask_fake_aug, dom)
                    g_loss = sum(hinge_g(fl) for fl in fake_logits)
                    g_loss_tgt = g_loss.detach().cpu()
                    
                g_opt.zero_grad(set_to_none=True)
                scaler.scale(g_loss).backward()
                scaler.step(g_opt)
                scaler.update()
                ema.update()
                
                # After warmup iterations, unfreeze early layers to allow full adaptation
                # Control-flow branch for conditional or iterative execution.
                if it == warmup:
                    print("--- Unfreezing early Generator layers ---")
                    # Control-flow branch for conditional or iterative execution.
                    for p in frozen:
                        p.requires_grad = True
            
            # --- End of Target Epoch ---
            # Save both raw and EMA generator checkpoints
            ema.copy_to(G_ema)
            torch.save({"G": G.state_dict(), "D": D_ensemble.state_dict()}, Path(outdir)/f"tgt_e{epoch}.pt")
            torch.save({"G": G_ema.state_dict()}, Path(outdir)/f"tgt_e{epoch}_emaG.pt")
            print(f"[TGT] epoch {epoch}/{epochs_tgt} d={d_loss_tgt.item():.3f} g={g_loss_tgt.item():.3f}")

            # MODIFIED SAMPLING FREQUENCY: sample every 50 epochs (and last epoch)
            # Control-flow branch for conditional or iterative execution.
            if epoch % 50 == 0 or epoch == epochs_tgt:
                print(f"Generating samples for target epoch {epoch}...")
                sample_with_masks(G, str(Path(outdir)/f"tgt_e{epoch}_samples_pdgm_dev"), 
                                  dev_samples_pdgm, domain_id=1, ema_copy=G_ema)
                sample_with_masks(G, str(Path(outdir)/f"tgt_e{epoch}_samples_gbm_dev"), 
                                  dev_samples_gbm, domain_id=0, ema_copy=G_ema)

        ema.copy_to(G_ema)
        
    print("--- Training Complete ---")

# ---------------------------
# 12. Main execution block
# ---------------------------
# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    # CLI lets you run the same script on different machines / datasets without edits
    ap = argparse.ArgumentParser()
    
    # --- Data paths ---
    ap.add_argument('--gbm_root', default='/home/j98my/Pre-Processing/prep/gbm_all')
    ap.add_argument('--pdgm_root', default='/home/j98my/Pre-Processing/prep/pdgm_target')
    ap.add_argument('--fewshot',   default='/home/j98my/Pre-Processing/prep/pdgm_fewshot.txt')
    
    # --- Output path ---
    ap.add_argument('--outdir',    default='/home/j98my/models/runs/cgan3d')
    
    # --- Training settings ---
    ap.add_argument('--epochs_src', type=int, default=5)
    ap.add_argument('--epochs_tgt', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    
    # --- Pre-processing settings ---
    ap.add_argument('--max_dim', type=int, default=128)
    ap.add_argument('--crop_dim', type=int, default=112)
    
    args = ap.parse_args()
    
    # START THE TRAINING
    train(args.gbm_root, args.pdgm_root, args.outdir, args.fewshot, args.epochs_src, args.epochs_tgt, args.seed,
          max_dim=args.max_dim, crop_dim=args.crop_dim)
