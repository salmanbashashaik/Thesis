#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
train_cgan_1D.py — Mask-conditioned 3D cGAN for few-/zero-shot cross-domain MRI generation

Aligned with the proposal's claims:
- Cross-domain: source=GBM (data-rich), target=PDGM (data-scarce).
- Few-shot & zero-shot: K subjects via a text list; K=0 supported (no target updates).
- Structure-aware conditioning: tumor mask (1ch) guides anatomy; domain token steers style.
- Modalities: synthesize (t1,t2,flair) only; mask used only as condition (no supervised seg loss).
- Full volumes: uses preprocessed, cropped full-res volumes; training uses random 3D crops for VRAM safety.
- Reproducibility: deterministic seed, saved config, EMA generator, checkpointing, sample exports.

Inputs expected (from the preprocessing step):
  /.../gbm_all/<subject>/{t1,t2,flair,mask}.nii.gz
  /.../pdgm_target/<subject>/{t1,t2,flair,mask}.nii.gz
"""
# ============================================================
# Imports
# ============================================================
# - os/json/random/argparse/pathlib: run configuration, filesystem, CLI args, reproducibility
# Import dependencies used by this module.
import os, json, random, argparse
from pathlib import Path

# - numpy: array handling + nibabel interop
# Import dependencies used by this module.
import numpy as np

# - torch: model building, training loops, losses, interpolation, autograd
# Import dependencies used by this module.
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AMP (automatic mixed precision):
# - torch.amp provides autocast + GradScaler (newer API than torch.cuda.amp)
# - note: scaler/autocast are enabled only on CUDA
# Import dependencies used by this module.
import torch.amp as amp

# Medical imaging IO:
# - nibabel loads NIfTI (.nii/.nii.gz) volumes and provides affine matrices
# Import dependencies used by this module.
import nibabel as nib

# ============================================================
# Repro & small utils
# ============================================================
# Function: `set_seed` implements a reusable processing step.
def set_seed(seed: int = 42):
    """
    Make runs reproducible (as much as PyTorch allows).
    This controls:
      - random flips/augmentations
      - z sampling
      - weight initialization
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN reduces nondeterminism but can be slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function: `to_torch` implements a reusable processing step.
def to_torch(x):
    """Convert numpy -> torch with a defensive copy (avoids non-contig memory issues)."""
    # Return the computed value to the caller.
    return torch.from_numpy(x.copy())

# Function: `load_vol` implements a reusable processing step.
def load_vol(path):  # float32 in [-1,1]
    """
    Load a modality volume as float32 and map to [-1, 1].

    Assumption:
      - preprocessing already scaled intensities into [0,1] range
      - then we remap: x = clip(x,0,1)*2-1 for tanh GAN training
    """
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    # Control-flow branch for conditional or iterative execution.
    if arr.ndim == 3:
        arr = arr[None]  # [1,D,H,W]
    arr = np.clip(arr, 0.0, 1.0) * 2.0 - 1.0
    # Return the computed value to the caller.
    return arr

# Function: `load_mask` implements a reusable processing step.
def load_mask(path):
    """
    Load tumor mask as float32 binary in {0,1}.
    Mask is conditioning only (no segmentation loss in this GAN).
    """
    m = nib.load(path).get_fdata(dtype=np.float32)
    # Control-flow branch for conditional or iterative execution.
    if m.ndim == 3:
        m = m[None]  # [1,D,H,W]
    m = (m > 0.5).astype(np.float32)
    # Return the computed value to the caller.
    return m

# Class definition: `EMA` encapsulates related model behavior.
class EMA:
    """
    Exponential Moving Average of Generator parameters.

    Why:
      - GAN training can be noisy; EMA often yields cleaner, more stable samples.
      - We use EMA weights for qualitative exports / final sampling.
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, model, beta=0.999):
        self.m = model
        self.b = beta
        # Store shadow copies for trainable parameters only
        self.shadow = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    # Function: `update` implements a reusable processing step.
    def update(self):
        """shadow = beta*shadow + (1-beta)*param"""
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
        """Copy EMA weights into a target model (e.g., G_ema)."""
        i = 0
        # Control-flow branch for conditional or iterative execution.
        for p in target.parameters():
            # Control-flow branch for conditional or iterative execution.
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[i])
            i += 1

# ============================================================
# Data
# ============================================================
# Class definition: `VolFolder` encapsulates related model behavior.
class VolFolder(Dataset):
    """
    Dataset for subjects stored as folders:
      root/SubjectID/t1.nii.gz
      root/SubjectID/t2.nii.gz
      root/SubjectID/flair.nii.gz
      root/SubjectID/mask.nii.gz

    Returns:
      x    : torch.FloatTensor (3,D,H,W) in [-1,1]
      mask : torch.FloatTensor (1,D,H,W) in {0,1}
      sid  : string subject id
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, root, subjects=None):
        self.root = root
        subs = sorted(next(os.walk(root))[1])
        # If subjects is passed (few-shot list), restrict to that subset
        self.subs = subjects if subjects else subs

    # Function: `__len__` implements a reusable processing step.
    def __len__(self):
        # Return the computed value to the caller.
        return len(self.subs)

    # Function: `__getitem__` implements a reusable processing step.
    def __getitem__(self, i):
        sid = self.subs[i]
        d = f"{self.root}/{sid}"

        # Load modalities (each is [1,D,H,W])
        imgs = [load_vol(f"{d}/{k}.nii.gz") for k in ["t1", "t2", "flair"]]
        x = np.concatenate(imgs, axis=0)       # (3,D,H,W)

        # Load mask (1,D,H,W)
        m = load_mask(f"{d}/mask.nii.gz")

        # Return the computed value to the caller.
        return to_torch(x), to_torch(m), sid

# ============================================================
# SPADE blocks (3D)
# ============================================================
# Class definition: `SPADE3D` encapsulates related model behavior.
class SPADE3D(nn.Module):
    """
    SPADE layer for 3D volumes:
      - normalize activations (InstanceNorm, no affine)
      - compute per-voxel gamma/beta from condition tensor
      - apply: norm(x) * (1 + gamma) + beta
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, norm_nc, cond_nc):
        super().__init__()
        self.norm = nn.InstanceNorm3d(norm_nc, affine=False, eps=1e-5)
        hidden = max(32, cond_nc * 2)

        # Condition encoder (shared trunk) then gamma/beta heads
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(cond_nc, hidden, 3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv3d(hidden, norm_nc, 3, padding=1)
        self.mlp_beta  = nn.Conv3d(hidden, norm_nc, 3, padding=1)

    # Function: `forward` implements a reusable processing step.
    def forward(self, x, cond):
        # 1) normalize x
        n = self.norm(x)

        # 2) ensure conditioning volume matches x spatial size
        # Control-flow branch for conditional or iterative execution.
        if x.shape[-3:] != cond.shape[-3:]:
            cond = F.interpolate(cond, size=x.shape[-3:], mode='nearest')

        # 3) compute gamma/beta from cond and modulate
        a = self.mlp_shared(cond)
        # Return the computed value to the caller.
        return n * (1 + self.mlp_gamma(a)) + self.mlp_beta(a)

# Class definition: `SPADEResBlk3D` encapsulates related model behavior.
class SPADEResBlk3D(nn.Module):
    """
    SPADE Residual Block (3D).

    If upsample=True:
      - upsample feature maps by 2x (trilinear) before SPADE+Conv
      - skip path also upsamples so shapes align
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, fin, fout, cond_nc, upsample=False):
        super().__init__()
        self.ups = upsample
        fmid = min(fin, fout)

        self.sp1 = SPADE3D(fin,  cond_nc)
        self.c1  = nn.Conv3d(fin,  fmid, 3, padding=1)

        self.sp2 = SPADE3D(fmid, cond_nc)
        self.c2  = nn.Conv3d(fmid, fout, 3, padding=1)

        # Optional 1x1 conv if channel count changes
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
            # Upsample features (condition is resized inside SPADE as needed)
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

        # Residual addition helps stability and gradient flow
        # Return the computed value to the caller.
        return h + xs

# ============================================================
# Generator (mask + domain token conditioning)
# ============================================================
# Class definition: `Generator` encapsulates related model behavior.
class Generator(nn.Module):
    """
    SPADE Generator.

    Conditioning:
      cond = concat([mask, domain_map]) -> (B,2,D,H,W)

    Notes:
      - This version starts from a learned 1x1x1 seed (after fc)
      - Repeated SPADE upsampling blocks expand spatial resolution
      - Final output is 3 channels (t1,t2,flair) in [-1,1]
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, z_dim=128, base=16, cond_nc=2, out_ch=3):
        super().__init__()

        # Maps z to an initial channel vector that will be reshaped to (B,C,1,1,1)
        self.fc = nn.Linear(z_dim, base * 8)

        # Channel plan through upsampling stages
        ch = [base * 8, base * 8, base * 4, base * 2, base]

        # Four upsample blocks -> 1 -> 2 -> 4 -> 8 -> 16 (then final interpolate to D/H/W)
        self.blocks = nn.ModuleList([
            SPADEResBlk3D(ch[0], ch[1], cond_nc, upsample=True),
            SPADEResBlk3D(ch[1], ch[2], cond_nc, upsample=True),
            SPADEResBlk3D(ch[2], ch[3], cond_nc, upsample=True),
            SPADEResBlk3D(ch[3], ch[4], cond_nc, upsample=True),
        ])

        # Output head: conv + tanh for normalized intensities
        self.to_img = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ch[4], out_ch, 3, padding=1),
            nn.Tanh()
        )

    # Function: `forward` implements a reusable processing step.
    def forward(self, z, cond):
        """
        z:    (B, z_dim)
        cond: (B, 2, D, H, W) = [mask, domain_map]
        """
        B, _, D, H, W = cond.shape

        # Start from a tiny learned "seed" volume (1x1x1)
        h = self.fc(z).view(B, -1, 1, 1, 1)

        # Grow spatial size using SPADE-conditioned residual upsampling blocks
        # Control-flow branch for conditional or iterative execution.
        for blk in self.blocks:
            h = blk(h, cond)

        # Force exact match to data crop size (important if starting resolution differs)
        # Control-flow branch for conditional or iterative execution.
        if h.shape[-3:] != (D, H, W):
            h = F.interpolate(h, size=(D, H, W), mode='trilinear', align_corners=False)

        # Return the computed value to the caller.
        return self.to_img(h)

# ============================================================
# Discriminator (PatchGAN-ish conv tower + projection)
# ============================================================
# Class definition: `Disc` encapsulates related model behavior.
class Disc(nn.Module):
    """
    Projection Discriminator with conditioning via concatenated mask.

    Input:
      - x:    (B,3,D,H,W)  real or fake MRI volume
      - mask: (B,1,D,H,W)  condition (tumor region)
      - domain: (B,)       integer domain id (0=GBM, 1=PDGM)

    Design:
      - Concatenate x and mask -> 4-channel input
      - Strided conv tower downsamples
      - Global average pool -> feature vector
      - Score = linear(h) + dot(proj(h), emb(domain))
    """
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, in_ch=4, base=16, emb_dim=32):
        super().__init__()

        # Function: `blk` implements a reusable processing step.
        def blk(ci, co):
            # Return the computed value to the caller.
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, True)
            )

        self.net = nn.Sequential(
            blk(in_ch,    base),      # downsample 2x
            blk(base,     base * 2),
            blk(base * 2, base * 4),
            blk(base * 4, base * 8),
            nn.Conv3d(base * 8, base * 8, 3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Real/Fake score head
        self.fc = nn.Linear(base * 8, 1)

        # Domain projection components
        self.emb  = nn.Embedding(2, emb_dim)
        self.proj = nn.Linear(base * 8, emb_dim)

        # Spectral normalization for stability (common in hinge/projection GANs)
        # Control-flow branch for conditional or iterative execution.
        for m in self.modules():
            # Control-flow branch for conditional or iterative execution.
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.utils.spectral_norm(m)

    # Function: `forward` implements a reusable processing step.
    def forward(self, x, mask, domain):
        # Concatenate mask as an additional input channel (conditional D)
        h = torch.cat([x, mask], dim=1)

        # Conv tower -> (B,C,*,*,*) then GAP -> (B,C)
        h = self.net(h).mean(dim=[2, 3, 4])

        out = self.fc(h)
        proj = (self.proj(h) * self.emb(domain)).sum(dim=1, keepdim=True)
        # Return the computed value to the caller.
        return out + proj

# ============================================================
# DiffAugment (minimal 3D)
# ============================================================
# Function: `diffaug3d` implements a reusable processing step.
def diffaug3d(x):
    """
    Lightweight augmentation to reduce discriminator overfitting:
      - random flips in depth/height/width
      - mild brightness/contrast jitter in [-1,1] space
    """
    # Control-flow branch for conditional or iterative execution.
    if random.random() < 0.5:
        x = torch.flip(x, [2])
    # Control-flow branch for conditional or iterative execution.
    if random.random() < 0.5:
        x = torch.flip(x, [3])
    # Control-flow branch for conditional or iterative execution.
    if random.random() < 0.5:
        x = torch.flip(x, [4])
    # Control-flow branch for conditional or iterative execution.
    if random.random() < 0.5:
        a = 1.0 + 0.1 * (2 * random.random() - 1)  # contrast-ish multiplier
        b = 0.1 * (2 * random.random() - 1)        # brightness-ish offset
        x = (x * a + b).clamp(-1, 1)
    # Return the computed value to the caller.
    return x

# ============================================================
# Losses
# ============================================================
# Function: `hinge_d` implements a reusable processing step.
def hinge_d(real_logits, fake_logits):
    """
    Discriminator hinge loss:
      L = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
    """
    # Return the computed value to the caller.
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()

# Function: `hinge_g` implements a reusable processing step.
def hinge_g(fake_logits):
    """Generator hinge loss: maximize D(fake) -> minimize -D(fake)."""
    # Return the computed value to the caller.
    return -fake_logits.mean()

# Function: `r1_grad_penalty` implements a reusable processing step.
def r1_grad_penalty(d_out, x):
    """
    R1 penalty on real images:
      gamma/2 * E[||∇_x D(x)||^2]

    Stabilizes D and prevents overly sharp decision boundaries.
    """
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x,
        create_graph=True
    )[0]
    # Return the computed value to the caller.
    return (grad.pow(2).flatten(1).sum(1)).mean()

# ============================================================
# Sampling (writes NIfTI for qualitative checks)
# ============================================================
@torch.no_grad()
# Function: `sample_with_masks` implements a reusable processing step.
def sample_with_masks(G, outdir, samples, domain_id=1, ema_copy=None):
    """
    Generate a set of samples given a list of masks:
      samples: list[(sid, mask_path)]

    Saves:
      outdir/sid/{t1,t2,flair}_synth.nii.gz
    """
    os.makedirs(outdir, exist_ok=True)

    # Use EMA weights if provided
    model = G if ema_copy is None else ema_copy
    model.eval()
    device = next(model.parameters()).device

    # Control-flow branch for conditional or iterative execution.
    for sid, mpath in samples:
        # Load mask -> (1,D,H,W), batch -> (1,1,D,H,W)
        m = load_mask(mpath)
        m_t = to_torch(m)[None].to(device)

        # Domain channel as a constant volume
        dom = torch.full_like(m_t, float(domain_id))

        # Conditioning tensor for SPADE: (1,2,D,H,W)
        cond = torch.cat([m_t, dom], dim=1)

        # Sample latent noise and generate synthetic volume
        z = torch.randn(1, 128, device=device)
        x = model(z, cond).cpu().numpy()[0]  # (3,D,H,W) in [-1,1]

        # Map back to [0,1] for saving/visualization
        x = (x + 1.0) / 2.0

        # Save into subject folder (NOTE: uses identity affine here)
        d = Path(outdir) / sid
        d.mkdir(parents=True, exist_ok=True)
        # Control-flow branch for conditional or iterative execution.
        for k, mod in enumerate(["t1", "t2", "flair"]):
            nib.save(nib.Nifti1Image(x[k], np.eye(4)), str(d / f"{mod}_synth.nii.gz"))

# ============================================================
# Train utilities (resize + random crop for VRAM safety)
# ============================================================
# Function: `_fit_to_maxdim` implements a reusable processing step.
def _fit_to_maxdim(x, m, max_dim=128):
    """
    If any spatial dim is bigger than max_dim, isotropically downscale.

    x: (B,3,D,H,W) in [-1,1]
    m: (B,1,D,H,W) in {0,1}
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
    newD, newH, newW = int(round(Dv * scale)), int(round(Hv * scale)), int(round(Wv * scale))

    # Images: trilinear (smooth), masks: nearest (binary)
    x = F.interpolate(x, size=(newD, newH, newW), mode='trilinear', align_corners=False)
    m = F.interpolate(m, size=(newD, newH, newW), mode='nearest')
    # Return the computed value to the caller.
    return x, m

# Function: `_random_crop` implements a reusable processing step.
def _random_crop(x, m, crop_dim=112):
    """
    Random 3D crop of size crop_dim^3.

    This is the main VRAM-saver:
      - full volumes might be bigger
      - discriminators in particular get expensive at high res
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

    x = x[:, :, d:d + crop_dim, h:h + crop_dim, w:w + crop_dim]
    m = m[:, :, d:d + crop_dim, h:h + crop_dim, w:w + crop_dim]
    # Return the computed value to the caller.
    return x, m

# ============================================================
# Train
# ============================================================
# Function: `train` implements a reusable processing step.
def train(
    gbm_root,
    pdgm_root,
    outdir,
    fewshot_path=None,
    epochs_src=5,
    epochs_tgt=5,
    seed=42,
    r1_gamma_src=10.0,
    r1_gamma_tgt=20.0,
    ema_beta=0.999,
    max_dim=128,
    crop_dim=112
):
    """
    Two-stage training:
      Stage A (source): train on GBM (domain 0)
      Stage B (target): fine-tune on PDGM few-shot list (domain 1) + rehearsal

    If fewshot list is empty or missing:
      - pdgm_fs=None -> Stage B is skipped (zero-shot setting)
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(outdir, exist_ok=True)

    # Save config for reproducibility and experiment tracking
    json.dump(
        {
            "gbm_root": gbm_root,
            "pdgm_root": pdgm_root,
            "fewshot": fewshot_path,
            "epochs_src": epochs_src,
            "epochs_tgt": epochs_tgt,
            "seed": seed,
            "r1_gamma_src": r1_gamma_src,
            "r1_gamma_tgt": r1_gamma_tgt,
            "ema_beta": ema_beta,
            "max_dim": max_dim,
            "crop_dim": crop_dim,
        },
        open(Path(outdir) / 'config.json', 'w'),
        indent=2
    )

    # ----------------------------
    # Data loaders
    # ----------------------------
    gbm_ds = VolFolder(gbm_root)
    gbm_dl = DataLoader(gbm_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    pdgm_all = sorted(next(os.walk(pdgm_root))[1])
    few = []
    # Control-flow branch for conditional or iterative execution.
    if fewshot_path and os.path.exists(fewshot_path):
        few = [l.strip() for l in open(fewshot_path) if l.strip()]

    # Dev IDs for qualitative sampling: PDGM subjects not in few-shot list
    dev = [s for s in pdgm_all if s not in set(few)][:min(16, len(pdgm_all))]

    # Few-shot PDGM loader (None => Stage B skipped)
    pdgm_fs = DataLoader(VolFolder(pdgm_root, few), batch_size=1, shuffle=True, num_workers=2, pin_memory=True) if few else None

    # ----------------------------
    # Models
    # ----------------------------
    G = Generator(z_dim=128, base=16, cond_nc=2, out_ch=3).to(device)
    D = Disc(in_ch=4, base=16, emb_dim=32).to(device)

    # EMA copy for stable sampling
    G_ema = Generator(z_dim=128, base=16, cond_nc=2, out_ch=3).to(device)
    G_ema.load_state_dict(G.state_dict())
    ema = EMA(G, beta=ema_beta)

    # Optimizers:
    # - lr=2e-4 is a common hinge GAN default
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.0, 0.99))

    # AMP grad scaler: only active on CUDA; on CPU it effectively becomes no-op
    scaler = amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Function: `domain_map` implements a reusable processing step.
    def domain_map(domain_id, mask):
        """
        Create generator conditioning tensor:
          cond = [mask, domain_volume]
        where domain_volume is a constant-valued 3D map with value domain_id.
        """
        B, _, Dv, Hv, Wv = mask.shape
        dom = torch.full((B, 1, Dv, Hv, Wv), float(domain_id), device=mask.device)
        # Return the computed value to the caller.
        return torch.cat([mask, dom], dim=1)

    # ============================================================
    # Stage A: Source pretrain (GBM)
    # ============================================================
    # Control-flow branch for conditional or iterative execution.
    for epoch in range(1, epochs_src + 1):
        # Control-flow branch for conditional or iterative execution.
        for x, m, _ in gbm_dl:
            x, m = x.to(device), m.to(device)

            # VRAM safety: optional downscale + random crop to fixed size
            x, m = _fit_to_maxdim(x, m, max_dim)
            x, m = _random_crop(x, m, crop_dim)

            dom0 = torch.zeros((x.size(0),), dtype=torch.long, device=device)  # GBM domain id
            cond = domain_map(0, m)

            # ----------------------------
            # Discriminator step
            # ----------------------------
            # Control-flow branch for conditional or iterative execution.
            for p in D.parameters():
                p.requires_grad = True

            # Control-flow branch for conditional or iterative execution.
            with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                z = torch.randn(x.size(0), 128, device=device)
                fake = G(z, cond).detach()  # detach: don't backprop into G on D step
                real_logits = D(diffaug3d(x), m, dom0)
                fake_logits = D(diffaug3d(fake), m, dom0)
                d_loss = hinge_d(real_logits, fake_logits)

            d_opt.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()

            # R1 regularization on real images
            x.requires_grad_(True)
            # Control-flow branch for conditional or iterative execution.
            with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                r1 = r1_grad_penalty(D(x, m, dom0), x) * (r1_gamma_src / 2.0)

            scaler.scale(r1).backward()
            scaler.step(d_opt)
            scaler.update()

            # ----------------------------
            # Generator step
            # ----------------------------
            # Control-flow branch for conditional or iterative execution.
            for p in D.parameters():
                p.requires_grad = False

            # Control-flow branch for conditional or iterative execution.
            with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                z = torch.randn(x.size(0), 128, device=device)
                fake = G(z, cond)
                g_loss = hinge_g(D(diffaug3d(fake), m, dom0))

            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()

            # Update EMA after G update
            ema.update()

        # Save checkpoints each epoch
        torch.save({"G": G.state_dict(), "D": D.state_dict()}, Path(outdir) / f"src_e{epoch}.pt")
        print(f"[SRC] epoch {epoch}/{epochs_src} d={d_loss.item():.3f} g={g_loss.item():.3f}")

    # Sync EMA weights into G_ema after source training
    ema.copy_to(G_ema)

    # ============================================================
    # Stage B: Few-shot target fine-tune (PDGM)
    # ============================================================
    # Control-flow branch for conditional or iterative execution.
    if epochs_tgt > 0 and pdgm_fs is not None:
        # Freeze early SPADE blocks briefly to preserve coarse priors
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

        # Rehearsal iterator over GBM
        gbm_iter = iter(gbm_dl)
        it = 0

        # Control-flow branch for conditional or iterative execution.
        for epoch in range(1, epochs_tgt + 1):
            # Control-flow branch for conditional or iterative execution.
            for x, m, _ in pdgm_fs:
                it += 1

                # Rehearsal mixing:
                # - 75% PDGM updates (domain=1)
                # - 25% GBM updates (domain=0) to reduce catastrophic forgetting
                # Control-flow branch for conditional or iterative execution.
                if random.random() < 0.25:
                    # Control-flow branch for conditional or iterative execution.
                    try:
                        x, m, _ = next(gbm_iter)
                    # Control-flow branch for conditional or iterative execution.
                    except StopIteration:
                        gbm_iter = iter(gbm_dl)
                        x, m, _ = next(gbm_iter)
                    dom = torch.zeros((x.size(0),), dtype=torch.long, device=device)
                else:
                    dom = torch.ones((x.size(0),), dtype=torch.long, device=device)

                x, m = x.to(device), m.to(device)
                x, m = _fit_to_maxdim(x, m, max_dim)
                x, m = _random_crop(x, m, crop_dim)

                # Condition uses the current domain label
                cond = domain_map(int(dom[0].item()), m)

                # ----------------------------
                # D step
                # ----------------------------
                # Control-flow branch for conditional or iterative execution.
                with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    z = torch.randn(x.size(0), 128, device=device)
                    fake = G(z, cond).detach()
                    d_loss = hinge_d(D(diffaug3d(x), m, dom), D(diffaug3d(fake), m, dom))

                d_opt.zero_grad(set_to_none=True)
                scaler.scale(d_loss).backward()

                # R1 penalty
                x.requires_grad_(True)
                # Control-flow branch for conditional or iterative execution.
                with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    r1 = r1_grad_penalty(D(x, m, dom), x) * (r1_gamma_tgt / 2.0)

                scaler.scale(r1).backward()
                scaler.step(d_opt)
                scaler.update()

                # ----------------------------
                # G step
                # ----------------------------
                # Control-flow branch for conditional or iterative execution.
                with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    z = torch.randn(x.size(0), 128, device=device)
                    fake = G(z, cond)
                    g_loss = hinge_g(D(diffaug3d(fake), m, dom))

                g_opt.zero_grad(set_to_none=True)
                scaler.scale(g_loss).backward()
                scaler.step(g_opt)
                scaler.update()
                ema.update()

                # Unfreeze early layers after warmup iterations
                # Control-flow branch for conditional or iterative execution.
                if it == warmup:
                    # Control-flow branch for conditional or iterative execution.
                    for p in frozen:
                        p.requires_grad = True

            # Save target checkpoints
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, Path(outdir) / f"tgt_e{epoch}.pt")
            ema.copy_to(G_ema)  # keep EMA in sync before saving
            torch.save({"G": G_ema.state_dict()}, Path(outdir) / f"tgt_e{epoch}_emaG.pt")
            print(f"[TGT] epoch {epoch}/{epochs_tgt} d={d_loss.item():.3f} g={g_loss.item():.3f}")

    # ============================================================
    # Final qualitative export (PDGM dev masks) using EMA weights
    # ============================================================
    dev_samples = [(sid, f"{pdgm_root}/{sid}/mask.nii.gz") for sid in dev]
    sample_with_masks(G, str(Path(outdir) / "samples_dev"), dev_samples, domain_id=1, ema_copy=G_ema)

# ============================================================
# CLI entry point
# ============================================================
# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Data paths
    ap.add_argument('--gbm_root', default='/home/j98my/Pre-Processing/prep/gbm_all')
    ap.add_argument('--pdgm_root', default='/home/j98my/Pre-Processing/prep/pdgm_target')
    ap.add_argument('--fewshot',   default='/home/j98my/Pre-Processing/prep/pdgm_fewshot.txt')

    # Output directory
    ap.add_argument('--outdir',    default='/home/j98my/models/runs/cgan3d')

    # Training knobs
    ap.add_argument('--epochs_src', type=int, default=5)
    ap.add_argument('--epochs_tgt', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)

    # Preprocessing knobs
    ap.add_argument('--max_dim', type=int, default=128)
    ap.add_argument('--crop_dim', type=int, default=112)

    args = ap.parse_args()

    train(
        args.gbm_root,
        args.pdgm_root,
        args.outdir,
        args.fewshot,
        args.epochs_src,
        args.epochs_tgt,
        args.seed,
        max_dim=args.max_dim,
        crop_dim=args.crop_dim
    )
