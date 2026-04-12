#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
Train an AlexNet-like classifier (AlexLite-DG) on diffuse glioma MR slices.

Data layout (patient-wise directories):
    /home/j98my/Pre-Processing/prep/pdgm_target/
        P001/
            t1.nii.gz
            t2.nii.gz
            flair.nii.gz
            mask.nii.gz   # binary mask (any >0 treated as tumor)
        P002/
            ...

Task: slice-level binary classification (tumor vs non-tumor) using 2.5D input
(T1,T2,FLAIR stacked as channels). Labels are derived automatically per slice:
    y=1 if any voxel in mask slice > 0 else y=0.

We do a patient-wise split to avoid leakage. Validation/test sets contain only
patients unseen in training. Evaluation is reported on the validation and test
sets using ROC-AUC and accuracy; Grad-CAM heatmaps are exported for a few
examples to sanity-check localization.

Usage (example):
    python3 train_alexlite_dg.py \
      --data_root /home/j98my/Pre-Processing/prep/pdgm_target \
      --outdir /home/j98my/models/runs/alexlite_dg \
      --epochs 15 --batch_size 32 --img_size 224 --seed 1337

Notes:
- Slices are taken along axial (last) dimension. Adjust with --plane if needed.
- Minimal augmentations; feel free to extend.
- Requires: nibabel, numpy, torch, scikit-learn, pillow, tqdm
"""

# Import dependencies used by this module.
from __future__ import annotations
import os
import sys
import math
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import nibabel as nib
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# ---------------------------
# Model: AlexLite-DG (2D)
# ---------------------------
# Class definition: `AlexLiteDG` encapsulates related model behavior.
class AlexLiteDG(nn.Module):
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    # Function: `forward` implements a reusable processing step.
    def forward(self, x):
        feats = self.features(x)              # (B,256,H',W')
        gap = F.adaptive_avg_pool2d(feats, 1) # (B,256,1,1)
        gap = gap.view(gap.size(0), -1)       # (B,256)
        x = self.dropout(gap)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)
        # Return the computed value to the caller.
        return logits, feats

# ---------------------------
# Data utilities
# ---------------------------

# Function: `load_nifti` implements a reusable processing step.
def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    # Return the computed value to the caller.
    return np.asarray(data, dtype=np.float32)


# Function: `zscore_per_volume` implements a reusable processing step.
def zscore_per_volume(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = float(np.mean(vol))
    s = float(np.std(vol))
    # Control-flow branch for conditional or iterative execution.
    if s < eps:
        # Return the computed value to the caller.
        return np.zeros_like(vol, dtype=np.float32)
    # Return the computed value to the caller.
    return (vol - m) / (s + eps)


# Function: `minmax_per_volume` implements a reusable processing step.
def minmax_per_volume(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    # Control-flow branch for conditional or iterative execution.
    if vmax - vmin < eps:
        # Return the computed value to the caller.
        return np.zeros_like(vol, dtype=np.float32)
    # Return the computed value to the caller.
    return (vol - vmin) / (vmax - vmin + eps)


# Function: `center_crop_or_pad` implements a reusable processing step.
def center_crop_or_pad(img: torch.Tensor, size: int) -> torch.Tensor:
    # img: (C,H,W). We resize with bilinear then center-crop/pad to square size.
    c, h, w = img.shape
    # First, shortest side -> size
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = TF.resize(img, [new_h, new_w], antialias=True)
    # Center crop or pad
    top = max((new_h - size) // 2, 0)
    left = max((new_w - size) // 2, 0)
    img = TF.crop(img, top, left, min(size, new_h), min(size, new_w))
    # If smaller, pad
    pad_h = size - img.shape[1]
    pad_w = size - img.shape[2]
    # Control-flow branch for conditional or iterative execution.
    if pad_h > 0 or pad_w > 0:
        img = TF.pad(img, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2])
    # Return the computed value to the caller.
    return img


# Class definition: `SliceIndex` encapsulates related model behavior.
class SliceIndex:
    __slots__ = ("patient", "k", "label")
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, patient: str, k: int, label: int):
        self.patient = patient
        self.k = k  # slice index along axial (Z)
        self.label = label  # 0/1


# Class definition: `GliomaSliceDataset` encapsulates related model behavior.
class GliomaSliceDataset(Dataset):
    # Function: `__init__` implements a reusable processing step.
    def __init__(
        self,
        patient_dirs: List[Path],
        img_size: int = 224,
        plane: str = "axial",  # future-proof; currently only axial
        norm: str = "zscore",   # or "minmax"
        augment: bool = False,
        balance_neg_pos: bool = False,
        pos_frac_cap: float = 1.0,  # optionally cap pos rate (<=1.0)
        rng: random.Random | None = None,
    ):
        self.patient_dirs = patient_dirs
        self.img_size = img_size
        self.plane = plane
        self.norm = norm
        self.augment = augment
        self.balance_neg_pos = balance_neg_pos
        self.pos_frac_cap = pos_frac_cap
        self.rng = rng or random.Random(1234)

        # Build slice index list per patient
        self.meta: Dict[str, Dict[str, Path]] = {}
        # Control-flow branch for conditional or iterative execution.
        for p in self.patient_dirs:
            paths = {
                "t1": p / "t1.nii.gz",
                "t2": p / "t2.nii.gz",
                "flair": p / "flair.nii.gz",
                "mask": p / "mask.nii.gz",
            }
            # Control-flow branch for conditional or iterative execution.
            if not all(pp.exists() for pp in paths.values()):
                continue
            self.meta[p.name] = paths

        self.slice_index: List[SliceIndex] = []
        self._build_index()

    # Function: `_build_index` implements a reusable processing step.
    def _build_index(self):
        tmp_pos: List[SliceIndex] = []
        tmp_neg: List[SliceIndex] = []
        # Control-flow branch for conditional or iterative execution.
        for patient, paths in self.meta.items():
            mask = load_nifti(paths["mask"]).astype(np.float32)
            # assume axial along last axis (Z)
            depth = mask.shape[-1]
            # Control-flow branch for conditional or iterative execution.
            for k in range(depth):
                msl = mask[:, :, k]
                label = int(np.any(msl > 0.0))
                # Control-flow branch for conditional or iterative execution.
                if label == 1:
                    tmp_pos.append(SliceIndex(patient, k, 1))
                else:
                    tmp_neg.append(SliceIndex(patient, k, 0))
        # Optional balancing
        # Control-flow branch for conditional or iterative execution.
        if self.balance_neg_pos and len(tmp_pos) > 0:
            self.rng.shuffle(tmp_neg)
            tmp_neg = tmp_neg[: len(tmp_pos)]
        # Optional cap on positive fraction
        # Control-flow branch for conditional or iterative execution.
        if self.pos_frac_cap < 1.0 and len(tmp_pos) > 0:
            cap = int(math.floor(len(tmp_pos) * self.pos_frac_cap))
            self.rng.shuffle(tmp_pos)
            tmp_pos = tmp_pos[:cap]
        self.slice_index = tmp_pos + tmp_neg
        self.rng.shuffle(self.slice_index)

    # Function: `__len__` implements a reusable processing step.
    def __len__(self) -> int:
        # Return the computed value to the caller.
        return len(self.slice_index)

    # Function: `_load_patient_modalities` implements a reusable processing step.
    def _load_patient_modalities(self, patient: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        paths = self.meta[patient]
        t1 = load_nifti(paths["t1"])  # HxWxZ
        t2 = load_nifti(paths["t2"])  # HxWxZ
        fl = load_nifti(paths["flair"])  # HxWxZ
        ms = load_nifti(paths["mask"]).astype(np.float32)
        # Return the computed value to the caller.
        return t1, t2, fl, ms

    # Function: `_norm_vol` implements a reusable processing step.
    def _norm_vol(self, vol: np.ndarray) -> np.ndarray:
        # Control-flow branch for conditional or iterative execution.
        if self.norm == "zscore":
            # Return the computed value to the caller.
            return zscore_per_volume(vol)
        # Control-flow branch for conditional or iterative execution.
        elif self.norm == "minmax":
            # Return the computed value to the caller.
            return minmax_per_volume(vol)
        else:
            # Return the computed value to the caller.
            return vol

    # Function: `_apply_augment` implements a reusable processing step.
    def _apply_augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W)
        # Control-flow branch for conditional or iterative execution.
        if self.augment:
            # Control-flow branch for conditional or iterative execution.
            if self.rng.random() < 0.5:
                x = TF.hflip(x)
            # Control-flow branch for conditional or iterative execution.
            if self.rng.random() < 0.5:
                x = TF.vflip(x)
            # small random affine
            # Control-flow branch for conditional or iterative execution.
            if self.rng.random() < 0.3:
                angle = self.rng.uniform(-7.0, 7.0)
                x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)
        # Return the computed value to the caller.
        return x

    # Function: `__getitem__` implements a reusable processing step.
    def __getitem__(self, idx: int):
        si = self.slice_index[idx]
        # Load volumes per patient lazily per sample;
        t1, t2, fl, ms = self._load_patient_modalities(si.patient)
        k = si.k
        # Extract axial slice
        t1 = self._norm_vol(t1)
        t2 = self._norm_vol(t2)
        fl = self._norm_vol(fl)
        s1 = t1[:, :, k]
        s2 = t2[:, :, k]
        s3 = fl[:, :, k]

        # Stack to (C,H,W)
        x = np.stack([s1, s2, s3], axis=0)
        x = torch.from_numpy(x).float()
        x = center_crop_or_pad(x, self.img_size)
        x = self._apply_augment(x)
        y = torch.tensor(si.label, dtype=torch.long)
        # Return the computed value to the caller.
        return x, y


# ---------------------------
# Grad-CAM helper
# ---------------------------
# Class definition: `GradCAM` encapsulates related model behavior.
class GradCAM:
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, model: AlexLiteDG, target_layer_name: str = "features"):
        self.model = model
        self.model.eval()
        self.features = None
        self.gradients = None
        # hook the last conv block output
        # Function: `fwd_hook` implements a reusable processing step.
        def fwd_hook(module, inp, out):
            self.features = out.detach()
        # Function: `bwd_hook` implements a reusable processing step.
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        # register on the last layer of features
        self.handle_fwd = model.features.register_forward_hook(fwd_hook)
        self.handle_bwd = model.features.register_backward_hook(bwd_hook)

    # Function: `__del__` implements a reusable processing step.
    def __del__(self):
        # Control-flow branch for conditional or iterative execution.
        try:
            self.handle_fwd.remove()
            self.handle_bwd.remove()
        # Control-flow branch for conditional or iterative execution.
        except Exception:
            pass

    @torch.no_grad()
    # Function: `_safe_softmax` implements a reusable processing step.
    def _safe_softmax(self, logits):
        # Return the computed value to the caller.
        return F.softmax(logits, dim=1)

    # Function: `generate` implements a reusable processing step.
    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> Tuple[torch.Tensor, int]:
        # x: (1,3,H,W)
        logits, feats = self.model(x)
        # Control-flow branch for conditional or iterative execution.
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        # backward for chosen class
        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        grads = self.gradients           # (1,C,h,w)
        fmap = self.features             # (1,C,h,w)
        weights = torch.mean(grads, dim=(2,3), keepdim=True)  # (1,C,1,1)
        cam = torch.sum(weights * fmap, dim=1)  # (1,h,w)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)  # (1,H,W)
        # Return the computed value to the caller.
        return cam, class_idx


# ---------------------------
# Training / Eval loops
# ---------------------------

# Function: `set_seed` implements a reusable processing step.
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Function: `patient_split` implements a reusable processing step.
def patient_split(root: Path, seed: int, train_ratio=0.7, val_ratio=0.15) -> Tuple[List[Path], List[Path], List[Path]]:
    patients = [p for p in root.iterdir() if p.is_dir()]
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = patients[:n_train]
    val = patients[n_train:n_train+n_val]
    test = patients[n_train+n_val:]
    # Return the computed value to the caller.
    return train, val, test


# Function: `run_epoch` implements a reusable processing step.
def run_epoch(model, loader, device, criterion, optimizer=None) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    ys, ps = [], []
    # Control-flow branch for conditional or iterative execution.
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        # Control-flow branch for conditional or iterative execution.
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        prob = F.softmax(logits, dim=1)[:, 1]
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    ys = np.concatenate(ys, axis=0)
    ps = np.concatenate(ps, axis=0)
    # Control-flow branch for conditional or iterative execution.
    try:
        auc = roc_auc_score(ys, ps)
    # Control-flow branch for conditional or iterative execution.
    except ValueError:
        auc = float("nan")
    pred = (ps >= 0.5).astype(np.int32)
    acc = accuracy_score(ys, pred)
    avg_loss = total_loss / max(1, len(loader.dataset))
    # Return the computed value to the caller.
    return avg_loss, auc, acc


# Function: `save_checkpoint` implements a reusable processing step.
def save_checkpoint(state: dict, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(outpath))


# Function: `save_gradcam_examples` implements a reusable processing step.
def save_gradcam_examples(model, loader, device, outdir: Path, max_batches: int = 2):
    model.eval()
    cammer = GradCAM(model)
    outdir.mkdir(parents=True, exist_ok=True)
    count = 0
    # Control-flow branch for conditional or iterative execution.
    for i, (xb, yb) in enumerate(loader):
            # Control-flow branch for conditional or iterative execution.
            if i >= max_batches:
                break
            # Control-flow branch for conditional or iterative execution.
            for j in range(xb.size(0)):
                x = xb[j:j+1].to(device)
                cam, cls_idx = cammer.generate(x, None)
                # Convert to image overlay (simple alpha blend)
                img = x[0].detach().cpu()
                # Use FLAIR channel as grayscale background (index 2)
                base = img[2]
                base = base - base.min()
                base = base / (base.max() + 1e-6)
                base_img = (base.numpy() * 255).astype(np.uint8)
                base_img = Image.fromarray(base_img)
                heat = (cam[0].detach().cpu().numpy() * 255).astype(np.uint8)
                heat_img = Image.fromarray(heat).resize(base_img.size, resample=Image.BILINEAR)
                heat_img = heat_img.convert("RGBA")
                base_img = base_img.convert("RGBA")
                # simple colormap by putting heat into red channel
                r = heat_img.split()[0]
                overlay = Image.merge("RGBA", (r, Image.new("L", r.size), Image.new("L", r.size), Image.new("L", r.size, 128)))
                blended = Image.alpha_composite(base_img, overlay)
                label = int(yb[j].item())
                fname = outdir / f"cam_{count:04d}_y{label}_pred{cls_idx}.png"
                blended.save(fname)
                count += 1


# ---------------------------
# Main
# ---------------------------

# Function: `main` implements a reusable processing step.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to patient directories (each contains t1/t2/flair/mask).")
    ap.add_argument("--outdir", type=str, required=True, help="Output dir for checkpoints and logs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--balance", action="store_true", help="Balance negatives to positives in training")
    ap.add_argument("--pos_frac_cap", type=float, default=1.0, help="Cap pos fraction (<=1.0) if desired")
    ap.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--plane", type=str, default="axial")
    ap.add_argument("--export_cam", action="store_true", help="Export a few Grad-CAM panels after training")
    args = ap.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Patient-wise split
    train_pat, val_pat, test_pat = patient_split(data_root, seed=args.seed)

    # Datasets
    train_ds = GliomaSliceDataset(
        train_pat, img_size=args.img_size, plane=args.plane, norm=args.norm,
        augment=True, balance_neg_pos=args.balance, pos_frac_cap=args.pos_frac_cap,
        rng=random.Random(args.seed + 1),
    )
    val_ds = GliomaSliceDataset(
        val_pat, img_size=args.img_size, plane=args.plane, norm=args.norm,
        augment=False, balance_neg_pos=False, rng=random.Random(args.seed + 2),
    )
    test_ds = GliomaSliceDataset(
        test_pat, img_size=args.img_size, plane=args.plane, norm=args.norm,
        augment=False, balance_neg_pos=False, rng=random.Random(args.seed + 3),
    )

    # DataLoaders
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = AlexLiteDG(num_classes=2).to(device)

    # Class weights (optional): infer from training set distribution
    # quick pass to count labels
    lbls = []
    # Control-flow branch for conditional or iterative execution.
    for _, y in DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0):
        lbls.append(y.numpy())
    lbls = np.concatenate(lbls) if len(lbls) else np.array([0,1])
    pos = max(1, int(np.sum(lbls == 1)))
    neg = max(1, int(np.sum(lbls == 0)))
    # bigger weight on minority class (pos)
    w_neg = 1.0
    w_pos = neg / max(1, pos)
    class_weights = torch.tensor([w_neg, w_pos], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = -1.0
    ckpt_path = outdir / "best.pt"

    # Control-flow branch for conditional or iterative execution.
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc, tr_acc = run_epoch(model, train_ld, device, criterion, optimizer)
        va_loss, va_auc, va_acc = run_epoch(model, val_ld, device, criterion, optimizer=None)
        scheduler.step()
        dt = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_auc": tr_auc,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_auc": va_auc,
            "val_acc": va_acc,
            "lr": scheduler.get_last_lr()[0]
        }
        print(json.dumps(log))

        # Control-flow branch for conditional or iterative execution.
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc": va_auc,
                "args": vars(args),
            }, ckpt_path)

        # Control-flow branch for conditional or iterative execution.
        if (epoch % args.save_every) == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc": va_auc,
                "args": vars(args),
            }, outdir / f"epoch_{epoch:03d}.pt")

    # Load best and evaluate on test
    # Control-flow branch for conditional or iterative execution.
    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state["model_state"]) 
    te_loss, te_auc, te_acc = run_epoch(model, test_ld, device, criterion, optimizer=None)
    print(json.dumps({"phase": "test", "loss": te_loss, "auc": te_auc, "acc": te_acc}))

    # Export a few Grad-CAM panels
    # Control-flow branch for conditional or iterative execution.
    if args.export_cam:
        cam_dir = outdir / "gradcam"
        save_gradcam_examples(model, val_ld, device, cam_dir, max_batches=2)
        print(json.dumps({"gradcam_dir": str(cam_dir)}))


# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    main()
