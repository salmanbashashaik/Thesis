#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
test_using_cnn_cv_DS.py (ENHANCED + FIXED FOR NAMING CONVENTIONS)

Enhanced version with:
1. Domain adaptation evaluation
2. Bootstrap confidence intervals
3. Feature space analysis
4. Stratified evaluation
5. Better statistical reporting
6. FIXED: Handles both naming conventions (with/without meta.json)

All improvements require NO extra synthetic image generation!
"""

# Import dependencies used by this module.
from __future__ import annotations

import argparse
import random
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm


# ---------------------------
# Model: AlexLite-DG
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
        feats = self.features(x)
        gap = F.adaptive_avg_pool2d(feats, 1)
        gap = gap.view(gap.size(0), -1)
        x = self.dropout(gap)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)
        # Return the computed value to the caller.
        return logits, feats


# ---------------------------
# IO + Normalization
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


# Function: `resample_to_shape_trilinear` implements a reusable processing step.
def resample_to_shape_trilinear(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    vol_t = torch.from_numpy(vol).float()
    vol_t = vol_t.permute(2, 0, 1).contiguous()
    vol_t = vol_t[None, None, ...]

    tgt_h, tgt_w, tgt_d = target_shape
    vol_rs = F.interpolate(
        vol_t,
        size=(tgt_d, tgt_h, tgt_w),
        mode="trilinear",
        align_corners=False
    )

    out = vol_rs[0, 0].permute(1, 2, 0).contiguous()
    # Return the computed value to the caller.
    return out.cpu().numpy().astype(np.float32)


# ---------------------------
# FIXED: Patient ID resolution helpers
# ---------------------------
# Function: `read_patient_id_from_meta` implements a reusable processing step.
def read_patient_id_from_meta(synth_dir: Path) -> Optional[str]:
    """
    Reads patient ID from synth_dir/meta.json (sid_img preferred, sid_mask fallback).
    If meta.json doesn't exist, returns the directory name itself.
    This handles both naming conventions seamlessly.
    """
    meta_path = synth_dir / "meta.json"
    # Control-flow branch for conditional or iterative execution.
    if not meta_path.exists():
        # No meta.json - assume directory name is the patient ID
        # Return the computed value to the caller.
        return synth_dir.name
    
    # Control-flow branch for conditional or iterative execution.
    try:
        obj = json.loads(meta_path.read_text())
        sid = obj.get("sid_img") or obj.get("sid_mask")
        # Control-flow branch for conditional or iterative execution.
        if isinstance(sid, str) and sid.strip():
            # Return the computed value to the caller.
            return sid.strip()
        # If meta.json exists but has no valid sid, fall back to directory name
        # Return the computed value to the caller.
        return synth_dir.name
    # Control-flow branch for conditional or iterative execution.
    except Exception:
        # If meta.json is corrupted, fall back to directory name
        # Return the computed value to the caller.
        return synth_dir.name


# Function: `normalize_patient_id` implements a reusable processing step.
def normalize_patient_id(pid: str) -> str:
    """
    Normalize patient IDs by removing common suffixes.
    E.g., 'UCSF-PDGM-0333_nifti' -> 'UCSF-PDGM-0333'
    """
    s = pid.strip()
    # Control-flow branch for conditional or iterative execution.
    for suffix in ["_nifti", ".nii.gz", ".nii"]:
        # Control-flow branch for conditional or iterative execution.
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    # Return the computed value to the caller.
    return s


# Function: `resolve_mask_dir_from_patient_id` implements a reusable processing step.
def resolve_mask_dir_from_patient_id(
    patient_id: str,
    mask_root: Path
) -> Optional[Path]:
    """
    Resolve mask directory from patient ID.
    Handles various naming conventions with multiple fallback strategies.
    """
    pid_norm = normalize_patient_id(patient_id)
    
    # Build mapping of mask directories
    mask_dirs = [d for d in mask_root.iterdir() if d.is_dir()]
    mask_name_to_dir = {d.name: d for d in mask_dirs}
    
    # Strategy 1: Exact match on normalized ID
    # Control-flow branch for conditional or iterative execution.
    if pid_norm in mask_name_to_dir:
        # Return the computed value to the caller.
        return mask_name_to_dir[pid_norm]
    
    # Strategy 2: Exact match on original ID
    # Control-flow branch for conditional or iterative execution.
    if patient_id in mask_name_to_dir:
        # Return the computed value to the caller.
        return mask_name_to_dir[patient_id]
    
    # Strategy 3: Prefix match (handles suffixes like _11)
    # Control-flow branch for conditional or iterative execution.
    for d in mask_dirs:
        # Control-flow branch for conditional or iterative execution.
        if d.name.startswith(pid_norm):
            # Return the computed value to the caller.
            return d
    
    # Strategy 4: Contains match (last resort)
    # Control-flow branch for conditional or iterative execution.
    for d in mask_dirs:
        # Control-flow branch for conditional or iterative execution.
        if pid_norm in d.name or d.name in pid_norm:
            # Return the computed value to the caller.
            return d
    
    # Return the computed value to the caller.
    return None


# ---------------------------
# Slice indexing
# ---------------------------
# Class definition: `SliceIndex` encapsulates related model behavior.
class SliceIndex:
    __slots__ = ("patient", "k", "label", "patient_idx")

    # Function: `__init__` implements a reusable processing step.
    def __init__(self, patient: str, k: int, label: int, patient_idx: int):
        self.patient = patient
        self.k = k
        self.label = label
        self.patient_idx = patient_idx


# -----------------------------------------------------------------
# Dataset for CV evaluation (FIXED)
# -----------------------------------------------------------------
# Class definition: `GliomaSliceDatasetCV` encapsulates related model behavior.
class GliomaSliceDatasetCV(Dataset):
    # Function: `__init__` implements a reusable processing step.
    def __init__(
        self,
        patient_dirs: List[Path],
        mask_root: Path,
        img_size: int = 224,
        norm: str = "zscore",
        use_resized: bool = True,
        allow_fallback: bool = True,
        real: bool = False,
        verbose: bool = False,
    ):
        self.mask_root = mask_root
        self.img_size = img_size
        self.norm = norm
        self.use_resized = use_resized
        self.allow_fallback = allow_fallback
        self.real = real
        self.verbose = verbose

        self.meta: Dict[str, Dict[str, Path]] = {}
        self.patients: List[str] = []
        self.patient_to_idx: Dict[str, int] = {}

        # Control-flow branch for conditional or iterative execution.
        for p in patient_dirs:
            synth_folder_name = p.name
            
            # FIXED: Get patient ID (from meta.json or folder name)
            patient_id = read_patient_id_from_meta(p)
            # Control-flow branch for conditional or iterative execution.
            if patient_id is None:
                # Control-flow branch for conditional or iterative execution.
                if self.verbose:
                    print(f"[WARN] Skipping {synth_folder_name}, could not determine patient ID")
                continue

            # Control-flow branch for conditional or iterative execution.
            if self.real:
                paths = {
                    "t1": p / "t1.nii.gz",
                    "t2": p / "t2.nii.gz",
                    "flair": p / "flair.nii.gz",
                }
            else:
                # Control-flow branch for conditional or iterative execution.
                if self.use_resized:
                    paths = {
                        "t1": p / "t1_synth_resized.nii.gz",
                        "t2": p / "t2_synth_resized.nii.gz",
                        "flair": p / "flair_synth_resized.nii.gz",
                    }
                    # Control-flow branch for conditional or iterative execution.
                    if self.allow_fallback and not all(pp.exists() for pp in paths.values()):
                        paths = {
                            "t1": p / "t1_synth.nii.gz",
                            "t2": p / "t2_synth.nii.gz",
                            "flair": p / "flair_synth.nii.gz",
                        }
                else:
                    paths = {
                        "t1": p / "t1_synth.nii.gz",
                        "t2": p / "t2_synth.nii.gz",
                        "flair": p / "flair_synth.nii.gz",
                    }

            # Control-flow branch for conditional or iterative execution.
            if not all(pp.exists() for pp in paths.values()):
                # Control-flow branch for conditional or iterative execution.
                if self.verbose:
                    print(f"[WARN] Skipping {synth_folder_name}, missing volumes")
                continue

            # FIXED: Resolve mask directory using patient ID
            mask_dir = resolve_mask_dir_from_patient_id(patient_id, self.mask_root)
            # Control-flow branch for conditional or iterative execution.
            if mask_dir is None:
                # Control-flow branch for conditional or iterative execution.
                if self.verbose:
                    print(f"[WARN] Skipping {synth_folder_name} (pid={patient_id}), no matching mask directory in {self.mask_root}")
                continue
            
            mask_path = mask_dir / "mask.nii.gz"
            # Control-flow branch for conditional or iterative execution.
            if not mask_path.exists():
                # Control-flow branch for conditional or iterative execution.
                if self.verbose:
                    print(f"[WARN] Skipping {synth_folder_name}, missing mask at {mask_path}")
                continue

            # Use synth folder name as key (for consistency)
            self.patients.append(synth_folder_name)
            self.patient_to_idx[synth_folder_name] = len(self.patients) - 1
            self.meta[synth_folder_name] = paths
            
            # Control-flow branch for conditional or iterative execution.
            if self.verbose and synth_folder_name != mask_dir.name:
                print(f"[MAP] {synth_folder_name} (pid={patient_id}) -> mask: {mask_dir.name}")

        self.slice_index: List[SliceIndex] = []
        self._build_index_from_masks()

    # Function: `_norm_vol` implements a reusable processing step.
    def _norm_vol(self, vol: np.ndarray) -> np.ndarray:
        # Control-flow branch for conditional or iterative execution.
        if self.norm == "zscore":
            # Return the computed value to the caller.
            return zscore_per_volume(vol)
        # Control-flow branch for conditional or iterative execution.
        if self.norm == "minmax":
            # Return the computed value to the caller.
            return minmax_per_volume(vol)
        # Return the computed value to the caller.
        return vol

    # Function: `_build_index_from_masks` implements a reusable processing step.
    def _build_index_from_masks(self) -> None:
        # Control-flow branch for conditional or iterative execution.
        for patient in self.patients:
            patient_idx = self.patient_to_idx[patient]
            
            # Resolve mask path
            patient_id = read_patient_id_from_meta(Path(self.meta[patient]["t1"]).parent)
            mask_dir = resolve_mask_dir_from_patient_id(patient_id, self.mask_root)
            # Control-flow branch for conditional or iterative execution.
            if mask_dir is None:
                continue
            
            mask_path = mask_dir / "mask.nii.gz"
            # Control-flow branch for conditional or iterative execution.
            if not mask_path.exists():
                continue
                
            mask = load_nifti(mask_path).astype(np.float32)

            depth = int(mask.shape[-1])
            # Control-flow branch for conditional or iterative execution.
            for k in range(depth):
                msl = mask[:, :, k]
                label = int(np.any(msl > 0.0))
                self.slice_index.append(SliceIndex(patient, k, label, patient_idx))

        pos = sum(1 for s in self.slice_index if s.label == 1)
        neg = len(self.slice_index) - pos
        print(f"Built index: {pos} positive slices, {neg} negative slices from {len(self.patients)} patients.")

    # Function: `__len__` implements a reusable processing step.
    def __len__(self) -> int:
        # Return the computed value to the caller.
        return len(self.slice_index)

    # Function: `get_patient_indices` implements a reusable processing step.
    def get_patient_indices(self) -> np.ndarray:
        """Return array of patient indices for each slice (for CV splitting)."""
        # Return the computed value to the caller.
        return np.array([s.patient_idx for s in self.slice_index])

    # Function: `__getitem__` implements a reusable processing step.
    def __getitem__(self, idx: int):
        si = self.slice_index[idx]
        paths = self.meta[si.patient]

        # Resolve mask path
        patient_id = read_patient_id_from_meta(Path(paths["t1"]).parent)
        mask_dir = resolve_mask_dir_from_patient_id(patient_id, self.mask_root)
        mask_path = mask_dir / "mask.nii.gz"
        
        mask = load_nifti(mask_path).astype(np.float32)
        target_shape = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))

        # Control-flow branch for conditional or iterative execution.
        try:
            t1 = load_nifti(paths["t1"]).astype(np.float32)
            t2 = load_nifti(paths["t2"]).astype(np.float32)
            fl = load_nifti(paths["flair"]).astype(np.float32)
        # Control-flow branch for conditional or iterative execution.
        except Exception as e:
            print(f"[ERR] Error loading {si.patient}: {e}")
            x = torch.zeros(3, self.img_size, self.img_size)
            y = torch.tensor(0, dtype=torch.long)
            # Return the computed value to the caller.
            return x, y, si.patient_idx

        # Control-flow branch for conditional or iterative execution.
        if not self.real:
            # Control-flow branch for conditional or iterative execution.
            if t1.shape != target_shape:
                t1 = resample_to_shape_trilinear(t1, target_shape)
            # Control-flow branch for conditional or iterative execution.
            if t2.shape != target_shape:
                t2 = resample_to_shape_trilinear(t2, target_shape)
            # Control-flow branch for conditional or iterative execution.
            if fl.shape != target_shape:
                fl = resample_to_shape_trilinear(fl, target_shape)

        depth = target_shape[2]
        k = min(si.k, depth - 1)

        t1 = self._norm_vol(t1)
        t2 = self._norm_vol(t2)
        fl = self._norm_vol(fl)

        s1 = t1[:, :, k]
        s2 = t2[:, :, k]
        s3 = fl[:, :, k]

        x = np.stack([s1, s2, s3], axis=0)
        x = torch.from_numpy(x.copy()).float()
        x = TF.resize(x, [self.img_size, self.img_size], antialias=True)

        y = torch.tensor(si.label, dtype=torch.long)
        # Return the computed value to the caller.
        return x, y, si.patient_idx


# ---------------------------
# Metrics helpers
# ---------------------------
# Function: `compute_metrics` implements a reusable processing step.
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    bal_acc = 0.5 * (sens + spec)
    
    # Control-flow branch for conditional or iterative execution.
    try:
        auc = roc_auc_score(y_true, y_prob)
    # Control-flow branch for conditional or iterative execution.
    except ValueError:
        auc = float("nan")
    
    # Control-flow branch for conditional or iterative execution.
    try:
        f1 = f1_score(y_true, y_pred)
    # Control-flow branch for conditional or iterative execution.
    except:
        f1 = float("nan")
    
    # Return the computed value to the caller.
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


# Function: `find_best_threshold` implements a reusable processing step.
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    best_thr = 0.5
    best_acc = -1.0
    
    # Control-flow branch for conditional or iterative execution.
    for t in np.linspace(0.0, 1.0, 201):
        y_pred = (y_prob >= t).astype(np.int32)
        acc = accuracy_score(y_true, y_pred)
        # Control-flow branch for conditional or iterative execution.
        if acc > best_acc:
            best_acc = acc
            best_thr = t
    
    # Return the computed value to the caller.
    return best_thr, best_acc


# Function: `run_fold_eval` implements a reusable processing step.
def run_fold_eval(model, loader, device, criterion):
    """Run evaluation on a fold and return predictions."""
    model.eval()
    total_loss = 0.0
    ys, ps = [], []

    # Control-flow branch for conditional or iterative execution.
    with torch.no_grad():
        # Control-flow branch for conditional or iterative execution.
        for batch in loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits, _ = model(xb)
            loss = criterion(logits, yb)

            total_loss += float(loss.item()) * xb.size(0)
            prob = F.softmax(logits, dim=1)[:, 1]
            ys.append(yb.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())

    ys = np.concatenate(ys, axis=0).astype(np.int32)
    ps = np.concatenate(ps, axis=0).astype(np.float32)
    avg_loss = total_loss / max(1, len(loader.dataset))

    # Return the computed value to the caller.
    return ys, ps, avg_loss


# ============================================================
# Domain Adaptation Evaluation
# ============================================================
# Function: `domain_adaptation_evaluation` implements a reusable processing step.
def domain_adaptation_evaluation(
    model,
    dataset,
    device,
    n_adapt=10,
    n_epochs=5,
    lr=1e-4,
    batch_size=8,
    seed=42,
    verbose=True
):
    """
    Evaluate domain gap by fine-tuning on small subset.
    
    Returns:
        dict with 'auc_no_adapt', 'auc_adapted', 'domain_gap'
    """
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"\n{'='*70}")
        print("DOMAIN ADAPTATION ANALYSIS")
        print(f"{'='*70}\n")
    
    # Split into adapt + test sets
    all_indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    
    adapt_indices = all_indices[:n_adapt]
    test_indices = all_indices[n_adapt:]
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"Adaptation set: {len(adapt_indices)} samples")
        print(f"Test set: {len(test_indices)} samples")
    
    adapt_subset = Subset(dataset, adapt_indices)
    test_subset = Subset(dataset, test_indices)
    
    adapt_dl = DataLoader(adapt_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    
    # ===== Test without adaptation =====
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print("\n1. Evaluating WITHOUT adaptation...")
    ys, ps, loss_no_adapt = run_fold_eval(model, test_dl, device, criterion)
    
    # Control-flow branch for conditional or iterative execution.
    try:
        auc_no_adapt = roc_auc_score(ys, ps)
    # Control-flow branch for conditional or iterative execution.
    except ValueError:
        auc_no_adapt = float("nan")
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"   AUC (no adaptation): {auc_no_adapt:.4f}")
    
    # ===== Adapt model =====
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"\n2. Adapting model on {n_adapt} samples...")
    
    # Create a copy to adapt
    adapted_model = copy.deepcopy(model)
    
    # Freeze feature extractor, only adapt classifier head
    # Control-flow branch for conditional or iterative execution.
    for param in adapted_model.features.parameters():
        param.requires_grad = False
    # Control-flow branch for conditional or iterative execution.
    for param in adapted_model.fc1.parameters():
        param.requires_grad = False
    
    # Only train final classification layer
    optimizer = torch.optim.Adam(
        adapted_model.fc2.parameters(),
        lr=lr
    )
    
    adapted_model.train()
    # Control-flow branch for conditional or iterative execution.
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        # Control-flow branch for conditional or iterative execution.
        for xb, yb, _ in adapt_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            
            logits, _ = adapted_model(xb)
            loss = criterion(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Control-flow branch for conditional or iterative execution.
        if verbose and (epoch + 1) % 2 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(adapt_dl):.4f}")
    
    # ===== Test with adaptation =====
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print("\n3. Evaluating WITH adaptation...")
    
    ys, ps, loss_adapted = run_fold_eval(adapted_model, test_dl, device, criterion)
    
    # Control-flow branch for conditional or iterative execution.
    try:
        auc_adapted = roc_auc_score(ys, ps)
    # Control-flow branch for conditional or iterative execution.
    except ValueError:
        auc_adapted = float("nan")
    
    domain_gap = auc_adapted - auc_no_adapt
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"   AUC (with adaptation): {auc_adapted:.4f}")
        print(f"\n{'='*70}")
        print("DOMAIN GAP ANALYSIS")
        print(f"{'='*70}")
        print(f"AUC without adaptation: {auc_no_adapt:.4f}")
        print(f"AUC with adaptation:    {auc_adapted:.4f}")
        print(f"Domain gap effect:      {domain_gap:+.4f}")
        print(f"\nInterpretation:")
        # Control-flow branch for conditional or iterative execution.
        if abs(domain_gap) > 0.10:
            print(f"  Large domain gap (>{domain_gap:.3f}) - much of the performance")
            print(f"  difference is due to distribution shift, not quality issues.")
        # Control-flow branch for conditional or iterative execution.
        elif abs(domain_gap) > 0.05:
            print(f"  Moderate domain gap - some performance loss is fixable")
            print(f"  with minimal adaptation.")
        else:
            print(f"  Small domain gap - performance is mostly limited by")
            print(f"  intrinsic quality, not distribution shift.")
        print(f"{'='*70}\n")
    
    # Return the computed value to the caller.
    return {
        'auc_no_adapt': auc_no_adapt,
        'auc_adapted': auc_adapted,
        'domain_gap': domain_gap,
        'n_adapt': n_adapt,
        'n_test': len(test_indices)
    }

# ============================================================
# Check Mode Collapse
# ============================================================
# Function: `diagnose_mode_collapse` implements a reusable processing step.
def diagnose_mode_collapse(samples_dir, feature_extractor):
    """
    Check if the model has mode collapse
    High similarity scores = mode collapse confirmed
    """
    # Import dependencies used by this module.
    import torch
    import numpy as np
    from pathlib import Path
    
    # Extract features from all samples
    features = []
    # Control-flow branch for conditional or iterative execution.
    for sample_path in Path(samples_dir).glob("*/t1_synth.nii.gz"):
        # Load and extract features (use your CNN or VAE encoder)
        feat = feature_extractor(sample_path)
        features.append(feat)
    
    features = torch.stack(features)
    
    # Compute pairwise cosine similarity
    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    similarity_matrix = features_norm @ features_norm.T
    
    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(len(features), dtype=bool)
    similarities = similarity_matrix[mask]
    
    avg_sim = similarities.mean().item()
    max_sim = similarities.max().item()
    min_sim = similarities.min().item()
    
    print(f"Average pairwise similarity: {avg_sim:.4f}")
    print(f"Max similarity: {max_sim:.4f}")
    print(f"Min similarity: {min_sim:.4f}")
    
    # Control-flow branch for conditional or iterative execution.
    if avg_sim > 0.9:
        print("⚠️  HIGH MODE COLLAPSE - Samples are too similar!")
    # Control-flow branch for conditional or iterative execution.
    elif avg_sim > 0.8:
        print("⚠️  MODERATE MODE COLLAPSE - Reduce CFG scale")
    else:
        print("✅ Good diversity")
    
    # Return the computed value to the caller.
    return avg_sim


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================
# Function: `bootstrap_confidence_intervals` implements a reusable processing step.
def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap=1000,
    seed=42,
    verbose=True
):
    """
    Compute bootstrap confidence intervals for AUC.
    
    Returns:
        dict with 'auc_mean', 'auc_std', 'ci_95'
    """
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"\n{'='*70}")
        print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap} iterations)")
        print(f"{'='*70}\n")
    
    rng = np.random.RandomState(seed)
    auc_scores = []
    
    # Control-flow branch for conditional or iterative execution.
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class present
        # Control-flow branch for conditional or iterative execution.
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        # Control-flow branch for conditional or iterative execution.
        try:
            auc = roc_auc_score(y_true_boot, y_prob_boot)
            auc_scores.append(auc)
        # Control-flow branch for conditional or iterative execution.
        except ValueError:
            pass
    
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    auc_ci_low = np.percentile(auc_scores, 2.5)
    auc_ci_high = np.percentile(auc_scores, 97.5)
    auc_median = np.median(auc_scores)
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"Bootstrap Results:")
        print(f"  Mean:      {auc_mean:.4f}")
        print(f"  Std Dev:   {auc_std:.4f}")
        print(f"  Median:    {auc_median:.4f}")
        print(f"  95% CI:    [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")
        print(f"  Range:     [{min(auc_scores):.4f}, {max(auc_scores):.4f}]")
        print(f"{'='*70}\n")
    
    # Return the computed value to the caller.
    return {
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'auc_median': auc_median,
        'ci_95': (auc_ci_low, auc_ci_high),
        'range': (min(auc_scores), max(auc_scores)),
        'n_bootstrap': len(auc_scores)
    }


# ============================================================
# Feature Space Analysis
# ============================================================
# Function: `extract_features` implements a reusable processing step.
def extract_features(model, dataloader, device, max_samples=None):
    """Extract CNN features for analysis."""
    model.eval()
    all_features = []
    all_labels = []
    n_samples = 0
    
    # Control-flow branch for conditional or iterative execution.
    with torch.no_grad():
        # Control-flow branch for conditional or iterative execution.
        for x, y, _ in dataloader:
            x = x.to(device)
            _, feats = model(x)  # (B, 256, H, W)
            feats = F.adaptive_avg_pool2d(feats, 1).squeeze()  # (B, 256)
            
            # Control-flow branch for conditional or iterative execution.
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            
            all_features.append(feats.cpu().numpy())
            all_labels.append(y.numpy())
            
            n_samples += len(y)
            # Control-flow branch for conditional or iterative execution.
            if max_samples and n_samples >= max_samples:
                break
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    # Control-flow branch for conditional or iterative execution.
    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
    
    # Return the computed value to the caller.
    return features, labels


# Function: `feature_space_analysis` implements a reusable processing step.
def feature_space_analysis(model, real_dl, synth_dl, device, verbose=True):
    """
    Analyze feature space distance between real and synthetic.
    """
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"\n{'='*70}")
        print("FEATURE SPACE ANALYSIS")
        print(f"{'='*70}\n")
    
    # Extract features (limit to 500 samples for speed)
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print("Extracting features from real images...")
    feats_real, _ = extract_features(model, real_dl, device, max_samples=500)
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print("Extracting features from synthetic images...")
    feats_synth, _ = extract_features(model, synth_dl, device, max_samples=500)
    
    # Compute statistics
    mean_real = feats_real.mean(axis=0)
    mean_synth = feats_synth.mean(axis=0)
    
    # Cosine distance
    # Import dependencies used by this module.
    from scipy.spatial.distance import cosine
    cosine_dist = cosine(mean_real, mean_synth)
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(mean_real - mean_synth)
    
    # Per-dimension statistics
    # Import dependencies used by this module.
    from scipy.stats import wasserstein_distance
    dim_distances = []
    # Control-flow branch for conditional or iterative execution.
    for i in range(feats_real.shape[1]):
        wd = wasserstein_distance(feats_real[:, i], feats_synth[:, i])
        dim_distances.append(wd)
    
    mean_wasserstein = np.mean(dim_distances)
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"Feature Space Metrics:")
        print(f"  Cosine distance:      {cosine_dist:.4f}")
        print(f"  Euclidean distance:   {euclidean_dist:.4f}")
        print(f"  Mean Wasserstein:     {mean_wasserstein:.4f}")
        print(f"\nInterpretation:")
        # Control-flow branch for conditional or iterative execution.
        if cosine_dist < 0.15:
            print(f"  Very small domain gap - synthetic closely matches real distribution")
        # Control-flow branch for conditional or iterative execution.
        elif cosine_dist < 0.30:
            print(f"  Moderate domain gap - some distribution differences present")
        else:
            print(f"  Large domain gap - substantial distribution differences")
        print(f"{'='*70}\n")
    
    # Return the computed value to the caller.
    return {
        'cosine_distance': cosine_dist,
        'euclidean_distance': euclidean_dist,
        'mean_wasserstein': mean_wasserstein,
    }


# ============================================================
# Stratified Evaluation
# ============================================================
# Function: `stratified_evaluation` implements a reusable processing step.
def stratified_evaluation(model, dataloader, device, verbose=True):
    """
    Evaluate performance across different subgroups.
    """
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"\n{'='*70}")
        print("STRATIFIED EVALUATION")
        print(f"{'='*70}\n")
    
    model.eval()
    
    # Collect predictions with metadata
    results = {
        'all': {'y_true': [], 'y_prob': []},
        'positive': {'y_true': [], 'y_prob': []},
        'negative': {'y_true': [], 'y_prob': []},
        'by_patient': {}
    }
    
    # Control-flow branch for conditional or iterative execution.
    with torch.no_grad():
        # Control-flow branch for conditional or iterative execution.
        for x, y, patient_idx in dataloader:
            x = x.to(device)
            logits, _ = model(x)
            prob = F.softmax(logits, dim=1)[:, 1]
            
            y_np = y.numpy()
            prob_np = prob.cpu().numpy()
            
            # Overall
            results['all']['y_true'].extend(y_np)
            results['all']['y_prob'].extend(prob_np)
            
            # By label
            # Control-flow branch for conditional or iterative execution.
            for i in range(len(y_np)):
                # Control-flow branch for conditional or iterative execution.
                if y_np[i] == 1:
                    results['positive']['y_true'].append(y_np[i])
                    results['positive']['y_prob'].append(prob_np[i])
                else:
                    results['negative']['y_true'].append(y_np[i])
                    results['negative']['y_prob'].append(prob_np[i])
                
                # By patient
                pid = int(patient_idx[i])
                # Control-flow branch for conditional or iterative execution.
                if pid not in results['by_patient']:
                    results['by_patient'][pid] = {'y_true': [], 'y_prob': []}
                results['by_patient'][pid]['y_true'].append(y_np[i])
                results['by_patient'][pid]['y_prob'].append(prob_np[i])
    
    # Compute metrics
    y_true = np.array(results['all']['y_true'])
    y_prob = np.array(results['all']['y_prob'])
    overall_auc = roc_auc_score(y_true, y_prob)
    
    # Sensitivity (on positive slices only)
    # Control-flow branch for conditional or iterative execution.
    if len(results['positive']['y_true']) > 0:
        sens_scores = np.array(results['positive']['y_prob'])
        mean_prob_positive = np.mean(sens_scores)
        median_prob_positive = np.median(sens_scores)
    else:
        mean_prob_positive = float('nan')
        median_prob_positive = float('nan')
    
    # Specificity (on negative slices only)
    # Control-flow branch for conditional or iterative execution.
    if len(results['negative']['y_true']) > 0:
        spec_scores = np.array(results['negative']['y_prob'])
        mean_prob_negative = np.mean(spec_scores)
        median_prob_negative = np.median(spec_scores)
    else:
        mean_prob_negative = float('nan')
        median_prob_negative = float('nan')
    
    # Patient-level aggregation
    patient_aucs = []
    # Control-flow branch for conditional or iterative execution.
    for pid, data in results['by_patient'].items():
        # Control-flow branch for conditional or iterative execution.
        if len(set(data['y_true'])) > 1:  # Only if has both classes
            # Control-flow branch for conditional or iterative execution.
            try:
                auc = roc_auc_score(data['y_true'], data['y_prob'])
                patient_aucs.append(auc)
            # Control-flow branch for conditional or iterative execution.
            except:
                pass
    
    # Control-flow branch for conditional or iterative execution.
    if patient_aucs:
        patient_auc_mean = np.mean(patient_aucs)
        patient_auc_std = np.std(patient_aucs)
    else:
        patient_auc_mean = float('nan')
        patient_auc_std = float('nan')
    
    # Control-flow branch for conditional or iterative execution.
    if verbose:
        print(f"Stratified Metrics:")
        print(f"  Overall AUC:                 {overall_auc:.4f}")
        print(f"  Patient-level AUC:           {patient_auc_mean:.4f} ± {patient_auc_std:.4f}")
        print(f"  Mean prob on positive:       {mean_prob_positive:.4f}")
        print(f"  Median prob on positive:     {median_prob_positive:.4f}")
        print(f"  Mean prob on negative:       {mean_prob_negative:.4f}")
        print(f"  Median prob on negative:     {median_prob_negative:.4f}")
        print(f"  N patients with both labels: {len(patient_aucs)}")
        print(f"{'='*70}\n")
    
    # Return the computed value to the caller.
    return {
        'overall_auc': overall_auc,
        'patient_auc_mean': patient_auc_mean,
        'patient_auc_std': patient_auc_std,
        'mean_prob_positive': mean_prob_positive,
        'median_prob_positive': median_prob_positive,
        'mean_prob_negative': mean_prob_negative,
        'median_prob_negative': median_prob_negative,
        'n_patients_both_labels': len(patient_aucs)
    }


# ---------------------------
# Main
# ---------------------------
# Function: `main` implements a reusable processing step.
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--synth_root", type=str, required=True,
                   help="Path to synthetic images directory")
    ap.add_argument("--mask_root", type=str, default="",
                   help="Path to ground truth masks (auto-detect if not provided)")
    ap.add_argument("--model_path", type=str, required=True,
                   help="Path to trained CNN model")

    # Optional: real data for feature space comparison
    ap.add_argument("--real_root", type=str, default="",
                   help="Path to real images for feature space analysis (optional)")

    # Domain detection
    ap.add_argument("--domain", type=str, default="auto", choices=["auto", "gbm", "pdgm"],
                   help="Domain: auto-detect, gbm, or pdgm (default: auto)")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--use_resized", action="store_true")
    ap.add_argument("--no_fallback", action="store_true")
    ap.add_argument("--real", action="store_true")

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    # Domain adaptation parameters
    ap.add_argument("--n_adapt", type=int, default=10,
                   help="Number of samples for domain adaptation")
    ap.add_argument("--adapt_epochs", type=int, default=5,
                   help="Epochs for domain adaptation")
    ap.add_argument("--adapt_lr", type=float, default=1e-4,
                   help="Learning rate for domain adaptation")

    # Bootstrap parameters
    ap.add_argument("--n_bootstrap", type=int, default=1000,
                   help="Number of bootstrap iterations")

    # Feature analysis
    ap.add_argument("--skip_feature_analysis", action="store_true",
                   help="Skip feature space analysis (faster)")

    args = ap.parse_args()
    
    # ============================================================
    # Auto-detect domain and set mask_root if not provided
    # ============================================================
    
    synth_root = Path(args.synth_root)
    
    # Control-flow branch for conditional or iterative execution.
    if args.domain == "auto":
        # Try to detect domain from path
        path_str = str(synth_root).lower()
        # Control-flow branch for conditional or iterative execution.
        if "gbm" in path_str:
            detected_domain = "gbm"
        # Control-flow branch for conditional or iterative execution.
        elif "pdgm" in path_str:
            detected_domain = "pdgm"
        else:
            # Check parent directory name
            # Control-flow branch for conditional or iterative execution.
            if "gbm" in synth_root.parent.name.lower():
                detected_domain = "gbm"
            # Control-flow branch for conditional or iterative execution.
            elif "pdgm" in synth_root.parent.name.lower():
                detected_domain = "pdgm"
            else:
                detected_domain = "pdgm"  # Default to PDGM
        print(f"[INFO] Auto-detected domain: {detected_domain}")
    else:
        detected_domain = args.domain
        print(f"[INFO] Using specified domain: {detected_domain}")
    
    # Set mask_root based on domain if not provided
    # Control-flow branch for conditional or iterative execution.
    if not args.mask_root:
        # Control-flow branch for conditional or iterative execution.
        if detected_domain == "gbm":
            mask_root = Path("/home/j98my/Pre-Processing/prep/gbm_all_aligned")
        else:
            mask_root = Path("/home/j98my/Pre-Processing/prep/pdgm_target_aligned")
        print(f"[INFO] Using default mask root for {detected_domain}: {mask_root}")
    else:
        mask_root = Path(args.mask_root)
        print(f"[INFO] Using provided mask root: {mask_root}")

    root = Path(args.synth_root)
    model_path = Path(args.model_path)

    # Control-flow branch for conditional or iterative execution.
    if not root.exists():
        raise FileNotFoundError(f"--synth_root not found: {root}")
    # Control-flow branch for conditional or iterative execution.
    if not mask_root.exists():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")
    # Control-flow branch for conditional or iterative execution.
    if not model_path.exists():
        raise FileNotFoundError(f"--model_path not found: {model_path}")

    # FIXED: Accept ALL directories (both with and without meta.json)
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name)
    print(f"Found {len(patient_dirs)} patient folders in {root}")
    print(f"Using masks from: {mask_root}")
    print(f"Domain: {detected_domain.upper()}")

    # Build dataset
    ds = GliomaSliceDatasetCV(
        patient_dirs=patient_dirs,
        mask_root=mask_root,
        img_size=args.img_size,
        norm=args.norm,
        use_resized=args.use_resized,
        allow_fallback=not args.no_fallback,
        real=args.real,
        verbose=True,
    )

    # Control-flow branch for conditional or iterative execution.
    if len(ds) == 0:
        print("Error: dataset is empty.")
        # Return the computed value to the caller.
        return

    device = torch.device(args.device)

    # Load model
    model = AlexLiteDG(num_classes=2).to(device)
    print(f"Loading model from {model_path}")
    state = torch.load(str(model_path), map_location=device)
    # Control-flow branch for conditional or iterative execution.
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # ENHANCED EVALUATION PIPELINE
    # ============================================================
    
    print(f"\n{'='*70}")
    print("ENHANCED EVALUATION WITH DOMAIN GAP ANALYSIS")
    print(f"{'='*70}\n")
    
    # ===== 1. DOMAIN ADAPTATION ANALYSIS =====
    print("STEP 1: Domain Adaptation Analysis")
    print("This separates domain gap from quality issues")
    print("-" * 70)
    
    domain_results = domain_adaptation_evaluation(
        model=model,
        dataset=ds,
        device=device,
        n_adapt=args.n_adapt,
        n_epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        seed=args.seed,
        verbose=True
    )
    
    # ===== 2. BOOTSTRAP CONFIDENCE INTERVALS =====
    print("STEP 2: Bootstrap Confidence Intervals")
    print("This provides robust uncertainty estimates")
    print("-" * 70)
    
    # Get predictions on full dataset
    full_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    ys, ps, _ = run_fold_eval(model, full_loader, device, criterion)
    
    bootstrap_results = bootstrap_confidence_intervals(
        y_true=ys,
        y_prob=ps,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        verbose=True
    )
    
    # ===== 3. STRATIFIED EVALUATION =====
    print("STEP 3: Stratified Evaluation")
    print("This shows performance breakdown by subgroups")
    print("-" * 70)

    print("Checking Mode collapse")

    # Function: `_cnn_feature_extractor` implements a reusable processing step.
    def _cnn_feature_extractor(sample_path: Path) -> torch.Tensor:
        vol = load_nifti(Path(sample_path)).astype(np.float32)

        # Control-flow branch for conditional or iterative execution.
        if args.norm == "zscore":
            vol = zscore_per_volume(vol)
        # Control-flow branch for conditional or iterative execution.
        elif args.norm == "minmax":
            vol = minmax_per_volume(vol)

        k = int(vol.shape[2] // 2)
        sl = vol[:, :, k]

        x = np.stack([sl, sl, sl], axis=0)
        x = torch.from_numpy(x.copy()).float()
        x = TF.resize(x, [args.img_size, args.img_size], antialias=True)
        x = x.unsqueeze(0).to(device)

        model.eval()
        # Control-flow branch for conditional or iterative execution.
        with torch.no_grad():
            _, feats = model(x)  # (1, 256, H, W)
            feats = F.adaptive_avg_pool2d(feats, 1).view(-1)  # (256,)
        # Return the computed value to the caller.
        return feats.cpu()

    mode_collapse_avg_sim = diagnose_mode_collapse(samples_dir=root, feature_extractor=_cnn_feature_extractor)

    stratified_results = stratified_evaluation(
        model=model,
        dataloader=full_loader,
        device=device,
        verbose=True
    )

    # ===== 4. FEATURE SPACE ANALYSIS (if real data provided) =====
    # Control-flow branch for conditional or iterative execution.
    if args.real_root and not args.skip_feature_analysis:
        print("STEP 4: Feature Space Analysis")
        print("This quantifies distribution differences")
        print("-" * 70)
        
        real_root = Path(args.real_root)
        # Control-flow branch for conditional or iterative execution.
        if real_root.exists():
            real_dirs = sorted([p for p in real_root.iterdir() if p.is_dir()])[:100]  # Sample
            
            real_ds = GliomaSliceDatasetCV(
                patient_dirs=real_dirs,
                mask_root=mask_root,
                img_size=args.img_size,
                norm=args.norm,
                real=True,
                verbose=False,
            )
            
            real_loader = DataLoader(
                real_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            
            feature_results = feature_space_analysis(
                model=model,
                real_dl=real_loader,
                synth_dl=full_loader,
                device=device,
                verbose=True
            )
        else:
            print(f"Real root not found: {real_root}, skipping feature analysis")
            feature_results = None
    else:
        feature_results = None
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY REPORT")
    print(f"{'='*70}\n")
    
    print(f"DOMAIN: {detected_domain.upper()}")
    print(f"Synthetic samples: {len(patient_dirs)}")
    print(f"Total slices evaluated: {len(ds)}")
    print()
    
    print("1. BASIC PERFORMANCE:")
    print(f"   AUC: {bootstrap_results['auc_mean']:.4f} ± {bootstrap_results['auc_std']:.4f}")
    print(f"   95% CI: [{bootstrap_results['ci_95'][0]:.4f}, {bootstrap_results['ci_95'][1]:.4f}]")
    
    print("\n2. DOMAIN GAP EFFECT:")
    print(f"   AUC without adaptation: {domain_results['auc_no_adapt']:.4f}")
    print(f"   AUC with adaptation:    {domain_results['auc_adapted']:.4f}")
    print(f"   Domain gap:             {domain_results['domain_gap']:+.4f}")
    print(f"   Interpretation: {abs(domain_results['domain_gap']):.3f} of potential improvement")
    print(f"                   is achievable with {args.n_adapt} samples")
    
    print("\n3. PATIENT-LEVEL PERFORMANCE:")
    print(f"   Patient AUC: {stratified_results['patient_auc_mean']:.4f} ± {stratified_results['patient_auc_std']:.4f}")
    
    # Control-flow branch for conditional or iterative execution.
    if feature_results:
        print("\n4. FEATURE SPACE DISTANCE:")
        print(f"   Cosine distance: {feature_results['cosine_distance']:.4f}")
        print(f"   (Lower is better, <0.15 is excellent, >0.30 is large gap)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR PAPER:")
    print(f"{'='*70}")
    acc = compute_metrics(y_true=ys, y_prob=ps, threshold=0.5)
    print(f"Report Accuracy as: {acc}")
    print(f"Report AUC as: {bootstrap_results['auc_mean']:.3f} ± {bootstrap_results['auc_std']:.3f}")
    print(f"               [95% CI: {bootstrap_results['ci_95'][0]:.3f}, {bootstrap_results['ci_95'][1]:.3f}]")
    print(f"\nNote that {abs(domain_results['domain_gap']):.3f} AUC points are due to domain")
    print(f"shift rather than quality, as evidenced by adaptation experiment.")
    print(f"{'='*70}\n")
    
    # Save results to JSON
    results_dict = {
        'domain': detected_domain,
        'n_patients': len(patient_dirs),
        'n_slices': len(ds),
        'bootstrap': bootstrap_results,
        'domain_adaptation': domain_results,
        'stratified': stratified_results,
        'feature_space': feature_results if feature_results else {},
        'args': vars(args)
    }
    
    # Convert numpy types to Python types for JSON serialization
    # Function: `convert_to_python_types` implements a reusable processing step.
    def convert_to_python_types(obj):
        # Control-flow branch for conditional or iterative execution.
        if isinstance(obj, np.integer):
            # Return the computed value to the caller.
            return int(obj)
        # Control-flow branch for conditional or iterative execution.
        elif isinstance(obj, np.floating):
            # Return the computed value to the caller.
            return float(obj)
        # Control-flow branch for conditional or iterative execution.
        elif isinstance(obj, np.ndarray):
            # Return the computed value to the caller.
            return obj.tolist()
        # Control-flow branch for conditional or iterative execution.
        elif isinstance(obj, dict):
            # Return the computed value to the caller.
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        # Control-flow branch for conditional or iterative execution.
        elif isinstance(obj, (list, tuple)):
            # Return the computed value to the caller.
            return [convert_to_python_types(item) for item in obj]
        else:
            # Return the computed value to the caller.
            return obj
    
    results_dict = convert_to_python_types(results_dict)
    
    output_json = Path(args.synth_root) / "enhanced_evaluation_results.json"
    # Control-flow branch for conditional or iterative execution.
    with open(output_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to: {output_json}")


# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    main()