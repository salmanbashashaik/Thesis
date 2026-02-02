#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
test_using_fid_ssim.py (ROBUST META.JSON MAPPING - FIXED)

Evaluate synthetic image quality using SSIM and FID metrics.

Key features:
- Robust patient mapping that works with BOTH naming conventions:
  1. Folders named after patients (e.g., UPENN-GBM-00011_11, UCSF-PDGM-0034_nifti)
  2. Folders with generic names (e.g., s00000) that have meta.json files
- Builds slice index using depth = min(orig_depth, synth_depth) per patient.
- Supports *_synth_resized.nii.gz with fallback to *_synth.nii.gz
"""

# Import dependencies used by this module.
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

# --- Import torchmetrics ---
# Control-flow branch for conditional or iterative execution.
try:
    # Import dependencies used by this module.
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.fid import FrechetInceptionDistance
# Control-flow branch for conditional or iterative execution.
except ImportError as e:
    print("\n--- Import error ---")
    print(f">>> {e}")
    print("Try: python3 -m pip install torch-fidelity torchmetrics")
    print("--------------------\n")
    sys.exit(1)


# ---------------------------
# Utils
# ---------------------------
# Function: `load_nifti` implements a reusable processing step.
def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    # Return the computed value to the caller.
    return np.asarray(data, dtype=np.float32)

# Function: `nifti_depth` implements a reusable processing step.
def nifti_depth(path: Path) -> int:
    # Return the computed value to the caller.
    return int(nib.load(str(path)).shape[-1])

# Function: `minmax_per_slice` implements a reusable processing step.
def minmax_per_slice(vol2d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vmin = float(np.min(vol2d))
    vmax = float(np.max(vol2d))
    # Control-flow branch for conditional or iterative execution.
    if vmax - vmin < eps:
        # Return the computed value to the caller.
        return np.zeros_like(vol2d, dtype=np.float32)
    # Return the computed value to the caller.
    return (vol2d - vmin) / (vmax - vmin + eps)

# Function: `center_crop_or_pad` implements a reusable processing step.
def center_crop_or_pad(img: torch.Tensor, size: int) -> torch.Tensor:
    # img: (C,H,W)
    c, h, w = img.shape
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = TF.resize(img, [new_h, new_w], antialias=True)

    top = max((new_h - size) // 2, 0)
    left = max((new_w - size) // 2, 0)
    img = TF.crop(img, top, left, min(size, new_h), min(size, new_w))

    pad_h = size - img.shape[1]
    pad_w = size - img.shape[2]
    # Control-flow branch for conditional or iterative execution.
    if pad_h > 0 or pad_w > 0:
        img = TF.pad(img, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2])
    # Return the computed value to the caller.
    return img


# ---------------------------
# meta.json mapping helpers
# ---------------------------
# Function: `list_original_patient_dirs` implements a reusable processing step.
def list_original_patient_dirs(original_root: Path) -> List[Path]:
    # Deterministic ordering (useful for debugging, but mapping is via meta.json)
    # Return the computed value to the caller.
    return sorted([p for p in original_root.iterdir() if p.is_dir()], key=lambda p: p.name)

# Function: `read_meta_sid` implements a reusable processing step.
def read_meta_sid(synth_dir: Path) -> Optional[str]:
    """
    Reads sid_img (preferred) or sid_mask from synth_dir/meta.json.
    If meta.json doesn't exist, returns the directory name itself (handles direct naming).
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

# Function: `normalize_sid` implements a reusable processing step.
def normalize_sid(sid: str) -> str:
    """
    Normalize IDs like 'UCSF-PDGM-0333_nifti' -> 'UCSF-PDGM-0333'
    Extend if your pipeline adds other suffixes.
    """
    s = sid.strip()
    # Control-flow branch for conditional or iterative execution.
    for suf in ["_nifti", ".nii.gz", ".nii"]:
        # Control-flow branch for conditional or iterative execution.
        if s.endswith(suf):
            s = s[: -len(suf)]
    # Return the computed value to the caller.
    return s

# Function: `resolve_original_dir_from_sid` implements a reusable processing step.
def resolve_original_dir_from_sid(
    sid_raw: str,
    orig_dirs_sorted: List[Path],
    orig_name_to_dir: Dict[str, Path],
) -> Optional[Path]:
    """
    Resolve original patient folder using sid from meta.json or directory name.
    """
    sid_norm = normalize_sid(sid_raw)

    # 1) exact match on normalized
    # Control-flow branch for conditional or iterative execution.
    if sid_norm in orig_name_to_dir:
        # Return the computed value to the caller.
        return orig_name_to_dir[sid_norm]

    # 2) exact match on raw
    # Control-flow branch for conditional or iterative execution.
    if sid_raw in orig_name_to_dir:
        # Return the computed value to the caller.
        return orig_name_to_dir[sid_raw]

    # 3) prefix match (handles e.g. original folder = UCSF-PDGM-0333_11)
    # Control-flow branch for conditional or iterative execution.
    for p in orig_dirs_sorted:
        # Control-flow branch for conditional or iterative execution.
        if p.name.startswith(sid_norm):
            # Return the computed value to the caller.
            return p

    # 4) contains match (last resort)
    # Control-flow branch for conditional or iterative execution.
    for p in orig_dirs_sorted:
        # Control-flow branch for conditional or iterative execution.
        if sid_norm in p.name or p.name in sid_norm:
            # Return the computed value to the caller.
            return p

    # Return the computed value to the caller.
    return None


# Class definition: `SliceIndex` encapsulates related model behavior.
class SliceIndex:
    __slots__ = ("patient_key", "k")
    # Function: `__init__` implements a reusable processing step.
    def __init__(self, patient_key: str, k: int):
        self.patient_key = patient_key
        self.k = k


# ---------------------------
# Dataset
# ---------------------------
# Class definition: `GenerationEvalDataset` encapsulates related model behavior.
class GenerationEvalDataset(Dataset):
    # Function: `__init__` implements a reusable processing step.
    def __init__(
        self,
        synth_patient_dirs: List[Path],
        original_data_root: Path,
        img_size: int = 224,
        use_resized: bool = True,
        allow_fallback: bool = True,
        verbose_map: bool = True,
    ):
        self.synth_patient_dirs = synth_patient_dirs
        self.original_data_root = original_data_root
        self.img_size = img_size
        self.use_resized = use_resized
        self.allow_fallback = allow_fallback

        # Prepare original dir index for robust mapping
        self.orig_dirs_sorted = list_original_patient_dirs(self.original_data_root)
        self.orig_name_to_dir = {p.name: p for p in self.orig_dirs_sorted}

        # Control-flow branch for conditional or iterative execution.
        if len(self.orig_dirs_sorted) == 0:
            raise RuntimeError(f"No patient folders found under original root: {self.original_data_root}")

        # meta maps patient_key (synth folder name) -> file paths
        self.meta: Dict[str, Dict[str, Path]] = {}

        mapped = 0
        skipped_nomap = 0
        skipped_missing = 0

        # Control-flow branch for conditional or iterative execution.
        for p_synth_dir in self.synth_patient_dirs:
            synth_name = p_synth_dir.name

            sid = read_meta_sid(p_synth_dir)
            # Control-flow branch for conditional or iterative execution.
            if sid is None:
                print(f"[WARN] Skipping {synth_name}, could not determine patient ID.")
                skipped_nomap += 1
                continue

            p_orig_dir = resolve_original_dir_from_sid(
                sid_raw=sid,
                orig_dirs_sorted=self.orig_dirs_sorted,
                orig_name_to_dir=self.orig_name_to_dir,
            )

            # Control-flow branch for conditional or iterative execution.
            if p_orig_dir is None:
                print(f"[WARN] Skipping {synth_name}, could not map sid='{sid}' to any original folder.")
                skipped_nomap += 1
                continue

            # Prefer resized synth if asked
            # Control-flow branch for conditional or iterative execution.
            if self.use_resized:
                synth_paths = {
                    "synth_t1": p_synth_dir / "t1_synth_resized.nii.gz",
                    "synth_t2": p_synth_dir / "t2_synth_resized.nii.gz",
                    "synth_flair": p_synth_dir / "flair_synth_resized.nii.gz",
                }
                # Control-flow branch for conditional or iterative execution.
                if self.allow_fallback and not all(x.exists() for x in synth_paths.values()):
                    synth_paths = {
                        "synth_t1": p_synth_dir / "t1_synth.nii.gz",
                        "synth_t2": p_synth_dir / "t2_synth.nii.gz",
                        "synth_flair": p_synth_dir / "flair_synth.nii.gz",
                    }
            else:
                synth_paths = {
                    "synth_t1": p_synth_dir / "t1_synth.nii.gz",
                    "synth_t2": p_synth_dir / "t2_synth.nii.gz",
                    "synth_flair": p_synth_dir / "flair_synth.nii.gz",
                }

            orig_paths = {
                "orig_t1": p_orig_dir / "t1.nii.gz",
                "orig_t2": p_orig_dir / "t2.nii.gz",
                "orig_flair": p_orig_dir / "flair.nii.gz",
            }

            paths = {**synth_paths, **orig_paths}

            # Control-flow branch for conditional or iterative execution.
            if not all(pp.exists() for pp in paths.values()):
                print(f"[WARN] Skipping {synth_name} (orig={p_orig_dir.name}), missing one or more files.")
                skipped_missing += 1
                continue

            self.meta[synth_name] = paths
            mapped += 1

            # Control-flow branch for conditional or iterative execution.
            if verbose_map and synth_name != p_orig_dir.name:
                print(f"[MAP] {synth_name} (sid={sid}) -> {p_orig_dir.name}")

        # Control-flow branch for conditional or iterative execution.
        if verbose_map:
            print(
                f"[MAP] Mapped {mapped}/{len(self.synth_patient_dirs)} synth folders "
                f"(unmapped={skipped_nomap}, missing_files={skipped_missing})."
            )

        self.slice_index: List[SliceIndex] = []
        self._build_index()

    # Function: `_build_index` implements a reusable processing step.
    def _build_index(self):
        print("Building slice index (depth=min(orig,synth))...")
        total = 0
        # Control-flow branch for conditional or iterative execution.
        for patient_key, paths in self.meta.items():
            # Control-flow branch for conditional or iterative execution.
            try:
                od = min(
                    nifti_depth(paths["orig_t1"]),
                    nifti_depth(paths["orig_t2"]),
                    nifti_depth(paths["orig_flair"]),
                )
                sd = min(
                    nifti_depth(paths["synth_t1"]),
                    nifti_depth(paths["synth_t2"]),
                    nifti_depth(paths["synth_flair"]),
                )
                depth = min(od, sd)

                # Control-flow branch for conditional or iterative execution.
                if od != sd:
                    print(f"[WARN] Depth mismatch {patient_key}: orig={od} synth={sd} -> using {depth}")

                # Control-flow branch for conditional or iterative execution.
                for k in range(depth):
                    self.slice_index.append(SliceIndex(patient_key, k))
                total += depth
            # Control-flow branch for conditional or iterative execution.
            except Exception as e:
                print(f"[WARN] Skipping {patient_key}, failed to read shapes: {e}")
                continue

        print(f"Built index: {total} total slices from {len(self.meta)} mapped patients.")

    # Function: `__len__` implements a reusable processing step.
    def __len__(self) -> int:
        # Return the computed value to the caller.
        return len(self.slice_index)

    # Function: `_load_and_prep_slice` implements a reusable processing step.
    def _load_and_prep_slice(self, paths: Dict[str, Path], k: int, key_prefix: str) -> torch.Tensor:
        t1 = load_nifti(paths[f"{key_prefix}_t1"])
        t2 = load_nifti(paths[f"{key_prefix}_t2"])
        fl = load_nifti(paths[f"{key_prefix}_flair"])

        s1 = minmax_per_slice(t1[:, :, k])
        s2 = minmax_per_slice(t2[:, :, k])
        s3 = minmax_per_slice(fl[:, :, k])

        x = np.stack([s1, s2, s3], axis=0)
        x = torch.from_numpy(x.copy()).float()
        x = center_crop_or_pad(x, self.img_size)
        # Return the computed value to the caller.
        return x

    # Function: `__getitem__` implements a reusable processing step.
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si = self.slice_index[idx]
        paths = self.meta[si.patient_key]
        real_img = self._load_and_prep_slice(paths, si.k, "orig")
        synth_img = self._load_and_prep_slice(paths, si.k, "synth")
        # Return the computed value to the caller.
        return {"real": real_img, "synth": synth_img}


# ---------------------------
# Eval helper
# ---------------------------
# Function: `run_evaluation` implements a reusable processing step.
def run_evaluation(
    synth_root_str: str,
    original_root_str: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_resized: bool,
    no_fallback: bool,
) -> Tuple[float, float]:
    synth_root = Path(synth_root_str)
    original_root = Path(original_root_str)

    print(f"\n--- Evaluating Dataset ---")
    print(f"  Synth:    {synth_root}")
    print(f"  Original: {original_root}")
    print("----------------------------")

    # Accept ALL directories (both with and without meta.json)
    synth_patients = sorted(
        [p for p in synth_root.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )
    print(f"Found {len(synth_patients)} synthetic patient folders in {synth_root}")

    test_ds = GenerationEvalDataset(
        synth_patient_dirs=synth_patients,
        original_data_root=original_root,
        img_size=img_size,
        use_resized=use_resized,
        allow_fallback=(not no_fallback),
        verbose_map=True,
    )

    # Control-flow branch for conditional or iterative execution.
    if len(test_ds) == 0:
        print("Error: Dataset is empty. Check paths, file names, or meta.json mapping.")
        # Return the computed value to the caller.
        return (float("nan"), float("nan"))

    test_ld = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    print(f"Running evaluation on {len(test_ds)} slices...")

    # Control-flow branch for conditional or iterative execution.
    for batch in tqdm(test_ld, desc=f"Evaluating {synth_root.name}"):
        real_imgs = batch["real"].to(device)     # float [0,1]
        synth_imgs = batch["synth"].to(device)   # float [0,1]

        ssim_metric.update(synth_imgs, real_imgs)

        real_u8 = (real_imgs * 255).clamp(0, 255).byte()
        synth_u8 = (synth_imgs * 255).clamp(0, 255).byte()
        fid_metric.update(real_u8, real=True)
        fid_metric.update(synth_u8, real=False)

    final_ssim = float(ssim_metric.compute().item())
    final_fid = float(fid_metric.compute().item())

    print("Evaluation complete.")
    # Return the computed value to the caller.
    return final_ssim, final_fid


# Function: `main` implements a reusable processing step.
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate synthetic medical images using FID and SSIM metrics. "
                    "Can evaluate GBM only, PDGM only, or both domains."
    )

    ap.add_argument("--synth_gbm_root", type=str, default="",
                    help="Path to synthetic GBM images (optional)")
    ap.add_argument("--original_gbm_root", type=str, default="",
                    help="Path to original GBM images (optional)")

    ap.add_argument("--synth_pdgm_root", type=str, default="",
                    help="Path to synthetic PDGM images (optional)")
    ap.add_argument("--original_pdgm_root", type=str, default="",
                    help="Path to original PDGM images (optional)")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--use_resized", action="store_true",
                    help="Prefer *_synth_resized.nii.gz (fallback to *_synth.nii.gz unless --no_fallback)")
    ap.add_argument("--no_fallback", action="store_true",
                    help="Disable fallback to *_synth.nii.gz")

    args = ap.parse_args()
    device = torch.device(args.device)

    # Determine which domains to evaluate
    eval_gbm = bool(args.synth_gbm_root and args.original_gbm_root)
    eval_pdgm = bool(args.synth_pdgm_root and args.original_pdgm_root)

    # Control-flow branch for conditional or iterative execution.
    if not eval_gbm and not eval_pdgm:
        print("Error: Must provide paths for at least one domain:")
        print("  GBM:  --synth_gbm_root AND --original_gbm_root")
        print("  PDGM: --synth_pdgm_root AND --original_pdgm_root")
        sys.exit(1)

    # Check for incomplete domain specifications
    # Control-flow branch for conditional or iterative execution.
    if args.synth_gbm_root and not args.original_gbm_root:
        print("Error: --synth_gbm_root provided but --original_gbm_root is missing")
        sys.exit(1)
    # Control-flow branch for conditional or iterative execution.
    if args.original_gbm_root and not args.synth_gbm_root:
        print("Error: --original_gbm_root provided but --synth_gbm_root is missing")
        sys.exit(1)
    # Control-flow branch for conditional or iterative execution.
    if args.synth_pdgm_root and not args.original_pdgm_root:
        print("Error: --synth_pdgm_root provided but --original_pdgm_root is missing")
        sys.exit(1)
    # Control-flow branch for conditional or iterative execution.
    if args.original_pdgm_root and not args.synth_pdgm_root:
        print("Error: --original_pdgm_root provided but --synth_pdgm_root is missing")
        sys.exit(1)

    print("\n" + "="*70)
    print("FID/SSIM EVALUATION")
    print("="*70)
    print(f"\nDomains to evaluate:")
    # Control-flow branch for conditional or iterative execution.
    if eval_gbm:
        print(f"  ✓ GBM")
    # Control-flow branch for conditional or iterative execution.
    if eval_pdgm:
        print(f"  ✓ PDGM")
    print()

    # Evaluate GBM if requested
    gbm_ssim, gbm_fid = float('nan'), float('nan')
    # Control-flow branch for conditional or iterative execution.
    if eval_gbm:
        gbm_ssim, gbm_fid = run_evaluation(
            synth_root_str=args.synth_gbm_root,
            original_root_str=args.original_gbm_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_resized=args.use_resized,
            no_fallback=args.no_fallback,
        )

    # Evaluate PDGM if requested
    pdgm_ssim, pdgm_fid = float('nan'), float('nan')
    # Control-flow branch for conditional or iterative execution.
    if eval_pdgm:
        pdgm_ssim, pdgm_fid = run_evaluation(
            synth_root_str=args.synth_pdgm_root,
            original_root_str=args.original_pdgm_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_resized=args.use_resized,
            no_fallback=args.no_fallback,
        )

    # Print summary
    print("\n\n--- 📊 Final Evaluation Summary ---")
    print("--------------------------------------")
    
    # Control-flow branch for conditional or iterative execution.
    if eval_gbm:
        print("Dataset: GBM")
        print(f"  SSIM (Higher is better):  {gbm_ssim:.6f}")
        print(f"  FID (Lower is better):    {gbm_fid:.4f}")
        print("--------------------------------------")
    
    # Control-flow branch for conditional or iterative execution.
    if eval_pdgm:
        print("Dataset: PDGM")
        print(f"  SSIM (Higher is better):  {pdgm_ssim:.6f}")
        print(f"  FID (Lower is better):    {pdgm_fid:.4f}")
        print("--------------------------------------")


# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    main()