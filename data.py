"""
ldm3d/data.py

Dataset + augmentation utilities for 3D MRI volumes.

This file is responsible for:
  - Discovering subject folders and locating modality + mask files on disk
  - Loading NIfTI volumes safely (via io_nifti helpers)
  - Resizing volumes to a consistent cubic shape (IMAGE_SIZE^3)
  - Returning tensors in a consistent format for training/inference

Key conventions used by the training code:
  - Each subject lives under: root/SubjectID/
  - Modalities are expected to include: t1, t2, flair  (3 channels total)
  - A tumor/lesion mask file exists and contains "mask" in its filename
  - Images are returned as float32 tensors shaped [3, D, H, W]
  - Masks are returned as float32 tensors shaped [1, D, H, W] with values in {0,1}
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ldm3d.config import IMAGE_SIZE
from ldm3d.io_nifti import load_vol_safe, load_mask_safe


# ----------------------------
# AUGMENTATIONS (MASK)
# ----------------------------
def augment_mask(mask: torch.Tensor, p: float = 0.7) -> torch.Tensor:
    """
    Apply random augmentations to a binary mask.

    Inputs:
      mask: [B, 1, D, H, W] float tensor with values {0,1}
      p: probability of applying augmentation at all

    Augmentations implemented here are intentionally *mask-only*:
      - Random translation (via torch.roll)
      - Random dilation or erosion (via max-pooling tricks)

    Why mask-only augmentations?
      - The diffusion model often uses the mask as a control signal (conditioning).
      - Augmenting masks encourages robustness to slight annotation shifts/shape changes.
      - These ops do NOT touch the underlying MRI intensities, only the control signal.

    Output:
      Augmented mask with the same shape and still binarized to {0,1}.
    """
    # With probability (1-p), return the input unchanged
    if random.random() > p:
        return mask

    # Work on a local variable so the original reference isn't accidentally reused
    m = mask

    # random translation (±4 voxels)
    # This simulates small misalignments / annotation imprecision.
    if random.random() < 0.5:
        shifts = [random.randint(-4, 4) for _ in range(3)]
        # dims=(2,3,4) correspond to (D,H,W) because mask is [B,1,D,H,W]
        m = torch.roll(m, shifts=shifts, dims=(2, 3, 4))

    # random dilation / erosion
    # k controls morphological strength. Larger k -> bigger shape changes.
    k = random.choice([3, 5, 7])
    if random.random() < 0.5:
        # dilation:
        # max_pool3d on a binary mask expands foreground regions
        m = F.max_pool3d(m, kernel_size=k, stride=1, padding=k // 2)
    else:
        # erosion:
        # erosion(mask) == 1 - dilation(1-mask)
        m = 1.0 - F.max_pool3d(1.0 - m, kernel_size=k, stride=1, padding=k // 2)

    # Re-binarize to avoid numerical artifacts from pooling
    return (m > 0.5).float()


def blur3d(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Simple mean blur using avg pooling.

    Inputs:
      x: tensor shaped like [B,C,D,H,W] (or compatible)
      k: kernel size (odd values are typical: 3/5/7)

    Notes:
      - This is a cheap low-pass filter.
      - Often used for smoothing masks or control signals (depending on how train.py calls it).
      - Uses stride=1 with padding to preserve spatial size.
    """
    return F.avg_pool3d(x, kernel_size=k, stride=1, padding=k // 2)


# ----------------------------
# DATASET
# ----------------------------
class VolFolder(Dataset):
    """
    A dataset that loads 3D MRI volumes (T1/T2/FLAIR) + a tumor mask from a folder structure.

    Expected filesystem layout:
        root/
          SubjectID_1/
            t1.nii.gz      (or any file containing 't1' in its name)
            t2.nii.gz      (or any file containing 't2' in its name)
            flair.nii.gz   (or any file containing 'flair' in its name)
            *mask*.nii.gz  (any file containing 'mask' in its name)
          SubjectID_2/
            ...

    Returns (per sample):
        x:    [3, D, H, W] float32 (assumed normalized by load_vol_safe; docstring says [-1,1])
        mask: [1, D, H, W] float32 values {0,1}
        sid:  subject identifier string

    Resizing behavior:
      - If any volume/mask is not exactly (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        it is resized to that cube:
          - MRI volumes: trilinear interpolation (smooth, intensity-preserving-ish)
          - Masks: nearest interpolation (preserves discrete labels)
    """

    def __init__(self, root: str, subjects: Optional[List[str]] = None):
        # Root directory containing subject subfolders
        self.root = root

        # Discover all subject directories (immediate subfolders of root)
        try:
            full_subs = sorted(next(os.walk(root))[1])
        except StopIteration:
            # os.walk yields nothing if root is invalid/empty
            raise ValueError(f"Root dir {root} is empty or invalid.")

        # If a subject list is provided, use it; otherwise use all discovered subjects
        self.subs = subjects if subjects else full_subs
        if not self.subs:
            raise ValueError(f"No subjects found in {root}")

        print(f"[DATA] Found {len(self.subs)} subjects in {root}")

    def __len__(self) -> int:
        # Number of subject folders available
        return len(self.subs)

    # ----------------------------
    # FILE DISCOVERY
    # ----------------------------
    def _find_modality(self, d: str, m: str) -> str:
        """
        Locate a modality file inside a subject directory.

        Priority:
          1) Exact match: {m}.nii.gz  (e.g., t1.nii.gz)
          2) Fallback: first file that contains the modality substring (case-insensitive)
             and ends with .nii.gz

        Raises:
          FileNotFoundError if no suitable file is found.
        """
        # Fast path: exact conventional filename
        p = os.path.join(d, f"{m}.nii.gz")
        if os.path.exists(p):
            return p

        # Fallback: more flexible naming (e.g., "sub-001_T1w.nii.gz")
        cands = [
            f for f in os.listdir(d)
            if (m in f.lower()) and f.endswith(".nii.gz")
        ]
        if cands:
            return os.path.join(d, cands[0])

        raise FileNotFoundError(f"Missing modality '{m}' in {d}")

    def _find_mask(self, d: str) -> str:
        """
        Locate the mask file inside a subject directory.

        Strategy:
          - Return the first file that contains "mask" (case-insensitive)
            and ends with .nii.gz

        Raises:
          FileNotFoundError if not found.
        """
        for name in os.listdir(d):
            if "mask" in name.lower() and name.endswith(".nii.gz"):
                return os.path.join(d, name)
        raise FileNotFoundError(f"No mask file found in {d}")

    # ----------------------------
    # GETITEM
    # ----------------------------
    def __getitem__(self, i: int):
        """
        Load a single subject by index.

        Steps:
          1) Determine subject id and directory path
          2) Load each modality as numpy arrays via load_vol_safe (expected shape [1,D,H,W])
          3) Concatenate into a 3-channel volume [3,D,H,W]
          4) Resize to IMAGE_SIZE cube if needed (trilinear)
          5) Load mask (expected shape [?,D,H,W]) and keep first channel [:1] -> [1,D,H,W]
          6) Resize mask to IMAGE_SIZE cube if needed (nearest)
          7) Return tensors and subject id

        Robustness:
          - Any exception triggers a warning and returns a safe fallback:
              x = all -1 (black-ish if [-1,1]) and mask = all 0
            This prevents training loops from crashing mid-epoch due to a single bad file.
        """
        sid = self.subs[i]
        d = os.path.join(self.root, sid)

        try:
            # ---- Load three MRI modalities as [1,D,H,W] each
            imgs = []
            for m in ("t1", "t2", "flair"):
                p = self._find_modality(d, m)
                imgs.append(load_vol_safe(p))  # [1,D,H,W]

            # ---- Stack to [3,D,H,W] in numpy then convert to torch
            x = np.concatenate(imgs, axis=0).astype(np.float32)  # [3,D,H,W]
            x = torch.from_numpy(x)

            # ---- Enforce consistent cube size for training stability
            # Interpolate expects [N,C,D,H,W], so temporarily add batch dim.
            if tuple(x.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE):
                x = x.unsqueeze(0)  # [1,3,D,H,W]
                x = F.interpolate(
                    x,
                    size=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE),
                    mode="trilinear",
                    align_corners=False,
                )
                x = x.squeeze(0)

            # ---- Load mask
            mask_path = self._find_mask(d)

            # load_mask_safe returns numpy; slicing [:1] ensures [1,D,H,W] even if multi-channel
            mask = load_mask_safe(mask_path)[:1]  # [1,D,H,W]
            mask = torch.from_numpy(mask)

            # ---- Resize mask with nearest neighbor to preserve binary labels
            if tuple(mask.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE):
                mask = mask.unsqueeze(0)  # [1,1,D,H,W]
                mask = F.interpolate(
                    mask,
                    size=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE),
                    mode="nearest",
                )
                mask = mask.squeeze(0)

            return x, mask, sid

        except Exception as e:
            # If a subject fails to load, don't crash training; return a known dummy sample
            print(f"[WARN] Error loading {sid}: {e}")
            x = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32) - 1.0
            m = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
            return x, m, sid


# ----------------------------
# UTILS
# ----------------------------
def load_subject_list(path: str) -> List[str]:
    """
    Read a newline-delimited subject list from `path`.

    Returns:
      - [] if path is empty or missing
      - otherwise a list of stripped non-empty lines

    Typical usage:
      - PDGM few-shot experiments where we only want a subset of subjects.
    """
    if not path or not os.path.exists(path):
        return []
    return [l.strip() for l in open(path) if l.strip()]
