# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
ldm3d/io_nifti.py

Robust NIfTI I/O utilities:
- load_vol_safe
- load_mask_safe
- save_nifti
- validate_input_data

These functions are intentionally framework-agnostic
(except for torch.Tensor in save_nifti).

Design goals of this module:
  - Be resilient to messy real-world medical files (NaNs, weird ranges, empty volumes).
  - Normalize intensities in a consistent, training-friendly way.
  - Provide a quick "sanity check" that writes an example volume to disk before training,
    so we can visually confirm preprocessing is correct (huge time-saver).
"""

# Import dependencies used by this module.
from __future__ import annotations

import os
import numpy as np
import nibabel as nib
import torch

from torch.utils.data import DataLoader

from ldm3d.config import IMAGE_SIZE, DEVICE


# ----------------------------
# LOAD VOLUME (ROBUST)
# ----------------------------
# Function: `load_vol_safe` implements a reusable processing step.
def load_vol_safe(path: str) -> np.ndarray:
    """
    Load a NIfTI image volume from disk and convert it into a normalized tensor-ready array.

    What this function does:
      1) Loads the NIfTI file with nibabel.
      2) Converts to float32 and replaces NaNs/Infs with safe finite numbers.
      3) Ensures a channel dimension exists:
           - If arr is [D,H,W], convert to [1,D,H,W]
      4) Applies robust intensity normalization to [-1, 1] using percentiles
         computed only over non-zero voxels.

    Why percentile normalization?
      - MRI intensity ranges are not standardized across scanners/sites.
      - Using (1st, 99th) percentiles is a common robust technique to reduce
        the influence of outliers while preserving most anatomical contrast.

    Non-zero mask logic:
      - Many MRI volumes have large zero-valued backgrounds.
      - We compute percentiles only where arr > 0 to avoid background dominating stats.

    Output:
      np.ndarray of shape [1, D, H, W], dtype float32, values ideally in [-1, 1].

    Failure behavior:
      - If anything goes wrong (corrupt file, missing file, etc.), returns a safe fallback
        volume of shape [1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE] filled with -1.0.
        This prevents training crashes due to a single bad input.
    """
    # Control-flow branch for conditional or iterative execution.
    try:
        # Load NIfTI file
        img = nib.load(path)

        # Convert to float32 (nibabel returns float64 by default) and sanitize NaNs/Infs
        arr = img.get_fdata(dtype=np.float32)
        arr = np.nan_to_num(arr)

        # Ensure channel dimension exists for consistent downstream handling
        # Control-flow branch for conditional or iterative execution.
        if arr.ndim == 3:
            arr = arr[None]  # [1, D, H, W]

        # Compute normalization statistics only on non-zero voxels (foreground-ish)
        mask = arr > 0
        # Control-flow branch for conditional or iterative execution.
        if mask.sum() > 0:
            # Robust range estimate using percentiles
            mn = np.percentile(arr[mask], 1.0)
            mx = np.percentile(arr[mask], 99.0)

            # Clip to robust range to suppress extreme outliers
            arr = np.clip(arr, mn, mx)

            # Control-flow branch for conditional or iterative execution.
            if mx - mn > 1e-6:
                # Scale to [0,1] then shift to [-1,1]
                arr = (arr - mn) / (mx - mn)   # [0,1]
                arr = arr * 2.0 - 1.0          # [-1,1]
            else:
                # Degenerate case: almost constant foreground -> return "all black" (-1)
                arr = np.zeros_like(arr) - 1.0
        else:
            # Entire volume is non-positive -> treat as empty/invalid and return "all black"
            arr = np.zeros_like(arr) - 1.0

        # Return the computed value to the caller.
        return arr.astype(np.float32)

    # Control-flow branch for conditional or iterative execution.
    except Exception as e:
        # Hard failure path: return safe fallback cube so the pipeline doesn't explode
        print(f"[ERR] Failed to load {path}: {e}")
        # Return the computed value to the caller.
        return (
            np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) - 1.0
        )


# ----------------------------
# LOAD MASK (ROBUST)
# ----------------------------
# Function: `load_mask_safe` implements a reusable processing step.
def load_mask_safe(path: str) -> np.ndarray:
    """
    Load a tumor/lesion mask from disk and return a clean binary mask.

    Assumptions:
      - Mask file may contain arbitrary positive values for foreground.
      - We binarize with (arr > 0).

    Steps:
      1) Load NIfTI mask with nibabel.
      2) Convert to float32 and sanitize NaNs/Infs.
      3) Ensure channel dimension exists:
           - If arr is [D,H,W], convert to [1,D,H,W]
      4) Binarize to {0,1}.

    Output:
      np.ndarray of shape [1, D, H, W], dtype float32, values in {0,1}

    Notes:
      - Unlike images, we do NOT normalize masks to [-1,1].
      - Masks typically should be resized using nearest-neighbor interpolation
        to preserve discrete labels (handled elsewhere in the dataset).
    """
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    arr = np.nan_to_num(arr)

    # Ensure channel dimension for consistency
    # Control-flow branch for conditional or iterative execution.
    if arr.ndim == 3:
        arr = arr[None]  # [1,D,H,W]

    # Binarize: anything > 0 is treated as foreground
    arr = (arr > 0).astype(np.float32)
    # Return the computed value to the caller.
    return arr


# ----------------------------
# SAVE NIFTI
# ----------------------------
# Function: `save_nifti` implements a reusable processing step.
def save_nifti(tensor_3d: torch.Tensor, path: str, verbose: bool = True) -> None:
    """
    Save a single-channel 3D torch tensor as a NIfTI file.

    Intended use:
      - Debugging / qualitative inspection of model inputs and outputs.
      - Sanity checking preprocessing and sampling results.

    Input:
      tensor_3d:
        Either:
          - [D, H, W]
          - [1, D, H, W]
        Values are assumed to be in [-1, 1] (the convention used by load_vol_safe).

    Output format:
      - Saved NIfTI contains values mapped to [0, 1] for easy viewing:
            arr = (arr + 1) / 2
      - Uses an identity affine (np.eye(4)).
        (So it does not preserve real-world spacing/orientation — this is deliberate
         for quick visualization, not for clinical coordinate accuracy.)

    Verbose logging:
      - Prints min/max/mean after mapping to [0,1].
      - Emits a "looks black" warning if mean is extremely low,
        which often indicates a normalization bug or collapsed output.
    """
    # Convert tensor to numpy for nibabel
    arr = tensor_3d.detach().cpu().numpy()
    # Control-flow branch for conditional or iterative execution.
    if arr.ndim == 4:
        # If [1,D,H,W], drop channel dimension
        arr = arr[0]

    # Map from [-1,1] to [0,1] for visualization-friendly saved output
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)

    # Control-flow branch for conditional or iterative execution.
    if verbose:
        mn, mx, mean = float(arr.min()), float(arr.max()), float(arr.mean())

        # A near-zero mean often means the saved image will appear mostly black
        # (common symptom of a bug in normalization or decoding)
        # Control-flow branch for conditional or iterative execution.
        if mean < 1e-3:
            print(f"[CRITICAL WARN] SAVED IMAGE LOOKS BLACK (mean={mean:.6f}): {path}")
        else:
            print(f"[SAVE] {path} | min={mn:.4f} max={mx:.4f} mean={mean:.4f}")

    # Save using identity affine (fast + simple)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


# ----------------------------
# DATA SANITY CHECK
# ----------------------------
# Function: `validate_input_data` implements a reusable processing step.
def validate_input_data(dataloader: DataLoader, outdir: str) -> None:
    """
    Perform an immediate end-to-end check of the dataset + dataloader pipeline.

    What it does:
      - Pulls exactly one batch from `dataloader`.
      - Saves the first sample's first channel (T1) as a NIfTI file in `outdir`.

    Why this exists:
      - In medical imaging pipelines, the #1 time-waster is training on broken preprocessing:
          * wrong intensity scaling (everything black/white)
          * wrong orientation / wrong axis ordering
          * wrong resizing (squished anatomy)
          * broken file discovery
      - This gives us a concrete artifact on disk to inspect BEFORE burning GPU hours.

    Output:
      outdir/sanity_check_input_{SubjectID}_t1.nii.gz

    Failure behavior:
      - If the dataloader crashes, raise the exception.
        That’s intentional: if our pipeline can’t load one batch, training will not work.
    """
    print("--- RUNNING IMMEDIATE DATA SANITY CHECK ---")
    # Control-flow branch for conditional or iterative execution.
    try:
        # Grab first batch from dataloader
        x, _, sid = next(iter(dataloader))

        # Move to DEVICE to ensure preprocessing + device transfers work as expected
        x = x.to(DEVICE)

        # Save the first sample's first modality (channel 0 assumed to be T1)
        out_path = os.path.join(
            outdir, f"sanity_check_input_{sid[0]}_t1.nii.gz"
        )
        save_nifti(x[0, 0], out_path, verbose=True)

        print("--- SANITY CHECK COMPLETE: CHECK FILE NOW ---")
    # Control-flow branch for conditional or iterative execution.
    except Exception as e:
        # If this fails, training will almost certainly fail too.
        print(f"[CRITICAL FAILURE] Data Loader crashed: {e}")
        raise
