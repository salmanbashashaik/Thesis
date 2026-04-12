"""
NIfTI I/O utilities for medical imaging.
"""

import nibabel as nib
import numpy as np
from pathlib import Path


def load_nifti(filepath):
    """
    Load NIfTI file.
    
    Args:
        filepath: Path to .nii or .nii.gz file
        
    Returns:
        volume: numpy array
        affine: affine transformation matrix
        header: NIfTI header
    """
    nii = nib.load(str(filepath))
    volume = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    return volume, affine, header


def save_nifti(volume, filepath, affine=None, header=None):
    """
    Save volume as NIfTI file.
    
    Args:
        volume: numpy array
        filepath: Output path
        affine: Optional affine matrix
        header: Optional NIfTI header
    """
    if affine is None:
        affine = np.eye(4)
    
    nii = nib.Nifti1Image(volume, affine, header)
    nib.save(nii, str(filepath))


def load_multimodal_mri(subject_dir, modalities=['t1', 't2', 'flair']):
    """
    Load multimodal MRI for a subject.
    
    Args:
        subject_dir: Path to subject directory
        modalities: List of modality names
        
    Returns:
        volume: (C, D, H, W) stacked modalities
        affine: affine matrix
    """
    subject_dir = Path(subject_dir)
    volumes = []
    
    for modality in modalities:
        filepath = subject_dir / f"{modality}.nii.gz"
        if not filepath.exists():
            filepath = subject_dir / f"{modality}.nii"
        
        volume, affine, _ = load_nifti(filepath)
        volumes.append(volume)
    
    # Stack along channel dimension
    volume = np.stack(volumes, axis=0)
    return volume, affine


def save_multimodal_mri(volume, output_dir, subject_id, modalities=['t1', 't2', 'flair'], affine=None):
    """
    Save multimodal MRI volume.
    
    Args:
        volume: (C, D, H, W) stacked modalities
        output_dir: Output directory
        subject_id: Subject identifier
        modalities: List of modality names
        affine: Affine matrix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, modality in enumerate(modalities):
        filepath = output_dir / f"{subject_id}_{modality}.nii.gz"
        save_nifti(volume[i], filepath, affine)


# TODO: Add your I/O implementations here
