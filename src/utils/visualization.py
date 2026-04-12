"""
Visualization utilities for MRI volumes.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_slice(volume, slice_idx=None, modality_names=['T1', 'T2', 'FLAIR'], save_path=None):
    """
    Visualize a single slice across all modalities.
    
    Args:
        volume: (3, D, H, W) MRI volume
        slice_idx: Slice index (default: middle slice)
        modality_names: Names of modalities
        save_path: Optional path to save figure
    """
    if slice_idx is None:
        slice_idx = volume.shape[1] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, name) in enumerate(zip(axes, modality_names)):
        ax.imshow(volume[i, slice_idx], cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(real, synthetic, slice_idx=None, save_path=None):
    """
    Compare real and synthetic MRI side-by-side.
    
    Args:
        real: (3, D, H, W) real MRI
        synthetic: (3, D, H, W) synthetic MRI
        slice_idx: Slice index
        save_path: Optional save path
    """
    if slice_idx is None:
        slice_idx = real.shape[1] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    modalities = ['T1', 'T2', 'FLAIR']
    
    for i, name in enumerate(modalities):
        # Real
        axes[0, i].imshow(real[i, slice_idx], cmap='gray')
        axes[0, i].set_title(f'Real {name}')
        axes[0, i].axis('off')
        
        # Synthetic
        axes[1, i].imshow(synthetic[i, slice_idx], cmap='gray')
        axes[1, i].set_title(f'Synthetic {name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_3d_volume(volume, threshold=0.5, save_path=None):
    """
    3D visualization of MRI volume.
    
    Args:
        volume: (D, H, W) single modality volume
        threshold: Threshold for surface rendering
        save_path: Optional save path
    """
    # TODO: Implement 3D volume rendering
    pass


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional save path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# TODO: Add your visualization implementations here
