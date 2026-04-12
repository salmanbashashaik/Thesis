#!/usr/bin/env python3
"""
Analysis and visualization script for 10-shot training results.

Usage:
    python analyze_10shot.py --outdir /path/to/output

Features:
- Plot training curves
- Analyze loss progression
- Compare source vs target adaptation
- Generate summary statistics
- Create comprehensive training report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data(outdir: Path):
    """Load all training data and hyperparameters."""
    log_dir = outdir / "logs"
    
    data = {}
    
    # Load hyperparameters
    hyperparam_file = log_dir / "hyperparameters.json"
    if hyperparam_file.exists():
        with open(hyperparam_file, 'r') as f:
            data['hyperparams'] = json.load(f)
    
    # Load VAE training data
    vae_file = log_dir / "vae_training.csv"
    if vae_file.exists():
        data['vae'] = pd.read_csv(vae_file)
    
    # Load LDM training data
    ldm_file = log_dir / "ldm_training.csv"
    if ldm_file.exists():
        data['ldm'] = pd.read_csv(ldm_file)
    
    return data


def plot_vae_training(vae_df: pd.DataFrame, save_path: Path):
    """Plot VAE training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reconstruction loss - source
    ax = axes[0, 0]
    src_data = vae_df[vae_df['stage'] == 'src']
    if len(src_data) > 0:
        ax.plot(src_data['epoch'], src_data['rec_loss_mean'], 'b-', label='Source', linewidth=2)
        ax.fill_between(
            src_data['epoch'],
            src_data['rec_loss_mean'] - src_data['rec_loss_std'],
            src_data['rec_loss_mean'] + src_data['rec_loss_std'],
            alpha=0.3, color='b'
        )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reconstruction Loss (L1)', fontsize=12)
    ax.set_title('VAE Reconstruction Loss - Source (GBM)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction loss - target
    ax = axes[0, 1]
    tgt_data = vae_df[vae_df['stage'] == 'tgt']
    if len(tgt_data) > 0:
        ax.plot(tgt_data['epoch'], tgt_data['rec_loss_mean'], 'r-', label='Target (10-shot)', linewidth=2)
        ax.fill_between(
            tgt_data['epoch'],
            tgt_data['rec_loss_mean'] - tgt_data['rec_loss_std'],
            tgt_data['rec_loss_mean'] + tgt_data['rec_loss_std'],
            alpha=0.3, color='r'
        )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reconstruction Loss (L1)', fontsize=12)
    ax.set_title('VAE Reconstruction Loss - Target (PDGM)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KL divergence - source
    ax = axes[1, 0]
    if len(src_data) > 0:
        ax.plot(src_data['epoch'], src_data['kl_loss_mean'], 'b-', label='KL Loss', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(src_data['epoch'], src_data['kl_weight'], 'g--', label='KL Weight', linewidth=1.5)
        ax2.set_ylabel('KL Weight', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('VAE KL Divergence - Source', fontsize=14, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    if len(src_data) > 0:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # KL divergence - target
    ax = axes[1, 1]
    if len(tgt_data) > 0:
        ax.plot(tgt_data['epoch'], tgt_data['kl_loss_mean'], 'r-', label='KL Loss', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(tgt_data['epoch'], tgt_data['kl_weight'], 'g--', label='KL Weight', linewidth=1.5)
        ax2.set_ylabel('KL Weight', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12, color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.set_title('VAE KL Divergence - Target', fontsize=14, fontweight='bold')
    if len(tgt_data) > 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"VAE training plot saved: {save_path}")
    plt.close()


def plot_ldm_training(ldm_df: pd.DataFrame, save_path: Path):
    """Plot LDM training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Diffusion loss - source
    ax = axes[0, 0]
    src_data = ldm_df[ldm_df['stage'] == 'ldm_src']
    if len(src_data) > 0:
        ax.plot(src_data['epoch'], src_data['diffusion_loss_mean'], 'b-', linewidth=2)
        ax.fill_between(
            src_data['epoch'],
            src_data['diffusion_loss_mean'] - src_data['diffusion_loss_std'],
            src_data['diffusion_loss_mean'] + src_data['diffusion_loss_std'],
            alpha=0.3, color='b'
        )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Diffusion Loss (MSE)', fontsize=12)
    ax.set_title('LDM Training - Source (GBM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Diffusion loss - target (THE KEY PLOT FOR 10-SHOT)
    ax = axes[0, 1]
    tgt_data = ldm_df[ldm_df['stage'] == 'ldm_tgt']
    if len(tgt_data) > 0:
        ax.plot(tgt_data['epoch'], tgt_data['diffusion_loss_mean'], 'r-', linewidth=2.5)
        ax.fill_between(
            tgt_data['epoch'],
            tgt_data['diffusion_loss_mean'] - tgt_data['diffusion_loss_std'],
            tgt_data['diffusion_loss_mean'] + tgt_data['diffusion_loss_std'],
            alpha=0.3, color='r'
        )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Diffusion Loss (MSE)', fontsize=12)
    ax.set_title('LDM Adaptation - Target (PDGM 10-Shot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Learning rate over time
    ax = axes[1, 0]
    if len(src_data) > 0:
        ax.plot(src_data['epoch'], src_data['lr'], 'b-', label='Source', linewidth=2)
    if len(tgt_data) > 0:
        # Offset target epoch numbers for visualization
        tgt_epoch_offset = src_data['epoch'].max() if len(src_data) > 0 else 0
        ax.plot(tgt_data['epoch'] + tgt_epoch_offset, tgt_data['lr'], 'r-', label='Target', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Training time per epoch
    ax = axes[1, 1]
    if len(src_data) > 0:
        ax.plot(src_data['epoch'], src_data['epoch_time_s'], 'b-', label='Source', alpha=0.7, linewidth=2)
    if len(tgt_data) > 0:
        tgt_epoch_offset = src_data['epoch'].max() if len(src_data) > 0 else 0
        ax.plot(tgt_data['epoch'] + tgt_epoch_offset, tgt_data['epoch_time_s'], 'r-', label='Target', alpha=0.7, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time per Epoch (seconds)', fontsize=12)
    ax.set_title('Training Speed', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"LDM training plot saved: {save_path}")
    plt.close()


def plot_combined_comparison(vae_df: pd.DataFrame, ldm_df: pd.DataFrame, save_path: Path):
    """Create a combined comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # VAE: Source vs Target final reconstruction loss
    ax = axes[0]
    src_vae = vae_df[vae_df['stage'] == 'src']
    tgt_vae = vae_df[vae_df['stage'] == 'tgt']
    
    stages = []
    final_losses = []
    colors = []
    
    if len(src_vae) > 0:
        stages.append('VAE\nSource')
        final_losses.append(src_vae['rec_loss_mean'].iloc[-1])
        colors.append('steelblue')
    
    if len(tgt_vae) > 0:
        stages.append('VAE\nTarget')
        final_losses.append(tgt_vae['rec_loss_mean'].iloc[-1])
        colors.append('coral')
    
    bars = ax.bar(stages, final_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Final Reconstruction Loss', fontsize=12)
    ax.set_title('VAE Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # LDM: Source vs Target final diffusion loss
    ax = axes[1]
    src_ldm = ldm_df[ldm_df['stage'] == 'ldm_src']
    tgt_ldm = ldm_df[ldm_df['stage'] == 'ldm_tgt']
    
    stages = []
    final_losses = []
    colors = []
    
    if len(src_ldm) > 0:
        stages.append('LDM\nSource')
        final_losses.append(src_ldm['diffusion_loss_mean'].iloc[-1])
        colors.append('steelblue')
    
    if len(tgt_ldm) > 0:
        stages.append('LDM\nTarget\n(10-Shot)')
        final_losses.append(tgt_ldm['diffusion_loss_mean'].iloc[-1])
        colors.append('coral')
    
    bars = ax.bar(stages, final_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Final Diffusion Loss', fontsize=12)
    ax.set_title('LDM Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {save_path}")
    plt.close()


def generate_summary_stats(data: dict, save_path: Path):
    """Generate summary statistics report."""
    report = []
    
    report.append("="*80)
    report.append("10-SHOT TRAINING SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    # Hyperparameters
    if 'hyperparams' in data:
        hp = data['hyperparams']
        report.append("HYPERPARAMETERS")
        report.append("-" * 80)
        report.append(f"  Seed:                 {hp.get('seed', 'N/A')}")
        report.append(f"  Device:               {hp.get('device', 'N/A')}")
        report.append(f"  Batch Size:           {hp.get('batch_size', 'N/A')}")
        report.append(f"  VAE Base:             {hp.get('vae_base', 'N/A')}")
        report.append(f"  UNet Base:            {hp.get('unet_base', 'N/A')}")
        report.append(f"  Latent Channels:      {hp.get('z_channels', 'N/A')}")
        report.append(f"  Latent Size:          {hp.get('latent_size', 'N/A')}")
        report.append("")
    
    # VAE Training
    if 'vae' in data:
        vae_df = data['vae']
        report.append("VAE TRAINING RESULTS")
        report.append("-" * 80)
        
        src_vae = vae_df[vae_df['stage'] == 'src']
        if len(src_vae) > 0:
            report.append(f"  Source (GBM):")
            report.append(f"    Total Epochs:       {len(src_vae)}")
            report.append(f"    Initial Rec Loss:   {src_vae['rec_loss_mean'].iloc[0]:.6f}")
            report.append(f"    Final Rec Loss:     {src_vae['rec_loss_mean'].iloc[-1]:.6f}")
            report.append(f"    Loss Reduction:     {(1 - src_vae['rec_loss_mean'].iloc[-1]/src_vae['rec_loss_mean'].iloc[0])*100:.2f}%")
            report.append(f"    Final KL:           {src_vae['kl_loss_mean'].iloc[-1]:.6f}")
            report.append(f"    Total Time:         {src_vae['epoch_time_s'].sum()/3600:.2f} hours")
        
        tgt_vae = vae_df[vae_df['stage'] == 'tgt']
        if len(tgt_vae) > 0:
            report.append(f"  Target (PDGM 10-shot):")
            report.append(f"    Total Epochs:       {len(tgt_vae)}")
            report.append(f"    Initial Rec Loss:   {tgt_vae['rec_loss_mean'].iloc[0]:.6f}")
            report.append(f"    Final Rec Loss:     {tgt_vae['rec_loss_mean'].iloc[-1]:.6f}")
            report.append(f"    Loss Reduction:     {(1 - tgt_vae['rec_loss_mean'].iloc[-1]/tgt_vae['rec_loss_mean'].iloc[0])*100:.2f}%")
            report.append(f"    Final KL:           {tgt_vae['kl_loss_mean'].iloc[-1]:.6f}")
            report.append(f"    Total Time:         {tgt_vae['epoch_time_s'].sum()/3600:.2f} hours")
        
        report.append("")
    
    # LDM Training
    if 'ldm' in data:
        ldm_df = data['ldm']
        report.append("LDM TRAINING RESULTS")
        report.append("-" * 80)
        
        src_ldm = ldm_df[ldm_df['stage'] == 'ldm_src']
        if len(src_ldm) > 0:
            report.append(f"  Source (GBM):")
            report.append(f"    Total Epochs:       {len(src_ldm)}")
            report.append(f"    Initial Loss:       {src_ldm['diffusion_loss_mean'].iloc[0]:.6f}")
            report.append(f"    Final Loss:         {src_ldm['diffusion_loss_mean'].iloc[-1]:.6f}")
            report.append(f"    Loss Reduction:     {(1 - src_ldm['diffusion_loss_mean'].iloc[-1]/src_ldm['diffusion_loss_mean'].iloc[0])*100:.2f}%")
            report.append(f"    Min Loss Achieved:  {src_ldm['diffusion_loss_mean'].min():.6f} (Epoch {src_ldm['diffusion_loss_mean'].idxmin()+1})")
            report.append(f"    Total Time:         {src_ldm['epoch_time_s'].sum()/3600:.2f} hours")
        
        tgt_ldm = ldm_df[ldm_df['stage'] == 'ldm_tgt']
        if len(tgt_ldm) > 0:
            report.append(f"  Target (PDGM 10-shot) *** KEY RESULTS ***:")
            report.append(f"    Total Epochs:       {len(tgt_ldm)}")
            report.append(f"    Initial Loss:       {tgt_ldm['diffusion_loss_mean'].iloc[0]:.6f}")
            report.append(f"    Final Loss:         {tgt_ldm['diffusion_loss_mean'].iloc[-1]:.6f}")
            report.append(f"    Loss Reduction:     {(1 - tgt_ldm['diffusion_loss_mean'].iloc[-1]/tgt_ldm['diffusion_loss_mean'].iloc[0])*100:.2f}%")
            report.append(f"    Min Loss Achieved:  {tgt_ldm['diffusion_loss_mean'].min():.6f} (Epoch {tgt_ldm['diffusion_loss_mean'].idxmin()+1})")
            report.append(f"    Total Time:         {tgt_ldm['epoch_time_s'].sum()/3600:.2f} hours")
            
            # Overfitting check
            min_loss_epoch = tgt_ldm['diffusion_loss_mean'].idxmin()
            final_epoch = len(tgt_ldm) - 1
            if final_epoch > min_loss_epoch + 10:
                report.append(f"    OVERFITTING CHECK:  Loss increased after epoch {min_loss_epoch+1}")
                report.append(f"                        Consider early stopping or reducing epochs")
        
        report.append("")
    
    # Total training time
    total_time = 0
    if 'vae' in data:
        total_time += data['vae']['epoch_time_s'].sum()
    if 'ldm' in data:
        total_time += data['ldm']['epoch_time_s'].sum()
    
    report.append("TOTAL TRAINING")
    report.append("-" * 80)
    report.append(f"  Total Time:           {total_time/3600:.2f} hours")
    report.append(f"  Total Time:           {total_time/86400:.2f} days")
    report.append("")
    
    report.append("="*80)
    
    # Write report
    report_text = "\n".join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSummary report saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze 10-shot training results")
    parser.add_argument("--outdir", required=True, help="Output directory from training")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    
    if not outdir.exists():
        print(f"Error: Output directory not found: {outdir}")
        return
    
    print("Loading training data...")
    data = load_data(outdir)
    
    if not data:
        print("No training data found!")
        return
    
    # Create analysis directory
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    print("Generating plots...")
    
    # Plot VAE training
    if 'vae' in data:
        plot_vae_training(data['vae'], analysis_dir / "vae_training.png")
    
    # Plot LDM training
    if 'ldm' in data:
        plot_ldm_training(data['ldm'], analysis_dir / "ldm_training.png")
    
    # Plot comparison
    if 'vae' in data and 'ldm' in data:
        plot_combined_comparison(data['vae'], data['ldm'], analysis_dir / "comparison.png")
    
    # Generate summary statistics
    generate_summary_stats(data, analysis_dir / "summary_report.txt")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {analysis_dir}")
    print("="*80)


if __name__ == "__main__":
    main()