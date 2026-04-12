# Training Guide

## Overview

ALDM training consists of two stages:
1. **Stage 1**: Train 3D VAE on data-rich GBM source domain
2. **Stage 2**: Train conditional latent diffusion model with few-shot PDGM adaptation

---

## Stage 1: VAE Training

### Training on GBM (Source Domain)

```bash
python scripts/train_vae.py \
    --config configs/vae_gbm.yaml \
    --data_dir ./data/processed/gbm \
    --output_dir ./checkpoints/vae \
    --epochs 200 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_workers 2
```

**Expected Training Time**: ~48 hours on single GPU (24GB VRAM)

**Checkpoints Saved**:
- `vae_epoch_050.pth`
- `vae_epoch_100.pth`
- `vae_epoch_150.pth`
- `vae_epoch_200.pth` (final)
- `vae_best.pth` (best validation loss)

### Optional: Fine-tune VAE on PDGM

```bash
python scripts/train_vae.py \
    --config configs/vae_pdgm_finetune.yaml \
    --data_dir ./data/processed/pdgm \
    --output_dir ./checkpoints/vae_finetuned \
    --resume ./checkpoints/vae/vae_best.pth \
    --epochs 50 \
    --batch_size 1 \
    --learning_rate 5e-5
```

---

## Stage 2: Diffusion Training

### Training on GBM Latents

```bash
python scripts/train_diffusion.py \
    --config configs/diffusion_gbm.yaml \
    --vae_checkpoint ./checkpoints/vae/vae_best.pth \
    --data_dir ./data/processed/gbm \
    --output_dir ./checkpoints/diffusion \
    --epochs 300 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --guidance_scale 3.0
```

**Expected Training Time**: ~72 hours on single GPU (24GB VRAM)

### Few-Shot Adaptation on PDGM (16-shot)

```bash
python scripts/train_diffusion.py \
    --config configs/diffusion_16shot.yaml \
    --vae_checkpoint ./checkpoints/vae/vae_best.pth \
    --diffusion_checkpoint ./checkpoints/diffusion/diffusion_gbm_epoch_300.pth \
    --data_dir ./data/processed/pdgm \
    --manifest ./data/manifests/pdgm_train_16shot.txt \
    --output_dir ./checkpoints/diffusion_16shot \
    --epochs 100 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --guidance_scale 3.0
```

**Expected Training Time**: ~12 hours on single GPU

---

## Hyperparameters

### VAE Configuration

```yaml
# configs/vae_gbm.yaml
model:
  latent_channels: 8
  base_channels: 64
  input_channels: 3  # T1, T2, FLAIR
  spatial_dims: [112, 112, 112]

training:
  learning_rate: 1.0e-4
  batch_size: 1
  epochs: 200
  optimizer: AdamW
  weight_decay: 1.0e-4
  beta1: 0.9
  beta2: 0.999

loss:
  reconstruction_weight: 1.0
  kl_weight: 1.0e-4
  kl_warmup_fraction: 0.35
  gradient_consistency_weight: 0.1

data:
  num_workers: 2
  pin_memory: true
  augmentation: true
```

### Diffusion Configuration

```yaml
# configs/diffusion_16shot.yaml
model:
  unet_base_channels: 64
  timestep_embedding_dim: 256
  attention_resolutions: [14, 7]
  
diffusion:
  timesteps: 1000
  beta_schedule: linear
  beta_start: 1.0e-4
  beta_end: 2.0e-2

training:
  learning_rate: 2.0e-4
  batch_size: 1
  epochs: 100
  optimizer: AdamW
  weight_decay: 1.0e-4
  ema_decay: 0.999

conditioning:
  guidance_scale: 3.0
  dropout_prob: 0.1
  control_weight: 1.0
  tumor_loss_weight: 2.0

latent_normalization:
  enabled: true
  num_batches: 400
  epsilon: 1.0e-6
```

---

## Training Monitoring

### TensorBoard

```bash
tensorboard --logdir ./checkpoints/diffusion_16shot/logs
```

**Metrics Logged**:
- Training loss (per epoch)
- Validation loss (per epoch)
- Reconstruction quality (SSIM, PSNR)
- Sample images (every 10 epochs)
- Learning rate schedule

### Checkpointing Strategy

Checkpoints are saved:
- Every 50 epochs
- When validation loss improves (best model)
- At the end of training (final model)

**Checkpoint Contents**:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'ema_state_dict': ema_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config
}
```

---

## Resuming Training

### Resume VAE Training

```bash
python scripts/train_vae.py \
    --config configs/vae_gbm.yaml \
    --resume ./checkpoints/vae/vae_epoch_100.pth \
    --data_dir ./data/processed/gbm \
    --output_dir ./checkpoints/vae
```

### Resume Diffusion Training

```bash
python scripts/train_diffusion.py \
    --config configs/diffusion_16shot.yaml \
    --resume ./checkpoints/diffusion_16shot/diffusion_epoch_050.pth \
    --data_dir ./data/processed/pdgm \
    --output_dir ./checkpoints/diffusion_16shot
```

---

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 24GB VRAM (e.g., RTX 3090, A5000)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space
- **CUDA**: 11.8 or higher

### Recommended Setup
- **GPU**: NVIDIA A100 (40GB) or H100
- **RAM**: 64GB system memory
- **Storage**: 500GB SSD

### Multi-GPU Training

```bash
# Use DataParallel (not recommended for batch_size=1)
python scripts/train_diffusion.py \
    --config configs/diffusion_16shot.yaml \
    --multi_gpu \
    --gpu_ids 0,1,2,3
```

---

## Training Tips

### Memory Optimization
- Use gradient checkpointing: `--gradient_checkpointing`
- Reduce base channels: `--base_channels 32`
- Use mixed precision: `--mixed_precision`

### Convergence Issues
- Increase KL warmup: `--kl_warmup_fraction 0.5`
- Reduce learning rate: `--learning_rate 5e-5`
- Enable gradient clipping: `--grad_clip_norm 1.0`

### Faster Training
- Disable EMA during initial epochs: `--ema_start_epoch 50`
- Reduce validation frequency: `--val_every 10`
- Use fewer diffusion timesteps: `--timesteps 500`

---

## Ablation Studies

### Guidance Scale Sensitivity

```bash
for scale in 0.3 0.5 1.0 3.0; do
    python scripts/train_diffusion.py \
        --config configs/diffusion_16shot.yaml \
        --guidance_scale $scale \
        --output_dir ./checkpoints/ablation_scale_${scale}
done
```

### Few-Shot Size Comparison

```bash
# 10-shot
python scripts/train_diffusion.py \
    --config configs/diffusion_10shot.yaml \
    --manifest ./data/manifests/pdgm_train_10shot.txt \
    --output_dir ./checkpoints/diffusion_10shot

# 16-shot
python scripts/train_diffusion.py \
    --config configs/diffusion_16shot.yaml \
    --manifest ./data/manifests/pdgm_train_16shot.txt \
    --output_dir ./checkpoints/diffusion_16shot
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce spatial dimensions
--spatial_dims 96 96 96

# Use gradient accumulation
--accumulation_steps 4
```

### NaN Loss
```bash
# Enable gradient clipping
--grad_clip_norm 1.0

# Reduce learning rate
--learning_rate 1e-5

# Check data normalization
python scripts/check_data_stats.py --data_dir ./data/processed/gbm
```

### Slow Training
```bash
# Enable mixed precision
--mixed_precision

# Increase num_workers
--num_workers 4

# Use faster data format (HDF5 instead of NIfTI)
--data_format hdf5
```

---

## Expected Results

### VAE (after 200 epochs on GBM)
- Reconstruction SSIM: >0.95
- KL divergence: ~0.01
- Latent space: Well-structured, continuous

### Diffusion (after 300 epochs on GBM + 100 on PDGM)
- FID: <90
- SSIM: >0.70
- Downstream AUC: >0.95

---

## References

- VAE architecture based on Stable Diffusion VAE
- Diffusion process follows DDPM (Ho et al., 2020)
- Classifier-free guidance (Ho & Salimans, 2022)
