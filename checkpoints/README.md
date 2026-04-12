# ALDM: Anatomically-Conditioned Latent Diffusion Model

## Checkpoints Directory

This directory contains trained model checkpoints for ALDM and baselines.

### Directory Structure

```
checkpoints/
├── vae/
│   ├── vae_epoch_050.pth
│   ├── vae_epoch_100.pth
│   ├── vae_epoch_150.pth
│   ├── vae_epoch_200.pth
│   ├── vae_best.pth              # Best validation loss
│   └── latent_stats.npz          # Latent normalization statistics
│
├── diffusion/
│   ├── diffusion_gbm_epoch_300.pth
│   ├── diffusion_gbm_best.pth
│   └── ema_gbm_epoch_300.pth
│
├── diffusion_16shot/
│   ├── diffusion_epoch_025.pth
│   ├── diffusion_epoch_050.pth
│   ├── diffusion_epoch_075.pth
│   ├── diffusion_epoch_100.pth
│   ├── diffusion_best.pth
│   ├── ema_epoch_100.pth         # EMA weights for inference
│   └── config.yaml
│
├── diffusion_10shot/
│   ├── diffusion_epoch_100.pth
│   ├── diffusion_best.pth
│   └── ema_epoch_100.pth
│
└── baselines/
    ├── cgan/
    │   └── cgan_best.pth
    ├── cgan_3disc/
    │   └── cgan3d_best.pth
    └── vaegan/
        └── vaegan_best.pth
```

### Download Pretrained Models

**TODO**: Add download links once models are uploaded

```bash
# Download all checkpoints
wget https://example.com/aldm_checkpoints.tar.gz
tar -xzf aldm_checkpoints.tar.gz -C ./checkpoints/
```

### Checkpoint Format

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'ema_state_dict': OrderedDict,  # For diffusion models
    'optimizer_state_dict': OrderedDict,
    'loss': float,
    'config': dict
}
```

### Loading Checkpoints

```python
import torch

# Load VAE
vae_checkpoint = torch.load('./checkpoints/vae/vae_best.pth')
vae.load_state_dict(vae_checkpoint['model_state_dict'])

# Load Diffusion (use EMA for inference)
diffusion_checkpoint = torch.load('./checkpoints/diffusion_16shot/ema_epoch_100.pth')
unet.load_state_dict(diffusion_checkpoint['model_state_dict'])
```

### Model Sizes

| Model | Parameters | File Size |
|-------|-----------|-----------|
| VAE | ~30M | ~120 MB |
| Diffusion U-Net | ~90M | ~360 MB |
| ControlNet | ~20M | ~80 MB |
| EMA Weights | ~90M | ~360 MB |

### Training Time

| Model | Epochs | GPU | Time |
|-------|--------|-----|------|
| VAE (GBM) | 200 | A100 | ~48h |
| Diffusion (GBM) | 300 | A100 | ~72h |
| Diffusion (16-shot) | 100 | A100 | ~12h |

---

**Note**: Checkpoints will be made publicly available upon paper publication.
