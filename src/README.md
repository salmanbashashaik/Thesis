# Source Code

This directory contains the implementation of ALDM.

## Structure

```
src/
├── models/              # Model architectures
│   ├── vae.py          # 3D Variational Autoencoder
│   ├── unet.py         # 3D U-Net for diffusion
│   ├── diffusion.py    # DDPM implementation
│   └── controlnet.py   # ControlNet conditioning
│
├── data/               # Data loading and preprocessing
│   ├── dataset.py      # Dataset classes
│   ├── preprocessing.py # Preprocessing utilities
│   └── augmentation.py # Data augmentation
│
├── training/           # Training loops
│   ├── train_vae.py    # VAE training
│   ├── train_diffusion.py # Diffusion training
│   └── ema.py          # Exponential Moving Average
│
├── evaluation/         # Evaluation metrics
│   ├── metrics.py      # FID, SSIM, etc.
│   ├── downstream_cnn.py # CNN classifier
│   └── evaluate.py     # Evaluation pipeline
│
└── utils/              # Utility functions
    ├── io_nifti.py     # NIfTI I/O
    ├── visualization.py # Visualization tools
    └── seed.py         # Reproducibility
```

## Usage

### Import Models

```python
from src.models.vae import VAE3D
from src.models.unet import UNet3D
from src.models.diffusion import LatentDDPM
from src.models.controlnet import ControlNet3D

# Initialize models
vae = VAE3D(latent_channels=8, base_channels=64)
unet = UNet3D(base_channels=64, timestep_emb_dim=256)
diffusion = LatentDDPM(unet, timesteps=1000)
controlnet = ControlNet3D(in_channels=1, base_channels=64)
```

### Load Data

```python
from src.data.dataset import MRIDataset
from torch.utils.data import DataLoader

dataset = MRIDataset(
    data_dir='./data/processed/gbm/train',
    load_masks=True
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2
)
```

### Training

```python
from src.training.train_vae import train_vae
from src.training.train_diffusion import train_diffusion

# Train VAE
vae = train_vae(
    config='configs/vae_gbm.yaml',
    data_dir='./data/processed/gbm',
    output_dir='./checkpoints/vae'
)

# Train Diffusion
diffusion = train_diffusion(
    config='configs/diffusion_16shot.yaml',
    vae_checkpoint='./checkpoints/vae/best.pth',
    data_dir='./data/processed/pdgm',
    output_dir='./checkpoints/diffusion'
)
```

### Evaluation

```python
from src.evaluation.metrics import compute_fid, compute_ssim
from src.evaluation.downstream_cnn import evaluate_downstream_cnn

# Compute fidelity metrics
fid = compute_fid(real_features, synthetic_features)
ssim = compute_ssim(real_volume, synthetic_volume)

# Evaluate downstream task
metrics = evaluate_downstream_cnn(model, test_loader)
```

## Implementation Notes

- All models use 3D convolutions for volumetric processing
- Batch size is typically 1 due to memory constraints
- Instance normalization is used instead of batch normalization
- EMA weights are used for inference in diffusion models
- Classifier-free guidance is implemented for controllable generation

## TODO

- [ ] Complete ControlNet implementation
- [ ] Add data augmentation strategies
- [ ] Implement latent normalization statistics computation
- [ ] Add visualization notebooks
- [ ] Write unit tests
