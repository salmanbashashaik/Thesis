# Anatomically-Conditioned Latent Diffusion Model (ALDM)

[![Paper](https://img.shields.io/badge/Paper-CAN--AI%202026-blue)](https://proceedings.mlr.press/v318/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

**Official implementation of "Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis"**

*Accepted at the 39th Canadian Conference on Artificial Intelligence (CAN-AI 2026)*

---

## Overview

ALDM is a novel framework for synthesizing high-fidelity 3D volumetric MRI scans in extreme few-shot settings. By transferring anatomical priors from a data-rich glioblastoma (GBM) source domain to a data-scarce preoperative diffuse glioma (PDGM) target domain, ALDM achieves superior performance with only **16 target images**.

### Key Features
- **Few-shot learning**: Generates realistic MRI volumes with only 10-16 target samples
- **Cross-domain transfer**: Leverages anatomical priors from GBM to synthesize PDGM scans
- **3D volumetric synthesis**: Produces spatially coherent 112×112×112 volumes across T1, T2, and FLAIR modalities
- **Anatomical conditioning**: Uses tumor masks via ControlNet for precise spatial control
- **State-of-the-art results**: FID 85.40, AUC 0.987 on downstream classification

---

## Architecture

ALDM consists of two stages:

1. **3D VAE**: Compresses MRI volumes (3×112×112×112) into latent space (8×28×28×28)
2. **Conditional Latent Diffusion**: U-Net-based DDPM with anatomical mask conditioning

```
Input MRI → VAE Encoder → Latent Space → Diffusion U-Net → VAE Decoder → Synthetic MRI
                                              ↑
                                         Tumor Mask (ControlNet)
```

---

## Results

### Quantitative Performance (PDGM Target Domain)

| Model | FID ↓ | SSIM ↑ | BAcc ↑ | F1 ↑ | AUC ↑ |
|-------|-------|--------|--------|------|-------|
| CGAN | 145.22 | 0.374 | 0.764 | 0.720 | 0.876 |
| 3M-CGAN | 116.48 | 0.680 | 0.780 | 0.731 | 0.866 |
| VAE-GAN | 88.18 | **0.750** | 0.751 | 0.675 | 0.882 |
| **ALDM (K=16, s=3.0)** | **85.40** | 0.712 | **0.875** | **0.836** | **0.987** |

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 24GB+ GPU memory (for 3D volumes)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/aldm-mri-synthesis.git
cd aldm-mri-synthesis

# Create conda environment
conda create -n aldm python=3.10
conda activate aldm

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Data Preparation

Download the datasets:
- **GBM**: [UPENN-GBM Dataset](https://doi.org/10.7937/TCIA.709X-DN49)
- **PDGM**: [UCSF-PDGM Dataset](https://doi.org/10.7937/3ec9-yk87)

Preprocess the data:
```bash
python scripts/preprocess_data.py \
    --input_dir /path/to/raw/data \
    --output_dir ./data/processed \
    --dataset gbm  # or pdgm
```

### 2. Training

#### Stage 1: Train VAE on GBM
```bash
python train_vae.py \
    --config configs/vae_gbm.yaml \
    --data_dir ./data/processed/gbm \
    --output_dir ./checkpoints/vae
```

#### Stage 2: Train Diffusion Model
```bash
python train_diffusion.py \
    --config configs/diffusion_10shot.yaml \
    --vae_checkpoint ./checkpoints/vae/best.pth \
    --data_dir ./data/processed/pdgm \
    --output_dir ./checkpoints/diffusion
```

### 3. Inference

Generate synthetic MRI volumes:
```bash
python generate.py \
    --vae_checkpoint ./checkpoints/vae/best.pth \
    --diffusion_checkpoint ./checkpoints/diffusion/best.pth \
    --mask_dir ./data/masks \
    --output_dir ./outputs/synthetic \
    --num_samples 16 \
    --guidance_scale 3.0
```

### 4. Evaluation

```bash
python evaluate.py \
    --synthetic_dir ./outputs/synthetic \
    --real_dir ./data/processed/pdgm/test \
    --output_dir ./results
```

---

## Repository Structure

```
aldm-mri-synthesis/
├── README.md                    # This file
├── LICENSE                      # CC BY 4.0
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
│
├── configs/                     # Training configurations
│   ├── vae_gbm.yaml
│   ├── diffusion_10shot.yaml
│   └── diffusion_16shot.yaml
│
├── src/                         # Source code
│   ├── models/                  # VAE, U-Net, Diffusion, ControlNet
│   ├── data/                    # Dataset and preprocessing
│   ├── training/                # Training loops and EMA
│   ├── evaluation/              # Metrics and downstream CNN
│   └── utils/                   # I/O and utilities
│
├── scripts/                     # Executable scripts
│   ├── preprocess_data.py
│   ├── train_vae.py
│   ├── train_diffusion.py
│   ├── generate.py
│   └── evaluate.py
│
├── baselines/                   # Baseline implementations
│   ├── cgan/
│   ├── cgan_3disc/
│   └── vaegan/
│
├── docs/                        # Detailed documentation
│   ├── DATASET.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   └── ARCHITECTURE.md
│
├── checkpoints/                 # Model weights (download separately)
├── data/                        # Datasets (download separately)
└── results/                     # Evaluation results
```

---

## Pretrained Models

Download pretrained checkpoints:

| Model | Dataset | Epochs | Download |
|-------|---------|--------|----------|
| VAE | GBM | 200 | [Link](#) |
| Diffusion (16-shot) | PDGM | 100 | [Link](#) |
| Diffusion (10-shot) | PDGM | 100 | [Link](#) |
| AlexLite-DG (CNN) | PDGM | - | [Link](#) |

---

## Datasets

### GBM (Source Domain)
- **Source**: [UPENN-GBM](https://doi.org/10.7937/TCIA.709X-DN49)
- **Size**: ~828,000 slices
- **Modalities**: T1, T2, FLAIR
- **Format**: NIfTI

### PDGM (Target Domain)
- **Source**: [UCSF-PDGM](https://doi.org/10.7937/3ec9-yk87)
- **Size**: ~12,000 images
- **Modalities**: T1, T2, FLAIR
- **Format**: NIfTI

See [DATASET.md](docs/DATASET.md) for detailed preprocessing instructions.

---

## Hyperparameters

### VAE Training
```yaml
learning_rate: 1e-4
batch_size: 1
epochs: 200
kl_weight: 1e-4
kl_warmup: 0.35
latent_channels: 8
base_channels: 64
```

### Diffusion Training
```yaml
learning_rate: 2e-4
batch_size: 1
epochs: 300 (GBM) + 100 (PDGM)
timesteps: 1000
beta_schedule: linear
beta_start: 1e-4
beta_end: 2e-2
guidance_scale: 3.0
ema_decay: 0.999
```

See [configs/](configs/) for complete configurations.

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{basha2026aldm,
  title={Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis},
  author={Basha, Shaik Salman and [Co-authors]},
  booktitle={Proceedings of the 39th Canadian Conference on Artificial Intelligence},
  year={2026},
  series={Proceedings of Machine Learning Research},
  volume={318},
  publisher={PMLR}
}
```

---

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Supervisor: Dr. Hung Cao, UNB
- Computing resources provided by UNB Laboratory Server
- Datasets: UPENN-GBM and UCSF-PDGM from The Cancer Imaging Archive

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: [salmanbasha.shaik@unb.ca]

---

---

**Note**: This repository contains the official implementation of our CAN-AI 2026 paper. 
