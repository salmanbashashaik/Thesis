# Anatomically-Conditioned Latent Diffusion Model (ALDM)

[![Paper](https://img.shields.io/badge/Paper-CAN--AI%202026-blue)](https://proceedings.mlr.press/v318/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

**Official implementation of _“Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis”_ — accepted at the 39th Canadian Conference on Artificial Intelligence (CAN-AI 2026).**

ALDM transfers anatomical priors from a data-rich glioblastoma source domain to a data-scarce diffuse-glioma target domain to synthesize spatially coherent 3D MRI volumes with only **10–16 target examples**.

## Highlights

- **Few-shot cross-domain generation:** target-domain synthesis with as few as 10–16 examples.
- **3D latent diffusion:** operates on compressed volumetric representations rather than independent 2D slices.
- **Anatomical conditioning:** tumor-mask conditioning with FiLM-style modulation and ControlNet-style residual injection.
- **Multimodal volumes:** T1, T2, and FLAIR channels at `112 × 112 × 112` spatial resolution.
- **Downstream utility:** best reported 16-shot configuration reaches **FID 85.40** and **AUC 0.987** for downstream classification.

## Architecture

```text
3D MRI volume
     ↓
  VAE Encoder
     ↓
Latent volume  ───────────────┐
     ↓                        │
Diffusion U-Net ← tumor mask / control features
     ↓              │
ControlNet-style residuals + FiLM conditioning
     ↓
  VAE Decoder
     ↓
Synthetic 3D MRI
```

The implementation uses a 3D VAE to compress `3 × 112 × 112 × 112` MRI volumes into an `8 × 28 × 28 × 28` latent space. A conditional U-Net-based DDPM then denoises those latents while incorporating anatomical mask information at multiple scales.

## Quantitative Results

### PDGM target domain

| Model | FID ↓ | SSIM ↑ | BAcc ↑ | F1 ↑ | AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CGAN | 145.22 | 0.374 | 0.764 | 0.720 | 0.876 |
| 3M-CGAN | 116.48 | 0.680 | 0.780 | 0.731 | 0.866 |
| VAE-GAN | 88.18 | **0.750** | 0.751 | 0.675 | 0.882 |
| **ALDM (K=16, s=3.0)** | **85.40** | 0.712 | **0.875** | **0.836** | **0.987** |

## Repository Structure

```text
anatomically-conditioned-ldm/
├── configs/                     # VAE and diffusion experiment configs
├── src/
│   ├── models/                  # 3D VAE, U-Net, diffusion components
│   ├── data/                    # dataset and preprocessing code
│   ├── training/                # training loops and EMA
│   ├── evaluation/              # metrics and downstream CNN evaluation
│   └── utils/                   # shared utilities
├── scripts/                     # preprocessing, training, generation, evaluation
├── baselines/                   # comparison models
├── docs/                        # dataset/training/evaluation documentation
├── checkpoints/                 # model weights stored separately
├── results/                     # experiment outputs
├── requirements.txt
└── setup.py
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ recommended
- 24 GB+ GPU memory recommended for full 3D training

```bash
git clone https://github.com/salmanbashashaik/anatomically-conditioned-ldm.git
cd anatomically-conditioned-ldm

conda create -n aldm python=3.10
conda activate aldm
pip install -r requirements.txt
```

## Data Preparation

The experiments use public MRI datasets from The Cancer Imaging Archive:

- **UPENN-GBM** — source domain
- **UCSF-PDGM** — target domain

Preprocess a dataset with:

```bash
python scripts/preprocess_data.py \
  --input_dir /path/to/raw/data \
  --output_dir ./data/processed \
  --dataset gbm
```

Use `--dataset pdgm` for the target domain. See [`docs/DATASET.md`](docs/DATASET.md) for the full preprocessing workflow.

## Training

### 1. Train the source-domain VAE

```bash
python train_vae.py \
  --config configs/vae_gbm.yaml \
  --data_dir ./data/processed/gbm \
  --output_dir ./checkpoints/vae
```

### 2. Train the conditional diffusion model

```bash
python train_diffusion.py \
  --config configs/diffusion_16shot.yaml \
  --vae_checkpoint ./checkpoints/vae/best.pth \
  --data_dir ./data/processed/pdgm \
  --output_dir ./checkpoints/diffusion
```

## Inference

```bash
python generate.py \
  --vae_checkpoint ./checkpoints/vae/best.pth \
  --diffusion_checkpoint ./checkpoints/diffusion/best.pth \
  --mask_dir ./data/masks \
  --output_dir ./outputs/synthetic \
  --num_samples 16 \
  --guidance_scale 3.0
```

## Evaluation

```bash
python evaluate.py \
  --synthetic_dir ./outputs/synthetic \
  --real_dir ./data/processed/pdgm/test \
  --output_dir ./results
```

Evaluation includes image-distribution, structural-similarity, and downstream-classification metrics used in the paper.

## Pretrained Models

Model checkpoints are kept separate from the repository because of their size. For checkpoint availability, contact **sbashash@uwaterloo.ca**.

## Citation

If you use this implementation, please cite the CAN-AI 2026 paper:

> Shaik Salman Basha et al. “Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis.” Proceedings of the 39th Canadian Conference on Artificial Intelligence, 2026.

For the canonical citation metadata, use the published proceedings entry linked above.

## License

This repository is released under the Creative Commons Attribution 4.0 International License. See [`LICENSE`](LICENSE).

## Acknowledgments

This work was conducted at the University of New Brunswick with public datasets from The Cancer Imaging Archive.

## Contact

For questions or issues, open a GitHub issue or email **sbashash@uwaterloo.ca**.
