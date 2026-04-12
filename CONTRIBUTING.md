# Contributing to ALDM

Thank you for your interest in our work!

## For Reviewers

This repository contains the official implementation of our CAN-AI 2026 paper. Key resources:

- **Paper Results**: See main [README.md](README.md)
- **Architecture Details**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Training Details**: See [docs/TRAINING.md](docs/TRAINING.md)
- **Evaluation Protocol**: See [docs/EVALUATION.md](docs/EVALUATION.md)
- **Dataset Information**: See [docs/DATASET.md](docs/DATASET.md)

## Code Structure

- `src/models/` - Model implementations (VAE, U-Net, Diffusion)
- `src/evaluation/` - Evaluation metrics and downstream CNN
- `baselines/` - Baseline implementations (CGAN, 3M-CGAN, VAE-GAN)
- `configs/` - Training configurations
- `scripts/` - Executable training and evaluation scripts

## Reproducing Results

Model checkpoints will be made available upon publication. See [docs/TRAINING.md](docs/TRAINING.md) for training instructions and [docs/EVALUATION.md](docs/EVALUATION.md) for evaluation protocols.

## Questions

For questions about the paper or implementation, please open an issue on GitHub.
