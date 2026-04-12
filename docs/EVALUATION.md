# Evaluation Guide

## Overview

ALDM is evaluated using two complementary approaches:
1. **Image-level fidelity**: FID and SSIM metrics
2. **Downstream classification**: CNN trained on synthetic data, tested on real data

---

## Quick Evaluation

```bash
python scripts/evaluate.py \
    --synthetic_dir ./outputs/synthetic \
    --real_dir ./data/processed/pdgm/test \
    --output_dir ./results \
    --metrics all
```

---

## Image-Level Fidelity Metrics

### Fréchet Inception Distance (FID)

Measures distributional similarity between real and synthetic images.

```bash
python scripts/compute_fid.py \
    --real_dir ./data/processed/pdgm/test \
    --synthetic_dir ./outputs/synthetic \
    --batch_size 8 \
    --output ./results/fid_scores.json
```

**Patient-wise FID Computation**:
- Each patient has 112 axial slices
- FID computed per patient (64 patients total)
- Final FID: mean across all patients

**Expected Results**:
- ALDM (K=16, s=3.0): **85.40**
- VAE-GAN: 88.18
- 3M-CGAN: 116.48
- CGAN: 145.22

### Structural Similarity Index (SSIM)

Measures local and global structural correspondence.

```bash
python scripts/compute_ssim.py \
    --real_dir ./data/processed/pdgm/test \
    --synthetic_dir ./outputs/synthetic \
    --output ./results/ssim_scores.json
```

**Computation**:
- Computed per slice, averaged per patient
- Reported as mean ± std across patients

**Expected Results**:
- ALDM (K=16, s=3.0): **0.712**
- VAE-GAN: 0.750
- 3M-CGAN: 0.680
- CGAN: 0.374

---

## Downstream Classification Evaluation

### AlexLite-DG CNN Classifier

A lightweight CNN for slice-level tumor classification.

**Architecture**:
- Input: 112×112×3 (T1, T2, FLAIR)
- 5 convolutional blocks
- Global average pooling
- Binary classification (tumor vs. non-tumor)

### Training CNN on Synthetic Data

```bash
python scripts/train_downstream_cnn.py \
    --synthetic_dir ./outputs/synthetic \
    --output_dir ./checkpoints/cnn \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3
```

### Testing CNN on Real Data

```bash
python scripts/test_downstream_cnn.py \
    --checkpoint ./checkpoints/cnn/best.pth \
    --test_dir ./data/processed/pdgm/test \
    --output ./results/downstream_metrics.json
```

**Metrics Reported**:
- Balanced Accuracy (BAcc)
- F1 Score
- AUC (Area Under ROC Curve)
- Precision
- Recall

**Expected Results (PDGM Domain)**:

| Model | BAcc ↑ | F1 ↑ | AUC ↑ |
|-------|--------|------|-------|
| ALDM (K=16, s=3.0) | **0.875** | **0.836** | **0.987** |
| VAE-GAN | 0.751 | 0.675 | 0.882 |
| 3M-CGAN | 0.780 | 0.731 | 0.866 |
| CGAN | 0.764 | 0.720 | 0.876 |

---

## Cross-Validation Protocol

### 3×5-Fold Cross-Validation

```bash
python scripts/evaluate_cv.py \
    --data_dir ./data/processed/pdgm \
    --model_dir ./checkpoints/diffusion_16shot \
    --num_folds 5 \
    --num_repeats 3 \
    --output_dir ./results/cv
```

**Protocol**:
1. Split 16 training subjects into 5 folds
2. Train diffusion model on 4 folds, validate on 1
3. Generate synthetic samples from each fold
4. Train CNN on synthetic, test on real
5. Repeat 3 times with different random seeds
6. Report mean ± std across all runs

---

## Ablation Studies

### Guidance Scale Sensitivity

```bash
python scripts/evaluate_ablation.py \
    --ablation guidance_scale \
    --scales 0.3 0.5 1.0 3.0 \
    --checkpoint_dir ./checkpoints/ablation \
    --output_dir ./results/ablation_guidance
```

**Results**:

| Guidance Scale | FID ↓ | SSIM ↑ | AUC ↑ |
|----------------|-------|--------|-------|
| s=0.3 | 88.02 | 0.716 | 0.871 |
| s=0.5 | 88.08 | 0.714 | 0.877 |
| s=1.0 | 87.52 | 0.715 | 0.897 |
| **s=3.0** | **85.40** | 0.712 | **0.987** |

### Few-Shot Size Comparison

```bash
python scripts/evaluate_ablation.py \
    --ablation few_shot_size \
    --num_shots 10 16 \
    --checkpoint_dir ./checkpoints \
    --output_dir ./results/ablation_fewshot
```

**Results**:

| Few-Shot Size | FID ↓ | SSIM ↑ | AUC ↑ |
|---------------|-------|--------|-------|
| K=10 | 95.08 | 0.699 | 0.948 |
| **K=16** | **85.40** | 0.712 | **0.987** |

---

## Qualitative Evaluation

### Generate Comparison Visualizations

```bash
python scripts/visualize_comparison.py \
    --real_dir ./data/processed/pdgm/test \
    --synthetic_dirs \
        ./outputs/cgan \
        ./outputs/3m_cgan \
        ./outputs/vaegan \
        ./outputs/aldm \
    --output_dir ./results/qualitative \
    --num_samples 10
```

**Outputs**:
- Side-by-side comparison images
- 3D volume renderings
- Tumor region close-ups
- Cross-modal consistency plots

### Training Progression Visualization

```bash
python scripts/visualize_training_progression.py \
    --checkpoint_dir ./checkpoints/diffusion_16shot \
    --epochs 50 100 200 \
    --output_dir ./results/progression
```

**Outputs**:
- Synthetic samples at different epochs
- Quality improvement over training
- Anatomical fidelity evolution

---

## Evaluation Metrics Details

### FID Computation

**Feature Extractor**: InceptionV3 pretrained on ImageNet
**Layer**: Pool3 (2048-dimensional features)
**Distance**: Fréchet distance between Gaussian distributions

```python
FID = ||μ_real - μ_syn||² + Tr(Σ_real + Σ_syn - 2√(Σ_real Σ_syn))
```

### SSIM Computation

**Window Size**: 11×11
**Gaussian Weights**: σ=1.5
**Constants**: C1=0.01², C2=0.03²

```python
SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
```

### Balanced Accuracy

```python
BAcc = (Sensitivity + Specificity) / 2
```

Accounts for class imbalance in tumor classification.

---

## Baseline Comparisons

### Evaluate All Baselines

```bash
bash scripts/evaluate_all_baselines.sh
```

**Baselines Evaluated**:
1. CGAN (Mirza & Osindero, 2014)
2. 3M-CGAN (Xin et al., 2020)
3. VAE-GAN (Larsen et al., 2016)

**Output**: Comparative table with all metrics

---

## Statistical Significance Testing

### Paired t-test

```bash
python scripts/statistical_tests.py \
    --results_dir ./results \
    --baseline vaegan \
    --proposed aldm \
    --test paired_ttest \
    --alpha 0.05
```

**Tests**:
- FID: Lower is better (one-tailed)
- SSIM: Higher is better (one-tailed)
- AUC: Higher is better (one-tailed)

---

## Evaluation on GBM Domain

### Source Domain Performance

```bash
python scripts/evaluate.py \
    --synthetic_dir ./outputs/synthetic_gbm \
    --real_dir ./data/processed/gbm/test \
    --output_dir ./results/gbm \
    --metrics all
```

**Purpose**: Verify model hasn't forgotten source domain knowledge

---

## Failure Mode Analysis

### Identify Poor Quality Samples

```bash
python scripts/analyze_failures.py \
    --synthetic_dir ./outputs/synthetic \
    --real_dir ./data/processed/pdgm/test \
    --threshold_fid 150 \
    --threshold_ssim 0.5 \
    --output_dir ./results/failures
```

**Outputs**:
- List of low-quality samples
- Common failure patterns
- Recommendations for improvement

---

## Reproducibility

### Fixed Random Seeds

All evaluation scripts use fixed seeds:
- Python: 42
- NumPy: 42
- PyTorch: 42

### Deterministic Evaluation

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/evaluate.py --deterministic
```

---

## Output Format

### JSON Results

```json
{
  "model": "ALDM_K16_s3.0",
  "fidelity": {
    "fid": 85.40,
    "ssim": 0.712
  },
  "downstream": {
    "balanced_accuracy": 0.875,
    "f1_score": 0.836,
    "auc": 0.987,
    "precision": 0.842,
    "recall": 0.830
  },
  "config": {
    "num_samples": 16,
    "guidance_scale": 3.0,
    "few_shot_size": 16
  }
}
```

---

## Troubleshooting

### FID Computation Fails
- Ensure InceptionV3 model is downloaded
- Check image dimensions (should be ≥75×75)
- Verify sufficient samples (≥50 recommended)

### SSIM Values Too Low
- Check intensity normalization
- Verify spatial alignment
- Ensure same resolution for real and synthetic

### CNN Overfitting
- Reduce model complexity
- Add dropout (p=0.5)
- Use data augmentation

---

## References

- FID: Heusel et al. (2017)
- SSIM: Wang et al. (2004)
- AlexLite-DG: Custom architecture for medical imaging
