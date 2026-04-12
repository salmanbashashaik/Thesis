# Data Directory

This directory contains raw and processed MRI datasets.

## Directory Structure

```
data/
├── raw/
│   ├── gbm/                      # UPENN-GBM raw data
│   │   ├── subject_001/
│   │   │   ├── t1.nii.gz
│   │   │   ├── t2.nii.gz
│   │   │   ├── flair.nii.gz
│   │   │   └── seg.nii.gz
│   │   ├── subject_002/
│   │   └── ...
│   │
│   └── pdgm/                     # UCSF-PDGM raw data
│       ├── subject_001/
│       └── ...
│
├── processed/
│   ├── gbm/
│   │   ├── train/
│   │   │   ├── subject_001_volume.npy    # (3, 112, 112, 112)
│   │   │   ├── subject_001_mask.npy      # (1, 112, 112, 112)
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   │
│   └── pdgm/
│       ├── train/                # 16 subjects for few-shot
│       ├── val/
│       └── test/                 # 64 subjects for evaluation
│
├── manifests/
│   ├── gbm_train.txt
│   ├── gbm_val.txt
│   ├── gbm_test.txt
│   ├── pdgm_train_16shot.txt
│   ├── pdgm_train_10shot.txt
│   ├── pdgm_val.txt
│   └── pdgm_test.txt
│
└── masks/
    ├── mask_001.npy
    ├── mask_002.npy
    └── ...
```

## Dataset Downloads

### UPENN-GBM (Source Domain)
- **URL**: https://doi.org/10.7937/TCIA.709X-DN49
- **Size**: ~828,000 slices
- **Download**: Use NBIA Data Retriever from TCIA

### UCSF-PDGM (Target Domain)
- **URL**: https://doi.org/10.7937/3ec9-yk87
- **Size**: ~12,000 images
- **Download**: Use NBIA Data Retriever from TCIA

## Preprocessing

After downloading raw data, run preprocessing:

```bash
# Preprocess GBM
python scripts/preprocess_data.py \
    --input_dir ./data/raw/gbm \
    --output_dir ./data/processed/gbm \
    --dataset gbm

# Preprocess PDGM
python scripts/preprocess_data.py \
    --input_dir ./data/raw/pdgm \
    --output_dir ./data/processed/pdgm \
    --dataset pdgm
```

## Data Format

### Processed Volumes
- **Format**: NumPy arrays (.npy)
- **Shape**: (3, 112, 112, 112) for volumes, (1, 112, 112, 112) for masks
- **Dtype**: float32
- **Range**: [-1, 1] (normalized)

### Manifest Files
Text files with one subject ID per line:
```
subject_001
subject_002
subject_003
...
```

## Storage Requirements

| Dataset | Raw | Processed | Total |
|---------|-----|-----------|-------|
| GBM | ~200 GB | ~50 GB | ~250 GB |
| PDGM | ~30 GB | ~5 GB | ~35 GB |
| **Total** | ~230 GB | ~55 GB | ~285 GB |

---

**Note**: Raw data is not included in this repository. Download from TCIA.
