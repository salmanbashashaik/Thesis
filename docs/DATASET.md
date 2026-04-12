# Dataset Documentation

## Overview

This project uses two publicly available glioma MRI datasets from The Cancer Imaging Archive (TCIA):

1. **UPENN-GBM**: Data-rich source domain (Glioblastoma)
2. **UCSF-PDGM**: Data-scarce target domain (Preoperative Diffuse Glioma)

---

## Dataset Details

### UPENN-GBM (Source Domain)

**Citation**: Bakas et al. "Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM)". The Cancer Imaging Archive, 2021.

**DOI**: [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)

**Characteristics**:
- **Size**: ~828,000 image slices
- **Subjects**: Multiple GBM patients
- **Modalities**: T1-weighted, T2-weighted, FLAIR
- **Format**: NIfTI (.nii.gz)
- **Resolution**: Variable (standardized to 112×112×112 in preprocessing)
- **Tumor Type**: WHO Grade IV Glioblastoma (IDH-wildtype)
- **Domain**: Homogeneous, high-grade tumors

**Download**:
```bash
# Visit TCIA website and download using NBIA Data Retriever
# https://wiki.cancerimagingarchive.net/display/Public/UPENN-GBM
```

---

### UCSF-PDGM (Target Domain)

**Citation**: Calabrese et al. "The University of California San Francisco preoperative diffuse glioma MRI dataset". Radiology: Artificial Intelligence, 2022.

**DOI**: [10.7937/3ec9-yk87](https://doi.org/10.7937/3ec9-yk87)

**Characteristics**:
- **Size**: ~12,000 images
- **Subjects**: Multiple diffuse glioma patients
- **Modalities**: T1-weighted, T2-weighted, FLAIR
- **Format**: NIfTI (.nii.gz)
- **Resolution**: Variable (standardized to 112×112×112 in preprocessing)
- **Tumor Type**: WHO Grade II-III diffuse gliomas (heterogeneous subtypes)
- **Domain**: Heterogeneous, mixed grades and molecular subtypes

**Download**:
```bash
# Visit TCIA website and download using NBIA Data Retriever
# https://wiki.cancerimagingarchive.net/display/Public/UCSF-PDGM
```

---

## Preprocessing Pipeline

### 1. Subject Manifest Creation

Create a manifest file listing all subjects and their available modalities:

```bash
python scripts/preprocess_data.py \
    --input_dir /path/to/raw/gbm \
    --output_dir ./data/processed/gbm \
    --dataset gbm \
    --step manifest
```

### 2. Spatial Standardization

All volumes are resampled to a uniform spatial resolution and orientation:
- **Target size**: 112×112×112
- **Voxel spacing**: Isotropic
- **Orientation**: RAS (Right-Anterior-Superior)

### 3. Intensity Normalization

Per-modality intensity normalization:
- **Method**: Z-score normalization within brain mask
- **Range**: Clipped to [-1, 1] for stable training

### 4. Foreground Cropping

Remove background regions to focus on brain tissue:
- **Method**: Otsu thresholding + morphological operations
- **Padding**: 5 voxels on each side

### 5. Mask Preparation

Tumor segmentation masks are processed:
- **Binary masks**: Whole tumor region (all tumor labels merged)
- **Format**: Same spatial dimensions as MRI volumes (112×112×112)
- **Conditioning**: Edge maps and distance transforms derived from masks

### 6. Data Splits

**GBM (Source)**:
- Training: 80% of subjects
- Validation: 10% of subjects
- Testing: 10% of subjects

**PDGM (Target - Few-Shot)**:
- Training: 16 subjects (few-shot adaptation)
- Validation: 10 subjects
- Testing: 64 subjects (for evaluation)

---

## Directory Structure

After preprocessing, the data directory should look like:

```
data/
├── raw/
│   ├── gbm/
│   │   ├── subject_001/
│   │   │   ├── t1.nii.gz
│   │   │   ├── t2.nii.gz
│   │   │   ├── flair.nii.gz
│   │   │   └── seg.nii.gz
│   │   └── ...
│   └── pdgm/
│       ├── subject_001/
│       └── ...
│
├── processed/
│   ├── gbm/
│   │   ├── train/
│   │   │   ├── subject_001_volume.npy
│   │   │   ├── subject_001_mask.npy
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── pdgm/
│       ├── train/  # 16 subjects for few-shot
│       ├── val/
│       └── test/   # 64 subjects for evaluation
│
└── manifests/
    ├── gbm_train.txt
    ├── gbm_val.txt
    ├── gbm_test.txt
    ├── pdgm_train_16shot.txt
    ├── pdgm_val.txt
    └── pdgm_test.txt
```

---

## Preprocessing Script Usage

### Full Pipeline

```bash
# Preprocess GBM dataset
python scripts/preprocess_data.py \
    --input_dir /path/to/raw/gbm \
    --output_dir ./data/processed/gbm \
    --dataset gbm \
    --target_size 112 112 112 \
    --num_workers 8

# Preprocess PDGM dataset
python scripts/preprocess_data.py \
    --input_dir /path/to/raw/pdgm \
    --output_dir ./data/processed/pdgm \
    --dataset pdgm \
    --target_size 112 112 112 \
    --num_workers 8
```

### Create Few-Shot Split

```bash
# Create 16-shot training split for PDGM
python scripts/create_fewshot_split.py \
    --data_dir ./data/processed/pdgm \
    --num_shots 16 \
    --seed 42 \
    --output_dir ./data/manifests
```

---

## Data Augmentation

During training, the following augmentations are applied:
- **Random flips**: Horizontal (50% probability)
- **Random rotations**: ±10 degrees
- **Elastic deformations**: Mild (for anatomical realism)
- **Intensity shifts**: ±10% (per modality)

---

## Notes

- All preprocessing is deterministic (fixed random seed: 42)
- Preprocessing takes ~2-4 hours per dataset on 8-core CPU
- Processed data requires ~50GB for GBM, ~5GB for PDGM
- Original NIfTI files are preserved in `data/raw/`

---

## Troubleshooting

**Issue**: Missing modalities for some subjects
- **Solution**: Subjects with incomplete modalities are excluded during manifest creation

**Issue**: Memory errors during preprocessing
- **Solution**: Reduce `--num_workers` or process subjects sequentially

**Issue**: Inconsistent voxel spacing
- **Solution**: All volumes are automatically resampled to isotropic spacing

---

## References

1. Bakas et al. (2021). UPENN-GBM Dataset. TCIA.
2. Calabrese et al. (2022). UCSF-PDGM Dataset. Radiology: AI.
