#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This shell script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

#
# 10-shot cross-domain training - IMPROVED (anti-collapse FIXED)
# 
# CRITICAL CHANGES FROM LAST RUN:
#   epochs_ldm_tgt:  120 -> 40   (prevent overfitting on 10 samples)
#   guidance_scale:  1.2 -> 3.0  (enforce conditioning)
#   cond_drop_p:     0.3 -> 0.05 (learn conditional generation)
#   x0_clip:         1.8 -> 3.0  (better dynamic range)
#   lr_unet:         2e-4 -> 5e-5 (preserve pretrained features during fine-tune)
#   noise_mult:      0.75 -> 1.0 (full noise for diversity)
#   noise_end_frac:  0.35 -> 0.0 (no early annealing)
#   ema_decay:       0.999 -> 0.9995 (more stable for few-shot)
#
set -e

# Set a configuration variable used later in the script.
OUTDIR=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_IMPROVED

# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo "10-SHOT IMPROVED RUN (MODE COLLAPSE FIXES APPLIED)"
# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "KEY CHANGES:"
# Print a status message for runtime visibility.
echo "  - epochs_ldm_tgt:  120 -> 40"
# Print a status message for runtime visibility.
echo "  - guidance_scale:  1.2 -> 3.0"
# Print a status message for runtime visibility.
echo "  - cond_drop_p:     0.3 -> 0.05"
# Print a status message for runtime visibility.
echo "  - x0_clip:         1.8 -> 3.0"
# Print a status message for runtime visibility.
echo "  - lr_unet:         2e-4 -> 5e-5"
# Print a status message for runtime visibility.
echo "  - noise_mult:      0.75 -> 1.0"
# Print a status message for runtime visibility.
echo "  - ema_decay:       0.999 -> 0.9995"
# Print a status message for runtime visibility.
echo ""

# Clean slate
# Print a status message for runtime visibility.
echo "Cleaning previous run..."
# Remove previous artifacts to ensure a clean run.
rm -rf ${OUTDIR}
# Create required output directories if they do not exist.
mkdir -p ${OUTDIR}

# Write an embedded file block to disk.
cat > ${OUTDIR}/run_pipeline.sh << 'PIPELINE_EOF'
#!/bin/bash
set -e

OUTDIR=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_IMPROVED

echo "============================================================"
echo "STEP 1: Validating 10-shot PDGM subjects..."
echo "============================================================"

cd /home/j98my/models/ldm3d
python3 validate_10shot_subjects.py \
  --pdgm_root /home/j98my/Pre-Processing/prep/pdgm_target_aligned \
  --fewshot /home/j98my/Pre-Processing/prep/pdgm_fewshot.txt

echo "✅ Validation passed!"
echo ""

echo "============================================================"
echo "STEP 2: Training VAE + Diffusion (IMPROVED HYPERPARAMETERS)"
echo "============================================================"
echo ""
echo ">>> MODE COLLAPSE FIXES APPLIED <<<"
echo ""

cd /home/j98my/models

python3 -u -m ldm3d.main_10shot \
  --gbm_root /home/j98my/Pre-Processing/prep/gbm_all_aligned \
  --pdgm_root /home/j98my/Pre-Processing/prep/pdgm_target_aligned \
  --fewshot /home/j98my/Pre-Processing/prep/pdgm_fewshot.txt \
  --outdir ${OUTDIR} \
  \
  `# === MODEL ARCHITECTURE (unchanged) ===` \
  --vae_base 64 \
  --unet_base 96 \
  --z_channels 8 \
  --latent_size 28 \
  \
  `# === TRAINING EPOCHS ===` \
  --epochs_vae_src 250 \
  --epochs_vae_tgt 80 \
  --epochs_ldm_src 300 \
  --epochs_ldm_tgt 40 \
  `# ^^^ CRITICAL: Was 120, now 40 - prevents overfitting on 10 samples` \
  \
  `# === LEARNING RATES ===` \
  --lr_vae 1e-4 \
  --lr_unet 5e-5 \
  `# ^^^ CRITICAL: Was 2e-4, now 5e-5 - preserves pretrained features` \
  \
  `# === DATA LOADING ===` \
  --batch_size 1 \
  --num_workers 2 \
  --seed 42 \
  \
  `# === VAE LOSSES ===` \
  --kl_w 1e-4 \
  --rec_w 1.0 \
  --kl_warmup_frac 0.25 \
  \
  `# === DIFFUSION SCHEDULE ===` \
  --timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 2e-2 \
  --ema_decay 0.9995 \
  `# ^^^ CRITICAL: Was 0.999, now 0.9995 - more stable for few-shot` \
  \
  `# === SAMPLING PARAMETERS (CRITICAL FIXES) ===` \
  --guidance_scale 3.0 \
  `# ^^^ CRITICAL: Was 1.2, now 3.0 - enforces conditioning` \
  --guidance_rescale 0.7 \
  `# ^^^ Prevents oversaturation from high guidance` \
  --noise_mult 1.0 \
  `# ^^^ CRITICAL: Was 0.75, now 1.0 - full noise for diversity` \
  --noise_end_frac 0.0 \
  `# ^^^ CRITICAL: Was 0.35, now 0.0 - no early noise annealing` \
  --x0_clip 3.0 \
  `# ^^^ CRITICAL: Was 1.8, now 3.0 - better dynamic range` \
  \
  `# === CONTROL TENSOR WEIGHTS (unchanged) ===` \
  --ctrl_mask_w 1.0 \
  --ctrl_edge_w 0.10 \
  --ctrl_dist_w 0.90 \
  \
  `# === CONDITIONING (CRITICAL FIX) ===` \
  --cond_drop_p 0.05 \
  `# ^^^ CRITICAL: Was 0.3, now 0.05 - model learns conditional generation` \
  --mask_aug_p 0.80 \
  `# ^^^ Slightly reduced from 0.95` \
  \
  `# === TUMOR-WEIGHTED LOSS ===` \
  --tumor_loss_alpha 0.30 \
  --tumor_dilate_k 3 \
  \
  `# === LATENT STATS ===` \
  --latent_stat_batches 400 \
  --force_recompute_latent_stats \
  \
  `# === SAMPLING CONFIG ===` \
  --sample_seed 42 \
  --use_ema_for_sampling \
  --final_dump_n 64 \
  --periodic_sample_every 50 \
  `# ^^^ More frequent sampling to catch issues early` \
  --periodic_sample_n 4

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ TRAINING COMPLETED!"
    echo "============================================================"
    echo ""
    
    echo "STEP 3: Running analysis..."
    echo "============================================================"
    echo ""
    
    cd /home/j98my/models/ldm3d
    python3 analyze_10shot_fixed.py --outdir ${OUTDIR} || true
    
    echo ""
    echo "============================================================"
    echo "✅ PIPELINE FINISHED"
    echo "============================================================"
    echo ""
    echo "Results available at:"
    echo "  - Samples: ${OUTDIR}/final_samples/"
    echo "  - Analysis: ${OUTDIR}/analysis/"
    echo "  - Logs: ${OUTDIR}/logs/"
    echo ""
    echo "REMINDER: Run evaluation with:"
    echo "  cd /home/j98my/Evaluation"
    echo "  ./evaluate_synthetic.sh ${OUTDIR}/final_samples"
    echo ""
else
    echo "❌ TRAINING FAILED with exit code $TRAIN_EXIT_CODE"
    exit 1
fi
PIPELINE_EOF

# Update file permissions so the script can be executed.
chmod +x ${OUTDIR}/run_pipeline.sh

# Save hyperparameters for reference
# Write an embedded file block to disk.
cat > ${OUTDIR}/hyperparams_changes.txt << 'HYPER_EOF'
============================================================
CRITICAL HYPERPARAMETER CHANGES (vs LAST_RUN)
============================================================

Parameter          | LAST_RUN | IMPROVED | Reason
-------------------|----------|----------|---------------------------
epochs_ldm_tgt     | 120      | 40       | Prevent overfitting (10 samples!)
guidance_scale     | 1.2      | 3.0      | Enforce mask conditioning
cond_drop_p        | 0.3      | 0.05     | Learn conditional generation
x0_clip            | 1.8      | 3.0      | Better dynamic range
lr_unet            | 2e-4     | 5e-5     | Preserve pretrained features
noise_mult         | 0.75     | 1.0      | Full noise for diversity
noise_end_frac     | 0.35     | 0.0      | No early annealing
ema_decay          | 0.999    | 0.9995   | Stable few-shot training
guidance_rescale   | (none)   | 0.7      | Prevent oversaturation
mask_aug_p         | 0.95     | 0.80     | Slightly reduced
periodic_sample    | 100      | 50       | Earlier issue detection

============================================================
EXPECTED IMPROVEMENTS
============================================================
Metric    | LAST_RUN | Target   | Notes
----------|----------|----------|---------------------------
AUC GBM   | 0.826    | 0.88-0.92| Better tumor definition
AUC PDGM  | 0.858    | 0.90-0.94| Better conditioning
FID GBM   | 71.65    | 35-50    | More diversity
FID PDGM  | 82.25    | 40-55    | Better distribution
SSIM      | 0.69     | 0.72-0.78| Sharper outputs

============================================================
ROOT CAUSE OF MODE COLLAPSE (fixed)
============================================================
1. 120 epochs on 10 samples = 1200 passes = massive overfitting
2. guidance_scale=1.2 too weak = model ignores conditioning
3. cond_drop_p=0.3 = 30% unconditional = learns to ignore masks
4. x0_clip=1.8 = limited dynamic range = washed out outputs

HYPER_EOF

# Run in background
# Launch the long-running job in the background.
nohup ${OUTDIR}/run_pipeline.sh > ${OUTDIR}/nohup.out 2>&1 &
# Set a configuration variable used later in the script.
PID=$!

# Print a status message for runtime visibility.
echo "Pipeline started with PID: $PID"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "Monitor progress:"
# Print a status message for runtime visibility.
echo "  tail -f ${OUTDIR}/nohup.out"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "Stop training:"
# Print a status message for runtime visibility.
echo "  kill ${PID}"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "Hyperparameters saved to:"
# Print a status message for runtime visibility.
echo "  ${OUTDIR}/hyperparams_changes.txt"
# Print a status message for runtime visibility.
echo ""
# Print a status message for runtime visibility.
echo "============================================================"