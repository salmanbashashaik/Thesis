#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This shell script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

#
# 10-shot cross-domain training - LAST RUN (anti-collapse tuned)
# Retrain VAE + Diffusion with strong anti-collapse knobs
#
set -e

# Set a configuration variable used later in the script.
OUTDIR=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_LAST_RUN

# Print a status message for runtime visibility.
echo "============================================================"
# Print a status message for runtime visibility.
echo "10-SHOT LAST RUN (VAE + Diffusion, anti-collapse tuned)"
# Print a status message for runtime visibility.
echo "============================================================"
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
OUTDIR=/home/j98my/models/runs/ldm_3d_diffuse_glioma/10_shot_LAST_RUN

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
echo "STEP 2: Training VAE + Diffusion from scratch (LAST RUN)"
echo "============================================================"

cd /home/j98my/models

python3 -u -m ldm3d.main_10shot \
  --gbm_root /home/j98my/Pre-Processing/prep/gbm_all_aligned \
  --pdgm_root /home/j98my/Pre-Processing/prep/pdgm_target_aligned \
  --fewshot /home/j98my/Pre-Processing/prep/pdgm_fewshot.txt \
  --outdir ${OUTDIR} \
  --vae_base 64 \
  --unet_base 96 \
  --z_channels 8 \
  --latent_size 28 \
  --epochs_vae_src 250 \
  --epochs_vae_tgt 80 \
  --epochs_ldm_src 300 \
  --epochs_ldm_tgt 120 \
  --lr_vae 1e-4 \
  --lr_unet 2e-4 \
  --batch_size 1 \
  --num_workers 2 \
  --seed 42 \
  --kl_w 1e-4 \
  --rec_w 1.0 \
  --kl_warmup_frac 0.25 \
  --timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 2e-2 \
  --ema_decay 0.999 \
  --guidance_scale 1.20 \
  --ctrl_mask_w 1.0 \
  --ctrl_edge_w 0.10 \
  --ctrl_dist_w 0.90 \
  --noise_mult 0.75 \
  --noise_end_frac 0.35 \
  --x0_clip 1.8 \
  --cond_drop_p 0.30 \
  --mask_aug_p 0.95 \
  --tumor_loss_alpha 0.30 \
  --tumor_dilate_k 3 \
  --latent_stat_batches 400 \
  --force_recompute_latent_stats \
  --sample_seed 42 \
  --use_ema_for_sampling \
  --final_dump_n 64 \
  --periodic_sample_every 100 \
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
    echo "✅ PIPELINE FINISHED (analysis may warn, training done)"
    echo "============================================================"
    echo ""
    echo "Results available at:"
    echo "  - Samples: ${OUTDIR}/final_samples/"
    echo "  - Analysis: ${OUTDIR}/analysis/"
    echo "  - Logs: ${OUTDIR}/logs/"
    echo ""
else
    echo "❌ TRAINING FAILED"
    exit 1
fi
PIPELINE_EOF

# Update file permissions so the script can be executed.
chmod +x ${OUTDIR}/run_pipeline.sh

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
echo "============================================================"
