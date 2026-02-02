#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This shell script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

#
# evaluate_synthetic_images.sh (FLEXIBLE VERSION)
#
# Comprehensive evaluation pipeline for synthetic medical images:
# 1. CNN transferability (with domain adaptation)
# 2. FID/SSIM perceptual metrics
# 3. Aggregate results into summary report
#
# FLEXIBLE: Can evaluate:
#   - Just GBM (provide --synth_gbm_root and --original_gbm_root)
#   - Just PDGM (provide --synth_pdgm_root and --original_pdgm_root)
#   - Both domains (provide all 4 paths)
#

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

# Paths (FLEXIBLE - not all required)
# Set a configuration variable used later in the script.
SYNTH_GBM_ROOT=""
# Set a configuration variable used later in the script.
SYNTH_PDGM_ROOT=""
# Set a configuration variable used later in the script.
ORIGINAL_GBM_ROOT=""
# Set a configuration variable used later in the script.
ORIGINAL_PDGM_ROOT=""

# Set a configuration variable used later in the script.
MASK_ROOT=""  # Auto-detect based on domain if not provided
# Set a configuration variable used later in the script.
CNN_MODEL_PATH=""
# Set a configuration variable used later in the script.
OUTPUT_DIR="/home/j98my/Evaluation/results"

# Optional: For feature space analysis
# Set a configuration variable used later in the script.
REAL_ROOT=""

# Domain detection
# Set a configuration variable used later in the script.
DOMAIN="auto"  # auto, gbm, or pdgm

# Default parameters
# Set a configuration variable used later in the script.
IMG_SIZE=224
# Set a configuration variable used later in the script.
BATCH_SIZE=32
# Set a configuration variable used later in the script.
NUM_WORKERS=4
# Set a configuration variable used later in the script.
DEVICE="cuda"
# Set a configuration variable used later in the script.
N_ADAPT=10
# Set a configuration variable used later in the script.
N_BOOTSTRAP=1000

# Script paths (adjust if needed)
# Set a configuration variable used later in the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set a configuration variable used later in the script.
CNN_EVAL_SCRIPT="${SCRIPT_DIR}/test_using_cnn_cv_DS.py"
# Set a configuration variable used later in the script.
FID_SSIM_SCRIPT="${SCRIPT_DIR}/test_using_fid_ssim.py"

# ============================================================
# Parse arguments
# ============================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Comprehensive evaluation of synthetic medical images.
Can evaluate GBM only, PDGM only, or both domains.

Required Arguments (at least one domain):
  GBM Domain:
    --synth_gbm_root PATH       Path to synthetic GBM images directory
    --original_gbm_root PATH    Path to original/real GBM images directory
  
  PDGM Domain:
    --synth_pdgm_root PATH      Path to synthetic PDGM images directory
    --original_pdgm_root PATH   Path to original/real PDGM images directory
  
  Common:
    --cnn_model PATH            Path to trained CNN model (.pt file)
    --output_dir PATH           Output directory for results

Optional Arguments:
  --mask_root PATH         Path to ground truth masks (auto-detect if not provided)
                           Default GBM: /home/j98my/Pre-Processing/prep/gbm_all_aligned
                           Default PDGM: /home/j98my/Pre-Processing/prep/pdgm_target_aligned
  --domain DOMAIN          Domain: auto (default), gbm, or pdgm
  --real_root PATH         Path to real images for feature analysis
  --img_size N             Image size (default: 224)
  --batch_size N           Batch size (default: 32)
  --num_workers N          DataLoader workers (default: 4)
  --device DEVICE          Device: cuda or cpu (default: cuda)
  --n_adapt N              Samples for domain adaptation (default: 10)
  --n_bootstrap N          Bootstrap iterations (default: 1000)
  --use_resized            Use *_resized.nii.gz files
  --skip_cnn               Skip CNN evaluation
  --skip_fid_ssim          Skip FID/SSIM evaluation
  --help                   Show this help message

Examples:
  # Evaluate GBM only
  $0 \\
    --synth_gbm_root /path/to/synth/gbm \\
    --original_gbm_root /path/to/real/gbm \\
    --cnn_model /path/to/model.pt \\
    --output_dir /path/to/results
  
  # Evaluate PDGM only
  $0 \\
    --synth_pdgm_root /path/to/synth/pdgm \\
    --original_pdgm_root /path/to/real/pdgm \\
    --cnn_model /path/to/model.pt \\
    --output_dir /path/to/results
  
  # Evaluate both domains
  $0 \\
    --synth_gbm_root /path/to/synth/gbm \\
    --synth_pdgm_root /path/to/synth/pdgm \\
    --original_gbm_root /path/to/real/gbm \\
    --original_pdgm_root /path/to/real/pdgm \\
    --cnn_model /path/to/model.pt \\
    --output_dir /path/to/results
EOF
    # Exit the script with the intended status code.
    exit 1
}

# Parse command-line arguments
# Set a configuration variable used later in the script.
USE_RESIZED=""
# Set a configuration variable used later in the script.
SKIP_CNN=false
# Set a configuration variable used later in the script.
SKIP_FID_SSIM=false

# Shell control-flow statement managing script execution.
while [[ $# -gt 0 ]]; do
    # Shell control-flow statement managing script execution.
    case $1 in
        --synth_gbm_root)
            # Set a configuration variable used later in the script.
            SYNTH_GBM_ROOT="$2"
            shift 2
            ;;
        --synth_pdgm_root)
            # Set a configuration variable used later in the script.
            SYNTH_PDGM_ROOT="$2"
            shift 2
            ;;
        --original_gbm_root)
            # Set a configuration variable used later in the script.
            ORIGINAL_GBM_ROOT="$2"
            shift 2
            ;;
        --original_pdgm_root)
            # Set a configuration variable used later in the script.
            ORIGINAL_PDGM_ROOT="$2"
            shift 2
            ;;
        --mask_root)
            # Set a configuration variable used later in the script.
            MASK_ROOT="$2"
            shift 2
            ;;
        --domain)
            # Set a configuration variable used later in the script.
            DOMAIN="$2"
            shift 2
            ;;
        --cnn_model)
            # Set a configuration variable used later in the script.
            CNN_MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            # Set a configuration variable used later in the script.
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --real_root)
            # Set a configuration variable used later in the script.
            REAL_ROOT="$2"
            shift 2
            ;;
        --img_size)
            # Set a configuration variable used later in the script.
            IMG_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            # Set a configuration variable used later in the script.
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            # Set a configuration variable used later in the script.
            NUM_WORKERS="$2"
            shift 2
            ;;
        --device)
            # Set a configuration variable used later in the script.
            DEVICE="$2"
            shift 2
            ;;
        --n_adapt)
            # Set a configuration variable used later in the script.
            N_ADAPT="$2"
            shift 2
            ;;
        --n_bootstrap)
            # Set a configuration variable used later in the script.
            N_BOOTSTRAP="$2"
            shift 2
            ;;
        --use_resized)
            # Set a configuration variable used later in the script.
            USE_RESIZED="--use_resized"
            shift
            ;;
        --skip_cnn)
            # Set a configuration variable used later in the script.
            SKIP_CNN=true
            shift
            ;;
        --skip_fid_ssim)
            # Set a configuration variable used later in the script.
            SKIP_FID_SSIM=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            # Print a status message for runtime visibility.
            echo "Unknown option: $1"
            usage
            ;;
    # Shell control-flow statement managing script execution.
    esac
# Shell control-flow statement managing script execution.
done

# ============================================================
# Validate required arguments
# ============================================================

# Determine which domains are being evaluated
# Set a configuration variable used later in the script.
EVAL_GBM=false
# Set a configuration variable used later in the script.
EVAL_PDGM=false

# Shell control-flow statement managing script execution.
if [ -n "$SYNTH_GBM_ROOT" ] && [ -n "$ORIGINAL_GBM_ROOT" ]; then
    # Set a configuration variable used later in the script.
    EVAL_GBM=true
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ -n "$SYNTH_PDGM_ROOT" ] && [ -n "$ORIGINAL_PDGM_ROOT" ]; then
    # Set a configuration variable used later in the script.
    EVAL_PDGM=true
# Shell control-flow statement managing script execution.
fi

# Check that at least one domain is being evaluated
# Shell control-flow statement managing script execution.
if [ "$EVAL_GBM" = false ] && [ "$EVAL_PDGM" = false ]; then
    # Print a status message for runtime visibility.
    echo "Error: Must provide paths for at least one domain:"
    # Print a status message for runtime visibility.
    echo "  GBM:  --synth_gbm_root AND --original_gbm_root"
    # Print a status message for runtime visibility.
    echo "  PDGM: --synth_pdgm_root AND --original_pdgm_root"
    usage
# Shell control-flow statement managing script execution.
fi

# Check for incomplete domain specifications
# Shell control-flow statement managing script execution.
if [ -n "$SYNTH_GBM_ROOT" ] && [ -z "$ORIGINAL_GBM_ROOT" ]; then
    # Print a status message for runtime visibility.
    echo "Error: --synth_gbm_root provided but --original_gbm_root is missing"
    usage
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ -z "$SYNTH_GBM_ROOT" ] && [ -n "$ORIGINAL_GBM_ROOT" ]; then
    # Print a status message for runtime visibility.
    echo "Error: --original_gbm_root provided but --synth_gbm_root is missing"
    usage
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ -n "$SYNTH_PDGM_ROOT" ] && [ -z "$ORIGINAL_PDGM_ROOT" ]; then
    # Print a status message for runtime visibility.
    echo "Error: --synth_pdgm_root provided but --original_pdgm_root is missing"
    usage
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ -z "$SYNTH_PDGM_ROOT" ] && [ -n "$ORIGINAL_PDGM_ROOT" ]; then
    # Print a status message for runtime visibility.
    echo "Error: --original_pdgm_root provided but --synth_pdgm_root is missing"
    usage
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ -z "$OUTPUT_DIR" ]; then
    # Print a status message for runtime visibility.
    echo "Error: --output_dir is required"
    usage
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ "$SKIP_CNN" = false ] && [ -z "$CNN_MODEL_PATH" ]; then
    # Print a status message for runtime visibility.
    echo "Error: --cnn_model is required (or use --skip_cnn)"
    usage
# Shell control-flow statement managing script execution.
fi

# Check paths exist
# Shell control-flow statement managing script execution.
if [ "$EVAL_GBM" = true ]; then
    # Shell control-flow statement managing script execution.
    if [ ! -d "$SYNTH_GBM_ROOT" ]; then
        # Print a status message for runtime visibility.
        echo "Error: Synthetic GBM root not found: $SYNTH_GBM_ROOT"
        # Exit the script with the intended status code.
        exit 1
    # Shell control-flow statement managing script execution.
    fi
    # Shell control-flow statement managing script execution.
    if [ ! -d "$ORIGINAL_GBM_ROOT" ]; then
        # Print a status message for runtime visibility.
        echo "Error: Original GBM root not found: $ORIGINAL_GBM_ROOT"
        # Exit the script with the intended status code.
        exit 1
    # Shell control-flow statement managing script execution.
    fi
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ "$EVAL_PDGM" = true ]; then
    # Shell control-flow statement managing script execution.
    if [ ! -d "$SYNTH_PDGM_ROOT" ]; then
        # Print a status message for runtime visibility.
        echo "Error: Synthetic PDGM root not found: $SYNTH_PDGM_ROOT"
        # Exit the script with the intended status code.
        exit 1
    # Shell control-flow statement managing script execution.
    fi
    # Shell control-flow statement managing script execution.
    if [ ! -d "$ORIGINAL_PDGM_ROOT" ]; then
        # Print a status message for runtime visibility.
        echo "Error: Original PDGM root not found: $ORIGINAL_PDGM_ROOT"
        # Exit the script with the intended status code.
        exit 1
    # Shell control-flow statement managing script execution.
    fi
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ "$SKIP_CNN" = false ] && [ ! -f "$CNN_MODEL_PATH" ]; then
    # Print a status message for runtime visibility.
    echo "Error: CNN model not found: $CNN_MODEL_PATH"
    # Exit the script with the intended status code.
    exit 1
# Shell control-flow statement managing script execution.
fi

# Create output directory
# Create required output directories if they do not exist.
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Log setup
# ============================================================

# Set a configuration variable used later in the script.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Set a configuration variable used later in the script.
LOG_FILE="${OUTPUT_DIR}/evaluation_${TIMESTAMP}.log"

log() {
    # Print a status message for runtime visibility.
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOG_FILE"
}

log "============================================================"
log "COMPREHENSIVE SYNTHETIC IMAGE EVALUATION"
log "============================================================"
log ""
log "Configuration:"
log "  Domains to evaluate:"
[ "$EVAL_GBM" = true ] && log "    - GBM" || log "    - GBM (skipped)"
[ "$EVAL_PDGM" = true ] && log "    - PDGM" || log "    - PDGM (skipped)"
log ""
# Shell control-flow statement managing script execution.
if [ "$EVAL_GBM" = true ]; then
    log "  GBM Paths:"
    log "    Synthetic: $SYNTH_GBM_ROOT"
    log "    Original:  $ORIGINAL_GBM_ROOT"
# Shell control-flow statement managing script execution.
fi
# Shell control-flow statement managing script execution.
if [ "$EVAL_PDGM" = true ]; then
    log "  PDGM Paths:"
    log "    Synthetic: $SYNTH_PDGM_ROOT"
    log "    Original:  $ORIGINAL_PDGM_ROOT"
# Shell control-flow statement managing script execution.
fi
log ""
log "  Mask root:          $MASK_ROOT"
log "  CNN model:          $CNN_MODEL_PATH"
log "  Output directory:   $OUTPUT_DIR"
log "  Image size:         $IMG_SIZE"
log "  Batch size:         $BATCH_SIZE"
log "  Device:             $DEVICE"
log "  Use resized:        $([ -n "$USE_RESIZED" ] && echo "Yes" || echo "No")"
log ""

# ============================================================
# Step 1: CNN Evaluation
# ============================================================

# Set a configuration variable used later in the script.
CNN_AUC_GBM="N/A"
# Set a configuration variable used later in the script.
CNN_DOMAIN_GAP_GBM="N/A"
# Set a configuration variable used later in the script.
CNN_AUC_PDGM="N/A"
# Set a configuration variable used later in the script.
CNN_DOMAIN_GAP_PDGM="N/A"

# Shell control-flow statement managing script execution.
if [ "$SKIP_CNN" = false ]; then
    log "============================================================"
    log "STEP 1: CNN Transferability Evaluation"
    log "============================================================"
    log ""
    
    # Evaluate GBM if requested
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_GBM" = true ]; then
        log "Evaluating GBM domain..."
        # Set a configuration variable used later in the script.
        CNN_OUTPUT_GBM="${OUTPUT_DIR}/cnn_evaluation_gbm_${TIMESTAMP}.txt"
        
        # Set a configuration variable used later in the script.
        CNN_CMD="python3 ${CNN_EVAL_SCRIPT} \
            --synth_root ${SYNTH_GBM_ROOT} \
            --model_path ${CNN_MODEL_PATH} \
            --img_size ${IMG_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --device ${DEVICE} \
            --n_adapt ${N_ADAPT} \
            --n_bootstrap ${N_BOOTSTRAP} \
            --domain gbm \
            ${USE_RESIZED}"
        
        # Shell control-flow statement managing script execution.
        if [ -n "$MASK_ROOT" ]; then
            # Set a configuration variable used later in the script.
            CNN_CMD="${CNN_CMD} --mask_root ${MASK_ROOT}"
        # Shell control-flow statement managing script execution.
        fi
        
        # Shell control-flow statement managing script execution.
        if [ -n "$REAL_ROOT" ]; then
            # Set a configuration variable used later in the script.
            CNN_CMD="${CNN_CMD} --real_root ${REAL_ROOT}"
        # Shell control-flow statement managing script execution.
        fi
        
        log "Command: $CNN_CMD"
        log ""
        
        # Shell control-flow statement managing script execution.
        if eval "$CNN_CMD" 2>&1 | tee "$CNN_OUTPUT_GBM"; then
            log "✅ GBM CNN evaluation completed"
            # Set a configuration variable used later in the script.
            CNN_AUC_GBM=$(grep "Report AUC as:" "$CNN_OUTPUT_GBM" | tail -1 | awk '{print $4}')
            # Set a configuration variable used later in the script.
            CNN_DOMAIN_GAP_GBM=$(grep "Domain gap:" "$CNN_OUTPUT_GBM" | tail -1 | awk '{print $3}')
            log "  GBM AUC: $CNN_AUC_GBM, Domain gap: $CNN_DOMAIN_GAP_GBM"
        # Shell control-flow statement managing script execution.
        else
            log "❌ GBM CNN evaluation failed"
        # Shell control-flow statement managing script execution.
        fi
        log ""
    # Shell control-flow statement managing script execution.
    fi
    
    # Evaluate PDGM if requested
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_PDGM" = true ]; then
        log "Evaluating PDGM domain..."
        # Set a configuration variable used later in the script.
        CNN_OUTPUT_PDGM="${OUTPUT_DIR}/cnn_evaluation_pdgm_${TIMESTAMP}.txt"
        
        # Set a configuration variable used later in the script.
        CNN_CMD="python3 ${CNN_EVAL_SCRIPT} \
            --synth_root ${SYNTH_PDGM_ROOT} \
            --model_path ${CNN_MODEL_PATH} \
            --img_size ${IMG_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --device ${DEVICE} \
            --n_adapt ${N_ADAPT} \
            --n_bootstrap ${N_BOOTSTRAP} \
            --domain pdgm \
            ${USE_RESIZED}"
        
        # Shell control-flow statement managing script execution.
        if [ -n "$MASK_ROOT" ]; then
            # Set a configuration variable used later in the script.
            CNN_CMD="${CNN_CMD} --mask_root ${MASK_ROOT}"
        # Shell control-flow statement managing script execution.
        fi
        
        # Shell control-flow statement managing script execution.
        if [ -n "$REAL_ROOT" ]; then
            # Set a configuration variable used later in the script.
            CNN_CMD="${CNN_CMD} --real_root ${REAL_ROOT}"
        # Shell control-flow statement managing script execution.
        fi
        
        log "Command: $CNN_CMD"
        log ""
        
        # Shell control-flow statement managing script execution.
        if eval "$CNN_CMD" 2>&1 | tee "$CNN_OUTPUT_PDGM"; then
            log "✅ PDGM CNN evaluation completed"
            # Set a configuration variable used later in the script.
            CNN_AUC_PDGM=$(grep "Report AUC as:" "$CNN_OUTPUT_PDGM" | tail -1 | awk '{print $4}')
            # Set a configuration variable used later in the script.
            CNN_DOMAIN_GAP_PDGM=$(grep "Domain gap:" "$CNN_OUTPUT_PDGM" | tail -1 | awk '{print $3}')
            log "  PDGM AUC: $CNN_AUC_PDGM, Domain gap: $CNN_DOMAIN_GAP_PDGM"
        # Shell control-flow statement managing script execution.
        else
            log "❌ PDGM CNN evaluation failed"
        # Shell control-flow statement managing script execution.
        fi
        log ""
    # Shell control-flow statement managing script execution.
    fi
# Shell control-flow statement managing script execution.
else
    log "⏩ Skipping CNN evaluation (--skip_cnn)"
    log ""
# Shell control-flow statement managing script execution.
fi

# ============================================================
# Step 2: FID/SSIM Evaluation
# ============================================================

# Set a configuration variable used later in the script.
SSIM_GBM="N/A"
# Set a configuration variable used later in the script.
FID_GBM="N/A"
# Set a configuration variable used later in the script.
SSIM_PDGM="N/A"
# Set a configuration variable used later in the script.
FID_PDGM="N/A"

# Shell control-flow statement managing script execution.
if [ "$SKIP_FID_SSIM" = false ]; then
    log "============================================================"
    log "STEP 2: FID/SSIM Perceptual Metrics"
    log "============================================================"
    log ""
    
    # Set a configuration variable used later in the script.
    FID_SSIM_OUTPUT="${OUTPUT_DIR}/fid_ssim_${TIMESTAMP}.txt"
    
    log "Running FID/SSIM evaluation..."
    log "Output will be saved to: $FID_SSIM_OUTPUT"
    log ""
    
    # Build command based on which domains are being evaluated
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_GBM" = true ] && [ "$EVAL_PDGM" = true ]; then
        # Both domains
        # Set a configuration variable used later in the script.
        FID_SSIM_CMD="python3 ${FID_SSIM_SCRIPT} \
            --synth_gbm_root ${SYNTH_GBM_ROOT} \
            --original_gbm_root ${ORIGINAL_GBM_ROOT} \
            --synth_pdgm_root ${SYNTH_PDGM_ROOT} \
            --original_pdgm_root ${ORIGINAL_PDGM_ROOT} \
            --img_size ${IMG_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --device ${DEVICE} \
            ${USE_RESIZED}"
    # Shell control-flow statement managing script execution.
    elif [ "$EVAL_GBM" = true ]; then
        # GBM only
        # Set a configuration variable used later in the script.
        FID_SSIM_CMD="python3 ${FID_SSIM_SCRIPT} \
            --synth_gbm_root ${SYNTH_GBM_ROOT} \
            --original_gbm_root ${ORIGINAL_GBM_ROOT} \
            --img_size ${IMG_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --device ${DEVICE} \
            ${USE_RESIZED}"
    # Shell control-flow statement managing script execution.
    else
        # PDGM only
        # Set a configuration variable used later in the script.
        FID_SSIM_CMD="python3 ${FID_SSIM_SCRIPT} \
            --synth_pdgm_root ${SYNTH_PDGM_ROOT} \
            --original_pdgm_root ${ORIGINAL_PDGM_ROOT} \
            --img_size ${IMG_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --device ${DEVICE} \
            ${USE_RESIZED}"
    # Shell control-flow statement managing script execution.
    fi
    
    log "Command: $FID_SSIM_CMD"
    log ""
    
    # Run FID/SSIM evaluation
    # Shell control-flow statement managing script execution.
    if eval "$FID_SSIM_CMD" 2>&1 | tee "$FID_SSIM_OUTPUT"; then
        log "✅ FID/SSIM evaluation completed successfully"
        log "Results saved to: $FID_SSIM_OUTPUT"
        
        # Extract metrics from final summary section
        # Shell control-flow statement managing script execution.
        if [ "$EVAL_GBM" = true ]; then
            # Set a configuration variable used later in the script.
            SSIM_GBM=$(grep -A 10 "Dataset: GBM" "$FID_SSIM_OUTPUT" | grep "SSIM" | grep -oP '\d+\.\d+' | head -1)
            # Set a configuration variable used later in the script.
            FID_GBM=$(grep -A 10 "Dataset: GBM" "$FID_SSIM_OUTPUT" | grep "FID" | grep -oP '\d+\.\d+' | head -1)
            log "  GBM  - SSIM: $SSIM_GBM, FID: $FID_GBM"
        # Shell control-flow statement managing script execution.
        fi
        
        # Shell control-flow statement managing script execution.
        if [ "$EVAL_PDGM" = true ]; then
            # Set a configuration variable used later in the script.
            SSIM_PDGM=$(grep -A 10 "Dataset: PDGM" "$FID_SSIM_OUTPUT" | grep "SSIM" | grep -oP '\d+\.\d+' | head -1)
            # Set a configuration variable used later in the script.
            FID_PDGM=$(grep -A 10 "Dataset: PDGM" "$FID_SSIM_OUTPUT" | grep "FID" | grep -oP '\d+\.\d+' | head -1)
            log "  PDGM - SSIM: $SSIM_PDGM, FID: $FID_PDGM"
        # Shell control-flow statement managing script execution.
        fi
    # Shell control-flow statement managing script execution.
    else
        log "❌ FID/SSIM evaluation failed"
    # Shell control-flow statement managing script execution.
    fi
    
    log ""
# Shell control-flow statement managing script execution.
else
    log "⏩ Skipping FID/SSIM evaluation (--skip_fid_ssim)"
    log ""
# Shell control-flow statement managing script execution.
fi

# ============================================================
# Step 3: Generate Summary Report
# ============================================================

log "============================================================"
log "STEP 3: Generating Summary Report"
log "============================================================"
log ""

# Set a configuration variable used later in the script.
SUMMARY_FILE="${OUTPUT_DIR}/EVALUATION_SUMMARY_${TIMESTAMP}.txt"

# Write an embedded file block to disk.
cat > "$SUMMARY_FILE" << EOF
========================================================================
SYNTHETIC IMAGE EVALUATION SUMMARY
========================================================================

Evaluation Date: $(date)

------------------------------------------------------------------------
CONFIGURATION
------------------------------------------------------------------------
Domains Evaluated:
EOF

# Shell control-flow statement managing script execution.
if [ "$EVAL_GBM" = true ]; then
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
  GBM:
    Synthetic Root: $SYNTH_GBM_ROOT
    Original Root:  $ORIGINAL_GBM_ROOT
EOF
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ "$EVAL_PDGM" = true ]; then
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
  PDGM:
    Synthetic Root: $SYNTH_PDGM_ROOT
    Original Root:  $ORIGINAL_PDGM_ROOT
EOF
# Shell control-flow statement managing script execution.
fi

# Write an embedded file block to disk.
cat >> "$SUMMARY_FILE" << EOF

CNN Model:           $CNN_MODEL_PATH
Output Directory:    $OUTPUT_DIR

Image Size:          $IMG_SIZE
Batch Size:          $BATCH_SIZE
Device:              $DEVICE
Use Resized:         $([ -n "$USE_RESIZED" ] && echo "Yes" || echo "No")

------------------------------------------------------------------------
CNN TRANSFERABILITY METRICS
------------------------------------------------------------------------
EOF

# Shell control-flow statement managing script execution.
if [ "$SKIP_CNN" = false ]; then
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_GBM" = true ]; then
        # Write an embedded file block to disk.
        cat >> "$SUMMARY_FILE" << EOF
GBM Domain:
  AUC (no adaptation):     $CNN_AUC_GBM
  Domain Gap:              $CNN_DOMAIN_GAP_GBM

EOF
    # Shell control-flow statement managing script execution.
    fi
    
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_PDGM" = true ]; then
        # Write an embedded file block to disk.
        cat >> "$SUMMARY_FILE" << EOF
PDGM Domain:
  AUC (no adaptation):     $CNN_AUC_PDGM
  Domain Gap:              $CNN_DOMAIN_GAP_PDGM

EOF
    # Shell control-flow statement managing script execution.
    fi
    
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
Interpretation:
  Domain gap indicates how much performance can be recovered with adaptation.

EOF
# Shell control-flow statement managing script execution.
else
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
Status: SKIPPED

EOF
# Shell control-flow statement managing script execution.
fi

# Write an embedded file block to disk.
cat >> "$SUMMARY_FILE" << EOF
------------------------------------------------------------------------
PERCEPTUAL QUALITY METRICS
------------------------------------------------------------------------
EOF

# Shell control-flow statement managing script execution.
if [ "$SKIP_FID_SSIM" = false ]; then
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_GBM" = true ]; then
        # Write an embedded file block to disk.
        cat >> "$SUMMARY_FILE" << EOF
GBM Domain:
  SSIM (Higher is better):  $SSIM_GBM
  FID  (Lower is better):   $FID_GBM

EOF
    # Shell control-flow statement managing script execution.
    fi
    
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_PDGM" = true ]; then
        # Write an embedded file block to disk.
        cat >> "$SUMMARY_FILE" << EOF
PDGM Domain:
  SSIM (Higher is better):  $SSIM_PDGM
  FID  (Lower is better):   $FID_PDGM

EOF
    # Shell control-flow statement managing script execution.
    fi
    
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
Interpretation:
  SSIM measures structural similarity (0-1 scale)
  FID measures distribution distance (lower is better)

EOF
# Shell control-flow statement managing script execution.
else
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
Status: SKIPPED

EOF
# Shell control-flow statement managing script execution.
fi

# Write an embedded file block to disk.
cat >> "$SUMMARY_FILE" << EOF
------------------------------------------------------------------------
FILES GENERATED
------------------------------------------------------------------------
This summary:      $SUMMARY_FILE
Master log:        $LOG_FILE
EOF

# Shell control-flow statement managing script execution.
if [ "$SKIP_CNN" = false ]; then
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_GBM" = true ]; then
        # Write an embedded file block to disk.
        cat >> "$SUMMARY_FILE" << EOF
GBM CNN results:   $CNN_OUTPUT_GBM
GBM CNN JSON:      ${SYNTH_GBM_ROOT}/enhanced_evaluation_results.json
EOF
    # Shell control-flow statement managing script execution.
    fi
    # Shell control-flow statement managing script execution.
    if [ "$EVAL_PDGM" = true ]; then
        # Write an embedded file block to disk.
        cat >> "$SUMMARY_FILE" << EOF
PDGM CNN results:  $CNN_OUTPUT_PDGM
PDGM CNN JSON:     ${SYNTH_PDGM_ROOT}/enhanced_evaluation_results.json
EOF
    # Shell control-flow statement managing script execution.
    fi
# Shell control-flow statement managing script execution.
fi

# Shell control-flow statement managing script execution.
if [ "$SKIP_FID_SSIM" = false ]; then
    # Write an embedded file block to disk.
    cat >> "$SUMMARY_FILE" << EOF
FID/SSIM results:  $FID_SSIM_OUTPUT
EOF
# Shell control-flow statement managing script execution.
fi

# Write an embedded file block to disk.
cat >> "$SUMMARY_FILE" << EOF

========================================================================
EOF

# Display summary
cat "$SUMMARY_FILE" | tee -a "$LOG_FILE"

log ""
log "============================================================"
log "✅ EVALUATION PIPELINE COMPLETED"
log "============================================================"
log ""
log "Summary report saved to: $SUMMARY_FILE"
log "Full log saved to: $LOG_FILE"
log ""

# Exit successfully
# Exit the script with the intended status code.
exit 0