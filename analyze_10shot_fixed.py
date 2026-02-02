#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

# analyze_10shot_fixed.py
# Import dependencies used by this module.
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------
# Function: `_read_text` implements a reusable processing step.
def _read_text(p: Path) -> str:
    # Return the computed value to the caller.
    return p.read_text(errors="ignore")


# Function: `_ensure_dir` implements a reusable processing step.
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# Function: `_read_csv_if_nonempty` implements a reusable processing step.
def _read_csv_if_nonempty(p: Path) -> pd.DataFrame | None:
    # Control-flow branch for conditional or iterative execution.
    if not p.exists():
        # Return the computed value to the caller.
        return None
    # Control-flow branch for conditional or iterative execution.
    try:
        df = pd.read_csv(p)
        # Control-flow branch for conditional or iterative execution.
        if df is None or len(df) == 0:
            # Return the computed value to the caller.
            return None
        # Return the computed value to the caller.
        return df
    # Control-flow branch for conditional or iterative execution.
    except Exception:
        # Return the computed value to the caller.
        return None


# Function: `_rolling` implements a reusable processing step.
def _rolling(y: np.ndarray, k: int) -> np.ndarray:
    # Control-flow branch for conditional or iterative execution.
    if len(y) < k:
        # Return the computed value to the caller.
        return y
    # Return the computed value to the caller.
    return pd.Series(y).rolling(k, min_periods=1).mean().to_numpy()


# -------------------------
# Parsing (YOUR log format)
# -------------------------
# Example lines:
# [VAE:src][Ep 1] rec(L1)=0.0885 kl=2.898661 kl_w=9.52e-07 time=76.6s
# [VAE:tgt][Ep 100] rec(L1)=0.0216 kl=2.523274 kl_w=1.00e-04 time=1.2s
# [LDM:ldm_src][Ep 1] mse=0.839637 time=29.4s
# [LDM:ldm_tgt][Ep 300] mse=0.324605 time=0.5s

# -------------------------
# Parsing (support OLD + NEW log formats)
# -------------------------

# OLD format (your previous runs)
RE_VAE_OLD = re.compile(
    r"\[VAE:(src|tgt)\]\[Ep\s+(\d+)\]\s+rec\(L1\)=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+"
    r"kl=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+"
    r"kl_w=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+"
    r"time=([0-9]*\.?[0-9]+)\s*s",
    re.IGNORECASE,
)

RE_LDM_OLD = re.compile(
    r"\[LDM:(ldm_src|ldm_tgt)\]\[Ep\s+(\d+)\]\s+mse=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+"
    r"time=([0-9]*\.?[0-9]+)\s*s",
    re.IGNORECASE,
)

# NEW format (your current run)
# Example:
# [VAE src] Ep 1/250 | Loss=0.1206 | Rec=0.1063 | KL=3.016670 | Edge=0.1264 | HF=0.0323 | KL_mult=0.016
RE_VAE_NEW = re.compile(
    r"\[VAE\s+(src|tgt)\]\s+Ep\s+(\d+)\s*/\s*\d+\s*\|\s*"
    r"Loss=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"Rec=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"KL=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"Edge=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"HF=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"KL_mult=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Example:
# [LDM ldm_src] Ep 1/300 | Loss=0.303454 | LR=2.00e-04
RE_LDM_NEW = re.compile(
    r"\[LDM\s+(ldm_src|ldm_tgt)\]\s+Ep\s+(\d+)\s*/\s*\d+\s*\|\s*"
    r"Loss=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"LR=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Some logs get glued; insert newlines before [VAE ...] or [LDM ...] even if mid-line
GLUE_FIX = re.compile(r"(?<!^)(\[(?:VAE|LDM)\s+[^\]]+\]\s+Ep\s+\d+)")



# Function: `parse_logs` implements a reusable processing step.
def parse_logs(outdir: Path) -> dict:
    """
    Returns dict with optional:
      vae_df: columns [stage, epoch, rec_loss_mean, kl_loss_mean, kl_weight, epoch_time_s]
      ldm_df: columns [stage, epoch, diffusion_loss_mean, epoch_time_s]
    """
    candidates = [
        outdir / "logs" / "training_log.txt",
        outdir / "nohup.out",
    ]

    text = ""
    # Control-flow branch for conditional or iterative execution.
    for p in candidates:
        # Control-flow branch for conditional or iterative execution.
        if p.exists():
            t = _read_text(p)
            # Control-flow branch for conditional or iterative execution.
            if t.strip():
                print(f"[DIAG] Using log source: {p} ({p.stat().st_size} bytes)")
                text += "\n" + t

    # Control-flow branch for conditional or iterative execution.
    if not text.strip():
        # Return the computed value to the caller.
        return {}

    # Fix glued tokens by adding newlines before any [VAE:...][Ep ...] or [LDM:...][Ep ...] sequences
    text = GLUE_FIX.sub(r"\n\1", text)

    vae_rows = []
    ldm_rows = []

        # --- VAE parsing: NEW then OLD ---
    # Control-flow branch for conditional or iterative execution.
    for m in RE_VAE_NEW.finditer(text):
        st, ep, loss, rec, kl, edge, hf, kl_mult = m.groups()
        vae_rows.append(
            {
                "timestamp": "",
                "stage": st.lower(),
                "epoch": int(ep),
                "rec_loss_mean": float(rec),
                "rec_loss_std": 0.0,
                "kl_loss_mean": float(kl),
                "kl_loss_std": 0.0,
                "kl_weight": np.nan,          # not explicitly logged in NEW format
                "epoch_time_s": np.nan,       # not logged per-epoch in NEW format
                "total_loss_mean": float(loss),
                "edge_loss_mean": float(edge),
                "hf_loss_mean": float(hf),
                "kl_mult": float(kl_mult),
            }
        )

    # Control-flow branch for conditional or iterative execution.
    for m in RE_VAE_OLD.finditer(text):
        st, ep, rec, kl, klw, ts = m.groups()
        vae_rows.append(
            {
                "timestamp": "",
                "stage": st.lower(),
                "epoch": int(ep),
                "rec_loss_mean": float(rec),
                "rec_loss_std": 0.0,
                "kl_loss_mean": float(kl),
                "kl_loss_std": 0.0,
                "kl_weight": float(klw),
                "epoch_time_s": float(ts),
                "total_loss_mean": np.nan,
                "edge_loss_mean": np.nan,
                "hf_loss_mean": np.nan,
                "kl_mult": np.nan,
            }
        )

    # --- LDM parsing: NEW then OLD ---
    # Control-flow branch for conditional or iterative execution.
    for m in RE_LDM_NEW.finditer(text):
        st, ep, loss, lr = m.groups()
        ldm_rows.append(
            {
                "timestamp": "",
                "stage": st.lower(),  # ldm_src / ldm_tgt
                "epoch": int(ep),
                "diffusion_loss_mean": float(loss),
                "diffusion_loss_std": 0.0,
                "lr": float(lr),
                "epoch_time_s": np.nan,   # not logged in NEW format
                "ema_decay": np.nan,
            }
        )

    # Control-flow branch for conditional or iterative execution.
    for m in RE_LDM_OLD.finditer(text):
        st, ep, mse, ts = m.groups()
        ldm_rows.append(
            {
                "timestamp": "",
                "stage": st.lower(),
                "epoch": int(ep),
                "diffusion_loss_mean": float(mse),
                "diffusion_loss_std": 0.0,
                "lr": np.nan,
                "epoch_time_s": float(ts),
                "ema_decay": np.nan,
            }
        )


    out = {}
    # Control-flow branch for conditional or iterative execution.
    if vae_rows:
        dfv = pd.DataFrame(vae_rows)
        dfv = dfv.sort_values(["stage", "epoch"]).drop_duplicates(["stage", "epoch"], keep="last")
        out["vae"] = dfv
        print(f"[OK] Parsed VAE rows: {len(dfv)} | stages={sorted(dfv['stage'].unique().tolist())}")
    else:
        print("[WARN] No VAE rows parsed (check log format?)")

    # Control-flow branch for conditional or iterative execution.
    if ldm_rows:
        dfl = pd.DataFrame(ldm_rows)
        dfl = dfl.sort_values(["stage", "epoch"]).drop_duplicates(["stage", "epoch"], keep="last")
        out["ldm"] = dfl
        print(f"[OK] Parsed LDM rows: {len(dfl)} | stages={sorted(dfl['stage'].unique().tolist())}")
    else:
        print("[WARN] No LDM rows parsed (check log format?)")

    # Return the computed value to the caller.
    return out


# -------------------------
# Plotting
# -------------------------
# Function: `plot_vae_training` implements a reusable processing step.
def plot_vae_training(vae_df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    src = vae_df[vae_df["stage"] == "src"]
    tgt = vae_df[vae_df["stage"] == "tgt"]

    # Rec loss src
    ax = axes[0, 0]
    # Control-flow branch for conditional or iterative execution.
    if len(src):
        ax.plot(src["epoch"], src["rec_loss_mean"], linewidth=2, label="Source")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No source VAE data", ha="center", va="center")
    ax.set_title("VAE Reconstruction Loss - Source (GBM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rec Loss (L1)")
    ax.grid(True, alpha=0.2)

    # Rec loss tgt
    ax = axes[0, 1]
    # Control-flow branch for conditional or iterative execution.
    if len(tgt):
        ax.plot(tgt["epoch"], tgt["rec_loss_mean"], linewidth=2, label="Target (10-shot)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No target VAE data", ha="center", va="center")
    ax.set_title("VAE Reconstruction Loss - Target (PDGM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rec Loss (L1)")
    ax.grid(True, alpha=0.2)

    # KL src + KL weight
    ax = axes[1, 0]
    # Control-flow branch for conditional or iterative execution.
    if len(src):
        ax.plot(src["epoch"], src["kl_loss_mean"], linewidth=2, label="KL")
        ax2 = ax.twinx()
        ax2.plot(src["epoch"], src["kl_weight"], linestyle="--", linewidth=1.5, label="KL Weight")
        ax2.set_ylabel("KL Weight")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No source VAE data", ha="center", va="center")
    ax.set_title("VAE KL - Source")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.grid(True, alpha=0.2)

    # KL tgt + KL weight
    ax = axes[1, 1]
    # Control-flow branch for conditional or iterative execution.
    if len(tgt):
        ax.plot(tgt["epoch"], tgt["kl_loss_mean"], linewidth=2, label="KL")
        ax2 = ax.twinx()
        ax2.plot(tgt["epoch"], tgt["kl_weight"], linestyle="--", linewidth=1.5, label="KL Weight")
        ax2.set_ylabel("KL Weight")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No target VAE data", ha="center", va="center")
    ax.set_title("VAE KL - Target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {save_path}")


# Function: `plot_ldm_training` implements a reusable processing step.
def plot_ldm_training(ldm_df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    src = ldm_df[ldm_df["stage"] == "ldm_src"]
    tgt = ldm_df[ldm_df["stage"] == "ldm_tgt"]

    ax = axes[0, 0]
    # Control-flow branch for conditional or iterative execution.
    if len(src):
        ax.plot(src["epoch"], src["diffusion_loss_mean"], linewidth=2, label="Source")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No source LDM data", ha="center", va="center")
    ax.set_title("LDM Training - Source (GBM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion Loss (MSE)")
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    # Control-flow branch for conditional or iterative execution.
    if len(tgt):
        ax.plot(tgt["epoch"], tgt["diffusion_loss_mean"], linewidth=2.5, label="Target (10-shot)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No target LDM data", ha="center", va="center")
    ax.set_title("LDM Adaptation - Target (PDGM 10-Shot)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion Loss (MSE)")
    ax.grid(True, alpha=0.2)

    # “Learning rate schedule” placeholder (your log lines don’t include LR)
    ax = axes[1, 0]
    did_lr = False
    # Control-flow branch for conditional or iterative execution.
    if len(src) and src["lr"].notna().any():
        ax.plot(src["epoch"], src["lr"], linewidth=2, label="Source LR")
        did_lr = True
    # Control-flow branch for conditional or iterative execution.
    if len(tgt) and tgt["lr"].notna().any():
        ax.plot(tgt["epoch"], tgt["lr"], linewidth=2, label="Target LR")
        did_lr = True
    # Control-flow branch for conditional or iterative execution.
    if did_lr:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "LR not logged", ha="center", va="center")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.2)


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {save_path}")


# Function: `plot_comparison` implements a reusable processing step.
def plot_comparison(vae_df: pd.DataFrame | None, ldm_df: pd.DataFrame | None, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    labels, vals = [], []
    # Control-flow branch for conditional or iterative execution.
    if vae_df is not None and len(vae_df):
        src = vae_df[vae_df["stage"] == "src"]
        tgt = vae_df[vae_df["stage"] == "tgt"]
        # Control-flow branch for conditional or iterative execution.
        if len(src):
            labels.append("VAE\nSource")
            vals.append(float(src["rec_loss_mean"].iloc[-1]))
        # Control-flow branch for conditional or iterative execution.
        if len(tgt):
            labels.append("VAE\nTarget")
            vals.append(float(tgt["rec_loss_mean"].iloc[-1]))
    # Control-flow branch for conditional or iterative execution.
    if labels:
        bars = ax.bar(labels, vals, alpha=0.85, edgecolor="black", linewidth=1.5)
        # Control-flow branch for conditional or iterative execution.
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.4f}", ha="center", va="bottom")
    else:
        ax.text(0.5, 0.5, "No VAE data", ha="center", va="center")
    ax.set_title("VAE Performance Comparison")
    ax.set_ylabel("Final Reconstruction Loss")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    labels, vals = [], []
    # Control-flow branch for conditional or iterative execution.
    if ldm_df is not None and len(ldm_df):
        src = ldm_df[ldm_df["stage"] == "ldm_src"]
        tgt = ldm_df[ldm_df["stage"] == "ldm_tgt"]
        # Control-flow branch for conditional or iterative execution.
        if len(src):
            labels.append("LDM\nSource")
            vals.append(float(src["diffusion_loss_mean"].iloc[-1]))
        # Control-flow branch for conditional or iterative execution.
        if len(tgt):
            labels.append("LDM\nTarget\n(10-shot)")
            vals.append(float(tgt["diffusion_loss_mean"].iloc[-1]))
    # Control-flow branch for conditional or iterative execution.
    if labels:
        bars = ax.bar(labels, vals, alpha=0.85, edgecolor="black", linewidth=1.5)
        # Control-flow branch for conditional or iterative execution.
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.4f}", ha="center", va="bottom")
    else:
        ax.text(0.5, 0.5, "No LDM data", ha="center", va="center")
    ax.set_title("LDM Performance Comparison")
    ax.set_ylabel("Final Diffusion Loss")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {save_path}")


# Function: `plot_loss_overlay` implements a reusable processing step.
def plot_loss_overlay(vae_df: pd.DataFrame | None, ldm_df: pd.DataFrame | None, save_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5))

    did = False
    # Control-flow branch for conditional or iterative execution.
    if vae_df is not None and len(vae_df):
        # Control-flow branch for conditional or iterative execution.
        for st, lab in [("src", "VAE Rec (src)"), ("tgt", "VAE Rec (tgt)")]:
            d = vae_df[vae_df["stage"] == st]
            # Control-flow branch for conditional or iterative execution.
            if len(d):
                ax.plot(d["epoch"], d["rec_loss_mean"], linewidth=2, label=lab)
                did = True

    # Control-flow branch for conditional or iterative execution.
    if ldm_df is not None and len(ldm_df):
        # Control-flow branch for conditional or iterative execution.
        for st, lab in [("ldm_src", "LDM MSE (src)"), ("ldm_tgt", "LDM MSE (tgt)")]:
            d = ldm_df[ldm_df["stage"] == st]
            # Control-flow branch for conditional or iterative execution.
            if len(d):
                ax.plot(d["epoch"], d["diffusion_loss_mean"], linewidth=2, label=lab)
                did = True

    # Control-flow branch for conditional or iterative execution.
    if not did:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")

    ax.set_title("Loss Overlay (VAE Rec + LDM MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2)
    # Control-flow branch for conditional or iterative execution.
    if did:
        ax.legend()
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {save_path}")


# Function: `plot_ldm_target_zoom` implements a reusable processing step.
def plot_ldm_target_zoom(ldm_df: pd.DataFrame, save_path: Path):
    tgt = ldm_df[ldm_df["stage"] == "ldm_tgt"].sort_values("epoch")
    fig, ax = plt.subplots(figsize=(14, 5))
    # Control-flow branch for conditional or iterative execution.
    if not len(tgt):
        ax.text(0.5, 0.5, "No target LDM data", ha="center", va="center")
    else:
        x = tgt["epoch"].to_numpy()
        y = tgt["diffusion_loss_mean"].to_numpy()
        ax.plot(x, y, linewidth=1.5, label="Target MSE (raw)")
        ax.plot(x, _rolling(y, 7), linewidth=2.5, label="Target MSE (rolling-7)")
        ax.set_title("LDM Target (PDGM 10-shot) — Zoom + Rolling Mean")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.2)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {save_path}")


# Function: `plot_ldm_histograms` implements a reusable processing step.
def plot_ldm_histograms(ldm_df: pd.DataFrame, save_path: Path):
    src = ldm_df[ldm_df["stage"] == "ldm_src"]["diffusion_loss_mean"].dropna().to_numpy()
    tgt = ldm_df[ldm_df["stage"] == "ldm_tgt"]["diffusion_loss_mean"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Control-flow branch for conditional or iterative execution.
    if len(src):
        axes[0].hist(src, bins=40, alpha=0.85)
        axes[0].set_title("LDM Source MSE Distribution")
    else:
        axes[0].text(0.5, 0.5, "No src data", ha="center", va="center")
        axes[0].set_title("LDM Source MSE Distribution")
    axes[0].set_xlabel("MSE"); axes[0].set_ylabel("Count"); axes[0].grid(True, alpha=0.2)

    # Control-flow branch for conditional or iterative execution.
    if len(tgt):
        axes[1].hist(tgt, bins=40, alpha=0.85)
        axes[1].set_title("LDM Target MSE Distribution")
    else:
        axes[1].text(0.5, 0.5, "No tgt data", ha="center", va="center")
        axes[1].set_title("LDM Target MSE Distribution")
    axes[1].set_xlabel("MSE"); axes[1].set_ylabel("Count"); axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {save_path}")


# Function: `generate_summary` implements a reusable processing step.
def generate_summary(hp: dict, vae_df: pd.DataFrame | None, ldm_df: pd.DataFrame | None, save_path: Path):
    r = []
    r.append("=" * 80)
    r.append("10-SHOT TRAINING SUMMARY REPORT (PARSED FROM TEXT LOGS)")
    r.append("=" * 80)
    r.append("")

    # Control-flow branch for conditional or iterative execution.
    if hp:
        r.append("HYPERPARAMETERS")
        r.append("-" * 80)
        keys = ["seed","device","batch_size","vae_base","unet_base","z_channels","latent_size","epochs_ldm_src","epochs_ldm_tgt","kl_w","rec_w"]
        # Control-flow branch for conditional or iterative execution.
        for k in keys:
            # Control-flow branch for conditional or iterative execution.
            if k in hp:
                r.append(f"  {k:22s}: {hp[k]}")
        r.append("")

    # Control-flow branch for conditional or iterative execution.
    if vae_df is not None and len(vae_df):
        r.append("VAE")
        r.append("-" * 80)
        # Control-flow branch for conditional or iterative execution.
        for st in ["src","tgt"]:
            d = vae_df[vae_df["stage"] == st].sort_values("epoch")
            # Control-flow branch for conditional or iterative execution.
            if len(d):
                r.append(f"  {st.upper()}: epochs={len(d)}  rec_start={d['rec_loss_mean'].iloc[0]:.6f}  rec_final={d['rec_loss_mean'].iloc[-1]:.6f}  kl_final={d['kl_loss_mean'].iloc[-1]:.6f}")
                r.append(f"       time_hours={d['epoch_time_s'].sum()/3600:.2f}")
        r.append("")

    # Control-flow branch for conditional or iterative execution.
    if ldm_df is not None and len(ldm_df):
        r.append("LDM")
        r.append("-" * 80)
        # Control-flow branch for conditional or iterative execution.
        for st in ["ldm_src","ldm_tgt"]:
            d = ldm_df[ldm_df["stage"] == st].sort_values("epoch")
            # Control-flow branch for conditional or iterative execution.
            if len(d):
                mn_i = d["diffusion_loss_mean"].idxmin()
                r.append(f"  {st.upper()}: epochs={len(d)}  mse_start={d['diffusion_loss_mean'].iloc[0]:.6f}  mse_final={d['diffusion_loss_mean'].iloc[-1]:.6f}")
                r.append(f"       mse_min={d['diffusion_loss_mean'].min():.6f} at epoch={int(d.loc[mn_i,'epoch'])}  time_hours={d['epoch_time_s'].sum()/3600:.2f}")
        r.append("")

    total_s = 0.0
    # Control-flow branch for conditional or iterative execution.
    if vae_df is not None and len(vae_df):
        total_s += float(vae_df["epoch_time_s"].sum())
    # Control-flow branch for conditional or iterative execution.
    if ldm_df is not None and len(ldm_df):
        total_s += float(ldm_df["epoch_time_s"].sum())
    r.append("TOTAL TIME (from per-epoch 'time=...s' lines)")
    r.append("-" * 80)
    r.append(f"  total_hours: {total_s/3600:.2f}")
    r.append(f"  total_days : {total_s/86400:.2f}")
    r.append("")
    r.append("=" * 80)

    save_path.write_text("\n".join(r))
    print(f"[OK] saved {save_path}")


# -------------------------
# Main
# -------------------------
# Function: `main` implements a reusable processing step.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    log_dir = outdir / "logs"
    analysis_dir = outdir / "analysis"
    _ensure_dir(analysis_dir)

    # Load hyperparams json (always present)
    hp = {}
    hp_path = log_dir / "hyperparameters.json"
    # Control-flow branch for conditional or iterative execution.
    if hp_path.exists():
        # Control-flow branch for conditional or iterative execution.
        try:
            hp = json.loads(hp_path.read_text())
        # Control-flow branch for conditional or iterative execution.
        except Exception:
            hp = {}

    # Parse from logs (don’t trust the empty CSVs)
    parsed = parse_logs(outdir)
    vae_df = parsed.get("vae")
    ldm_df = parsed.get("ldm")

    # Save recovered CSVs
    # Control-flow branch for conditional or iterative execution.
    if vae_df is not None and len(vae_df):
        vae_df.to_csv(analysis_dir / "vae_training_recovered.csv", index=False)
        print(f"[OK] wrote {analysis_dir / 'vae_training_recovered.csv'}")
    # Control-flow branch for conditional or iterative execution.
    if ldm_df is not None and len(ldm_df):
        ldm_df.to_csv(analysis_dir / "ldm_training_recovered.csv", index=False)
        print(f"[OK] wrote {analysis_dir / 'ldm_training_recovered.csv'}")

    # Plot everything
    # Control-flow branch for conditional or iterative execution.
    if vae_df is not None and len(vae_df):
        plot_vae_training(vae_df, analysis_dir / "vae_training.png")
    else:
        print("[WARN] No VAE data parsed; vae_training.png will not be created.")

    # Control-flow branch for conditional or iterative execution.
    if ldm_df is not None and len(ldm_df):
        plot_ldm_training(ldm_df, analysis_dir / "ldm_training.png")
        plot_ldm_target_zoom(ldm_df, analysis_dir / "ldm_target_zoom.png")
        plot_ldm_histograms(ldm_df, analysis_dir / "ldm_histograms.png")
    else:
        print("[WARN] No LDM data parsed; LDM plots will not be created.")

    plot_comparison(vae_df, ldm_df, analysis_dir / "comparison.png")
    plot_loss_overlay(vae_df, ldm_df, analysis_dir / "loss_overlay.png")
    generate_summary(hp, vae_df, ldm_df, analysis_dir / "summary_report.txt")

    print("\n[DONE] All plots + recovered CSVs are in:", analysis_dir)


# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == "__main__":
    main()
