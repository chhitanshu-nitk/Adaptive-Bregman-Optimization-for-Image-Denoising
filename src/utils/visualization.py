"""Visualization utilities for nmPPMDC and Bregman ablation study."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_publication_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })


def plot_ablation_bars(summary_df: pd.DataFrame, save_path: Path) -> None:
    """Generate bar charts for PSNR, SSIM, and Runtime across all 7 penalties."""
    set_publication_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    penalties = summary_df["penalty"].tolist()
    display_names = [p.upper().replace("_", "-") for p in penalties]
    x = np.arange(len(penalties))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    color_euc = "#386cb0"
    color_ada = "#f0027f"

    # PSNR
    axes[0].bar(x - width / 2, summary_df["psnr_euclidean_mean"], width,
                yerr=summary_df.get("psnr_euclidean_std", None),
                capsize=4, label="Euclidean", color=color_euc, alpha=0.9, edgecolor="black", lw=0.7)
    axes[0].bar(x + width / 2, summary_df["psnr_adaptive_mean"], width,
                yerr=summary_df.get("psnr_adaptive_std", None),
                capsize=4, label="Adaptive", color=color_ada, alpha=0.9, edgecolor="black", lw=0.7)
    axes[0].set_title("PSNR (dB)", fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=25, ha="right")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, facecolor="white", framealpha=0.9)

    # SSIM
    axes[1].bar(x - width / 2, summary_df["ssim_euclidean_mean"], width,
                yerr=summary_df.get("ssim_euclidean_std", None),
                capsize=4, label="Euclidean", color=color_euc, alpha=0.9, edgecolor="black", lw=0.7)
    axes[1].bar(x + width / 2, summary_df["ssim_adaptive_mean"], width,
                yerr=summary_df.get("ssim_adaptive_std", None),
                capsize=4, label="Adaptive", color=color_ada, alpha=0.9, edgecolor="black", lw=0.7)
    axes[1].set_title("SSIM", fontweight="bold")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=25, ha="right")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, facecolor="white", framealpha=0.9)

    # Runtime
    axes[2].bar(x - width / 2, summary_df["time_euclidean_mean"], width,
                label="Euclidean", color=color_euc, alpha=0.9, edgecolor="black", lw=0.7)
    axes[2].bar(x + width / 2, summary_df["time_adaptive_mean"], width,
                label="Adaptive", color=color_ada, alpha=0.9, edgecolor="black", lw=0.7)
    axes[2].set_title("Average CPU Runtime (s)", fontweight="bold")
    axes[2].set_ylabel("Runtime (seconds)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display_names, rotation=25, ha="right")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[2].legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.suptitle("7-Penalty Ablation: Euclidean vs. Adaptive Bregman (nmPPMDC)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_all_penalties(
    histories: Dict[str, Dict[str, Any]],
    save_path: Path
) -> None:
    """Plot energy and stopping metric convergence across penalties."""
    set_publication_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    penalties = ["atan", "log", "mcp", "scad", "geman", "fractional", "capped_l1"]
    cmap = plt.get_cmap("tab10")

    for idx, pen in enumerate(penalties):
        color = cmap(idx)
        key_euc = f"{pen}_euclidean"
        key_ada = f"{pen}_adaptive"

        if key_euc in histories:
            h_euc = histories[key_euc]
            axes[0].plot(h_euc["energy_history"], linestyle="--", color=color, alpha=0.7, lw=1.5,
                         label=f"{pen.upper()} (Euc)")
            if h_euc["stop_rule_history"]:
                axes[1].semilogy(h_euc["stop_rule_history"], linestyle="--", color=color, alpha=0.7, lw=1.5)

        if key_ada in histories:
            h_ada = histories[key_ada]
            axes[0].plot(h_ada["energy_history"], linestyle="-", color=color, lw=2.0,
                         label=f"{pen.upper()} (Ada)")
            if h_ada["stop_rule_history"]:
                axes[1].semilogy(h_ada["stop_rule_history"], linestyle="-", color=color, lw=2.0)

    axes[0].set_title("Objective Energy f(x^k)", fontweight="bold")
    axes[0].set_xlabel("Iteration k")
    axes[0].set_ylabel("f(x^k)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(ncol=2, fontsize=8, loc="upper right")

    axes[1].set_title("||x^{k+1} - x^k||_2", fontweight="bold")
    axes[1].set_xlabel("Iteration k")
    axes[1].set_ylabel("||x^{k+1} - x^k||_2 (log scale)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("nmPPMDC Convergence Dynamics", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_grid(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    save_path: Path
) -> None:
    """Plot visual comparison grid across reconstructed images."""
    set_publication_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_items = len(reconstructions) + 2
    cols = min(4, n_items)
    rows = (n_items + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    axes_flat[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes_flat[0].set_title("Clean", fontweight="bold")
    axes_flat[0].axis("off")

    axes_flat[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes_flat[1].set_title("Noisy", fontweight="bold")
    axes_flat[1].axis("off")

    for idx, (label, img) in enumerate(reconstructions.items(), start=2):
        if idx < len(axes_flat):
            axes_flat[idx].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes_flat[idx].set_title(label, fontweight="bold")
            axes_flat[idx].axis("off")

    for j in range(n_items, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
