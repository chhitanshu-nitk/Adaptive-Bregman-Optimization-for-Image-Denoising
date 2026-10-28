"""Module to benchmark Euclidean vs Adaptive Bregman and generate publication plots and tables."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.image_io import load_grayscale_image, save_image
from src.noise.gaussian import add_gaussian_noise
from src.utils.metrics import compute_psnr, compute_ssim, compute_mse, compute_rmse
from src.optimizers.nmPPMDC import nmPPMDCOptimizer


def plot_convergence(
    euc_history: Dict[str, Any],
    ada_history: Dict[str, Any],
    save_path: Path
) -> None:
    """Generate convergence graphs showing Objective Value and Relative Error vs Iterations."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Objective energy curve
    axes[0].plot(euc_history["energy_history"], label="Euclidean nmPPMDC (Baseline)", color="#d95f02", lw=2)
    axes[0].plot(ada_history["energy_history"], label="Adaptive nmPPMDC (Proposed)", color="#1b9e77", lw=2)
    axes[0].set_title("Objective Energy Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration", fontsize=10)
    axes[0].set_ylabel("F(u)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Relative difference curve
    axes[1].semilogy(euc_history["rel_diff_history"], label="Euclidean nmPPMDC", color="#d95f02", lw=2)
    axes[1].semilogy(ada_history["rel_diff_history"], label="Adaptive nmPPMDC", color="#1b9e77", lw=2)
    axes[1].set_title("Iterate Relative Change ||u_{k+1} - u_k|| / ||u_k||", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Iteration", fontsize=10)
    axes[1].set_ylabel("Relative Change (log scale)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close(fig)


def plot_comparison_bar(df: pd.DataFrame, save_path: Path) -> None:
    """Create comparison bar charts for PSNR and SSIM across test images."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images = df["image"].tolist()
    x = np.arange(len(images))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # PSNR Plot
    axes[0].bar(x - width/2, df["psnr_euclidean"], width, label="Euclidean", color="#386cb0")
    axes[0].bar(x + width/2, df["psnr_adaptive"], width, label="Adaptive (Proposed)", color="#f0027f")
    axes[0].set_title("PSNR (dB) Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(images, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("PSNR (dB)", fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # SSIM Plot
    axes[1].bar(x - width/2, df["ssim_euclidean"], width, label="Euclidean", color="#386cb0")
    axes[1].bar(x + width/2, df["ssim_adaptive"], width, label="Adaptive (Proposed)", color="#f0027f")
    axes[1].set_title("SSIM Comparison", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(images, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("SSIM", fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close(fig)


def run_experiments(config: Dict[str, Any]) -> pd.DataFrame:
    """Execute evaluation pipeline on all images specified in config."""
    raw_dir = Path(config["dataset"]["raw_dir"])
    base_out = Path(config["output"]["base_dir"])
    sigma = float(config["noise"]["sigma"])
    seed = int(config["experiment"]["seed"])
    max_images = config["dataset"].get("max_images", None)

    noisy_dir = base_out / "noisy"
    recon_dir = base_out / "reconstructed"
    euclidean_recon_dir = recon_dir / "euclidean"
    adaptive_recon_dir = recon_dir / "adaptive"
    metrics_dir = base_out / "metrics"
    plots_dir = base_out / "plots"

    for d in [noisy_dir, euclidean_recon_dir, adaptive_recon_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    optimizer = nmPPMDCOptimizer(config)
    image_paths = sorted(list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.bmp")))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {raw_dir.resolve()}")

    if max_images is not None:
        image_paths = image_paths[:int(max_images)]

    records: List[Dict[str, Any]] = []

    print(f"\n[Starting Experiment]: {len(image_paths)} images from {raw_dir.name} (Noise sigma={sigma})")

    first_euc_history = None
    first_ada_history = None

    for idx, img_p in enumerate(image_paths, 1):
        stem = img_p.stem
        img_clean = load_grayscale_image(img_p)

        # Generate Gaussian noise reproducibly
        img_noisy, _ = add_gaussian_noise(img_clean, sigma=sigma, seed=seed + idx)

        # Save noisy reference
        if config["output"]["save_images"]:
            save_image(img_noisy, noisy_dir / f"{stem}_sigma{int(sigma)}_noisy.png")

        noisy_psnr = compute_psnr(img_clean, img_noisy)
        noisy_ssim = compute_ssim(img_clean, img_noisy)

        # 1. Run Euclidean Baseline
        u_euc, hist_euc = optimizer.optimize(img_noisy, mode="euclidean")
        psnr_euc = compute_psnr(img_clean, u_euc)
        ssim_euc = compute_ssim(img_clean, u_euc)
        mse_euc = compute_mse(img_clean, u_euc)
        rmse_euc = compute_rmse(img_clean, u_euc)

        # 2. Run Proposed Adaptive Novelty
        u_ada, hist_ada = optimizer.optimize(img_noisy, mode="adaptive")
        psnr_ada = compute_psnr(img_clean, u_ada)
        ssim_ada = compute_ssim(img_clean, u_ada)
        mse_ada = compute_mse(img_clean, u_ada)
        rmse_ada = compute_rmse(img_clean, u_ada)

        if idx == 1:
            first_euc_history = hist_euc
            first_ada_history = hist_ada

        # Save reconstructed images
        if config["output"]["save_images"]:
            save_image(u_euc, euclidean_recon_dir / f"{stem}_sigma{int(sigma)}.png")
            save_image(u_ada, adaptive_recon_dir / f"{stem}_sigma{int(sigma)}.png")

        record = {
            "image": stem,
            "sigma": sigma,
            "noisy_psnr": noisy_psnr,
            "noisy_ssim": noisy_ssim,
            "psnr_euclidean": psnr_euc,
            "ssim_euclidean": ssim_euc,
            "mse_euclidean": mse_euc,
            "rmse_euclidean": rmse_euc,
            "time_euclidean": hist_euc["runtime"],
            "iter_euclidean": hist_euc["iterations"],
            "psnr_adaptive": psnr_ada,
            "ssim_adaptive": ssim_ada,
            "mse_adaptive": mse_ada,
            "rmse_adaptive": rmse_ada,
            "time_adaptive": hist_ada["runtime"],
            "iter_adaptive": hist_ada["iterations"],
            "psnr_gain": psnr_ada - psnr_euc,
            "ssim_gain": ssim_ada - ssim_euc
        }
        records.append(record)

        print(f"[{idx:02d}/{len(image_paths):02d}] {stem:<12} | "
              f"Euclidean PSNR: {psnr_euc:6.2f} dB, SSIM: {ssim_euc:5.4f} | "
              f"Adaptive PSNR: {psnr_ada:6.2f} dB, SSIM: {ssim_ada:5.4f} | "
              f"Gain: +{record['psnr_gain']:4.2f} dB")

    df = pd.DataFrame(records)

    # Save summary tables
    csv_path = metrics_dir / f"comparison_results_sigma{int(sigma)}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved Metrics]: {csv_path}")

    # Generate comparison plots
    if config["output"]["save_plots"]:
        if first_euc_history and first_ada_history:
            plot_convergence(first_euc_history, first_ada_history, plots_dir / f"convergence_sample_sigma{int(sigma)}.png")
        plot_comparison_bar(df, plots_dir / f"comparison_metrics_sigma{int(sigma)}.png")
        print(f"[Saved Plots]: {plots_dir}")

    return df