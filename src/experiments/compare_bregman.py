"""Ablation study: compare 7 non-convex penalties under Euclidean and Adaptive Bregman distances."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from src.utils.image_io import load_grayscale_image, save_image
from src.noise.gaussian import add_gaussian_noise
from src.utils.metrics import compute_psnr, compute_ssim, compute_mse, compute_rmse
from src.optimizers.nmPPMDC import nmPPMDCOptimizer
from src.utils.visualization import (
    plot_ablation_bars,
    plot_convergence_all_penalties,
    plot_reconstruction_grid,
)


def run_experiments(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the 7-penalty x 2-Bregman ablation study across benchmark images."""
    raw_dir = Path(config["dataset"]["raw_dir"])
    base_out = Path(config["output"]["base_dir"])
    sigma = float(config["noise"]["sigma"])
    seed = int(config["experiment"]["seed"])
    max_images = config["dataset"].get("max_images", None)

    penalties = config.get("penalties", [
        "atan", "log", "mcp", "scad", "geman", "fractional", "capped_l1"
    ])
    modes = config.get("bregman", {}).get("modes", ["euclidean", "adaptive"])

    noisy_dir = base_out / "noisy"
    recon_base = base_out / "reconstructed"
    metrics_dir = base_out / "metrics"
    plots_dir = base_out / "plots"

    for d in [noisy_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    optimizer = nmPPMDCOptimizer(config)

    image_paths = sorted(
        list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.bmp"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {raw_dir.resolve()}")

    if max_images is not None:
        image_paths = image_paths[:int(max_images)]

    records_per_image: List[Dict[str, Any]] = []
    records_comparison: List[Dict[str, Any]] = []

    print(f"\nRunning 7-penalty x 2-Bregman ablation on {len(image_paths)} images (sigma={sigma})...\n")

    first_image_histories: Dict[str, Dict[str, Any]] = {}
    first_image_recons: Dict[str, np.ndarray] = {}
    first_clean: np.ndarray = None
    first_noisy: np.ndarray = None

    for img_idx, img_p in enumerate(image_paths, start=1):
        stem = img_p.stem
        img_clean = load_grayscale_image(img_p)
        img_noisy, _ = add_gaussian_noise(img_clean, sigma=sigma, seed=seed + img_idx)

        if config["output"].get("save_images", True):
            save_image(img_noisy, noisy_dir / f"{stem}_sigma{int(sigma)}_noisy.png")

        noisy_psnr = compute_psnr(img_clean, img_noisy)
        noisy_ssim = compute_ssim(img_clean, img_noisy)

        if img_idx == 1:
            first_clean = img_clean
            first_noisy = img_noisy

        print(f"[{img_idx:02d}/{len(image_paths):02d}] {stem} (Input PSNR: {noisy_psnr:.2f} dB, SSIM: {noisy_ssim:.4f})")

        for pen in penalties:
            results_pen: Dict[str, Any] = {}

            for mode in modes:
                u_recon, hist = optimizer.optimize(
                    b=img_noisy,
                    mode=mode,
                    penalty=pen,
                    x0=None
                )

                psnr_val = compute_psnr(img_clean, u_recon)
                ssim_val = compute_ssim(img_clean, u_recon)
                mse_val = compute_mse(img_clean, u_recon)
                rmse_val = compute_rmse(img_clean, u_recon)

                results_pen[mode] = {
                    "u": u_recon,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "mse": mse_val,
                    "rmse": rmse_val,
                    "time": hist["runtime"],
                    "iter": hist["iterations"],
                    "stop_rule": hist["final_stop_rule"],
                    "obj": hist["energy_history"][-1] if hist["energy_history"] else 0.0,
                }

                if config["output"].get("save_images", True):
                    out_dir = recon_base / pen / mode
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_image(u_recon, out_dir / f"{stem}_sigma{int(sigma)}.png")

                if img_idx == 1:
                    first_image_histories[f"{pen}_{mode}"] = hist
                    first_image_recons[f"{pen[:4].upper()}_{mode[:3].title()}"] = u_recon

                records_per_image.append({
                    "image": stem,
                    "sigma": sigma,
                    "penalty": pen,
                    "mode": mode,
                    "noisy_psnr": noisy_psnr,
                    "noisy_ssim": noisy_ssim,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "mse": mse_val,
                    "rmse": rmse_val,
                    "runtime_sec": hist["runtime"],
                    "iterations": hist["iterations"],
                    "stop_rule": hist["final_stop_rule"],
                })

            euc = results_pen.get("euclidean", {})
            ada = results_pen.get("adaptive", {})

            psnr_gain = ada["psnr"] - euc["psnr"]
            ssim_gain = ada["ssim"] - euc["ssim"]
            time_diff = ada["time"] - euc["time"]

            records_comparison.append({
                "image": stem,
                "sigma": sigma,
                "penalty": pen,
                "noisy_psnr": noisy_psnr,
                "noisy_ssim": noisy_ssim,
                "psnr_euclidean": euc["psnr"],
                "psnr_adaptive": ada["psnr"],
                "psnr_gain": psnr_gain,
                "ssim_euclidean": euc["ssim"],
                "ssim_adaptive": ada["ssim"],
                "ssim_gain": ssim_gain,
                "time_euclidean": euc["time"],
                "time_adaptive": ada["time"],
                "time_diff": time_diff,
                "iter_euclidean": euc["iter"],
                "iter_adaptive": ada["iter"],
            })

            print(f"  * {pen.upper():<11} | Euc: {euc['psnr']:5.2f} dB, {euc['ssim']:.4f} | "
                  f"Ada: {ada['psnr']:5.2f} dB, {ada['ssim']:.4f} | Gain: {psnr_gain:+4.2f} dB, {ssim_gain:+.4f}")

    df_per_image = pd.DataFrame(records_per_image)
    df_comparison = pd.DataFrame(records_comparison)

    summary_rows = []
    for pen in penalties:
        sub = df_comparison[df_comparison["penalty"] == pen]
        summary_rows.append({
            "penalty": pen,
            "psnr_euclidean_mean": float(sub["psnr_euclidean"].mean()),
            "psnr_euclidean_std": float(sub["psnr_euclidean"].std()),
            "psnr_adaptive_mean": float(sub["psnr_adaptive"].mean()),
            "psnr_adaptive_std": float(sub["psnr_adaptive"].std()),
            "psnr_gain_mean": float(sub["psnr_gain"].mean()),
            "ssim_euclidean_mean": float(sub["ssim_euclidean"].mean()),
            "ssim_euclidean_std": float(sub["ssim_euclidean"].std()),
            "ssim_adaptive_mean": float(sub["ssim_adaptive"].mean()),
            "ssim_adaptive_std": float(sub["ssim_adaptive"].std()),
            "ssim_gain_mean": float(sub["ssim_gain"].mean()),
            "time_euclidean_mean": float(sub["time_euclidean"].mean()),
            "time_adaptive_mean": float(sub["time_adaptive"].mean()),
            "iter_euclidean_mean": float(sub["iter_euclidean"].mean()),
            "iter_adaptive_mean": float(sub["iter_adaptive"].mean()),
        })

    df_summary = pd.DataFrame(summary_rows)

    df_per_image.to_csv(metrics_dir / "ablation_per_image.csv", index=False)
    df_comparison.to_csv(metrics_dir / "ablation_comparison_table.csv", index=False)
    df_summary.to_csv(metrics_dir / "ablation_aggregate_summary.csv", index=False)

    if config["output"].get("save_plots", True):
        plot_ablation_bars(df_summary, plots_dir / "ablation_metrics_comparison.png")
        if first_image_histories:
            plot_convergence_all_penalties(first_image_histories, plots_dir / "convergence_curves_sample.png")
        if first_clean is not None and first_noisy is not None and first_image_recons:
            plot_reconstruction_grid(first_clean, first_noisy, first_image_recons, plots_dir / "reconstruction_gallery_sample.png")

    return df_comparison, df_summary