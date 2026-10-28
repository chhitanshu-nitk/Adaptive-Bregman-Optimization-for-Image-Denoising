"""Entrypoint for the nmPPMDC 7-penalty x 2-Bregman ablation study."""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd

from src.experiments.compare_bregman import run_experiments


def load_config(config_path: Path) -> dict:
    """Read YAML configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Execute ablation study pipeline and display summary."""
    parser = argparse.ArgumentParser(description="nmPPMDC Non-Convex TV Reconstruction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    config_file = root_dir / args.config

    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"[Error] Failed reading config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.max_images is not None:
        config["dataset"]["max_images"] = args.max_images

    df_comparison, df_summary = run_experiments(config)

    # Print summary report
    print("\n" + "=" * 80)
    print("      7-PENALTY x 2-BREGMAN ABLATION SUMMARY (nmPPMDC)")
    print("=" * 80)
    print(f"Dataset         : {config['dataset']['name']}")
    print(f"Noise Sigma     : {config['noise']['sigma']} (AWGN)")
    print(f"Regularization  : mu={config['model']['mu']}, tau={config['model']['tau']}")
    print("-" * 80)
    print(f"{'Penalty':<12} | {'PSNR Euc':<10} {'PSNR Ada':<10} {'Gain':<8} | {'SSIM Euc':<9} {'SSIM Ada':<9} | {'Time Euc':<8} {'Time Ada':<8}")
    print("-" * 80)

    for _, row in df_summary.iterrows():
        p_name = str(row["penalty"]).upper()
        print(f"{p_name:<12} | {row['psnr_euclidean_mean']:10.2f} {row['psnr_adaptive_mean']:10.2f} {row['psnr_gain_mean']:+8.2f} | "
              f"{row['ssim_euclidean_mean']:9.4f} {row['ssim_adaptive_mean']:9.4f} | {row['time_euclidean_mean']:7.2f}s {row['time_adaptive_mean']:7.2f}s")

    print("-" * 80)
    best_psnr = df_summary.loc[df_summary["psnr_adaptive_mean"].idxmax()]
    best_ssim = df_summary.loc[df_summary["ssim_adaptive_mean"].idxmax()]
    print(f"Best PSNR : {best_psnr['penalty'].upper()} ({best_psnr['psnr_adaptive_mean']:.2f} dB)")
    print(f"Best SSIM : {best_ssim['penalty'].upper()} ({best_ssim['ssim_adaptive_mean']:.4f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()