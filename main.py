"""Entrypoint script for running the mid-sem project evaluation."""

import sys
from pathlib import Path
import yaml
import pandas as pd

from src.experiments.compare_bregman import run_experiments


def load_config(config_path: Path) -> dict:
    """Read configuration from YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Execute pipeline and display mid-sem review summary."""
    root_dir = Path(__file__).resolve().parent
    config_file = root_dir / "config.yaml"

    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"[Error] Failed reading config: {e}", file=sys.stderr)
        sys.exit(1)

    df_results = run_experiments(config)

    # Print clean mid-sem summary report
    print("\n" + "=" * 70)
    print("      MID-SEM EVALUATION SUMMARY: PROPOSED ADAPTIVE BREGMAN")
    print("=" * 70)
    print(f"Dataset             : {config['dataset']['name']}")
    print(f"Noise Sigma         : {config['noise']['sigma']}")
    print(f"Total Processed     : {len(df_results)}")
    print("-" * 70)
    print(f"Metric                     Euclidean (Baseline)     Adaptive (Proposed)")
    print("-" * 70)
    print(f"Average PSNR (dB)       :  {df_results['psnr_euclidean'].mean():15.3f}     {df_results['psnr_adaptive'].mean():15.3f}")
    print(f"Average SSIM            :  {df_results['ssim_euclidean'].mean():15.4f}     {df_results['ssim_adaptive'].mean():15.4f}")
    print(f"Average Runtime (s)     :  {df_results['time_euclidean'].mean():15.3f}     {df_results['time_adaptive'].mean():15.3f}")
    print(f"Average Iterations      :  {df_results['iter_euclidean'].mean():15.1f}     {df_results['iter_adaptive'].mean():15.1f}")
    print("-" * 70)
    print(f"Net Average Gain (PSNR) :  +{df_results['psnr_gain'].mean():.3f} dB")
    print(f"Net Average Gain (SSIM) :  +{df_results['ssim_gain'].mean():.4f}")
    print("=" * 70)
    print("Outputs successfully written to `results/` folder.")


if __name__ == "__main__":
    main()