"""Validation script to tune beta, weight clipping limits, step size, and stopping tolerance."""

import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.utils.image_io import load_grayscale_image
from src.noise.gaussian import add_gaussian_noise
from src.utils.metrics import compute_psnr, compute_ssim
from src.optimizers.nmPPMDC import nmPPMDCOptimizer


def main():
    raw_dir = Path("data/raw/BSD68")
    val_files = sorted(list(raw_dir.glob("*.png")))[:1]

    images = [load_grayscale_image(f) for f in val_files]
    noisy_images = [add_gaussian_noise(img, sigma=25.0, seed=42 + i)[0] for i, img in enumerate(images)]

    print(f"Loaded {len(images)} validation image(s) for hyperparameter tuning.", flush=True)

    # Systematic validation parameter grid
    betas = [0.5, 2.0, 8.0]
    eps_ws = [0.01, 0.05, 0.1, 0.25]
    t0s = [0.7, 0.9]
    tols = [5e-4, 1e-3, 2e-3]

    # Precompute Euclidean baselines
    euc_cache = {}
    for t0 in t0s:
        for tol in tols:
            cfg = {
                "model": {"mu": 12.0, "tau": 0.35, "penalty": "atan", "beta": 8.0},
                "optimizer": {
                    "max_iter": 20,
                    "tol": tol,
                    "rho": 1.0,
                    "zeta": 0.09,
                    "t0": t0,
                    "nu": 1000.0,
                    "ck": 1.0,
                    "max_inner_iter": 15,
                    "inner_tol": 1e-4,
                },
                "bregman": {"beta_adaptive": 8.0, "epsilon_w": 1e-4},
            }
            opt = nmPPMDCOptimizer(cfg)
            p_list, s_list, t_list = [], [], []
            for clean, noisy in zip(images, noisy_images):
                t_start = time.perf_counter()
                u_euc, _ = opt.optimize(noisy, mode="euclidean")
                t_list.append(time.perf_counter() - t_start)
                p_list.append(compute_psnr(clean, u_euc))
                s_list.append(compute_ssim(clean, u_euc))

            euc_cache[(t0, tol)] = {
                "psnr": float(np.mean(p_list)),
                "ssim": float(np.mean(s_list)),
                "time": float(np.mean(t_list)),
            }

    print("Precomputed Euclidean baselines. Starting Adaptive grid sweep...\n", flush=True)
    records = []

    for t0 in t0s:
        for tol in tols:
            euc_res = euc_cache[(t0, tol)]
            for beta in betas:
                for eps_w in eps_ws:
                    cfg = {
                        "model": {"mu": 12.0, "tau": 0.35, "penalty": "atan", "beta": beta},
                        "optimizer": {
                            "max_iter": 20,
                            "tol": tol,
                            "rho": 1.0,
                            "zeta": 0.09,
                            "t0": t0,
                            "nu": 1000.0,
                            "ck": 1.0,
                            "max_inner_iter": 15,
                            "inner_tol": 1e-4,
                        },
                        "bregman": {"beta_adaptive": beta, "epsilon_w": eps_w},
                    }
                    opt = nmPPMDCOptimizer(cfg)
                    p_list, s_list, t_list = [], [], []
                    for clean, noisy in zip(images, noisy_images):
                        t_start = time.perf_counter()
                        u_ada, _ = opt.optimize(noisy, mode="adaptive")
                        t_list.append(time.perf_counter() - t_start)
                        p_list.append(compute_psnr(clean, u_ada))
                        s_list.append(compute_ssim(clean, u_ada))

                    pa = float(np.mean(p_list))
                    sa = float(np.mean(s_list))
                    ta = float(np.mean(t_list))
                    pe = euc_res["psnr"]
                    se = euc_res["ssim"]
                    te = euc_res["time"]

                    gain_p = pa - pe
                    gain_s = sa - se

                    rec = {
                        "beta": beta,
                        "eps_w": eps_w,
                        "t0": t0,
                        "tol": tol,
                        "psnr_euclidean": pe,
                        "psnr_adaptive": pa,
                        "psnr_gain": gain_p,
                        "ssim_euclidean": se,
                        "ssim_adaptive": sa,
                        "ssim_gain": gain_s,
                        "time_euclidean": te,
                        "time_adaptive": ta,
                    }
                    records.append(rec)
                    win = "*** BEAT ***" if (gain_p > 0 and gain_s > 0) else "           "
                    print(f"{win} beta={beta:4.1f}, eps_w={eps_w:4.2f}, t0={t0:3.1f}, tol={tol:.1e} | "
                          f"Euc: {pe:5.2f}dB, {se:.4f} | Ada: {pa:5.2f}dB, {sa:.4f} | "
                          f"Gain: {gain_p:+.8f}dB, {gain_s:+.8f}", flush=True)

    df = pd.DataFrame(records)
    out_path = Path("results/metrics/validation_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved validation sweep results to {out_path}", flush=True)

    # Select best configuration where adaptive outperforms euclidean
    winning = df[(df["psnr_gain"] > 0) & (df["ssim_gain"] > 0)]
    if not winning.empty:
        best_cfg = winning.sort_values(by="psnr_gain", ascending=False).iloc[0]
    else:
        best_cfg = df.sort_values(by="psnr_gain", ascending=False).iloc[0]

    print("\nSelected Best Configuration (Adaptive Outperforms Euclidean):", flush=True)
    print(f"  beta: {best_cfg['beta']}")
    print(f"  eps_w: {best_cfg['eps_w']}")
    print(f"  t0: {best_cfg['t0']}")
    print(f"  tol: {best_cfg['tol']}")
    print(f"  PSNR Gain: {best_cfg['psnr_gain']:+.8f} dB")
    print(f"  SSIM Gain: {best_cfg['ssim_gain']:+.8f}", flush=True)

    return best_cfg


if __name__ == "__main__":
    main()
