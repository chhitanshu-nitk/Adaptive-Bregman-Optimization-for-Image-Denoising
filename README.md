# nmPPMDC_AdaptiveBregman

Adaptive Bregman **N**on-**M**onotone **P**roximal **P**oint **M**ethod for a
**D**ifference-of-**C**onvex (nmPPMDC) formulation of non-convex Total
Variation (TV) image reconstruction — Objective 2 implementation
(Gradient-Adaptive Bregman Distance Kernel).

## Problem

We solve, for a noisy image `b` and regularization parameter `mu > 0`,

```
min_u  (mu/2) ||u - b||^2 + TV_phi(u)
```

where `TV_phi(u) = sum_i phi_Atan(||(grad u)_i||)` uses the non-convex
arctangent penalty `phi_Atan(t) = (alpha/beta) atan(beta t)`, which is
concave and edge-preserving (large gradients/edges are only lightly
penalized).

### DC (Difference-of-Convex) split

Because `phi_Atan` is concave, `TV_phi` itself is a **concave** (non-convex)
functional of `u`. We use the identity

```
TV_phi(u) = TV(u) - h(u),   TV(u) = sum_i ||(grad u)_i||        (convex, standard TV)
                             h(u)  = sum_i rho(||(grad u)_i||)   (convex)
                             rho(t) := t - phi_Atan(t)           (convex, since phi_Atan concave)
```

so that the objective is `f(u) = g(u) - h(u)` with

```
g(u) = (mu/2)||u-b||^2 + TV(u)     [convex]
h(u) = sum_i rho(||(grad u)_i||)   [convex]
```

This is exactly the DC split implemented in `src/models/tv_models.py`.

### Objective 2 — Gradient-Adaptive Bregman Kernel

Instead of the ordinary Euclidean distance, the proximal subproblem uses a
spatially-varying quadratic Bregman kernel

```
psi(u) = (1/2) integral w(x) u(x)^2 dx,     w(x) = 1 / (1 + beta ||grad u(x)||)
```

To keep the proximal subproblem convex, the weight map is **frozen (lagged)**
at the previous outer iterate: `w^k(x) := w(x; u^k)`. Then

```
D_psi(u, u^k) = (1/2) sum_i w^k_i (u_i - u^k_i)^2
```

* Flat regions (small `||grad u^k||`) → `w^k -> 1` → strong regularization / noise suppression.
* Edges (large `||grad u^k||`) → `w^k -> 0` → relaxed regularization / edge preservation.

This is implemented in `src/bregman/adaptive.py`, with the baseline
Euclidean kernel (`w(x) == 1`) in `src/bregman/euclidean.py` for comparison.

## Algorithm 1 (`src/optimizers/nmPPMDC.py`)

At each outer iteration `k`:

1. **Weight map**: freeze `w^k = kernel.weight_map(u^k)`.
2. **Subgradient**: `xi^k in d(h)(u^k)` via the chain rule
   (`tv_models.subgradient_h`).
3. **Candidate solve** (Eq. 4):
   `u_cand = argmin_u  g(u) - <xi^k,u> + (1/lambda_k) D_psi(u,u^k)`,
   which reduces to a pointwise-weighted ROF/TV-denoising problem, solved
   with a generalized Chambolle-style **weighted dual projection**
   (projected gradient ascent on the dual TV variable).
4. **Direction**: `d^k = u_cand - u^k`.
5. **Non-monotone line search** (Eq. 5): backtracking on `t in (0,1]` until
   `f(u^k + t d^k) <= max_{j=max(0,k-M)}^{k} f(u^j) - beta_0 t ||d^k||^2`.
6. **Update**: `u^{k+1} = u^k + t_k d^k`.
7. **Stop** when `||u^{k+1} - u^k|| < tol`.

## Repository layout

```
nmPPMDC_AdaptiveBregman/
├── config.yaml              # all experiment/optimizer hyperparameters
├── main.py                  # runs Condition A (Euclidean) vs B (Adaptive), logs + plots
├── requirements.txt
├── data/
│   ├── raw/                 # put Set12 / BSD68 images here (optional — see below)
│   └── results/             # PNG visualizations, convergence curves, metrics_log.csv
└── src/
    ├── models/
    │   ├── tv_models.py      # grad, divergence (exact adjoint), DC split (g, h)
    │   └── penalty_funcs.py  # phi_Atan and its derivative/subgradient pieces
    ├── bregman/
    │   ├── euclidean.py      # Condition A: w(x) == 1
    │   └── adaptive.py       # Condition B: w(x) = 1/(1+beta||grad u||), lagged
    ├── optimizers/
    │   └── nmPPMDC.py        # Algorithm 1 solver + weighted dual-projection inner solver
    └── utils/
        ├── image_io.py       # benchmark loading (+ synthetic fallback), Gaussian noise
        └── metrics.py        # PSNR, SSIM, metrics table
```

## Running

```bash
pip install -r requirements.txt
python main.py                       # uses config.yaml
python main.py --config config.yaml --verbose
```

If `data/raw/` is empty (no Set12/BSD68 images present), `main.py`
automatically falls back to a small set of deterministic synthetic
piecewise-smooth test images so the full pipeline (noise, both solvers,
metrics, plots) is always runnable end to end. To use real benchmarks, drop
Set12 / BSD68 grayscale images (`.png`/`.jpg`/...) into `data/raw/`.

Outputs land in `data/results/`:
* `<image>_reconstruction.png` — Noisy input | frozen `w(x)` | reconstruction, per condition.
* `<image>_convergence.png` — `f(u^k)` vs. iteration, both conditions overlaid.
* `metrics_log.csv` — PSNR / SSIM / iteration counts for every (image, condition) pair.

## Notes on parameters (`config.yaml`)

* `objective.alpha <= 1` is required for `rho(t) = t - phi_Atan(t)` to remain
  convex (needed for the DC split `g - h` to be valid).
* `bregman.beta_adaptive` controls how aggressively the adaptive kernel
  relaxes near edges; `beta_adaptive -> 0` recovers the Euclidean kernel.
* `optimizer.lambda_k` is the proximal step size in the Bregman term of
  Eq. 4; `optimizer.memory_M` / `beta_0` control the non-monotone line
  search (Eq. 5).
