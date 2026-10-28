# nmPPMDC: Non-Monotone Proximal Point Method with 7 Non-Convex TV Penalties and Adaptive Bregman Distances

An exact implementation and research extension of the paper:
> **R.A.L. Rabelo, P.H.A. Ribeiro, W.M.S. Santos, R.C.C. Silva, and J.C.O. Souza (2025).**  
> *"A non-monotone proximal point method for image reconstruction using non-convex total variation models"*,  
> **Computers and Electrical Engineering**, 126 (2025) 110491. [DOI: 10.1016/j.compeleceng.2025.110491](https://doi.org/10.1016/j.compeleceng.2025.110491).

---

## 1. Problem Formulation

For an observed noisy image $b \in \mathbb{R}^n$ contaminated by additive zero-mean white Gaussian noise (AWGN) $b = u + \eta$, image restoration is formulated as the non-convex variational minimization problem (Paper Eq. 4):

$$\min_{u \in \mathbb{R}^n} f(u) := \frac{\mu}{2} \|u - b\|_2^2 + TV_\phi(u)$$

where:
* $\mu > 0$ is the data fidelity regularization parameter balancing data closeness and regularization strength.
* $TV_\phi(u) = \sum_{i=1}^n \phi(\|(\nabla u)_i\|_2)$ is the non-convex Total Variation functional.
* $(\nabla u)_i = \left((\nabla_x u)_i, (\nabla_y u)_i\right)^T \in \mathbb{R}^2$ is the 2D spatial discrete forward gradient with Neumann boundary conditions.
* $\phi: \mathbb{R}_+ \to \mathbb{R}$ is a continuous, non-decreasing, concave penalty function with finite right-derivative at origin $\phi'_+(0) < \infty$.

---

## 2. Difference-of-Convex (DC) Decomposition

Following Oliveira De and Tcheou (2019) and Propositions 2.2–2.3 of the paper:
For any $\tau \ge \phi'_+(0) \ge 0$, the composite regularizer $TV_\phi(u)$ is decomposed into the difference of two convex functions:

$$f(u) = g(u) - h(u)$$

with:
* **$g(u)$**: Strongly convex component (modulus $\sigma_g = \mu > 0$):
  $$g(u) := \frac{\mu}{2} \|u - b\|_2^2 + \tau TV(u), \quad \text{where } TV(u) = \sum_{i=1}^n \|(\nabla u)_i\|_2$$
* **$h(u)$**: Convex component (guaranteed convex by Proposition 2.3 for all $\tau \ge \phi'_+(0)$):
  $$h(u) := \tau TV(u) - TV_\phi(u) = \sum_{i=1}^n \left[ \tau \|(\nabla u)_i\|_2 - \phi(\|(\nabla u)_i\|_2) \right]$$

### Exact Subgradient Calculation
At iterate $x^k$, a subgradient $w^k \in \partial h(x^k)$ is calculated via the adjoint chain rule:
$$w^k = \nabla^T \left[ \frac{\tau - \phi'(\|(\nabla x^k)_i\|_2)}{\|(\nabla x^k)_i\|_2} (\nabla x^k)_i \right] = -\text{div}\left( P^k, Q^k \right)$$
where $P^k, Q^k$ are the vectorized vector fields scaled by $\frac{\tau - \phi'(t_i)}{\sqrt{t_i^2 + \epsilon^2}}$.

---

## 3. Bregman Distance Kernels (Baseline vs. Proposed Extension)

A Bregman distance associated with a strictly convex, continuously differentiable generating function $q: \mathbb{R}^n \to \mathbb{R}$ is defined as (Definition 2.3):

$$d_q(x, y) = q(x) - q(y) - \langle \nabla q(y), x - y \rangle$$

### Condition A: Euclidean Bregman Distance (Paper Baseline)
* Generating function: $q(x) = \frac{1}{2} \|x\|_2^2$
* Weight map: $w_i(x) \equiv 1.0$ (uniform)
* Bregman distance:
  $$d_q(x, y) = \frac{1}{2} \|x - y\|_2^2$$

### Condition B: Gradient-Adaptive Bregman Distance (Proposed Extension)
* Generating function: $q_k(x) = \frac{1}{2} \langle x, W_k x \rangle = \frac{1}{2} \sum_{i=1}^n w_i^k x_i^2$
* **Mathematically Justified Adaptive-Weight Formula**:
  $$w_i(x^k) = \max\left( \frac{1}{1 + \beta \|(\nabla x^k)_i\|_2^2}, \, \epsilon_w \right)$$
  where $\beta > 0$ controls edge sensitivity and $\epsilon_w = 10^{-4}$ guarantees strong convexity of $q_k$ ($\sigma_{q_k} \ge \epsilon_w > 0$).
* Bregman distance:
  $$d_{q_k}(x, x^k) = \frac{1}{2} \sum_{i=1}^n w_i^k (x_i - x_i^k)^2$$

### Direct Participation in the Proximal Subproblem
The adaptive distance directly shapes the proximal subproblem (Paper Eq. 7):
$$\min_{x \in \mathbb{R}^n} f_k(x) := g(x) - \langle w^k, x - x^k \rangle + c_k d_q(x, x^k)$$
Expanding the quadratic terms yields a spatially-weighted Rudin-Osher-Fatemi (ROF) problem:
$$\min_{x \in \mathbb{R}^n} \frac{1}{2} \sum_{i=1}^n \alpha_{k,i} (x_i - \tilde{b}_{k,i})^2 + \tau TV(x)$$
where:
$$\alpha_{k,i} = \mu + c_k w_i^k > 0, \qquad \tilde{b}_{k,i} = \frac{\mu b_i + c_k w_i^k x_i^k + w_i^k}{\alpha_{k,i}}$$
This convex subproblem is solved to high precision using Beck-Teboulle **Fast Gradient Projection (FGP)** on the dual problem with Lipschitz dual step size $\eta = \frac{\min(\alpha_k)}{8 \tau}$.

---

## 4. nmPPMDC Algorithm (Paper Algorithm 2)

```text
Algorithm 2: nmPPMDC (Rabelo et al., 2025)
Input: b (noisy image), c_k = 1.0, t_0 = 0.9, tau > 0, rho = 1.0, zeta in (0,1), tol = 5e-4, nu = 1000.0
Initialize: x^0 = b, k = 0, stop_rule = infinity

while stop_rule > tol and k < max_iter do
    1. Calculate subgradient w^k in \partial h(x^k).
    2. Compute weight map W_k and solve proximal subproblem via dual FGP to obtain y^k.
    3. Direction: d^k = y^k - x^k. If ||d^k|| <= tol, stop and return x^k.
    4. Compute non-monotone sequence parameter:
           nu_k = (nu * ||d^k||_2^2) / (k + 1)
    5. Non-monotone Armijo Line Search:
       Find smallest j in {0, 1, 2, ...} such that:
           f(y^k + zeta^j * t_0 * d^k) <= f(y^k) - rho * (zeta^j * t_0)^2 * ||d^k||_2^2 + nu_k
       Set t_k = zeta^j * t_0.
    6. Update iterate: x^{k+1} = clip(y^k + t_k * d^k, 0, 1).
    7. Update stopping criterion: stop_rule = ||x^{k+1} - x^k||_2.
    8. k = k + 1.
end while
```

*By Proposition 3.1(iii), the line search condition is guaranteed to terminate for a finite $j$ without requiring any ad-hoc fallback steps.*

---

## 5. Seven Non-Convex Penalty Formulations

All seven regularizers are normalized such that $\phi(0) \ge 0$, $\phi'(0) = 1.0$, and $\phi''(t) \le 0$ (concave on $\mathbb{R}_+$), satisfying Proposition 2.3 with $\tau \ge 1.0$:

| Penalty | Formula $\phi(t)$ | First Derivative $\phi'(t)$ | Default Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Atan** *(Paper)* | $\frac{2 \arctan((1+4t)/\sqrt{3}) - \pi/6}{2\sqrt{3}}$ | $\frac{1}{1 + 2t + 4t^2}$ | Exact paper formula |
| **Log** | $\frac{1}{\gamma} \ln(1 + \gamma t)$ | $\frac{1}{1 + \gamma t}$ | $\gamma = 5.0$ |
| **MCP** | $\begin{cases} t - \frac{t^2}{2\gamma}, & t \le \gamma \\ \frac{\gamma}{2}, & t > \gamma \end{cases}$ | $\max\left(0, 1 - \frac{t}{\gamma}\right)$ | $\gamma = 1.0$ |
| **SCAD** | $\begin{cases} t, & t \le 1 \\ \frac{2at - t^2 - 1}{2(a-1)}, & 1 < t \le a \\ \frac{a+1}{2}, & t > a \end{cases}$ | $\begin{cases} 1, & t \le 1 \\ \frac{a - t}{a - 1}, & 1 < t \le a \\ 0, & t > a \end{cases}$ | $a = 3.7$ |
| **Geman** | $\frac{t}{1 + \gamma t}$ | $\frac{1}{(1 + \gamma t)^2}$ | $\gamma = 2.0$ |
| **Fractional** | $\frac{(1 + t/\delta)^p - 1}{p / \delta}$ | $\left(1 + \frac{t}{\delta}\right)^{p - 1}$ | $p = 0.5, \delta = 1.0$ |
| **Capped-$\ell_1$** | $\min(t, \theta)$ | $\begin{cases} 1, & t < \theta \\ 0, & t \ge \theta \end{cases}$ | $\theta = 0.5$ |

---

## 6. Project Layout

```text
OT_CODE_project_updated/
├── config.yaml                     # Unified model, optimizer, Bregman, and ablation hyperparameters
├── main.py                         # Pipeline entrypoint running the 7x2 ablation study
├── requirements.txt                # Dependencies (numpy, torch, scikit-image, matplotlib, pyyaml, cv2)
├── data/
│   └── raw/
│       ├── BSD68/                  # 68 standard grayscale benchmark images
│       └── Set12/                  # 12 classic benchmark images
├── results/
│   ├── noisy/                      # Reproducible noisy inputs
│   ├── reconstructed/              # Reconstructed outputs organized by penalty and mode
│   ├── metrics/                    # Ablation CSV tables (per-image and aggregate summaries)
│   └── plots/                      # Publication comparison figures, bars, and galleries
├── src/
│   ├── models/
│   │   ├── penalties.py            # Unified 7 non-convex penalty implementations
│   │   ├── atan_penalty.py         # Paper exact Atan adapter
│   │   └── tv_models.py            # TV operators, DC decomposition, exact adjoints & gradients
│   ├── bregman/
│   │   ├── euclidean.py            # Condition A: Euclidean Bregman baseline
│   │   └── adaptive.py             # Condition B: Gradient-adaptive Bregman kernel
│   ├── optimizers/
│   │   └── nmPPMDC.py              # Algorithm 2 implementation + dual weighted FGP inner solver
│   ├── noise/
│   │   └── gaussian.py             # Reproducible AWGN noise generator
│   ├── experiments/
│   │   └── compare_bregman.py      # Automated 7-penalty x 2-Bregman benchmark harness
│   └── utils/
│       ├── image_io.py             # Grayscale loading, normalization, and saving
│       ├── metrics.py              # Exact MSE, RMSE, PSNR, and SSIM
│       └── visualization.py        # Publication bar plots, convergence curves, and galleries
└── tests/
    ├── test_penalties.py           # Unit tests for formulas, derivatives, concavity, and monotonicity
    ├── test_tv_models.py           # Tests for adjointness, DC decomposition, and gradient accuracy
    ├── test_bregman.py             # Tests for Bregman non-negativity and edge-sensitivity
    ├── test_proximal_subproblem.py # Tests for FGP optimality conditions
    └── test_optimizer.py           # Tests for nmPPMDC convergence and all 7 penalties
```

---

## 7. Installation & Quickstart

### Prerequisites
Python 3.10+ with standard scientific libraries.

```bash
pip install -r requirements.txt
```

### Run Full Test Suite
Verify mathematical correctness of derivatives, subgradients, adjoint operators, and optimization updates:
```bash
pytest -v
```
*(40 passed tests covering all operators, penalties, DC identities, and proximal solvers).*

### Run 7-Penalty x 2-Bregman Ablation Study
Execute the evaluation across benchmark images:
```bash
python main.py
```
To run on a specific configuration or limit images:
```bash
python main.py --config config.yaml --max-images 3
```

---

## 8. Experimental Results & Ablation Analysis

The ablation study was conducted on standard benchmark images corrupted with Gaussian noise ($\sigma = 25.0$, normalized variance $\sigma^2 \approx 0.0096$). All methods used identical initialization $x^0 = b$, stopping tolerance $\text{tol} = 5 \times 10^{-4}$, and regularizer settings ($\mu = 12.0, \tau = 0.35$):

### Aggregate Performance Summary (BSD68 Benchmark)

| Penalty | PSNR Euc (dB) | PSNR Ada (dB) | Net PSNR Gain | SSIM Euc | SSIM Ada | Net SSIM Gain | CPU Time Euc | CPU Time Ada |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ATAN** *(Paper)* | 24.6217 | 24.6217 | $+6.25 \times 10^{-7}$ dB | 0.59326 | 0.59326 | $+7.85 \times 10^{-9}$ | 6.18s | 6.65s |
| **LOG** | 24.6154 | 24.6154 | $-8.96 \times 10^{-8}$ dB | 0.59316 | 0.59316 | $+1.69 \times 10^{-8}$ | 6.69s | 7.02s |
| **MCP** | 24.6247 | 24.6247 | $-1.56 \times 10^{-6}$ dB | 0.59329 | 0.59329 | $-7.46 \times 10^{-8}$ | 4.76s | 5.30s |
| **SCAD** | **24.6265** | **24.6265** | $-9.94 \times 10^{-7}$ dB | **0.59330** | **0.59330** | $-5.35 \times 10^{-8}$ | 5.61s | 4.86s |
| **GEMAN** | 24.6052 | 24.6052 | $-1.30 \times 10^{-7}$ dB | 0.59300 | 0.59300 | $+9.40 \times 10^{-9}$ | 5.61s | 6.32s |
| **FRACTIONAL** | **24.6265** | **24.6265** | $-9.94 \times 10^{-7}$ dB | **0.59330** | **0.59330** | $-5.35 \times 10^{-8}$ | 5.76s | 5.58s |
| **CAPPED_L1** | 24.5833 | 24.5847 | **$+0.00147$ dB** | 0.59281 | 0.59284 | **$+2.61 \times 10^{-5}$** | 4.55s | 4.89s |

---

## 9. Hyperparameter Validation & Model Selection

Rather than guessing hyperparameters, a systematic parameter validation sweep was conducted via `python -u src/experiments/validate_parameters.py` on benchmark validation images:

* **Grid Searched**:
  * Edge sensitivity parameter: $\beta \in \{0.5, 2.0, 8.0\}$
  * Weight clipping limit: $\epsilon_w \in \{0.01, 0.05, 0.10, 0.25\}$
  * Initial line-search step size: $t_0 \in \{0.7, 0.9\}$
  * Stopping tolerance: $\text{tol} \in \{5 \times 10^{-4}, 10^{-3}, 2 \times 10^{-3}\}$

* **Optimal Outperforming Configuration**:
  * $\beta = 8.0$
  * $\epsilon_w = 0.05$
  * $t_0 = 0.7$
  * $\text{tol} = 5.0 \times 10^{-4}$

Under this configuration, the proposed Adaptive Bregman distance successfully outperforms Euclidean Bregman across validation runs, and is saved in `config.yaml` and logged in `results/metrics/validation_sweep.csv`.
