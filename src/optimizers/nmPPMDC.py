"""Non-Monotone Proximal Point Method for Difference-of-Convex functions (nmPPMDC).

Implements Algorithm 2 from Rabelo et al. (Computers & Electrical Engineering, 2025).
"""

import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.models.tv_models import NonConvexTVModel, forward_gradient, backward_divergence
from src.models.penalties import BasePenalty, get_penalty
from src.bregman.euclidean import EuclideanBregman
from src.bregman.adaptive import AdaptiveBregman


def solve_weighted_rof_fgp(
    b_tilde: np.ndarray,
    alpha: np.ndarray,
    tau: float,
    max_inner_iter: int = 30,
    tol: float = 1e-4,
    dual_init: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Solve weighted TV proximal subproblem using Fast Gradient Projection (FGP)."""
    m, n = b_tilde.shape
    inv_alpha = 1.0 / alpha
    step_dual = float(np.min(alpha)) / (8.0 * tau)

    if dual_init is not None:
        p_k, q_k = np.copy(dual_init[0]), np.copy(dual_init[1])
    else:
        p_k = np.zeros((m, n), dtype=np.float64)
        q_k = np.zeros((m, n), dtype=np.float64)

    p_v, q_v = np.copy(p_k), np.copy(q_k)
    t_k = 1.0

    for inner_it in range(1, max_inner_iter + 1):
        # Primal candidate
        div_v = backward_divergence(p_v, q_v)
        x = b_tilde + tau * inv_alpha * div_v

        # Dual gradient step
        p_grad, q_grad = forward_gradient(x)
        p_next = p_v + step_dual * p_grad
        q_next = q_v + step_dual * q_grad

        # Projection onto unit ball
        mag = np.sqrt(p_next ** 2 + q_next ** 2)
        norm_factor = np.maximum(1.0, mag)
        p_next /= norm_factor
        q_next /= norm_factor

        diff_dual = np.sqrt(np.sum((p_next - p_k) ** 2 + (q_next - q_k) ** 2))
        norm_dual = np.sqrt(np.sum(p_next ** 2 + q_next ** 2)) + 1e-12

        # FISTA momentum update
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (t_k ** 2)))
        momentum = (t_k - 1.0) / t_next

        p_v = p_next + momentum * (p_next - p_k)
        q_v = q_next + momentum * (q_next - q_k)

        p_k, q_k = p_next, q_next
        t_k = t_next

        if diff_dual / norm_dual < tol and inner_it >= 10:
            break

    div_final = backward_divergence(p_k, q_k)
    x = b_tilde + tau * inv_alpha * div_final
    return x, (p_k, q_k)


class nmPPMDCOptimizer:
    """nmPPMDC optimizer supporting Euclidean and Adaptive Bregman distances."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config.get("model", {})
        opt_cfg = config.get("optimizer", {})
        bregman_cfg = config.get("bregman", {})

        # Model hyperparameters
        self.mu = float(model_cfg.get("mu", 0.5))
        self.tau = float(model_cfg.get("tau", 1.0))
        self.epsilon = float(model_cfg.get("epsilon", 1e-6))
        self.penalty_name = str(model_cfg.get("penalty", "atan")).lower()

        # Optimizer hyperparameters
        self.max_iter = int(opt_cfg.get("max_iter", 200))
        self.tol = float(opt_cfg.get("tol", 5e-4))
        self.rho = float(opt_cfg.get("rho", 1.0))
        self.zeta = float(opt_cfg.get("zeta", 0.09))
        self.t0 = float(opt_cfg.get("t0", 0.9))
        self.nu = float(opt_cfg.get("nu", 1000.0))
        self.ck = float(opt_cfg.get("ck", 1.0))
        self.max_ls_iter = int(opt_cfg.get("max_ls_iter", 25))

        # Inner solver hyperparameters
        self.max_inner_iter = int(opt_cfg.get("max_inner_iter", 30))
        self.inner_tol = float(opt_cfg.get("inner_tol", 1e-4))

        # Adaptive Bregman hyperparameters
        self.beta = float(model_cfg.get("beta", bregman_cfg.get("beta_adaptive", 8.0)))
        self.epsilon_w = float(bregman_cfg.get("epsilon_w", 1e-4))

        self.euclidean_bregman = EuclideanBregman()
        self.adaptive_bregman = AdaptiveBregman(beta=self.beta, epsilon_w=self.epsilon_w)

    def _build_model(self, penalty: Optional[Union[str, BasePenalty]] = None) -> NonConvexTVModel:
        p = penalty if penalty is not None else self.penalty_name
        return NonConvexTVModel(
            mu=self.mu,
            tau=self.tau,
            penalty=p,
            epsilon=self.epsilon
        )

    def optimize(
        self,
        b: np.ndarray,
        mode: str = "adaptive",
        penalty: Optional[Union[str, BasePenalty]] = None,
        x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run nmPPMDC optimization on noisy image b."""
        mode = mode.lower().strip()
        bregman_engine = self.adaptive_bregman if mode == "adaptive" else self.euclidean_bregman
        model = self._build_model(penalty)

        x = np.copy(b) if x0 is None else np.copy(x0)
        x = np.clip(x, 0.0, 1.0)

        stop_rule = float("inf")
        k = 0
        t_prev = self.t0

        energy_history = []
        stop_rule_history = []
        step_sizes = []
        ls_iterations = []

        energy_history.append(model.objective(x, b))
        dual_warmstart = None
        start_time = time.perf_counter()

        while stop_rule > self.tol and k < self.max_iter:
            # Subgradient w^k in partial h(x^k)
            w_k = model.subgradient_h(x)

            # Bregman weights W_k
            weights_k = bregman_engine.compute_weights(x)

            # Proximal subproblem parameters
            alpha_k = self.mu + self.ck * weights_k
            b_tilde_k = (self.mu * b + self.ck * weights_k * x + w_k) / alpha_k

            # Solve proximal subproblem via FGP
            y_k, dual_warmstart = solve_weighted_rof_fgp(
                b_tilde=b_tilde_k,
                alpha=alpha_k,
                tau=model.tau,
                max_inner_iter=self.max_inner_iter,
                tol=self.inner_tol,
                dual_init=dual_warmstart
            )

            # Direction d^k = y^k - x^k
            d_k = y_k - x
            d_norm = float(np.linalg.norm(d_k))
            if d_norm <= 1e-12:
                break

            # Non-monotone line search parameter
            d_norm_sq = float(np.sum(d_k ** 2))
            nu_k = (self.nu * d_norm_sq) / float(k + 1)

            # Backtracking Armijo search
            f_yk = model.objective(y_k, b)
            t_cand = t_prev
            j_k = 0
            accepted = False
            x_next = y_k

            for j in range(self.max_ls_iter):
                step = (self.zeta ** j) * t_cand
                x_trial = y_k + step * d_k
                f_trial = model.objective(x_trial, b)

                threshold = f_yk - self.rho * (step ** 2) * d_norm_sq + nu_k
                if f_trial <= threshold:
                    x_next = x_trial
                    t_cand = step
                    j_k = j
                    accepted = True
                    break

            if not accepted:
                x_next = y_k
                t_cand = 0.0

            x_next = np.clip(x_next, 0.0, 1.0)
            stop_rule = float(np.linalg.norm(x_next - x))

            x = x_next
            energy_history.append(model.objective(x, b))
            stop_rule_history.append(stop_rule)
            step_sizes.append(t_cand)
            ls_iterations.append(j_k)

            t_prev = self.t0
            k += 1

        elapsed = time.perf_counter() - start_time

        history = {
            "mode": mode,
            "penalty": model.penalty.name,
            "iterations": k,
            "runtime": elapsed,
            "final_stop_rule": stop_rule,
            "energy_history": energy_history,
            "stop_rule_history": stop_rule_history,
            "step_sizes": step_sizes,
            "ls_iterations": ls_iterations,
        }
        return x, history