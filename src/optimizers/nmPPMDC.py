"""Adaptive Bregman-Regularized Non-Monotone Proximal Point Method (nmPPMDC)."""

import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.models.tv_models import NonConvexTVModel
from src.bregman.euclidean import EuclideanBregman
from src.bregman.adaptive import AdaptiveBregman


class nmPPMDCOptimizer:
    def __init__(self, config: Dict[str, Any]):
        model_cfg = config.get("model", {})
        opt_cfg = config.get("optimizer", {})

        self.mu = float(model_cfg.get("mu", 0.15))
        self.alpha = float(model_cfg.get("alpha", 5.0))
        self.beta = float(model_cfg.get("beta", 50.0))
        self.epsilon = float(model_cfg.get("epsilon", 1e-4))

        self.max_iter = int(opt_cfg.get("max_iter", 50))
        self.tol = float(opt_cfg.get("tol", 1e-5))
        self.gamma = float(opt_cfg.get("gamma", 1e-4))
        self.sigma_ls = float(opt_cfg.get("sigma_ls", 0.5))
        self.M = int(opt_cfg.get("M", 5))
        self.step_size = float(opt_cfg.get("step_size", 0.5))

        self.model = NonConvexTVModel(mu=self.mu, alpha=self.alpha, epsilon=self.epsilon)
        self.euclidean_bregman = EuclideanBregman()
        self.adaptive_bregman = AdaptiveBregman(beta=self.beta, epsilon=self.epsilon)

    def optimize(self, f: np.ndarray, mode: str = "adaptive") -> Tuple[np.ndarray, Dict[str, Any]]:
        mode = mode.lower()
        bregman_engine = self.adaptive_bregman if mode == "adaptive" else self.euclidean_bregman

        u = np.copy(f)
        energy_history: List[float] = []
        rel_diff_history: List[float] = []

        start_time = time.perf_counter()
        current_energy = self.model.objective(u, f)
        energy_history.append(current_energy)
        iterations = 0

        for it in range(1, self.max_iter + 1):
            iterations = it
            grad = self.model.gradient(u, f)

            # Search direction
            direction = -bregman_engine.apply_metric_inv(grad, u)
            dir_deriv = float(np.sum(grad * direction))

            if dir_deriv >= 0 or np.isnan(dir_deriv):
                direction = -grad
                dir_deriv = float(np.sum(grad * direction))

            # Non-monotone threshold
            R_k = max(energy_history[-self.M:])

            # Line search
            s = self.step_size
            accepted = False
            u_trial = u
            trial_energy = current_energy
            for _ in range(12):  # Max 12 backtracking reductions
                u_trial = np.clip(u + s * direction, 0.0, 1.0)
                trial_energy = self.model.objective(u_trial, f)

                if trial_energy <= R_k + self.gamma * s * dir_deriv:
                    accepted = True
                    break
                s *= self.sigma_ls

            if not accepted:
                # If Armijo is overly strict, take a guaranteed small descent step
                s = 0.02
                u_trial = np.clip(u + s * direction, 0.0, 1.0)
                trial_energy = self.model.objective(u_trial, f)

            diff_norm = np.linalg.norm(u_trial - u)
            u_norm = np.linalg.norm(u)
            rel_diff = float(diff_norm / max(u_norm, 1e-8))
            rel_diff_history.append(rel_diff)

            u = u_trial
            energy_history.append(trial_energy)
            current_energy = trial_energy

            # Enforce minimum of 20 iterations to ensure full noise removal
            if it >= 20 and rel_diff < self.tol:
                break

        elapsed = time.perf_counter() - start_time
        history = {
            "mode": mode,
            "iterations": iterations,
            "runtime": elapsed,
            "energy_history": energy_history,
            "rel_diff_history": rel_diff_history,
        }
        return u, history