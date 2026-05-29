"""
error_estimator.py
==================
A posteriori error estimation using Monte Carlo methods and stochastic
sampling strategies. Synthesized from monty_hall_simulation (probabilistic
analysis and conditional probability) and jumping_bean_simulation
(stochastic agent-based transport).
Provides statistical error estimators, convergence diagnostics, and
uncertainty quantification for DG solutions.
"""

import numpy as np
from typing import Callable, Tuple, List


# ---------------------------------------------------------------------------
# Monte Carlo error estimation for DG integrals
# ---------------------------------------------------------------------------

def mc_error_estimate_integral(f: Callable, domain_volume: float,
                                n_samples_list: List[int]) -> Tuple[float, float, List[float]]:
    """
    Estimate integral and its standard error using multiple Monte Carlo sample sizes.
    Implements the Monty-Hall-style probabilistic strategy: run multiple independent
    experiments and use the law of total variance.
    """
    estimates = []
    errors = []
    for n in n_samples_list:
        if n <= 0:
            continue
        samples = np.random.rand(n)
        vals = np.array([f(s) for s in samples], dtype=np.float64)
        mean = domain_volume * np.mean(vals)
        std_err = domain_volume * np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        estimates.append(mean)
        errors.append(std_err)
    # Richardson-like extrapolation
    if len(estimates) >= 2:
        refined_estimate = 2.0 * estimates[-1] - estimates[-2]
    else:
        refined_estimate = estimates[-1] if estimates else 0.0
    final_error = errors[-1] if errors else 0.0
    return refined_estimate, final_error, estimates


# ---------------------------------------------------------------------------
# Stochastic particle transport error estimator (from jumping_bean)
# ---------------------------------------------------------------------------

class StochasticParticleErrorEstimator:
    """
    Agent-based stochastic particles that sample the solution space
    and estimate local error via temperature relaxation (analogy to jumping beans).
    """
    def __init__(self, n_particles: int = 100, n_steps: int = 50):
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.positions = np.random.rand(n_particles, 3)
        self.temperatures = np.ones(n_particles, dtype=np.float64) * 20.0

    def estimate(self, solution_func: Callable,
                 reference_func: Callable,
                 domain: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
        """
        Particles perform random walks; temperature relaxes toward local error.
        Returns (mean_error, max_error).
        """
        a, b = domain
        for step in range(self.n_steps):
            for i in range(self.n_particles):
                x, y, z = self.positions[i]
                # Local ground temperature = local error
                try:
                    u_num = solution_func(x, y, z)
                    u_ref = reference_func(x, y, z)
                    if hasattr(u_num, '__len__'):
                        u_num = float(u_num[0]) if len(u_num) > 0 else 0.0
                    if hasattr(u_ref, '__len__'):
                        u_ref = float(u_ref[0]) if len(u_ref) > 0 else 0.0
                    T_ground = abs(float(u_num) - float(u_ref))
                except Exception:
                    T_ground = 0.0
                # Relax temperature toward ground
                if self.temperatures[i] < T_ground:
                    self.temperatures[i] += min(1.0, T_ground - self.temperatures[i])
                else:
                    self.temperatures[i] -= min(1.0, self.temperatures[i] - T_ground)
                # Jump probability based on temperature
                p_jump = np.clip((self.temperatures[i] - 0.0) / (50.0 + 1e-10), 0.0, 1.0)
                if np.random.rand() < p_jump:
                    self.positions[i] += 0.1 * np.random.randn(3)
                    self.positions[i] = np.mod(self.positions[i] - a, b - a) + a
        mean_error = float(np.mean(self.temperatures))
        max_error = float(np.max(self.temperatures))
        return mean_error, max_error


# ---------------------------------------------------------------------------
# Dual-weighted residual error estimator
# ---------------------------------------------------------------------------

def dual_weighted_residual_estimate(element_residuals: np.ndarray,
                                    dual_weights: np.ndarray,
                                    element_volumes: np.ndarray) -> float:
    """
    Compute dual-weighted residual (DWR) error estimate:
    eta = sum_K |rho_K(u_h)| * |z - z_h|_K * |K|
    where rho_K is the element residual and z is the dual solution.
    """
    if len(element_residuals) != len(dual_weights) or len(element_residuals) != len(element_volumes):
        raise ValueError("Array length mismatch in DWR estimator.")
    eta = np.sum(np.abs(element_residuals) * np.abs(dual_weights) * element_volumes)
    return float(eta)


# ---------------------------------------------------------------------------
# Convergence rate estimation
# ---------------------------------------------------------------------------

def estimate_convergence_rate(errors: np.ndarray, resolutions: np.ndarray) -> Tuple[float, float]:
    """
    Estimate convergence rate p from errors ~ C * h^p via log-linear regression.
    Returns (p, C).
    """
    if len(errors) < 2 or len(resolutions) < 2:
        return 0.0, 0.0
    log_h = np.log(resolutions)
    log_err = np.log(errors + 1e-30)
    # Linear regression
    n = len(log_h)
    sum_x = np.sum(log_h)
    sum_y = np.sum(log_err)
    sum_xy = np.sum(log_h * log_err)
    sum_x2 = np.sum(log_h * log_h)
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-30:
        return 0.0, 0.0
    p = (n * sum_xy - sum_x * sum_y) / denom
    logC = (sum_y - p * sum_x) / n
    return float(p), float(np.exp(logC))


# ---------------------------------------------------------------------------
# Statistical hypothesis test for convergence (Monte Carlo strategy)
# ---------------------------------------------------------------------------

def monte_carlo_convergence_test(solver_func: Callable,
                                  n_trials: int = 30,
                                  perturbation_scale: float = 1e-4) -> Tuple[float, float]:
    """
    Run solver multiple times with random perturbations to assess
    statistical stability of the solution. Returns (mean, std).
    Inspired by Monty Hall conditional probability analysis.
    """
    results = []
    for _ in range(n_trials):
        try:
            res = solver_func(perturbation_scale * np.random.randn())
            if hasattr(res, '__len__'):
                res = float(res[0]) if len(res) > 0 else 0.0
            results.append(float(res))
        except Exception:
            results.append(np.nan)
    results = np.array(results, dtype=np.float64)
    valid = ~np.isnan(results)
    if np.sum(valid) == 0:
        return 0.0, 0.0
    mean_val = float(np.mean(results[valid]))
    std_val = float(np.std(results[valid], ddof=1))
    return mean_val, std_val
