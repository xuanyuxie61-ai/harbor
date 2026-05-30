
import numpy as np
from typing import Callable, Optional, Tuple, List


def analyze_ms_stability_region(
    lambda_range: Tuple[float, float],
    mu_range: Tuple[float, float],
    dt: float,
    n_lambda: int = 50,
    n_mu: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lvals = np.linspace(lambda_range[0], lambda_range[1], n_lambda)
    muvals = np.linspace(mu_range[0], mu_range[1], n_mu)
    L, MU = np.meshgrid(lvals, muvals)

    stable_mask = (1.0 + L * dt) ** 2 + (MU ** 2) * dt < 1.0
    return L, MU, stable_mask


def estimate_convergence_rate(
    h_vals: np.ndarray,
    err_vals: np.ndarray,
) -> Tuple[float, float, float]:

    valid = (h_vals > 0) & (err_vals > 0) & np.isfinite(h_vals) & np.isfinite(err_vals)
    if np.sum(valid) < 2:
        return 0.0, 0.0, np.inf

    log_h = np.log(h_vals[valid])
    log_e = np.log(err_vals[valid])

    A = np.vstack([np.ones(len(log_h)), log_h]).T
    sol, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)

    logC = sol[0]
    p = sol[1]
    residual = float(np.linalg.norm(A @ sol - log_e))
    return float(p), float(logC), residual


def lyapunov_exponential_decay_rate(
    t: np.ndarray,
    y: np.ndarray,
    norm_order: int = 2,
) -> float:
    norms = np.linalg.norm(y, ord=norm_order, axis=1)
    valid = (norms > 1e-12) & np.isfinite(norms)
    if np.sum(valid) < 2:
        return 0.0

    log_norms = np.log(norms[valid])
    t_valid = t[valid]


    A = np.vstack([np.ones(len(t_valid)), t_valid]).T
    sol, _, _, _ = np.linalg.lstsq(A, log_norms, rcond=None)
    lambda_est = -sol[1]
    return float(lambda_est)


def compute_maximum_lyapunov_exponent(
    f: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tspan: Tuple[float, float],
    n_steps: int = 1000,
    n_perturbations: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps


    x_ref = x0.copy()
    lyapunov_estimates = []

    for _ in range(n_perturbations):

        delta = rng.normal(0, 1e-8, len(x0))
        delta_norm0 = np.linalg.norm(delta)
        if delta_norm0 < 1e-15:
            continue

        x_pert = x_ref + delta

        for step in range(n_steps):

            x_ref = x_ref + dt * f(x_ref)

            x_pert = x_pert + dt * f(x_pert)


            delta = x_pert - x_ref
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-15:
                break
            ratio = delta_norm / delta_norm0
            lyapunov_estimates.append(np.log(ratio) / (t0 + (step + 1) * dt))
            delta = delta / delta_norm * delta_norm0
            x_pert = x_ref + delta

    if len(lyapunov_estimates) == 0:
        return 0.0
    return float(np.mean(lyapunov_estimates))


def perform_stability_sweep(
    integrator: Callable,
    lambda_list: List[float],
    mu_list: List[float],
    dt: float,
    n_paths: int = 1000,
    tmax: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    nL = len(lambda_list)
    nM = len(mu_list)
    stability = np.zeros((nL, nM), dtype=int)

    for i, lam in enumerate(lambda_list):
        for j, mu in enumerate(mu_list):

            theory_stable = (1.0 + lam * dt) ** 2 + (mu ** 2) * dt < 1.0


            n_steps = int(tmax / dt)
            x0 = 1.0
            x_final_sq = []
            for _ in range(n_paths):
                xtemp = x0
                for _ in range(n_steps):
                    dW = np.sqrt(dt) * rng.standard_normal()
                    xtemp = xtemp + lam * xtemp * dt + mu * xtemp * dW
                x_final_sq.append(xtemp ** 2)

            mean_sq = np.mean(x_final_sq)
            num_stable = mean_sq < x0 ** 2


            if num_stable:
                stability[i, j] = 1

    return stability
