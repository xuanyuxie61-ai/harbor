# -*- coding: utf-8 -*-
"""
inverse_kappa.py
----------------
Inverse problem: infer the radial profile of the parallel diffusion
coefficient  kappa_||(r)  from "observed" cosmic-ray intensities at
a set of detector positions.

The forward model is the focused Parker transport equation solved
on the (r, mu) grid by explicit time integration; the inverse is
posed as a least-squares problem

    min_{kappa}  J(kappa) = (1/2) sum_i (f_model(r_i) - f_obs(r_i))^2
                            + (lambda / 2) || L kappa ||^2

with a Tikhonov regularisation of the second radial derivative of
log kappa.  The parameter vector is  theta_j = log kappa(r_j),
j = 0, ..., N-1.

The optimiser is a hand-written Adam loop (Kingma & Ba 2014) with
Polyak averaging (inspired by the Glow optimiser module) --
the averaging provides a low-noise trajectory of the parameters
during the final iterations and greatly improves robustness on
the ill-conditioned inverse problem.

Gradient computation uses central finite differences in parameter
space (no adjoint code required for the small-scale reproducible
experiments targeted here).
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional, Tuple

import cosmic_ray_physics as crp
import heliocentric_mesh as hm


# =====================================================================
# Synthetic "observations"
# =====================================================================
def synthetic_observations(r_det: np.ndarray,
                           kappa_true_func: Callable[[float], float],
                           mu_idx: int = 8,
                           noise_rel: float = 0.02,
                           seed: int = 7) -> np.ndarray:
    """Generate synthetic observations of f(r, mu=mu_idx) using the
    true kappa profile and add log-normal noise.
    """
    import focused_transport_eq as fte
    mu = hm.pitch_angle_mesh(Nmu=max(2 * mu_idx + 4, 16))
    Ek = 1.0e9 * crp.q_e
    op = fte.FocusedTransportOperator(
        r_det, mu, Ek, kappa_par_func=kappa_true_func,
        stencil_order="upwind2")
    f = np.ones((r_det.size, mu.size)) * 1.0
    hm.apply_all_boundaries(f,
                            np.ones(mu.size) * 1.0,
                            np.zeros(mu.size))
    dt = op.cfl_timestep() * 0.5
    n_steps = min(int(1.0 / dt) if dt > 0 else 100, 500)
    for _ in range(n_steps):
        dfdt = op(f)
        f = f + dt * dfdt
        hm.apply_all_boundaries(f,
                                np.ones(mu.size) * 1.0,
                                np.zeros(mu.size))
    f_obs = f[:, min(mu_idx, mu.size - 1)].copy()
    rng = np.random.default_rng(seed)
    f_obs *= np.exp(noise_rel * rng.standard_normal(f_obs.size))
    return f_obs


# =====================================================================
# Forward model
# =====================================================================
def forward_model(r: np.ndarray, mu: np.ndarray,
                  kappa_func: Callable[[float], float],
                  Ek: float,
                  n_steps: int = 500,
                  stencil: str = "upwind2") -> np.ndarray:
    """Solve the forward problem and return f(r, mu)."""
    import focused_transport_eq as fte
    op = fte.FocusedTransportOperator(
        r, mu, Ek, kappa_par_func=kappa_func, stencil_order=stencil)
    f = np.ones((r.size, mu.size)) * 1.0
    hm.apply_all_boundaries(f, np.ones(mu.size), np.zeros(mu.size))
    dt = op.cfl_timestep() * 0.5
    for _ in range(n_steps):
        dfdt = op(f)
        f = f + dt * dfdt
        hm.apply_all_boundaries(f, np.ones(mu.size), np.zeros(mu.size))
    return f


# =====================================================================
# Loss and gradient
# =====================================================================
def loss_function(theta: np.ndarray, r: np.ndarray, mu: np.ndarray,
                  Ek: float, f_obs: np.ndarray,
                  lambda_reg: float = 1.0e-4,
                  n_steps: int = 500) -> float:
    """Return the regularised least-squares loss.

    theta[j] = log kappa(r_j).
    """
    kappa_r = np.exp(theta)
    def kfunc(rr):
        return float(np.interp(rr, r, kappa_r))
    f = forward_model(r, mu, kfunc, Ek, n_steps=n_steps)
    # compare at detector mu
    j_det = mu.size // 2
    residual = f[:, j_det] - f_obs
    data_term = 0.5 * float(np.sum(residual ** 2))
    # Tikhonov on second derivative of log kappa
    d2theta = theta[:-2] - 2.0 * theta[1:-1] + theta[2:]
    reg_term = 0.5 * lambda_reg * float(np.sum(d2theta ** 2))
    return data_term + reg_term


def loss_gradient(theta: np.ndarray, r: np.ndarray, mu: np.ndarray,
                  Ek: float, f_obs: np.ndarray,
                  lambda_reg: float = 1.0e-4,
                  n_steps: int = 500,
                  eps: float = 1.0e-3) -> np.ndarray:
    """Central-difference gradient of the loss w.r.t. theta."""
    g = np.zeros_like(theta)
    L0 = loss_function(theta, r, mu, Ek, f_obs, lambda_reg, n_steps)
    for k in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp[k] += eps
        tm[k] -= eps
        Lp = loss_function(tp, r, mu, Ek, f_obs, lambda_reg, n_steps)
        Lm = loss_function(tm, r, mu, Ek, f_obs, lambda_reg, n_steps)
        g[k] = (Lp - Lm) / (2.0 * eps)
    return g


# =====================================================================
# Adam + Polyak averaging
# =====================================================================
class AdamPolyak:
    """Adam optimiser with Polyak (exponential moving average) of the
    parameters, inspired by the Glow codebase.

    The Polyak average  theta_bar_k = beta2 * theta_bar_{k-1}
                                    + (1 - beta2) theta_k
    provides a low-noise estimator of the parameters during the
    final iterations.
    """

    def __init__(self, theta0: np.ndarray, alpha: float = 1.0e-2,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1.0e-8):
        self.theta = theta0.copy()
        self.theta_bar = theta0.copy()
        self.m = np.zeros_like(theta0)
        self.v = np.zeros_like(theta0)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def step(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        self.theta = self.theta - self.alpha * m_hat / (
            np.sqrt(v_hat) + self.epsilon)
        # Polyak averaging
        self.theta_bar = self.beta2 * self.theta_bar + (
            1.0 - self.beta2) * self.theta
        return self.theta.copy(), self.theta_bar.copy()


# =====================================================================
# Inverse solver
# =====================================================================
def solve_inverse(r: np.ndarray, mu: np.ndarray, Ek: float,
                  f_obs: np.ndarray,
                  theta0: Optional[np.ndarray] = None,
                  n_iter: int = 20,
                  alpha: float = 1.0e-2,
                  lambda_reg: float = 1.0e-4,
                  n_steps: int = 100,
                  verbose: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                                  list]:
    """Minimise  J(theta)  with Adam+Polyak.

    Returns (theta_opt, theta_bar_opt, loss_history).
    """
    if theta0 is None:
        theta0 = np.full(r.size, np.log(1.0e22))
    opt = AdamPolyak(theta0, alpha=alpha)
    history = []
    for it in range(n_iter):
        g = loss_gradient(opt.theta, r, mu, Ek, f_obs,
                          lambda_reg=lambda_reg, n_steps=n_steps)
        theta, theta_bar = opt.step(g)
        L = loss_function(theta, r, mu, Ek, f_obs,
                          lambda_reg=lambda_reg, n_steps=n_steps)
        history.append(L)
        if verbose:
            print(f"  iter {it:3d}  L = {L:.4e}  "
                  f"|g| = {np.linalg.norm(g):.3e}")
    return theta, theta_bar, history


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    r = hm.logarithmic_radial_mesh(Nr=16)
    mu = hm.pitch_angle_mesh(Nmu=8)
    Ek = 1.0e9 * crp.q_e
    kappa_true = lambda rr: 1.0e22 * (rr / crp.AU) ** 0.3
    f_obs = synthetic_observations(r, kappa_true, mu_idx=4, seed=3)
    theta0 = np.full(r.size, np.log(5.0e21))
    theta, theta_bar, hist = solve_inverse(
        r, mu, Ek, f_obs, theta0=theta0,
        n_iter=3, n_steps=50, verbose=True)
    print(f"[inverse_kappa] loss history: {hist}")


if __name__ == "__main__":
    _demo()
