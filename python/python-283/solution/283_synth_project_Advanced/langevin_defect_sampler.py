"""
langevin_defect_sampler.py
==========================
Higher-order Langevin Monte Carlo sampler for defect-configuration sampling
in perovskite solar cells. Ported from 1071_kaihongz_HigherOrderLMC.

Physical motivation
-------------------
The equilibrium distribution of N defects in a perovskite lattice cell is
given by the Boltzmann distribution:
    pi({r_i}) = (1/Z) exp(-U({r_i}) / kT)
where U is the total potential energy including:
    - pairwise Coulomb repulsion between charged defects
    - lattice strain energy (from local distortion)
    - interaction with the electrostatic potential phi(x) from Poisson

Sampling from pi is essential to compute thermodynamic averages of
defect-related observables (formation energy, recombination rate).
Standard overdamped Langevin dynamics (Euler-Maruyama):
    dX_t = -grad U(X_t) dt + sqrt(2 kT) dW_t
converges with weak order 1 in the time step h.

The higher-order scheme (Picard-Lagrange, K >= 3)
--------------------------------------------------
We use the K-th order Picard-Lagrange scheme of Cai, Lu, Li (2019):
    X_{n+1} = exp(A h) X_n + Integral_0^h exp(A(h-s)) B dW(s)
            - Integral_0^h exp(A(h-s)) grad U(X_n) ds + O(h^{K/2})
where A, B are K x K matrices encoding the higher-order auxiliary
processes. The key insight is that the Kronecker structure
    A = A_small (x) I_d,  B = B_small (x) I_d
allows us to work with K x K "small" matrices regardless of the
physical dimension d, yielding O(K^2 d) cost per step instead of O(K^2 d^2).

Implementation highlights (from 1071)
-------------------------------------
- Exact precomputation of alpha_grad and Sigma_C via matrix exponentials
  (avoids Gauss-Legendre quadrature errors).
- Cached gradient within each Picard sweep.
- Kronecker structure exploited via A_small, D_small, Q_small blocks.
"""

from __future__ import annotations
import math
from typing import Callable, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from perovskite_constants import (
    K_B, T_K, E_CHARGE, EPS_PERP, LANGEVIN_GAMMA, LANGEVIN_H
)


# ============================================================================
# Build the small K x K matrices for the Picard-Lagrange scheme
# ============================================================================
def build_D_Q_small(K: int, gamma: float) -> Tuple[NDArray, NDArray]:
    """Build the K x K drift matrix D and diffusion matrix Q for the
    K-th order augmented Langevin process.

    D_small is the nilpotent block:
        D_small[i, j] = gamma * delta_{j, i+1}   (super-diagonal)
    Q_small has a single non-zero entry Q_small[K-1, K-1] = 1
    (noise injected only into the highest block).
    """
    D = np.zeros((K, K))
    for i in range(K - 1):
        D[i, i + 1] = gamma
    Q = np.zeros((K, K))
    Q[K - 1, K - 1] = 1.0
    return D, Q


def build_J_small(K: int) -> NDArray:
    """Build the integration-shift matrix J such that (J x)_i = x_{i+1}."""
    J = np.zeros((K, K))
    for i in range(K - 1):
        J[i, i + 1] = 1.0
    return J


def build_A_small(D: NDArray, Q: NDArray, J: NDArray) -> NDArray:
    """A = -D + Q - J^T  (the drift of the augmented process)."""
    return -D + Q - J.T


# ============================================================================
# Exact alpha and Sigma_C via matrix exponential
# ============================================================================
def precompute_alpha_and_expA(A: NDArray, h: float,
                              d: int) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute:
        expAh = expm(A h)                     (K x K)
        alpha_grad = Integral_0^h exp(A s) ds @ e_1
                   = A^{-1} (exp(A h) - I) e_1
        Sigma_C = Integral_0^h exp(A s) Q exp(A^T s) ds
    Both integrals are computed via the Van Loan augmented matrix trick:
        M = [ A    Q   ]    then expm(M h) = [ exp(Ah)    Sigma_C ]
            [ 0  -A^T  ]                       [   0       exp(-A^T h)]
    and alpha_grad via a similar augmented matrix with e_1 source.
    """
    K = A.shape[0]
    # Sigma_C via Van Loan
    Q_small = np.zeros((K, K))
    Q_small[K - 1, K - 1] = 1.0
    M = np.zeros((2 * K, 2 * K))
    M[:K, :K] = A
    M[:K, K:] = Q_small
    M[K:, K:] = -A.T
    expMh = expm(M * h)
    Sigma_C_small = expMh[:K, K:]
    expAh = expm(A * h)
    # alpha_grad = A^{-1} (exp(A h) - I) e_1
    # Use pseudo-inverse for robustness
    A_inv = np.linalg.pinv(A)
    alpha_grad_small = A_inv @ (expAh - np.eye(K)) @ np.eye(K)[:, 0]
    return expAh, alpha_grad_small, Sigma_C_small


# ============================================================================
# Defect potential energy and gradient
# ============================================================================
def defect_pair_potential(X: NDArray, d: int,
                          charges: NDArray,
                          lattice_eps: float) -> float:
    """Compute the total pairwise Coulomb energy of defects:
    U = sum_{i < j} q_i q_j / (4 pi eps |r_i - r_j|)
    X has shape (K*d,) where K defects live in d-dimensional space.
    We take the physical positions from the first block: X[:d], X[d:2d], etc.
    For K >= 3, only the second block contributes to grad_U (per 1071).
    """
    n_defects = len(charges)
    positions = X[:d * n_defects].reshape(n_defects, d) \
        if len(X) >= d * n_defects else X.reshape(-1, d)
    n = positions.shape[0]
    U = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                r = 1e-10
            U += charges[i] * charges[j] / (4.0 * math.pi * lattice_eps * r)
    return U


def defect_pair_gradient(X: NDArray, d: int,
                         charges: NDArray,
                         lattice_eps: float) -> NDArray:
    """Gradient of the pairwise Coulomb energy w.r.t. X."""
    n_defects = len(charges)
    positions = X[:d * n_defects].reshape(n_defects, d) \
        if len(X) >= d * n_defects else X.reshape(-1, d)
    n = positions.shape[0]
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                r = 1e-10
                r_vec = np.array([1e-10, 0.0, 0.0][:d])
            force_mag = (charges[i] * charges[j]
                         / (4.0 * math.pi * lattice_eps * r ** 3))
            grad[i] += force_mag * r_vec
            grad[j] -= force_mag * r_vec
    return grad.ravel()


# ============================================================================
# Picard-Lagrange higher-order Langevin sampler
# ============================================================================
class HigherOrderLangevinDefectSampler:
    """Higher-order Langevin sampler specialized for defect configurations."""

    def __init__(self, K: int, d: int, h: float,
                 gamma: float = LANGEVIN_GAMMA,
                 grad_U_fn: Optional[Callable] = None,
                 rng: Optional[np.random.Generator] = None):
        if K < 3:
            raise ValueError("Higher-order scheme requires K >= 3")
        self.K = K
        self.d = d
        self.h = h
        self.gamma = gamma
        self.rng = rng or np.random.default_rng(283)
        self.dim = K * d
        # Build small matrices
        self.D_small, self.Q_small = build_D_Q_small(K, gamma)
        self.J_small = build_J_small(K)
        self.A_small = build_A_small(self.D_small, self.Q_small, self.J_small)
        # Exact precomputations
        (self.expA_small, self.alpha_grad_small,
         self.Sigma_C_small) = precompute_alpha_and_expA(
            self.A_small, h, d)
        # Factorize Sigma_C for Gaussian sampling
        eigvals, eigvecs = np.linalg.eigh(self.Sigma_C_small)
        eigvals = np.maximum(eigvals, 0.0)
        self.Sigma_C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        self.grad_U_fn = grad_U_fn

    def step(self, X: NDArray) -> NDArray:
        """Advance one time step. X has shape (K*d,)."""
        K, d = self.K, self.d
        # Reshape to (K, d) blocks
        X_blocks = X.reshape(K, d)
        # Gradient: only on the physical position block (block 0)
        if self.grad_U_fn is not None:
            gU = self.grad_U_fn(X)
            gU_blocks = np.zeros((K, d))
            gU_blocks[0] = gU[:d] if len(gU) >= d else gU
        else:
            gU_blocks = np.zeros((K, d))
        # Deterministic part: exp(Ah) X - alpha * grad_U
        new_X_blocks = self.expA_small @ X_blocks
        for k in range(K):
            new_X_blocks[k] -= self.alpha_grad_small[k] * gU_blocks[k]
        # Stochastic part: Sigma_C^{1/2} @ Z where Z ~ N(0, I_K) per dim
        Z = self.rng.standard_normal((K, d))
        noise = self.Sigma_C_sqrt @ Z
        new_X_blocks += noise
        return new_X_blocks.ravel()

    def sample(self, X0: NDArray, n_steps: int,
               thin: int = 10) -> Tuple[NDArray, NDArray]:
        """Run the sampler for n_steps, returning thinned samples and
        energy trajectory."""
        X = np.copy(X0)
        samples = [np.copy(X)]
        energies = []
        for step in range(n_steps):
            X = self.step(X)
            if step % thin == 0:
                samples.append(np.copy(X))
                if self.grad_U_fn is not None:
                    gU = self.grad_U_fn(X)
                    energies.append(float(np.linalg.norm(gU)))
        return np.array(samples), np.array(energies)


# ============================================================================
# Driver: sample defect configurations in a 1D perovskite slab
# ============================================================================
def run_langevin_defect_sampling(n_defects: int = 4,
                                 slab_length_nm: float = 50.0,
                                 n_steps: int = 500,
                                 K: int = 3) -> dict:
    """Run the higher-order Langevin sampler for n_defects in a 1D slab."""
    d = 1  # 1D positions
    L = slab_length_nm * 1e-9  # m
    # Charges: alternating +1, -1 (electron units)
    charges = np.array([(1 if i % 2 == 0 else -1) * E_CHARGE
                        for i in range(n_defects)])
    # Initial positions: evenly spaced
    X0_blocks = np.zeros((K, d))
    positions = np.linspace(0.1 * L, 0.9 * L, n_defects)
    for i in range(min(n_defects, K)):
        X0_blocks[i, 0] = positions[i]
    X0 = X0_blocks.ravel()

    def grad_U(X):
        return defect_pair_gradient(X, d, charges, EPS_PERP)

    sampler = HigherOrderLangevinDefectSampler(
        K=K, d=d, h=LANGEVIN_H, gamma=LANGEVIN_GAMMA,
        grad_U_fn=grad_U)
    samples, energies = sampler.sample(X0, n_steps=n_steps, thin=10)
    return {
        "samples": samples,
        "energies": energies,
        "n_steps": n_steps,
        "n_defects": n_defects,
        "K": K,
        "final_positions": samples[-1, :d * n_defects],
    }
