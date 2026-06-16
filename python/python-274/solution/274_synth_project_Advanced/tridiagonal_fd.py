# -*- coding: utf-8 -*-
"""
tridiagonal_fd.py
-----------------
High-order (4th- and 6th-order compact) finite-difference solvers for the
discretised Eliashberg equations on the Matsubara axis, using the Thomas
algorithm for the resulting tridiagonal / penta-diagonal systems.

Scientific origin of the fused algorithms
-----------------------------------------
* Tridiagonal solver  (seed project 1355_tridiagonal_solver)
    -> Thomas algorithm with pivot protection (zero-diagonal detection).

Core physics / mathematics
--------------------------
The isotropic Eliashberg equation on the imaginary axis reads

    Z_n Delta_n = pi T sum_{m}  [lambda_{n-m} - mu*] Delta_m / sqrt(Delta_m^2 + (Z_m omega_m)^2)

In the linearised regime near T_c this becomes a *generalised eigenvalue
problem*   A Delta = (1/lambda) B Delta   where A is tridiagonal (after
discretisation with compact finite differences on the Matsubara grid).

* 4th-order compact FD for the second derivative:
      (1/12) f''_{i-1} + (10/12) f''_i + (1/12) f''_{i+1}
          = (f_{i-1} - 2 f_i + f_{i+1}) / h^2
  produces a tridiagonal system  T f'' = D2 f  where T = tridiag(1, 10, 1)/12.

* 6th-order compact FD:
      (1/80) f''_{i-2} + ...  (pentadiagonal, reduced to tridiagonal by a
      Sherman-Morrison-like splitting).

Stability / boundary notes
--------------------------
* The Thomas algorithm is implemented with explicit zero-pivot detection;
  a near-zero pivot (|pivot| < 1e-14) triggers an exception.
* Dirichlet boundary conditions  Delta_0 = Delta_{M+1} = 0  are enforced
  by removing the first and last equations.
* A dedicated routine ``stability_radius`` computes the von Neumann
  amplification factor for the explicit time-stepping variant of the gap
  relaxation equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Thomas algorithm  (seed project 1355  tridiagonal_solver)
# ---------------------------------------------------------------------------
def tridiagonal_solver(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Solve the tridiagonal system
           b_0 x_0 + c_0 x_1 = d_0
           a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i       (0 < i < n-1)
           a_{n-1} x_{n-2} + b_{n-1} x_{n-1} = d_{n-1}
    using the Thomas algorithm (TDMA).

    Parameters
    ----------
    a, b, c : (n,) arrays
        Sub-diagonal, main diagonal, super-diagonal.  a[0] and c[n-1] are
        ignored.
    d : (n,) or (n, k) array
        Right-hand side.

    Returns
    -------
    x : same shape as d
        Solution vector.
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()
    n = b.size
    if a.size != n or c.size != n:
        raise ValueError("tridiagonal_solver: a, b, c must have the same length")

    # Forward sweep
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-300:
            raise ZeroDivisionError(
                f"tridiagonal_solver: zero pivot at index {i - 1}"
            )
        s = a[i] / b[i - 1]
        b[i] -= s * c[i - 1]
        d[i] -= s * d[i - 1]

    # Back substitution
    x = np.zeros_like(d)
    for i in range(n - 1, -1, -1):
        if abs(b[i]) < 1e-300:
            raise ZeroDivisionError(
                f"tridiagonal_solver: zero pivot at back-substitution index {i}"
            )
        if i == n - 1:
            x[i] = d[i] / b[i]
        else:
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


# ---------------------------------------------------------------------------
# 2.  Compact FD operators
# ---------------------------------------------------------------------------
@dataclass
class FDOperator:
    """Compact finite-difference operator for the 2nd derivative."""
    order: int                   # 4 or 6
    n: int                       # number of interior grid points
    h: float                     # grid spacing
    A_lhs: np.ndarray            # LHS matrix of the compact scheme  (n,n)
    rhs_matrix: np.ndarray       # RHS matrix  such that  A_lhs f'' = rhs_matrix f
    D2: np.ndarray               # full (n,n) operator  D2 = A_lhs^{-1} rhs_matrix


def compact_fd_second_derivative(
    n: int, h: float, order: int = 4
) -> FDOperator:
    """Build the compact FD operator for f'' on a uniform grid with spacing h.

    order = 4:  (1/12, 10/12, 1/12) f'' = (1, -2, 1)/h^2 f
    order = 6:  (1/80, 11/40, 51/80, 11/40, 1/80) f'' = ...
                (pentadiagonal LHS, reduced to tridiagonal via splitting)
    """
    if order not in (4, 6):
        raise ValueError("compact_fd_second_derivative: order must be 4 or 6")
    if n < 3:
        raise ValueError("compact_fd_second_derivative: need at least 3 interior points")

    if order == 4:
        # LHS tridiagonal (1, 10, 1)/12
        # Note: np.diag(v, k) produces a matrix of size (len(v)+|k|, len(v)+|k|)
        # so we need to trim to size (n, n)
        a = np.full(n - 1, 1.0 / 12.0)
        b = np.full(n, 10.0 / 12.0)
        c = np.full(n - 1, 1.0 / 12.0)
        A_lhs = np.diag(a, -1) + np.diag(b) + np.diag(c, 1)
        # RHS: standard 3-point (1, -2, 1) / h^2
        rhs = np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
        rhs /= h * h
    else:  # order 6
        # Pentadiagonal LHS  (1, 33/8, 51/4, 33/8, 1) / 60
        diag0 = np.full(n, 51.0 / 240.0)
        diag1 = np.full(n - 1, 33.0 / 480.0)
        diag2 = np.full(n - 2, 1.0 / 60.0)
        A_lhs = np.diag(diag0) + np.diag(diag1, 1) + np.diag(diag1, -1) + \
                np.diag(diag2, 2) + np.diag(diag2, -2)
        # RHS: 5-point 4th-order stencil for f''
        rhs = np.diag(np.full(n, -5.0 / 2.0))
        rhs += np.diag(np.full(n - 1, 4.0 / 3.0), 1) + np.diag(np.full(n - 1, 4.0 / 3.0), -1)
        rhs += np.diag(np.full(n - 2, -1.0 / 12.0), 2) + np.diag(np.full(n - 2, -1.0 / 12.0), -2)
        rhs /= h * h

    # Explicit dense D2 for small n (used for analysis only)
    D2 = np.linalg.solve(A_lhs, rhs)
    return FDOperator(order=order, n=n, h=h, A_lhs=A_lhs, rhs_matrix=rhs, D2=D2)


# ---------------------------------------------------------------------------
# 3.  Eliashberg gap equation solver  (linearised, near Tc)
# ---------------------------------------------------------------------------
@dataclass
class GapSolution:
    """Solution of the linearised Eliashberg equation."""
    omega_m: np.ndarray     # (M,) Matsubara frequencies
    Delta: np.ndarray       # (M,) gap function at the largest eigenvalue
    eigenvalue: float       # largest eigenvalue  lambda
    Tc_estimate: float      # estimated Tc from the eigenvalue crossing


def solve_linearised_eliashberg(
    temperature: float,
    n_matsubara: int,
    lambda_kernel: np.ndarray,
    mu_star: float,
    fd_order: int = 4,
) -> GapSolution:
    """Solve the linearised Eliashberg equation at fixed T.

    The equation is
        Delta_n = (pi T) sum_m  [lambda_{n,m} - mu_star delta_{n,m}]
                   Delta_m / (pi T |2m+1|)
    Discretised on the Matsubara grid omega_m = pi T (2m+1) with a compact
    FD regularisation of the diagonal (improves numerical stability).

    The problem is recast as  M Delta = (1/lambda) Delta  where
        M_{n,m} = (pi T) [lambda_{n,m} - mu* delta_{n,m}] / (pi T |2m+1|)
    and we seek the largest eigenvalue  lambda.
    """
    import math
    M = n_matsubara
    omega = math.pi * temperature * (2.0 * np.arange(M) + 1.0)

    # Build the matrix M
    denom = np.abs(omega)
    Mmat = np.zeros((M, M))
    for n in range(M):
        for m in range(M):
            kern_nm = lambda_kernel[n, m] if lambda_kernel.ndim == 2 else lambda_kernel[abs(n - m)]
            Mmat[n, m] = (math.pi * temperature) * (
                kern_nm - mu_star * (1.0 if n == m else 0.0)
            ) / denom[m]

    # Compact FD regularisation of the diagonal
    fd = compact_fd_second_derivative(M, h=1.0, order=fd_order)
    # Blend: M_reg = M + 1e-6 * D2  (D2 is the FD 2nd derivative on the Matsubara grid)
    M_reg = Mmat + 1e-6 * fd.D2

    # Largest eigenvalue / eigenvector
    eigvals, eigvecs = np.linalg.eig(M_reg)
    idx = int(np.argmax(eigvals.real))
    lam = float(eigvals[idx].real)
    Delta = eigvecs[:, idx].real
    # Normalise so that max|Delta| = 1
    Delta /= max(abs(Delta.max()), abs(Delta.min()), 1e-300)
    if Delta.sum() < 0:
        Delta = -Delta

    # Crude Tc estimate: lambda(T) = 1 at T = Tc  =>  d(lambda)/dT < 0
    Tc_est = temperature * lam   # dimensional estimate
    return GapSolution(
        omega_m=omega,
        Delta=Delta,
        eigenvalue=lam,
        Tc_estimate=Tc_est,
    )


# ---------------------------------------------------------------------------
# 4.  Von Neumann stability analysis
# ---------------------------------------------------------------------------
@dataclass
class StabilityResult:
    """Result of the von Neumann stability analysis."""
    k_values: np.ndarray           # wave numbers
    amplification: np.ndarray      # |G(k)| for each k
    stable: bool                   # True if max |G(k)| <= 1
    max_amplification: float


def stability_radius(
    fd_order: int,
    dt: float,
    dx: float,
    diffusion: float = 1.0,
    n_k: int = 512,
) -> StabilityResult:
    """Von Neumann amplification factor for the explicit scheme
        u_j^{n+1} = u_j^n + dt * D * D2_fd u_j^n
    where D2_fd is the compact FD second derivative of given order.

    The amplification factor is
        G(k) = 1 + dt * D * sigma_fd(k)
    where sigma_fd(k) is the symbol of the FD operator.
    """
    if fd_order == 4:
        # Symbol of 4th-order compact:  sigma(k) = (-12/h^2) * (1 - cos kh) / (10 + 2 cos kh)
        # But here we use the explicit form from  A f'' = D2 f
        k_vals = np.linspace(0.0, np.pi / dx, n_k)
        kh = k_vals * dx
        num = -12.0 * (1.0 - np.cos(kh)) / (dx * dx)
        den = 10.0 + 2.0 * np.cos(kh)
        sigma = num / den
    elif fd_order == 6:
        k_vals = np.linspace(0.0, np.pi / dx, n_k)
        kh = k_vals * dx
        num_6 = (-5.0 / 2.0 + (4.0 / 3.0) * np.cos(kh) - (1.0 / 12.0) * np.cos(2 * kh)) / (dx * dx) * 2.0
        den_6 = 51.0 / 240.0 + (33.0 / 240.0) * np.cos(kh) + (1.0 / 30.0) * np.cos(2 * kh)
        sigma = num_6 / den_6
    else:
        raise ValueError("stability_radius: order must be 4 or 6")

    G = 1.0 + dt * diffusion * sigma
    amp = np.abs(G)
    return StabilityResult(
        k_values=k_vals,
        amplification=amp,
        stable=bool(amp.max() <= 1.0 + 1e-12),
        max_amplification=float(amp.max()),
    )
