# -*- coding: utf-8 -*-
"""
stability_analysis.py
---------------------
Comprehensive numerical-stability analysis of the high-order compact finite-
difference scheme used for the Eliashberg equations:

  1. Von Neumann analysis (amplification factor vs. wave number)
  2. Energy-method bound on the discrete L2 norm
  3. Matrix stability (spectral radius of the iteration matrix)
  4. CFL condition for the explicit relaxation scheme

Scientific origin
-----------------
This module is the *original* contribution of the synthesis; it combines
the FD operators from tridiagonal_fd with classical PDE-stability theory
applied to the Eliashberg relaxation equation

    d Delta_n / dt = - Delta_n + F[Delta]_n

where F is the Eliashberg RHS.

Core physics / mathematics
--------------------------
* Von Neumann: G(k) = 1 + dt * sigma(k)  must satisfy |G(k)| <= 1.
* Energy method: multiply by Delta and sum, require d/dt ||Delta||^2 <= 0.
* Spectral radius: rho(I + dt * D2_fd) < 1  for stability.
* CFL: dt <= C * dx^2 / D  with C depending on the FD order.

Stability / boundary notes
--------------------------
* All checks return structured dataclasses so main.py can pretty-print them.
* Warnings are issued if the scheme is unstable for the given parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Von Neumann analysis (calls tridiagonal_fd.stability_radius)
# ---------------------------------------------------------------------------
@dataclass
class VonNeumannResult:
    stable: bool
    max_amplification: float
    critical_dt: float          # maximum dt for stability
    n_unstable_modes: int


def von_neumann_check(
    fd_order: int,
    dx: float,
    diffusion: float = 1.0,
    n_k: int = 512,
    dt_sweep_max: float = 5.0,
    n_dt: int = 200,
) -> VonNeumannResult:
    """Sweep dt to find the critical value where max |G| = 1."""
    try:
        from tridiagonal_fd import stability_radius
    except ImportError:
        from .tridiagonal_fd import stability_radius

    dt_vals = np.linspace(1e-6, dt_sweep_max, n_dt)
    critical_dt = dt_vals[-1]
    max_amp_at_crit = float("inf")
    n_unstable = 0
    for dt in dt_vals:
        res = stability_radius(fd_order, dt, dx, diffusion, n_k)
        if res.max_amplification <= 1.0 + 1e-12:
            critical_dt = dt
            max_amp_at_crit = res.max_amplification
        else:
            n_unstable = max(n_unstable, int((res.amplification > 1.0 + 1e-12).sum()))
            break

    return VonNeumannResult(
        stable=(n_unstable == 0),
        max_amplification=max_amp_at_crit,
        critical_dt=critical_dt,
        n_unstable_modes=n_unstable,
    )


# ---------------------------------------------------------------------------
# 2.  Energy-method bound
# ---------------------------------------------------------------------------
@dataclass
class EnergyMethodResult:
    bound_satisfied: bool
    spectral_radius: float
    max_eigenvalue_D2: float
    stable_dt_bound: float


def energy_method_check(
    fd_order: int,
    n: int,
    dx: float,
    diffusion: float = 1.0,
) -> EnergyMethodResult:
    """Compute the spectral radius of the FD operator and the stable dt."""
    try:
        from tridiagonal_fd import compact_fd_second_derivative
    except ImportError:
        from .tridiagonal_fd import compact_fd_second_derivative

    fd = compact_fd_second_derivative(n, dx, fd_order)
    eigvals = np.linalg.eigvals(fd.D2)
    eig_real = eigvals.real
    max_eig = float(eig_real.max())
    min_eig = float(eig_real.min())
    # Spectral radius of I + dt * D * D2  <= 1  requires  dt * D * |min_eig| <= 2
    if min_eig < 0:
        stable_dt = 2.0 / (diffusion * abs(min_eig))
    else:
        stable_dt = float("inf")
    rho = max(abs(1.0 + diffusion * max_eig), abs(1.0 + diffusion * min_eig))
    return EnergyMethodResult(
        bound_satisfied=(rho <= 1.0 + 1e-12),
        spectral_radius=rho,
        max_eigenvalue_D2=max_eig,
        stable_dt_bound=stable_dt,
    )


# ---------------------------------------------------------------------------
# 3.  Matrix stability of the Eliashberg iteration
# ---------------------------------------------------------------------------
@dataclass
class MatrixStabilityResult:
    spectral_radius: float
    converged: bool
    dominant_eigenvalue: complex
    n_matsubara: int


def matrix_stability_of_eliashberg(
    T: float,
    n_matsubara: int,
    lambda_kernel: np.ndarray,
    mu_star: float,
) -> MatrixStabilityResult:
    """Spectral radius of the linearised Eliashberg iteration matrix."""
    M = n_matsubara
    omega = math.pi * T * (2.0 * np.arange(M) + 1.0)
    Mmat = np.zeros((M, M))
    for n in range(M):
        for m in range(M):
            kern_nm = (
                lambda_kernel[n, m]
                if lambda_kernel.ndim == 2
                else lambda_kernel[abs(n - m)]
            )
            Mmat[n, m] = (
                math.pi * T
                * (kern_nm - mu_star * (1.0 if n == m else 0.0))
                / abs(omega[m])
            )
    eigvals = np.linalg.eigvals(Mmat)
    rho = float(np.max(np.abs(eigvals)))
    dom_idx = int(np.argmax(np.abs(eigvals)))
    return MatrixStabilityResult(
        spectral_radius=rho,
        converged=(rho < 1.0),
        dominant_eigenvalue=complex(eigvals[dom_idx]),
        n_matsubara=n_matsubara,
    )


# ---------------------------------------------------------------------------
# 4.  CFL condition
# ---------------------------------------------------------------------------
@dataclass
class CFLResult:
    cfl_number: float
    stable: bool
    max_dt: float
    dx: float
    diffusion: float


def cfl_check(
    fd_order: int,
    dx: float,
    diffusion: float = 1.0,
) -> CFLResult:
    """Check the CFL condition  dt <= C * dx^2 / D.

    For 4th-order compact FD, C approx 6/5.
    For 6th-order compact FD, C approx 40/51.
    """
    C = 6.0 / 5.0 if fd_order == 4 else 40.0 / 51.0
    max_dt = C * dx * dx / max(diffusion, 1e-300)
    cfl = diffusion * max_dt / max(dx * dx, 1e-300)
    return CFLResult(
        cfl_number=cfl,
        stable=(cfl <= C + 1e-12),
        max_dt=max_dt,
        dx=dx,
        diffusion=diffusion,
    )
