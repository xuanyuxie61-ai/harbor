# -*- coding: utf-8 -*-
"""
================================================================================
Cable Diffusion Module for Synaptic Protein Transport
================================================================================

This module implements discrete Laplacian operators for modeling the
diffusion of plasticity-related proteins (PRPs) along dendritic cables.

Mathematical Model:
-------------------
The dendritic cable equation for membrane potential V(x,t):

    λ² · ∂²V/∂x² = τ_m · ∂V/∂t + V - r_m · I_syn(x,t)

where:
    λ  = sqrt(r_m / r_a)      [space constant, μm]
    τ_m = r_m · c_m           [time constant, ms]
    r_m = membrane resistance [Ω·cm²]
    r_a = axial resistance    [Ω·cm]
    c_m = membrane capacitance[μF/cm²]

For protein concentration c(x,t) (e.g., CamKII, PKMζ, BDNF):

    ∂c/∂t = D · ∂²c/∂x² + f(c) - γ·c + S(x,t)

where:
    D   = diffusion coefficient [μm²/ms]
    γ   = degradation rate      [1/ms]
    f(c)= local synthesis rate  [concentration/ms]
    S   = synaptic source term  [concentration/ms]

Discrete Laplacian (1D, Dirichlet-Dirichlet):
---------------------------------------------
For N grid points with spacing h, the discrete Laplacian L ∈ ℝ^(N×N) is:

    L_ii     =  2/h²        for i = 1,...,N
    L_{i,i+1}= -1/h²        for i = 1,...,N-1
    L_{i,i-1}= -1/h²        for i = 2,...,N

This corresponds to the finite-difference approximation:

    ∂²c/∂x² ≈ (c_{i-1} - 2c_i + c_{i+1}) / h²

Boundary Conditions:
--------------------
- Dirichlet-Dirichlet (DD): c_0 = c_{N+1} = 0
- Dirichlet-Neumann (DN): c_0 = 0, ∂c/∂x|_{N+1} = 0
- Neumann-Dirichlet (ND): ∂c/∂x|_0 = 0, c_{N+1} = 0
- Neumann-Neumann (NN): ∂c/∂x = 0 at both ends
- Periodic (PP): c_0 = c_N, c_{N+1} = c_1

The module also provides Cholesky, eigenvalue, and LU decompositions
for stability analysis of the implicit time-stepping schemes.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional


def build_laplacian_1d(n: int, h: float, bc: str = "DD") -> np.ndarray:
    """
    Build the 1D discrete Laplacian matrix with specified boundary conditions.

    Parameters
    ----------
    n : int
        Number of interior grid points. Must be >= 3.
    h : float
        Grid spacing. Must be positive.
    bc : str
        Boundary condition type: 'DD', 'DN', 'ND', 'NN', or 'PP'.

    Returns
    -------
    L : np.ndarray
        The (n, n) Laplacian matrix.

    Raises
    ------
    ValueError
        If n < 3 or h <= 0 or bc is invalid.
    """
    if n < 3:
        raise ValueError(f"Number of points n={n} must be >= 3.")
    if h <= 0.0:
        raise ValueError(f"Grid spacing h={h} must be positive.")
    valid_bcs = ("DD", "DN", "ND", "NN", "PP")
    if bc not in valid_bcs:
        raise ValueError(f"Boundary condition '{bc}' not in {valid_bcs}.")

    L = np.zeros((n, n))
    inv_h2 = 1.0 / (h * h)

    if bc == "DD":
        # Dirichlet at both ends: standard tridiagonal
        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    elif bc == "DN":
        # Dirichlet at left, Neumann at right
        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        # Neumann: ghost point c_{n+1} = c_n  =>  ∂c/∂x = 0
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 1.0 * inv_h2

    elif bc == "ND":
        # Neumann at left, Dirichlet at right
        # Ghost point c_0 = c_1
        L[0, 0] = 1.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    elif bc == "NN":
        # Neumann at both ends
        L[0, 0] = 1.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 1.0 * inv_h2

    elif bc == "PP":
        # Periodic
        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        L[0, n - 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, 0] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    return L


def apply_laplacian_1d(n: int, h: float, u: np.ndarray, bc: str = "DD") -> np.ndarray:
    """
    Apply the 1D discrete Laplacian to a vector without forming the full matrix.

    For large dendritic trees, matrix-free application reduces memory from O(n²)
    to O(n) and improves cache efficiency.

    Parameters
    ----------
    n : int
        Number of grid points.
    h : float
        Grid spacing.
    u : np.ndarray
        Input vector of length n.
    bc : str
        Boundary condition type.

    Returns
    -------
    Lu : np.ndarray
        The Laplacian applied to u.
    """
    if u.shape[0] != n:
        raise ValueError(f"Input length {u.shape[0]} does not match n={n}.")
    if h <= 0.0:
        raise ValueError("Grid spacing h must be positive.")

    Lu = np.zeros_like(u)
    inv_h2 = 1.0 / (h * h)

    if bc == "DD":
        Lu[0] = (2.0 * u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + 2.0 * u[n - 1]) * inv_h2
    elif bc == "DN":
        Lu[0] = (2.0 * u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + u[n - 1]) * inv_h2
    elif bc == "ND":
        Lu[0] = (u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + 2.0 * u[n - 1]) * inv_h2
    elif bc == "NN":
        Lu[0] = (u[0] - u[1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + u[n - 1]) * inv_h2
    elif bc == "PP":
        Lu[0] = (2.0 * u[0] - u[1] - u[n - 1]) * inv_h2
        Lu[1:n - 1] = (-u[0:n - 2] + 2.0 * u[1:n - 1] - u[2:n]) * inv_h2
        Lu[n - 1] = (-u[n - 2] + 2.0 * u[n - 1] - u[0]) * inv_h2
    else:
        raise ValueError(f"Invalid boundary condition: {bc}")

    return Lu


def cable_diffusion_step(
    c: np.ndarray,
    D: float,
    h: float,
    dt: float,
    gamma: float,
    source: Optional[np.ndarray] = None,
    bc: str = "DD",
) -> np.ndarray:
    """
    Perform one forward-Euler time step of the cable diffusion equation:

        c^{n+1} = c^n + dt · [D · L(c^n) - γ · c^n + S]

    Stability requires: dt ≤ h² / (2D)

    Parameters
    ----------
    c : np.ndarray
        Current protein concentration.
    D : float
        Diffusion coefficient [μm²/ms]. Must be non-negative.
    h : float
        Grid spacing [μm].
    dt : float
        Time step [ms]. Must be positive.
    gamma : float
        Degradation rate [1/ms]. Must be non-negative.
    source : np.ndarray, optional
        Synaptic source term S(x).
    bc : str
        Boundary condition.

    Returns
    -------
    c_new : np.ndarray
        Concentration at next time step.
    """
    if D < 0.0:
        raise ValueError("Diffusion coefficient D must be non-negative.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")
    if gamma < 0.0:
        raise ValueError("Degradation rate gamma must be non-negative.")

    n = c.shape[0]
    Lu = apply_laplacian_1d(n, h, c, bc)

    c_new = c + dt * (D * Lu - gamma * c)
    if source is not None:
        if source.shape[0] != n:
            raise ValueError("Source term length must match concentration length.")
        c_new = c_new + dt * source

    # Clip negative concentrations (physical constraint)
    c_new = np.maximum(c_new, 0.0)

    return c_new


def laplacian_eigenvalues(n: int, h: float, bc: str = "DD") -> np.ndarray:
    """
    Compute eigenvalues of the 1D discrete Laplacian.

    For Dirichlet boundary conditions, the eigenvalues are:

        λ_k = (2/h²) · [1 - cos(kπ/(n+1))],  k = 1,2,...,n

    These determine the stability and decay rates of diffusion modes.

    Parameters
    ----------
    n : int
        Number of grid points.
    h : float
        Grid spacing.
    bc : str
        Boundary condition.

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues in ascending order.
    """
    if n < 3:
        raise ValueError("n must be >= 3.")
    if h <= 0.0:
        raise ValueError("h must be positive.")

    L = build_laplacian_1d(n, h, bc)
    eigvals = np.linalg.eigvalsh(L)
    return eigvals


def stability_limit(n: int, h: float, D: float, bc: str = "DD") -> float:
    """
    Compute the CFL stability limit for explicit Euler diffusion:

        dt_max = h² / (2D)

    For safety, we use the more conservative estimate based on the
    maximum eigenvalue magnitude.

    Parameters
    ----------
    n : int
        Number of grid points.
    h : float
        Grid spacing.
    D : float
        Diffusion coefficient.
    bc : str
        Boundary condition.

    Returns
    -------
    dt_max : float
        Maximum stable time step.
    """
    if D <= 0.0:
        raise ValueError("D must be positive for stability analysis.")
    eigvals = laplacian_eigenvalues(n, h, bc)
    lambda_max = np.max(np.abs(eigvals))
    dt_max = 2.0 / (D * lambda_max) if lambda_max > 0 else np.inf
    return dt_max


def simulate_protein_diffusion(
    n: int = 64,
    length: float = 100.0,
    D: float = 0.1,
    gamma: float = 0.01,
    dt: float = 0.1,
    t_final: float = 50.0,
    bc: str = "DD",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate diffusion of plasticity-related proteins along a dendritic cable.

    Initial condition: Gaussian pulse at center representing synaptic stimulation.

    Parameters
    ----------
    n : int
        Number of spatial grid points.
    length : float
        Dendrite length [μm].
    D : float
        Diffusion coefficient [μm²/ms].
    gamma : float
        Degradation rate [1/ms].
    dt : float
        Time step [ms].
    t_final : float
        Final simulation time [ms].
    bc : str
        Boundary condition.

    Returns
    -------
    x : np.ndarray
        Spatial grid coordinates.
    t : np.ndarray
        Time points.
    c_history : np.ndarray
        Concentration history of shape (nt, n).
    """
    if n < 3:
        raise ValueError("n must be >= 3.")
    if length <= 0.0:
        raise ValueError("length must be positive.")
    if dt <= 0.0 or t_final <= 0.0:
        raise ValueError("dt and t_final must be positive.")

    h = length / (n + 1.0)
    x = np.linspace(h, length - h, n)

    # Check stability
    dt_max = stability_limit(n, h, D, bc)
    if dt > dt_max:
        dt = dt_max * 0.9
        print(f"[cable_diffusion] dt adjusted to {dt:.4f} for stability (limit={dt_max:.4f})")

    nt = int(np.ceil(t_final / dt))
    t = np.linspace(0.0, t_final, nt + 1)

    # Initial condition: Gaussian pulse at center
    x0 = length / 2.0
    sigma0 = length / 20.0
    c = np.exp(-((x - x0) ** 2) / (2.0 * sigma0 ** 2))

    # Synaptic source: localized at x = 0.3L and x = 0.7L
    source = np.zeros(n)
    source += 0.5 * np.exp(-((x - 0.3 * length) ** 2) / (2.0 * (h * 2.0) ** 2))
    source += 0.5 * np.exp(-((x - 0.7 * length) ** 2) / (2.0 * (h * 2.0) ** 2))

    c_history = np.zeros((nt + 1, n))
    c_history[0, :] = c

    for step in range(nt):
        c = cable_diffusion_step(c, D, h, dt, gamma, source, bc)
        c_history[step + 1, :] = c

    return x, t, c_history


if __name__ == "__main__":
    x, t, c_hist = simulate_protein_diffusion()
    print(f"Cable diffusion: max concentration at t={t[-1]:.1f} ms = {np.max(c_hist[-1]):.6f}")
