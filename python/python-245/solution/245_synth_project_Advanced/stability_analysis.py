"""
stability_analysis.py
=====================
Von Neumann stability analysis for the finite-difference schemes
used in fission dynamics.

Maps from: 390_fem1d_heat_explicit (CFL stability check,
           mass/stiffness assembly, quadrature)

Physical context
----------------
For the explicit time-stepping of the collective Schrodinger equation:
  dPsi/dt = -i/hbar * H * Psi

The von Neumann stability condition for the FTCS scheme applied to
the diffusion-like part is:

  dt * lambda_max < 2  (for pure imaginary eigenvalues)

where lambda_max is the largest eigenvalue of H.

For the heat-equation analogue (dissipative fission dynamics):
  dT/dt = D * d2T/dx2

the CFL condition is:
  D * dt / dx^2 <= 0.5

For the 6th-order spatial stencil, the modified wavenumber is:
  k_modified^2 = (1/180*h^2) * [490 - 540*cos(k*h) + 54*cos(2k*h) - 4*cos(3k*h)]

The maximum eigenvalue occurs at k*h = pi:
  k_modified_max^2 = (490 + 540 + 54 + 4) / (180 * h^2) = 1088 / (180 * h^2)
                   = 6.044 / h^2

So the stability limit becomes:
  dt <= 0.5 * h^2 / (D * 6.044)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from high_order_finite_difference import build_laplacian_matrix


def modified_wavenumber_fd2(kh: np.ndarray) -> np.ndarray:
    """
    Modified wavenumber for the 6th-order 2nd derivative stencil:

    k_mod^2 * h^2 = (1/180) * [490 - 540*cos(k*h) + 54*cos(2*k*h) - 4*cos(3*k*h)]

    Compare to exact: k^2 * h^2.
    """
    return (490.0 - 540.0 * np.cos(kh) + 54.0 * np.cos(2.0 * kh)
            - 4.0 * np.cos(3.0 * kh)) / 180.0


def modified_wavenumber_fd4(kh: np.ndarray) -> np.ndarray:
    """
    Modified wavenumber for the 4th derivative (9-point) stencil.
    Returns k_mod^4 * h^4 as a function of k*h.
    """
    # 9-point stencil applied to exp(ikx)
    c = np.array([7.0 / 240.0, -2.0 / 9.0, 169.0 / 60.0, -122.0 / 15.0,
                  91.0 / 8.0, -122.0 / 15.0, 169.0 / 60.0, -2.0 / 9.0, 7.0 / 240.0])
    result = np.zeros_like(kh)
    for j in range(9):
        offset = j - 4
        result += c[j] * np.cos(offset * kh)
    return result


def von_neumann_amplification(r: float, kh: np.ndarray) -> np.ndarray:
    """
    Amplification factor for FTCS applied to du/dt = D * d2u/dx2:
      G(k) = 1 - 4*r*sin^2(k*h/2)   (2nd-order)

    For 6th-order stencil:
      G(k) = 1 - r * k_mod^2 * h^2

    where r = D*dt/h^2.

    Stability requires |G(k)| <= 1 for all k.
    """
    k2h2 = modified_wavenumber_fd2(kh)
    return 1.0 - r * k2h2


def cfl_limit_fd6(diffusivity: float, dx: float) -> float:
    """
    Compute CFL stability limit for 6th-order spatial discretisation:
      dt_max = 0.5 * dx^2 / (D * k_mod_max^2 * h^2)

    where k_mod_max^2 * h^2 = 1088/180 = 6.044 for the 6th-order stencil.
    """
    k2_max = 1088.0 / 180.0  # = 6.0444...
    if diffusivity <= 0 or dx <= 0:
        return float('inf')
    dt_max = 0.5 * dx * dx / (diffusivity * k2_max)
    return dt_max


def cfl_limit_fd2_standard(diffusivity: float, dx: float) -> float:
    """Standard 2nd-order CFL limit: dt <= dx^2 / (2*D)."""
    if diffusivity <= 0 or dx <= 0:
        return float('inf')
    return dx * dx / (2.0 * diffusivity)


def spectral_radius(hamiltonian: np.ndarray) -> float:
    """
    Compute the spectral radius (max |eigenvalue|) of the Hamiltonian matrix.
    Uses numpy for small matrices.
    """
    eigenvalues = np.linalg.eigvals(hamiltonian)
    return float(np.max(np.abs(eigenvalues)))


def stability_check_collective(mass_param: float, dx: float,
                               v_max: float) -> Dict[str, float]:
    """
    Stability analysis for the collective Schrodinger equation:
      i*dPsi/dt = [-1/(2M)*d2/dx2 + V(x)] * Psi

    The eigenvalue spectrum of H = -1/(2M)*L + V determines stability.

    For explicit Euler: dt < 2/rho(H)
    For Crank-Nicolson: unconditionally stable (but dispersive errors)

    Returns dict with stability limits for different schemes.
    """
    # Build Hamiltonian for typical parameters
    n = 31  # small grid for analysis
    L = build_laplacian_matrix(n, dx, 'dirichlet')
    x = np.linspace(-1, 1, n)
    V_diag = v_max * np.sin(np.pi * x) ** 2  # barrier-like potential

    H = -1.0 / (2.0 * mass_param) * L + np.diag(V_diag)
    rho_H = spectral_radius(H)

    # Euler stability limit
    dt_euler = 2.0 / rho_H if rho_H > 0 else float('inf')

    # For the kinetic part alone
    hbar2_2m = 1.0 / (2.0 * mass_param)
    dt_kinetic = 2.0 * dx * dx / (hbar2_2m * 1088.0 / 180.0)

    return {
        'spectral_radius': rho_H,
        'dt_euler_max': dt_euler,
        'dt_kinetic_max': dt_kinetic,
        'dt_recommended': 0.5 * dt_euler,
        'n_grid': n,
        'dx': dx,
        'mass_param': mass_param,
    }


def dispersion_analysis_fd6(n_points: int = 1000) -> Dict[str, np.ndarray]:
    """
    Dispersion analysis comparing exact vs modified wavenumber.

    For the 6th-order stencil, the ratio k_mod^2/k^2 -> 1 as h -> 0,
    but deviates at large k*h (short wavelengths).

    Returns arrays of kh, k_mod^2*h^2, exact k^2*h^2, and error.
    """
    kh = np.linspace(0, math.pi, n_points)
    k2h2_exact = kh * kh
    k2h2_mod = modified_wavenumber_fd2(kh)

    # Relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.where(k2h2_exact > 1e-10,
                            (k2h2_mod - k2h2_exact) / k2h2_exact,
                            0.0)

    return {
        'kh': kh,
        'k2h2_exact': k2h2_exact,
        'k2h2_modified': k2h2_mod,
        'relative_error': rel_error,
    }


def compute_fem_mass_matrix_1d(n_elements: int, length: float) -> np.ndarray:
    """
    Assemble consistent mass matrix for 1D linear FEM (from 390 concept):
      M_ij = integral phi_i * phi_j dx

    For uniform mesh with element length h = length/n_elements:
      M = h/6 * [4 1 0 ...; 1 4 1 ...; ...]  (lumped: h*I)
    """
    n_nodes = n_elements + 1
    h = length / n_elements
    M = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        # Element mass matrix: h/6 * [2 1; 1 2]
        M[e, e] += 2.0 * h / 6.0
        M[e, e + 1] += h / 6.0
        M[e + 1, e] += h / 6.0
        M[e + 1, e + 1] += 2.0 * h / 6.0

    return M


def compute_fem_stiffness_matrix_1d(n_elements: int,
                                     length: float,
                                     diffusivity: float = 1.0) -> np.ndarray:
    """
    Assemble stiffness matrix for 1D FEM (from 390 concept):
      K_ij = D * integral d(phi_i)/dx * d(phi_j)/dx dx

    For uniform mesh: K = D/h * [2 -1 0 ...; -1 2 -1 ...; ...]
    """
    n_nodes = n_elements + 1
    h = length / n_elements
    K = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        K[e, e] += diffusivity / h
        K[e, e + 1] -= diffusivity / h
        K[e + 1, e] -= diffusivity / h
        K[e + 1, e + 1] += diffusivity / h

    return K


def quadrature_gauss_legendre(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre quadrature points and weights on [-1, 1].
    From 390_fem1d_heat_explicit quadrature_set concept.
    """
    points, weights = np.polynomial.legendre.leggauss(n_points)
    return points, weights


def quadrature_on_element(x_left: float, x_right: float,
                          n_quad: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform Gauss-Legendre quadrature to element [x_left, x_right].
    """
    pts_ref, wts_ref = quadrature_gauss_legendre(n_quad)
    half_width = 0.5 * (x_right - x_left)
    mid = 0.5 * (x_left + x_right)
    pts = mid + half_width * pts_ref
    wts = half_width * wts_ref
    return pts, wts
