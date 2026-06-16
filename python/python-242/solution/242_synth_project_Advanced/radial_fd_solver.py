"""
radial_fd_solver.py  --  Finite-difference solution of the radial
                         Schrödinger equation
===========================================================================
We solve the reduced radial Schrödinger equation for u(r) = r R(r):

    - (hbar^2 / 2 m_red) d^2 u / dr^2 + V_eff(r) u = E u

with
    V_eff(r) = V_central(r) + V_so(r) + [V_C(r) if proton] + l(l+1) hbar^2 / (2 m_red r^2)

using an O(h^2) symmetric finite-difference discretisation on a uniform mesh
r_i = i h,  i = 0, ..., N.

Discretisation
--------------
    - u_{i-1} + (2 + h^2 (2m/hbar^2) V_eff(r_i)) u_i - u_{i+1}
        = (2m/hbar^2) E h^2 u_i

Or equivalently, in matrix form:
    H u = lambda u   with lambda = (2m / hbar^2) E

    H_{ii}     = 2/h^2 + (2m/hbar^2) V_eff(r_i)
    H_{i,i+1}  = H_{i+1,i} = -1/h^2

Boundary conditions (physical):
    u_0 = 0       (regularity at origin, u(r) ~ r^{l+1} near 0)
    u_N = 0       (box quantisation, R_max >> R_Woods-Saxon)

The eigensolver targets the LOWEST eigenvalues of H, which correspond to
the MOST BOUND physical states. The mesh must be fine enough that the
de Broglie wavelength of the deepest state is resolved by at least 10
mesh points; we use h ~ 0.06 fm for a well of depth ~ 50 MeV.

For the TIME-INDEPENDENT problem this yields a standard symmetric
eigenvalue problem, which we solve with scipy's eigh_banded or
eigh_tridiagonal.

References:
    Coz, M., Ann. Phys. (NY) 44 (1967) 286  (Numerov scheme)
    Buck, Perey, Phys. Rev. Lett. 48 (1982) 568  (Woods-Saxon parametrisation)
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import (
    HBAR_C, M_PROTON, M_NEUTRON, MAX_RADIAL_GRID, NR_RADIAL, R_EPSILON,
)
from woods_saxon_potential import total_single_particle_potential, centrifugal_barrier


def reduced_mass_nucleon(A_core: int, is_proton: bool) -> float:
    """Reduced mass of nucleon + core in MeV/c^2."""
    m_core = A_core * 0.5 * (M_PROTON + M_NEUTRON)
    m_nuc = M_PROTON if is_proton else M_NEUTRON
    return m_core * m_nuc / (m_core + m_nuc)


def build_radial_grid(
    r_max: float = MAX_RADIAL_GRID, n_r: int = NR_RADIAL,
) -> tuple[np.ndarray, float]:
    """Uniform radial mesh [0, r_max] with n_r points; returns (r, h)."""
    r = np.linspace(0.0, r_max, n_r)
    h = r[1] - r[0]
    return r, h


def effective_potential_vector(
    r: np.ndarray,
    A: int,
    Z_core: int,
    l_q: int,
    j_q: float,
    is_proton: bool,
    m_red: float,
) -> np.ndarray:
    """Effective radial potential V_eff(r) = V_total + centrifugal."""
    hbar2_over_2m = (HBAR_C ** 2) / (2.0 * m_red)
    V = total_single_particle_potential(r, A, Z_core, l_q, j_q, is_proton)
    V_cent = centrifugal_barrier(r, l_q, hbar2_over_2m)
    return V + V_cent


def fd_hamiltonian_tridiag(
    r: np.ndarray,
    V_eff: np.ndarray,
    m_red: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the symmetric tridiagonal finite-difference Hamiltonian.

    H_{ii}    = 2/h^2 + (2m/hbar^2) V_eff(r_i)
    H_{i,i+1} = -1/h^2

    Only the interior points i = 1, ..., N-1 are retained (boundary
    conditions u_0 = u_N = 0 imposed by removing the first and last row
    and column).

    Returns (diag, off) for use with scipy.linalg.eigh_tridiagonal.
    """
    h = r[1] - r[0]
    N = len(r) - 1
    h2 = h * h
    k_factor = 2.0 * m_red / (HBAR_C ** 2)
    # Interior diagonal
    diag = 2.0 / h2 + k_factor * V_eff[1:N]
    off = np.full(N - 2, -1.0 / h2)
    return diag, off


def solve_tridiag_lowest(
    diag: np.ndarray, off: np.ndarray, n_eig: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lowest n_eig eigenvalues and eigenvectors of the
    symmetric tridiagonal matrix (diag, off).

    We target the LOWEST (most negative) eigenvalues since these
    correspond to the most-bound physical states.
    """
    from scipy.linalg import eigh_tridiagonal
    n = len(diag)
    if n == 0:
        return np.array([]), np.zeros((0, 0))
    n_eig = min(n_eig, n)
    eigs, vecs = eigh_tridiagonal(
        diag, off, select="i", select_range=(0, n_eig - 1),
    )
    return eigs, vecs


def solve_radial_levels(
    A: int,
    Z_core: int,
    l_q: int,
    j_q: float,
    is_proton: bool,
    n_r: int = NR_RADIAL,
    r_max: float = MAX_RADIAL_GRID,
    n_levels: int = 6,
) -> dict:
    """Solve for the lowest n_levels bound states of orbital (l, j).

    Returns a dict with keys:
        'energies'      : array of MeV eigenvalues, sorted ascending
        'wavefunctions' : shape (n_levels, n_r), reduced u(r) = r R(r)
        'r'             : radial mesh
        'norms'         : L2 norms (unity after normalisation)
        'n_nodes'       : number of radial nodes per level
        'h'             : mesh spacing in fm
        'V_eff'         : effective potential on the full mesh (for diagnostics)
    """
    m_red = reduced_mass_nucleon(A - 1, is_proton)
    r, h = build_radial_grid(r_max, n_r)
    V_eff = effective_potential_vector(r, A, Z_core, l_q, j_q, is_proton, m_red)
    diag, off = fd_hamiltonian_tridiag(r, V_eff, m_red)
    lambda_eig, vecs = solve_tridiag_lowest(diag, off, n_eig=n_levels)
    # Convert eigenvalues to MeV: lambda = (2m / hbar^2) E  =>  E = lambda hbar^2 / (2 m)
    conv = (HBAR_C ** 2) / (2.0 * m_red)
    energies = lambda_eig * conv
    # Reconstruct full u(r) with boundary values
    u_all = np.zeros((len(energies), n_r))
    norms = np.zeros(len(energies))
    n_nodes = np.zeros(len(energies), dtype=int)
    for n in range(len(energies)):
        v_int = vecs[:, n]
        u_full = np.zeros(n_r)
        u_full[1:n_r - 1] = v_int
        # Normalise: int |u(r)|^2 dr = 1
        norm_sq = np.trapz(u_full * u_full, r)
        if norm_sq < R_EPSILON:
            norm_sq = R_EPSILON
        u_full = u_full / math.sqrt(norm_sq)
        # Count nodes (zero crossings, excluding endpoints)
        signs = np.sign(u_full[1:-1])
        # Remove exact zeros for node-counting purposes
        signs = np.where(signs == 0, 1, signs)
        n_nodes[n] = int(np.sum(np.abs(np.diff(signs)) > 0))
        u_all[n, :] = u_full
        norms[n] = 1.0
    return {
        "energies": energies,
        "wavefunctions": u_all,
        "r": r,
        "norms": norms,
        "n_nodes": n_nodes,
        "h": h,
        "V_eff": V_eff,
    }


def assign_nl_to_energies(results_list: list) -> list:
    """Merge (l, j, result_dict) entries into a sorted level scheme."""
    levels = []
    for idx, (l, j, res) in enumerate(results_list):
        for n in range(len(res["energies"])):
            levels.append({
                "E_MeV": float(res["energies"][n]),
                "l": l,
                "j": j,
                "n_rad": int(res["n_nodes"][n]),
                "idx": idx,
                "n_level": n,
            })
    levels.sort(key=lambda x: x["E_MeV"])
    return levels
