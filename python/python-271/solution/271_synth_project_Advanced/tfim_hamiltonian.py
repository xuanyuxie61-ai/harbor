# -*- coding: utf-8 -*-
"""
Exact diagonalization of the 1D Transverse-Field Ising Model

    H = - J * sum_{i=1}^{L-1} sigma^z_i sigma^z_{i+1}  -  h * sum_{i=1}^{L} sigma^x_i

Optionally periodic boundary conditions (PBC) may be selected,
adding the coupling - J sigma^z_L sigma^z_1.

The Hilbert space dimension is 2^L; we restrict to L <= 16 so that
the full dense spectrum can be computed reproducibly on a laptop.
This is the small-scale reproducible-experiment regime required by
the project charter.

We also expose the Jordan-Wigner mapped free-fermion form so that
large-L ground-state energies can be obtained in O(L^3) via an
elliptic-determinant evaluation (see ``elliptic_determinants.py``).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigh

try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# sigma matrices
# ---------------------------------------------------------------------------
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
EYE2 = np.eye(2, dtype=np.complex128)


def _kron_n(matrices: list[np.ndarray]) -> np.ndarray:
    """Tensor product of a list of 2x2 matrices.

    We hand-roll the loop instead of functools.reduce(np.kron, ...)
    so that intermediate blow-up is explicit and the routine stays
    numerically stable for L up to 14 on modest hardware.
    """
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out


def build_hamiltonian_dense(L: int, J: float = 1.0, h: float = 1.0,
                             periodic: bool = False) -> np.ndarray:
    """Construct the full 2^L x 2^L Hamiltonian as a dense Hermitian.

    The matrix is real symmetric because sigma^z sigma^z and sigma^x
    are both real in the computational basis; we therefore store it
    as float64 to halve memory and accelerate diagonalisation.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1, got L={L}")
    if L > 16:
        raise ValueError(f"L={L} exceeds the exact-diag cap (16). "
                         "Use the JW fermion route for larger systems.")

    dim = 1 << L
    H = np.zeros((dim, dim), dtype=np.float64)

    # ZZ couplings
    n_bonds = L if periodic else (L - 1)
    for b in range(n_bonds):
        j2 = (b + 1) % L
        ops = [EYE2] * L
        ops[b] = SIGMA_Z
        ops[j2] = SIGMA_Z
        H -= J * _kron_n(ops).real

    # Transverse field
    for i in range(L):
        ops = [EYE2] * L
        ops[i] = SIGMA_X
        H -= h * _kron_n(ops).real

    # Symmetrise to kill round-off asymmetry (important for eigh)
    H = 0.5 * (H + H.T)
    return H


# ---------------------------------------------------------------------------
# Spectrum cache
# ---------------------------------------------------------------------------
_SPECTRUM_CACHE = {}
_GS_VECTOR_CACHE = {}


def _cache_key(L, J, h, periodic):
    return (int(L), float(J), float(h), bool(periodic))


def spectrum(L: int, J: float = 1.0, h: float = 1.0,
              periodic: bool = False) -> np.ndarray:
    """Return the sorted eigenvalues of H(L, J, h).  Cached."""
    key = _cache_key(L, J, h, periodic)
    if key in _SPECTRUM_CACHE:
        return _SPECTRUM_CACHE[key]
    H = build_hamiltonian_dense(L, J, h, periodic)
    ev = np.linalg.eigvalsh(H)
    _SPECTRUM_CACHE[key] = ev
    return ev


def ground_state_energy(L: int, J: float, h: float,
                         periodic: bool = False) -> float:
    """Ground-state energy E_0(L, J, h).  Returns a plain float."""
    ev = spectrum(L, J, h, periodic)
    return float(ev[0])


def _ground_state_vector(L: int, J: float, h: float,
                           periodic: bool = False) -> np.ndarray:
    """Cached ground-state eigenvector."""
    key = _cache_key(L, J, h, periodic)
    if key in _GS_VECTOR_CACHE:
        return _GS_VECTOR_CACHE[key]
    H = build_hamiltonian_dense(L, J, h, periodic)
    ev, vecs = np.linalg.eigh(H)
    _SPECTRUM_CACHE[key] = ev
    gs = vecs[:, 0]
    _GS_VECTOR_CACHE[key] = gs
    return gs


def gap(L: int, J: float, h: float,
        periodic: bool = False) -> float:
    """Many-body gap Delta = E_1 - E_0.  Robust against near-degeneracy
    by explicitly skipping numerical zeros below EPS_NUM."""
    ev = spectrum(L, J, h, periodic)
    delta = ev[1] - ev[0]
    return max(float(delta), C.EPS_NUM)


# ---------------------------------------------------------------------------
# Jordan-Wigner free-fermion route
# ---------------------------------------------------------------------------
def jw_single_particle_energies(L: int, J: float, h: float,
                                 periodic: bool = False) -> np.ndarray:
    """After Jordan-Wigner + Fourier + Bogoliubov, the TFIM reduces
    to L independent fermion modes with dispersion

        eps_k = 2 J * sqrt(1 + (h/J)^2 - 2 (h/J) cos k)

    for OBC with k in the appropriate half-Brillouin set, and with a
    subtle L-dependent shift under PBC (even/odd parity sectors).

    We follow Pfeuty (1970) and return the positive branch only.
    """
    if J == 0.0:
        return np.full(L, abs(h) + C.EPS_NUM)
    lam = h / J
    if periodic:
        # Even-fermion-parity sector uses k = 2 pi (n + 1/2) / L
        ns = np.arange(L)
        k = 2.0 * C.PI * (ns + 0.5) / L
    else:
        ns = np.arange(L)
        k = C.PI * (2 * ns + 1) / (2 * L + 1)
    eps = 2.0 * abs(J) * np.sqrt(np.maximum(
        1.0 + lam * lam - 2.0 * lam * np.cos(k), 0.0))
    return np.sort(eps)


def ground_state_energy_jw(L: int, J: float, h: float,
                             periodic: bool = False) -> float:
    """Free-fermion ground-state energy via Jordan-Wigner:
        E_0 = - 1/2 sum_k eps_k   (plus a J-dependent constant for PBC).
    """
    eps = jw_single_particle_energies(L, J, h, periodic)
    e0 = -0.5 * float(np.sum(eps))
    if periodic:
        e0 += J * float(L)
    return e0


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def magnetization_x(L: int, J: float, h: float,
                     periodic: bool = False) -> float:
    """Per-site transverse magnetization <sigma^x>/L in the ground state.
    Computed from the exact dense ground state vector."""
    gs = _ground_state_vector(L, J, h, periodic)
    mx_total = 0.0
    for i in range(L):
        ops = [EYE2] * L
        ops[i] = SIGMA_X
        O = _kron_n(ops).real
        mx_total += float(gs @ O @ gs)
    return mx_total / L


def correlation_zz(L: int, J: float, h: float, r: int,
                    periodic: bool = False) -> float:
    """Longitudinal correlation <sigma^z_0 sigma^z_r> in the GS."""
    if r < 0 or r >= L:
        raise ValueError(f"r must be in [0, L-1], got r={r}, L={L}")
    gs = _ground_state_vector(L, J, h, periodic)
    ops = [EYE2] * L
    ops[0] = SIGMA_Z
    ops[r] = SIGMA_Z
    O = _kron_n(ops).real
    return float(gs @ O @ gs)
