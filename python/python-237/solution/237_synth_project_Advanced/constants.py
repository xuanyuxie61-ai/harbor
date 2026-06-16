"""
Physical constants, lattice parameters, and SU(2) group utilities for
Lattice QCD computations.

This module centralizes the dimensionless physical constants used across
the project, the lattice geometry specification (Nt, Nx, Ny, Nz, a),
and the SU(2) gauge group algebra.

Key equations:
    Wilson gauge action:
        S_G = beta * sum_{x, mu<nu} [ 1 - (1/2) Tr U_{mu nu}(x) ]
    where beta = 4 / g^2 for SU(2), U_{mu nu} is the plaquette.

    Wilson-Dirac operator:
        D_W = (m_0 + 4/a) delta_{xy}
              - (1/(2a)) sum_mu [ (1 - gamma_mu) U_mu(x) delta_{x+mu,y}
                                 + (1 + gamma_mu) U_mu^dag(x-mu) delta_{x-mu,y} ]
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Dimensionless physical constants (natural units: hbar = c = 1)
# ---------------------------------------------------------------------------

PI = np.pi
EUCLIDEAN_DIM = 4           # Euclidean spacetime dimensions (mu = 0,1,2,3)

# SU(2) generators: tau^a = sigma^a / 2 (sigma^a = Pauli matrices)
TAU1 = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
TAU2 = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
TAU3 = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
TAU = [TAU1, TAU2, TAU3]

# Dirac gamma matrices in Euclidean chiral representation (4x4 complex)
# {gamma_mu, gamma_nu} = 2 delta_{mu,nu}
def _build_gamma_euclidean():
    """Construct Euclidean gamma matrices satisfying the Clifford algebra."""
    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    Z2 = np.zeros((2, 2), dtype=complex)

    g0 = np.block([[Z2, I2], [I2, Z2]])
    g1 = np.block([[Z2, -1j * sigma1], [1j * sigma1, Z2]])
    g2 = np.block([[Z2, -1j * sigma2], [1j * sigma2, Z2]])
    g3 = np.block([[Z2, -1j * sigma3], [1j * sigma3, Z2]])
    return [g0, g1, g2, g3]

GAMMA = _build_gamma_euclidean()

# ---------------------------------------------------------------------------
# Lattice parameters
# ---------------------------------------------------------------------------

@dataclass
class LatticeParams:
    """
    Euclidean hypercubic lattice parameters.

    Attributes:
        Ns : spatial extent (Nx = Ny = Nz = Ns)
        Nt : temporal extent
        a  : lattice spacing in fm (default ~ 0.1 fm)
        beta : gauge coupling beta = 4/g^2 for SU(2)
        kappa : hopping parameter kappa = 1/(2*(m_0*a + 4))
        m0 : bare quark mass in lattice units
    """
    Ns: int = 4
    Nt: int = 4
    a: float = 0.1           # fm
    beta: float = 2.5        # SU(2) coupling
    m0: float = 0.1          # bare quark mass (lattice units)
    kappa: Optional[float] = None

    def __post_init__(self):
        if self.kappa is None:
            self.kappa = 1.0 / (2.0 * (self.m0 + 4.0))
        if self.Ns < 2 or self.Nt < 2:
            raise ValueError(f"Lattice extents must be >= 2, got Ns={self.Ns}, Nt={self.Nt}")
        if self.beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.m0 < 0.0:
            raise ValueError(f"bare mass m0 must be >= 0, got {self.m0}")

    @property
    def V(self) -> int:
        return self.Ns ** 3 * self.Nt

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.Nt, self.Ns, self.Ns, self.Ns)

    @property
    def g_sq(self) -> float:
        return 4.0 / self.beta

    def critical_kappa(self) -> float:
        return 1.0 / 8.0


# ---------------------------------------------------------------------------
# SU(2) group operations
# ---------------------------------------------------------------------------

def su2_random(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-distributed SU(2) matrix via Cayley transform."""
    while True:
        v = rng.standard_normal(4)
        n2 = float(np.dot(v, v))
        if n2 > 1e-12 and n2 < 1.0:
            break
    a0 = (1.0 - n2) / (1.0 + n2)
    fac = 2.0 / (1.0 + n2)
    a_vec = fac * v
    U = a0 * np.eye(2, dtype=complex) + 1j * sum(a_vec[i] * TAU[i] for i in range(3))
    return _project_su2(U)


def _project_su2(U: np.ndarray) -> np.ndarray:
    """Project a 2x2 complex matrix onto SU(2) via polar decomposition."""
    a0 = 0.5 * (U[0, 0] + U[1, 1]).real
    a1 = 0.5 * (U[0, 1] + U[1, 0]).imag
    a2 = 0.5 * (-U[0, 1] + U[1, 0]).real
    a3 = 0.5 * (U[0, 0] - U[1, 1]).imag
    norm = np.sqrt(a0**2 + a1**2 + a2**2 + a3**2 + 1e-30)
    a0, a1, a2, a3 = a0 / norm, a1 / norm, a2 / norm, a3 / norm
    return np.array([
        [a0 + 1j * a3, a2 + 1j * a1],
        [-a2 + 1j * a1, a0 - 1j * a3],
    ], dtype=complex)


def su2_identity() -> np.ndarray:
    return np.eye(2, dtype=complex)


def su2_trace(U: np.ndarray) -> float:
    return float(U[0, 0].real + U[1, 1].real)
