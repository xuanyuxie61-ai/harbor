"""
sparse_green_defect.py — Sparse Green's function for the defect perturbation
============================================================================

The central object in the embedded-cluster treatment of a point defect is
the single-particle Green's function
    G(z) = (z I - H)^{-1},     z = E + i η,
where H is the Kohn–Sham Hamiltonian of the host crystal. In the presence
of a defect the Hamiltonian changes by a *local* perturbation ΔV centred
on the defect site:
    H_d = H + ΔV.
Instead of inverting the full (large) matrix z I - H_d we solve the
**Dyson equation**
    G_d = G + G T G,      T = ΔV (I - G ΔV)^{-1},
which only requires inverting a small sub-block of size N_c × N_c where
N_c is the number of cluster sites around the defect (typically 10–50).

This module implements the *Sparse Green's Function (SGF) formalism* of
Willem & Wybo (seed project 1221) adapted to the electronic-structure
context:

  1. Build the host Green's function G on a small cluster around the defect
     by direct inversion of the sparse (z I - H) sub-block in the RI format
     of seed project 992_r8ri.
  2. Form the T-matrix  T = ΔV (I - G ΔV)^{-1}.
  3. Compute the change in the integrated density of states (IDOS) via the
     trace formula
         ΔN(E) = -(1/π) Im Tr ln(I - G ΔV)
     and the corresponding change in the band energy
         ΔE_band = ∫ E d(ΔN) .
  4. (Optional) Partial-fraction approximation of G(z) for efficient
     imaginary-time propagation à la Vector Fitting.

The physics is standard; see, e.g., M. Lannoo & P. Friedel, *Defects in
Semiconductors* (Springer, 1984), and E. Kaxiras, *Atomic and Electronic
Structure of Solids* (Cambridge, 2003), Ch. 11.
"""

from __future__ import annotations
import math
import cmath
import numpy as np
from typing import Tuple, List

from high_order_fd import RISparseLaplacian


# -------------------------------------------------------------------------
# (1) Host Hamiltonian on a real-space grid
# -------------------------------------------------------------------------
class HostHamiltonian:
    """Discretised Kohn–Sham Hamiltonian of the perfect host crystal.

        H = -(1/2) ∇² + V_bulk
    where V_bulk is a periodic effective potential (model: sum of Gaussian
    ion cores at lattice sites).
    """
    def __init__(self, Ny: int, Nx: int, h: float, order: int,
                 positions: np.ndarray, Z_eff: float, r_core: float):
        self.Ny, self.Nx, self.h = Ny, Nx, h
        self.N = Ny * Nx
        self.order = order
        self.positions = positions
        self.Z_eff = Z_eff
        self.r_core = r_core
        # kinetic operator (sparse RI)
        self.T_ri = RISparseLaplacian(Ny, Nx, h, order)
        self.T_diag = -0.5 * self.T_ri.diag        # -(1/2)∇² diagonal
        self.V_bulk = self._build_potential()

    # ------------------------------------------------------------------
    def _build_potential(self) -> np.ndarray:
        """Build V_bulk(r) = Σ_i (-Z_eff) erf(|r - R_i| / r_core) / |r - R_i|.

        We evaluate V_bulk on the real-space grid using minimum-image
        convention. The Gaussian-core model is soft at the origin, so no
        divergence occurs.
        """
        N = self.N
        h = self.h
        Ny, Nx = self.Ny, self.Nx
        xs = (np.arange(Nx) - Nx / 2) * h
        ys = (np.arange(Ny) - Ny / 2) * h
        X, Y = np.meshgrid(xs, ys)
        Lx = Nx * h
        Ly = Ny * h
        V = np.zeros((Ny, Nx))
        for (Rx, Ry) in self.positions:
            # minimum image
            dx = X - Rx
            dy = Y - Ry
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            r = np.sqrt(dx * dx + dy * dy + 1e-20)
            from math import erf
            vcore = np.vectorize(erf)
            V += -self.Z_eff * vcore(r / self.r_core) / r
        return V

    # ------------------------------------------------------------------
    def apply(self, psi: np.ndarray) -> np.ndarray:
        """Apply H to a wave function (flattened) using the RI format."""
        psi_2d = psi.reshape(self.Ny, self.Nx)
        Tpsi_2d = -0.5 * np.zeros_like(psi_2d)
        # kinetic part via dense RI mv
        Tpsi_flat = self.T_ri.mv(psi_2d.ravel())
        Tpsi_2d = -0.5 * Tpsi_flat.reshape(self.Ny, self.Nx)
        # wrong sign — T_ri already stores ∇²; multiply by -1/2
        # (we re-multiply because T_ri.diag is ∇² diag, so -0.5 * lap = -0.5*T_ri)
        # Correct the sign:  T_ri.mv(psi) = ∇² psi, so -(1/2)∇² = -0.5*T_ri.mv
        Hpsi_2d = Tpsi_2d + self.V_bulk * psi_2d
        return Hpsi_2d.ravel()


# -------------------------------------------------------------------------
# (2) Cluster selection — atoms within radius R_c of the defect
# -------------------------------------------------------------------------
def select_cluster(positions: np.ndarray, defect_idx: int,
                   R_cut: float) -> np.ndarray:
    """Return indices of sites within R_cut of positions[defect_idx]."""
    R0 = positions[defect_idx]
    dists = np.linalg.norm(positions - R0, axis=1)
    return np.where(dists < R_cut)[0]


# -------------------------------------------------------------------------
# (3) Build defect perturbation ΔV on the real-space grid
# -------------------------------------------------------------------------
def build_defect_potential(Ny: int, Nx: int, h: float,
                           defect_site_grid: Tuple[int, int],
                           Z_eff: float, r_core: float,
                           kind: str = "vacancy") -> np.ndarray:
    """Build ΔV = V_defect - V_bulk.

    For a vacancy, the ion core at `defect_site_grid` is removed:
        ΔV(r) = + Z_eff erf(|r - R_0| / r_core) / |r - R_0|.
    For an interstitial, an *additional* core is placed at the site:
        ΔV(r) = - Z_eff erf(|r - R_0| / r_core) / |r - R_0|.

    (The sign convention is ΔV = V_def - V_host.)
    """
    xs = (np.arange(Nx) - Nx / 2) * h
    ys = (np.arange(Ny) - Ny / 2) * h
    X, Y = np.meshgrid(xs, ys)
    Rx = (defect_site_grid[1] - Nx / 2) * h
    Ry = (defect_site_grid[0] - Ny / 2) * h
    dx = X - Rx
    dy = Y - Ry
    r = np.sqrt(dx * dx + dy * dy + 1e-20)
    from math import erf
    vcore = np.vectorize(erf)
    core = -Z_eff * vcore(r / r_core) / r
    if kind == "vacancy":
        dV = +core           # remove the ion ⇒ raise the potential
    else:
        dV = -core           # add an ion ⇒ lower the potential
    return dV


# -------------------------------------------------------------------------
# (4) Dyson equation solver for the cluster T-matrix
# -------------------------------------------------------------------------
class DysonSolver:
    """Solve the Dyson equation on a cluster of N_c sites.

    The host Green's function is obtained by direct inversion of the
    (z I - H_c) sub-block. The defect Hamiltonian on the same cluster is
    H_c + ΔV_c. We compute
        G_c(z)     = (z I - H_c)^{-1}
        G_d,c(z)   = (z I - H_c - ΔV_c)^{-1}
        T_c(z)     = ΔV_c + ΔV_c G_c ΔV_c + ... = ΔV_c (I - G_c ΔV_c)^{-1}
    and the cluster contribution to the band-energy shift
        ΔE_band = -(1/π) ∫ dE Im Tr [ T_c(E + i η) G_c(E + i η) ] f(E)
    approximated by Gauss–Hermite quadrature (linking to seed project 464).
    """
    def __init__(self, H_host: HostHamiltonian, cluster_idx: np.ndarray,
                 dV_grid: np.ndarray):
        self.H = H_host
        self.idx = cluster_idx
        self.Nc = cluster_idx.size
        # extract the cluster sub-block of V_bulk and ΔV
        iy = self.idx // H_host.Nx
        ix = self.idx % H_host.Nx
        self.V_c = np.array([H_host.V_bulk[a, b] for (a, b) in zip(iy, ix)])
        self.dV_c = np.array([dV_grid[a, b] for (a, b) in zip(iy, ix)])
        # build cluster kinetic sub-block via RI
        self.T_c = self._build_cluster_kinetic()

    def _build_cluster_kinetic(self) -> np.ndarray:
        """Extract the (Nc, Nc) kinetic sub-matrix from T_ri."""
        Nc = self.Nc
        T = np.zeros((Nc, Nc))
        for i, gi in enumerate(self.idx):
            T[i, i] = self.H.T_ri.diag[gi]
            for k in range(self.H.T_ri.row_ptr[gi],
                           self.H.T_ri.row_ptr[gi + 1]):
                col = self.H.T_ri.off_cols[k]
                # find j such that idx[j] == col
                js = np.where(self.idx == col)[0]
                if js.size:
                    T[i, js[0]] += self.H.T_ri.off_vals[k]
        return -0.5 * T

    # ------------------------------------------------------------------
    def green_cluster(self, z: complex) -> np.ndarray:
        """G_c(z) = (z I - H_c)^{-1} for the cluster."""
        Hc = self.T_c + np.diag(self.V_c)
        M = z * np.eye(self.Nc) - Hc
        return np.linalg.inv(M)

    def green_defect_cluster(self, z: complex) -> np.ndarray:
        """G_d,c(z) = (z I - H_c - ΔV_c)^{-1}."""
        Hd = self.T_c + np.diag(self.V_c + self.dV_c)
        M = z * np.eye(self.Nc) - Hd
        return np.linalg.inv(M)

    def t_matrix(self, z: complex) -> np.ndarray:
        """T_c(z) = ΔV_c (I - G_c ΔV_c)^{-1}."""
        Gc = self.green_cluster(z)
        dV = np.diag(self.dV_c)
        I_GdV = np.eye(self.Nc) - Gc @ dV
        return dV @ np.linalg.inv(I_GdV)

    # ------------------------------------------------------------------
    def band_energy_shift(self, E_min: float, E_max: float,
                          n_quad: int = 32, eta: float = 0.05,
                          kT: float = 0.025) -> float:
        """Compute the cluster band-energy shift ΔE_band.

        We use Gauss–Legendre quadrature over [E_min, E_max] of
            Δρ(E) = -(1/π) Im Tr [ G_d(E + iη) - G(E + iη) ]
        weighted by the Fermi function f(E) = 1 / (exp((E - μ)/kT) + 1)
        and by E:
            ΔE_band = ∫ dE  E Δρ(E) f(E).
        The chemical potential μ is set to 0 (half-filling).
        """
        # Gauss–Legendre nodes
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        Es = 0.5 * (E_max - E_min) * nodes + 0.5 * (E_max + E_min)
        ws = 0.5 * (E_max - E_min) * weights
        dE = 0.0
        for (E, w) in zip(Es, ws):
            z = E + 1j * eta
            Gc = self.green_cluster(z)
            Gd = self.green_defect_cluster(z)
            delta_rho = -(1.0 / math.pi) * np.trace(Gd - Gc).imag
            fE = 1.0 / (math.exp(min(E / kT, 500.0)) + 1.0)
            dE += w * E * delta_rho * fE
        return float(dE)


# -------------------------------------------------------------------------
# (5) Partial-fraction approximation (à la Vector Fitting / SGF)
# -------------------------------------------------------------------------
def partial_fraction_fit(G_diag: np.ndarray, zs: np.ndarray,
                         n_poles: int = 4
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit G_diag(z) ≈ Σ_{k=1}^{n_poles} α_k / (z - z_k).

    We implement a simplified version of the SGF Vector Fitting: pick
    `n_poles` logarithmically spaced along the imaginary axis, solve a
    least-squares problem for the residues α_k.

    This is used downstream to accelerate imaginary-time propagation since
        exp(-τ H) ψ ≈ Σ_k α_k exp(-τ z_k) (z_k I - H)^{-1} ψ.
    """
    n_z = zs.size
    poles = 1j * np.logspace(-2, 1, n_poles)
    A = np.zeros((n_z, n_poles), dtype=np.complex128)
    for k, zk in enumerate(poles):
        A[:, k] = 1.0 / (zs - zk)
    # least-squares solution for residues
    alpha, *_ = np.linalg.lstsq(A, G_diag, rcond=None)
    return alpha, poles
