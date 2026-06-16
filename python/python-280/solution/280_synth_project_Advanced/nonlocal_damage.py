"""
nonlocal_damage.py — Implicit gradient / integral-type nonlocal damage model.

Core physics
============
Local strain-softening damage models suffer from ill-posedness: the
governing PDE loses ellipticity and numerical results depend on mesh size
(Bažant & Jirássek, 2002).  The nonlocal regularization replaces the
local strain at a point by a spatially averaged (nonlocal) counterpart:

    ε̄_nl(x) = ∫_Ω α(x,ξ) ε(ξ) dξ  /  ∫_Ω α(x,ξ) dξ

where the weight function α is typically Gaussian:

    α(x,ξ) = exp( -||x-ξ||² / R² )

with R = characteristic length related to the fracture process zone.

Damage evolution law (Mazars, 1986; Peerlings et al., 1996):
    D(κ) = 0                                       if κ ≤ κ_0
    D(κ) = 1 - (κ_0/κ) [1 - ν + ν exp(-(κ-κ_0)/ν)]  if κ > κ_0

    where κ = max_t ε̃_eq  (history variable)
          ε̃_eq = equivalent strain (von Mises type)
          κ_0 = damage threshold
          ν   = softening control parameter

The governing system is solved by a staggered scheme:
  1. Given displacement field u^n, compute strains ε^n.
  2. Compute nonlocal equivalent strain ε̄_nl via weighted integration.
  3. Update history variable κ^{n+1} = max(κ^n, ε̄_nl).
  4. Update damage D^{n+1} = D(κ^{n+1}).
  5. Update stiffness and iterate to convergence.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional
from config import SimulationConfig, MaterialParams


# ===================================================================
# Equivalent strain measure
# ===================================================================

def equivalent_strain_von_mises(eps_xx: np.ndarray,
                                eps_yy: np.ndarray,
                                eps_xy: np.ndarray,
                                nu: float) -> np.ndarray:
    """Modified von Mises equivalent strain for tension-compression asymmetry.

    ε̃_eq = (1/(2(1-2ν))) * max(0, ε_I + ε_II + ε_III)
            + (1/(2(1+ν))) * sqrt( max(0, (ε_I-ε_II)² + (ε_II-ε_III)² + (ε_I-ε_III)²) )

    For 2D plane strain, ε_III = 0 and we use the principal strains.

    Simplified form for plane strain:
        ε̃_eq = (k-1)/(2k) * (ε_xx + ε_yy)
               + sqrt( ((k-1)/(2k))² * (ε_xx+ε_yy)²
                       + 1/(2(1+ν))² * ((ε_xx-ε_yy)² + 4*eps_xy²) )
    where k = f_c / f_t (strength ratio).
    """
    # Principal strains
    avg = 0.5 * (eps_xx + eps_yy)
    diff_half = 0.5 * (eps_xx - eps_yy)
    r = np.sqrt(diff_half ** 2 + eps_xy ** 2)
    eps1 = avg + r   # major principal strain
    eps2 = avg - r   # minor principal strain

    # Macaulay brackets <ε>_+ = max(ε, 0)
    eps1_pos = np.maximum(eps1, 0.0)
    eps2_pos = np.maximum(eps2, 0.0)

    # Von Mises equivalent (2D version with tension-only contribution)
    # Using the Mazars definition:
    #   ε̃_eq = sqrt( <ε_1>_+² + <ε_2>_+² )
    eq_strain = np.sqrt(eps1_pos ** 2 + eps2_pos ** 2)

    # Regularize to avoid division by zero
    eq_strain = np.maximum(eq_strain, 0.0)

    return eq_strain


def compute_strain_components(u: np.ndarray, v: np.ndarray,
                              dx: float, dy: float) -> Dict[str, np.ndarray]:
    """Compute small-strain tensor components from displacements.

        ε_xx = ∂u/∂x,   ε_yy = ∂v/∂y,   ε_xy = ½(∂u/∂y + ∂v/∂x)

    Uses 2nd-order central differences (sufficient for the staggered scheme).
    """
    ny, nx = u.shape
    eps_xx = np.zeros_like(u)
    eps_yy = np.zeros_like(v)
    eps_xy = np.zeros_like(u)

    # Interior points
    eps_xx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    eps_yy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy)
    eps_xy[1:-1, 1:-1] = 0.5 * (
        (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy) +
        (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx)
    )

    # Boundaries: forward/backward differences
    eps_xx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    eps_xx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    eps_yy[0, :] = (v[1, :] - v[0, :]) / dy
    eps_yy[-1, :] = (v[-1, :] - v[-2, :]) / dy

    return {"eps_xx": eps_xx, "eps_yy": eps_yy, "eps_xy": eps_xy}


# ===================================================================
# Nonlocal averaging
# ===================================================================

def gaussian_weight(r_sq: np.ndarray, R: float) -> np.ndarray:
    """Gaussian weight function:  α(r) = exp(-r²/R²)."""
    return np.exp(-r_sq / (R ** 2))


def build_nonlocal_kernel(x: np.ndarray, y: np.ndarray,
                          R: float,
                          cutoff_factor: float = 3.0) -> np.ndarray:
    """Build the nonlocal weight matrix for all pairs of grid points.

    For an N×M grid, this creates an (NM) × (NM) sparse-like weight matrix.
    We apply a cutoff at r_cutoff = cutoff_factor * R for efficiency.

    Returns the normalised weight matrix W such that:
        ε̄_nl = W @ ε_local
    where rows of W sum to 1.
    """
    ny, nx = x.shape
    n_total = ny * nx
    x_flat = x.ravel()
    y_flat = y.ravel()

    # Pairwise distance squared
    dx_mat = x_flat[:, None] - x_flat[None, :]
    dy_mat = y_flat[:, None] - y_flat[None, :]
    r_sq = dx_mat ** 2 + dy_mat ** 2

    # Cutoff
    r_cutoff_sq = (cutoff_factor * R) ** 2
    mask = r_sq <= r_cutoff_sq

    # Weight
    W = np.zeros((n_total, n_total))
    W[mask] = np.exp(-r_sq[mask] / (R ** 2))

    # Normalise rows
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0e-30)
    W /= row_sums

    return W


def apply_nonlocal_averaging(local_field: np.ndarray,
                             weight_matrix: np.ndarray,
                             shape: Tuple[int, int]) -> np.ndarray:
    """Apply the nonlocal weight matrix to a local field.

    ε̄_nl(x) = ∫ α(x,ξ) ε(ξ) dξ  /  ∫ α(x,ξ) dξ
             ≈ Σ_j W_{ij} ε_j
    """
    flat = local_field.ravel()
    nonlocal_flat = weight_matrix @ flat
    return nonlocal_flat.reshape(shape)


def fast_nonlocal_averaging(local_field: np.ndarray,
                            x: np.ndarray, y: np.ndarray,
                            R: float) -> np.ndarray:
    """Vectorized nonlocal averaging with Gaussian kernel.

    This is a faster approximation that avoids building the full matrix.
    Uses the separability of the Gaussian:
        exp(-(x²+y²)/R²) = exp(-x²/R²) * exp(-y²/R²)

    Applied as successive 1-D convolutions along each axis.
    """
    ny, nx = local_field.shape
    dx = x[0, 1] - x[0, 0] if nx > 1 else 1.0
    dy = y[1, 0] - y[0, 0] if ny > 1 else 1.0

    # 1-D Gaussian kernel
    cutoff_pts_x = min(int(3.0 * R / dx) + 1, nx // 2)
    cutoff_pts_y = min(int(3.0 * R / dy) + 1, ny // 2)

    # Kernel along x
    kx = np.arange(-cutoff_pts_x, cutoff_pts_x + 1)
    gauss_x = np.exp(-(kx * dx) ** 2 / (R ** 2))
    gauss_x /= gauss_x.sum()

    # Kernel along y
    ky = np.arange(-cutoff_pts_y, cutoff_pts_y + 1)
    gauss_y = np.exp(-(ky * dy) ** 2 / (R ** 2))
    gauss_y /= gauss_y.sum()

    # Convolve: first along columns (axis=1), then along rows (axis=0)
    result = np.zeros_like(local_field)
    # x-direction convolution
    padded_x = np.pad(local_field, ((0, 0), (cutoff_pts_x, cutoff_pts_x)), mode='edge')
    for k in range(len(gauss_x)):
        result += gauss_x[k] * padded_x[:, k:k + nx]

    # y-direction convolution
    padded_y = np.pad(result, ((cutoff_pts_y, cutoff_pts_y), (0, 0)), mode='edge')
    final = np.zeros_like(local_field)
    for k in range(len(gauss_y)):
        final += gauss_y[k] * padded_y[k:k + ny, :]

    return final


# ===================================================================
# Damage evolution law
# ===================================================================

def mazars_damage_law(kappa: np.ndarray,
                      kappa_0: float,
                      kappa_c: float,
                      n_soft: float = 2.0) -> np.ndarray:
    """Compute damage variable D from the history variable κ.

    D(κ) = 0                                          for κ ≤ κ_0
    D(κ) = 1 - (κ_0/κ) * exp(-(κ - κ_0) / (κ_c - κ_0) * n_soft)
         = 1 - (κ_0/κ) * exp(-β(κ - κ_0))             for κ > κ_0

    where β = n_soft / (κ_c - κ_0) controls the softening rate.

    This ensures:
      - D(κ_0) = 0    (onset of damage)
      - D(κ→∞) → 1   (complete failure)
      - Smooth transition with exponential tail
    """
    D = np.zeros_like(kappa)
    mask = kappa > kappa_0
    if not np.any(mask):
        return D

    beta = n_soft / max(kappa_c - kappa_0, 1.0e-15)
    kappa_m = kappa[mask]
    D[mask] = 1.0 - (kappa_0 / kappa_m) * np.exp(-beta * (kappa_m - kappa_0))

    # Clamp to [0, 1-ε] to avoid singularity
    D = np.clip(D, 0.0, 1.0 - 1.0e-8)

    return D


def damage_energy_release_rate(D: np.ndarray,
                               eps_xx: np.ndarray,
                               eps_yy: np.ndarray,
                               eps_xy: np.ndarray,
                               material: MaterialParams) -> np.ndarray:
    """Compute the energy release rate Y conjugate to damage.

    Y = -∂ψ/∂D = ½ ε : C : ε

    where ψ = ½ (1-D) ε : C : ε is the damaged strain energy density.

    For isotropic plane strain:
        Y = ½ [ (λ+2μ)(ε_xx² + ε_yy²) + 2λ ε_xx ε_yy + 4μ ε_xy² ]

    Wait — more precisely:
        C_ijkl ε_ij ε_kl = (λ+2μ)(ε_xx² + ε_yy²) + 2λ ε_xx ε_yy + 2μ (2ε_xy)²
                          = (λ+2μ)(ε_xx² + ε_yy²) + 2λ ε_xx ε_yy + 8μ ε_xy²

    Actually, for engineering strain with ε_xy as tensor component:
        ψ = ½(λ+2μ)(ε_xx² + ε_yy²) + λ ε_xx ε_yy + 2μ ε_xy²

    So Y = ψ (the undamaged energy density).
    """
    lam = material.lame_lambda
    mu = material.lame_mu

    Y = (0.5 * (lam + 2.0 * mu) * (eps_xx ** 2 + eps_yy ** 2)
         + lam * eps_xx * eps_yy
         + 2.0 * mu * eps_xy ** 2)

    return np.maximum(Y, 0.0)


# ===================================================================
# Full nonlocal damage update step
# ===================================================================

class NonlocalDamageState:
    """Mutable state container for the nonlocal damage field."""

    def __init__(self, ny: int, nx: int, cfg: SimulationConfig):
        self.cfg = cfg
        self.shape = (ny, nx)
        self.damage = np.zeros((ny, nx))
        self.kappa_history = np.zeros((ny, nx))
        self.equivalent_strain = np.zeros((ny, nx))
        self.nonlocal_strain = np.zeros((ny, nx))
        self.energy_release = np.zeros((ny, nx))
        self._kernel_built = False
        self._weight_matrix = None

    def build_kernel(self, x: np.ndarray, y: np.ndarray):
        """Pre-compute the nonlocal weight matrix."""
        R = self.cfg.material.characteristic_length
        n_total = x.size
        # For small grids, build full matrix; otherwise use fast method
        if n_total <= 2500:
            self._weight_matrix = build_nonlocal_kernel(x, y, R)
            self._kernel_built = True
        else:
            self._kernel_built = False  # will use fast method

    def update(self, u: np.ndarray, v: np.ndarray,
               x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform one nonlocal damage update.

        Returns convergence info dictionary.
        """
        dx = self.cfg.dx()
        dy = self.cfg.dy()
        mat = self.cfg.material

        # 1. Compute strains
        strains = compute_strain_components(u, v, dx, dy)
        eq_strain = equivalent_strain_von_mises(
            strains["eps_xx"], strains["eps_yy"], strains["eps_xy"],
            mat.poisson_ratio
        )
        self.equivalent_strain = eq_strain

        # 2. Nonlocal averaging
        if self._kernel_built and self._weight_matrix is not None:
            nl_strain = apply_nonlocal_averaging(eq_strain, self._weight_matrix, self.shape)
        else:
            nl_strain = fast_nonlocal_averaging(eq_strain, x, y,
                                                mat.characteristic_length)
        self.nonlocal_strain = nl_strain

        # 3. Update history
        self.kappa_history = np.maximum(self.kappa_history, nl_strain)

        # 4. Compute damage
        kappa_c = mat.damage_threshold_strain * 10.0  # κ_c controls ductility
        self.damage = mazars_damage_law(
            self.kappa_history, mat.damage_threshold_strain,
            kappa_c, mat.softening_exponent
        )

        # 5. Energy release rate
        self.energy_release = damage_energy_release_rate(
            self.damage, strains["eps_xx"], strains["eps_yy"],
            strains["eps_xy"], mat
        )

        # Convergence metrics
        max_damage = float(self.damage.max())
        mean_damage = float(self.damage.mean())
        damaged_frac = float((self.damage > 0.01).sum()) / self.damage.size

        return {
            "max_damage": max_damage,
            "mean_damage": mean_damage,
            "damaged_fraction": damaged_frac,
            "max_equiv_strain": float(eq_strain.max()),
            "max_energy_release": float(self.energy_release.max()),
        }
