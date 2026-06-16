"""
stochastic_defects.py — Stochastic micro-defect generation and analysis.

Seed reference: 442_fly_simulation (Monte Carlo uniform sampling in disk,
correct area-weighting via sqrt(rand) for radial distribution).

Core idea
=========
Real materials contain micro-defects (micro-voids, micro-cracks, inclusions)
that act as damage nucleation sites. We generate a random distribution of
micro-defects using:

  1. Poisson point process for defect locations
     - Defect density ρ_d [defects/m²]
     - Number of defects: N ~ Poisson(ρ_d * A)

  2. Defect radii from a truncated power-law distribution
     - p(r) ∝ r^{-α}  for r_min ≤ r ≤ r_max
     - Typical α ≈ 3 for brittle materials (Griffith crack distribution)

  3. Defect interaction via Eshelby's inclusion theory
     - Each defect creates a local stress perturbation
     - Stress concentration factor: K_t = 1 + 2(a/b) for elliptical void

  4. Monte Carlo sampling (seed 442_fly_simulation)
     - For circular defects, use correct area weighting
     - r = r_min * (r_max/r_min)^{sqrt(U)} for uniform area distribution

The defects modify the local damage threshold:
    κ_0^eff(x) = κ_0(x) * (1 - Σ_j φ_j(x))

where φ_j(x) is the influence function of defect j at point x:
    φ_j(x) = (r_j / |x - x_j|)²  for |x - x_j| > r_j
           = 1                     for |x - x_j| ≤ r_j
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SimulationConfig


# ===================================================================
# Poisson point process for defect locations
# ===================================================================

def generate_poisson_defect_locations(domain_x: Tuple[float, float],
                                      domain_y: Tuple[float, float],
                                      intensity: float,
                                      seed: int = 42
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate defect locations as a homogeneous Poisson point process.

    For a Poisson process with intensity λ [points/m²] on domain area A:
        N ~ Poisson(λA)

    Given N, the locations are uniformly distributed in the domain.

    Returns (x_coords, y_coords) arrays of length N.
    """
    rng = np.random.RandomState(seed)

    Lx = domain_x[1] - domain_x[0]
    Ly = domain_y[1] - domain_y[0]
    area = Lx * Ly
    expected_N = intensity * area

    # Sample N from Poisson distribution
    N = rng.poisson(expected_N)
    N = max(N, 0)

    if N == 0:
        return np.array([]), np.array([])

    # Uniform locations
    x_coords = domain_x[0] + rng.random(N) * Lx
    y_coords = domain_y[0] + rng.random(N) * Ly

    return x_coords, y_coords


# ===================================================================
# Defect radii from power-law distribution
# ===================================================================

def generate_defect_radii(n_defects: int,
                          r_min: float, r_max: float,
                          exponent: float = 3.0,
                          seed: int = 42) -> np.ndarray:
    """Generate defect radii from a truncated power-law distribution.

    p(r) = C * r^{-α}  for r_min ≤ r ≤ r_max

    CDF inversion:
        F(r) = (r^{1-α} - r_min^{1-α}) / (r_max^{1-α} - r_min^{1-α})

    Inverse:
        r = [r_min^{1-α} + U * (r_max^{1-α} - r_min^{1-α})]^{1/(1-α)}

    For α = 2 (Griffith cracks):
        r = r_min * r_max / (r_max - U * (r_max - r_min))

    For α = 3:
        1/(1-α) = -1/2
        r = [r_min^{-2} + U * (r_max^{-2} - r_min^{-2})]^{-1/2}
    """
    if n_defects == 0:
        return np.array([])

    rng = np.random.RandomState(seed + 1)  # offset seed
    U = rng.random(n_defects)

    if abs(exponent - 1.0) < 1.0e-10:
        # Log-uniform (α = 1)
        radii = r_min * (r_max / r_min) ** U
    else:
        alpha_m1 = 1.0 - exponent
        r_min_a = r_min ** alpha_m1
        r_max_a = r_max ** alpha_m1
        radii = (r_min_a + U * (r_max_a - r_min_a)) ** (1.0 / alpha_m1)

    return radii


# ===================================================================
# Monte Carlo disk sampling  (seed 442_fly_simulation)
# ===================================================================

def monte_carlo_defect_positions_disk(cx: float, cy: float,
                                      radius: float,
                                      n_samples: int,
                                      seed: int = 42
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate uniform random points inside a circular defect zone.

    Uses the correct area-weighting transformation (seed 442):
        r = R * sqrt(U_1)     (not R * U_1, which would cluster near center)
        θ = 2π * U_2

    This ensures uniform area density: P(r < r_0) = (r_0/R)²

    Returns (x, y) coordinate arrays.
    """
    rng = np.random.RandomState(seed + 2)

    U1 = rng.random(n_samples)
    U2 = rng.random(n_samples)

    r = radius * np.sqrt(U1)
    theta = 2.0 * math.pi * U2

    x_pts = cx + r * np.cos(theta)
    y_pts = cy + r * np.sin(theta)

    return x_pts, y_pts


def expected_distance_disk(radius: float, n_samples: int = 10000) -> float:
    """Expected distance from center for uniform point in disk.

    Theoretical: E[d] = 2R/3

    We verify via Monte Carlo (seed 442_fly_simulation).
    """
    rng = np.random.RandomState(12345)
    U1 = rng.random(n_samples)
    r = radius * np.sqrt(U1)
    return float(np.mean(r))


# ===================================================================
# Eshelby inclusion stress perturbation
# ===================================================================

def eshelby_stress_concentration(defect_x: np.ndarray,
                                 defect_y: np.ndarray,
                                 defect_r: np.ndarray,
                                 grid_x: np.ndarray,
                                 grid_y: np.ndarray,
                                 poisson_ratio: float) -> np.ndarray:
    """Compute the stress concentration field from elliptical micro-defects.

    For a circular hole of radius a in an infinite plate under remote
    tension σ_∞, the stress concentration factor is:
        K_t = 1 + 2(a/r)   at the hole boundary (maximum)

    More generally, for an elliptical hole with semi-axes a, b:
        K_t = 1 + 2(a/b)

    We compute the total stress concentration as a superposition:
        K_t(x) = 1 + Σ_j 2 * (r_j / max(d_j(x), r_j))

    where d_j(x) = ||x - x_j|| is the distance to defect j.

    For a more accurate Eshelby solution, the perturbation decays as:
        σ'_ij(x) ∝ (r_j/d_j)² * f(θ_j, ν)

    where f is an angular function depending on the loading direction
    and Poisson's ratio.
    """
    ny, nx = grid_x.shape
    K_t = np.ones((ny, nx))

    for j in range(len(defect_x)):
        dx_j = grid_x - defect_x[j]
        dy_j = grid_y - defect_y[j]
        d_j = np.sqrt(dx_j ** 2 + dy_j ** 2)
        d_j = np.maximum(d_j, 1.0e-15)

        # Stress concentration contribution
        ratio = defect_r[j] / d_j
        contribution = 2.0 * ratio ** 2

        # Angular modulation (simplified Eshelby)
        theta_j = np.arctan2(dy_j, dx_j)
        angular_factor = 1.0 + 0.5 * (1.0 - 2.0 * poisson_ratio) * np.cos(2.0 * theta_j)

        K_t += contribution * angular_factor

    return K_t


# ===================================================================
# Defect influence on damage threshold
# ===================================================================

def compute_defect_influence_field(defect_x: np.ndarray,
                                   defect_y: np.ndarray,
                                   defect_r: np.ndarray,
                                   grid_x: np.ndarray,
                                   grid_y: np.ndarray,
                                   influence_range_factor: float = 5.0
                                   ) -> np.ndarray:
    """Compute the reduction in damage threshold due to micro-defects.

    Each defect reduces the local damage threshold by:
        Δκ_0(x) / κ_0 = Σ_j φ_j(x)

    where φ_j(x) = (r_j / d_j(x))² * exp(-d_j / (influence_range * r_j))

    This creates zones of weakened material around each defect.
    """
    ny, nx = grid_x.shape
    influence = np.zeros((ny, nx))

    for j in range(len(defect_x)):
        dx_j = grid_x - defect_x[j]
        dy_j = grid_y - defect_y[j]
        d_j = np.sqrt(dx_j ** 2 + dy_j ** 2)
        d_j = np.maximum(d_j, defect_r[j])  # regularize inside defect

        r_j = defect_r[j]
        influence_range = influence_range_factor * r_j

        phi_j = (r_j / d_j) ** 2 * np.exp(-d_j / influence_range)
        influence += phi_j

    # Clamp to [0, 1)
    influence = np.clip(influence, 0.0, 0.99)

    return influence


# ===================================================================
# Complete stochastic defect generation
# ===================================================================

def generate_stochastic_defects(cfg: SimulationConfig,
                                grid_x: np.ndarray,
                                grid_y: np.ndarray
                                ) -> Dict:
    """Generate a complete stochastic defect population and its effects.

    Returns:
      defect_x, defect_y: coordinates
      defect_r: radii
      n_defects: number of defects
      stress_concentration: K_t field
      damage_threshold_reduction: influence field
      monte_carlo_verification: E[d] from MC simulation
    """
    domain_x = cfg.numerical.domain_x
    domain_y = cfg.numerical.domain_y
    intensity = cfg.numerical.defect_intensity
    r_max = cfg.numerical.defect_max_radius
    r_min = r_max * 0.01  # smallest defect is 1% of largest
    seed = cfg.numerical.defect_seed

    # Generate locations
    dx_locs, dy_locs = generate_poisson_defect_locations(
        domain_x, domain_y, intensity, seed
    )
    n_defects = len(dx_locs)

    # Generate radii
    radii = generate_defect_radii(n_defects, r_min, r_max, exponent=3.0, seed=seed)

    # Stress concentration
    if n_defects > 0:
        K_t = eshelby_stress_concentration(
            dx_locs, dy_locs, radii, grid_x, grid_y,
            cfg.material.poisson_ratio
        )
        influence = compute_defect_influence_field(
            dx_locs, dy_locs, radii, grid_x, grid_y
        )
    else:
        K_t = np.ones_like(grid_x)
        influence = np.zeros_like(grid_x)

    # Monte Carlo verification
    mc_expected_d = expected_distance_disk(r_max)
    theoretical_d = 2.0 * r_max / 3.0

    return {
        "defect_x": dx_locs,
        "defect_y": dy_locs,
        "defect_r": radii,
        "n_defects": n_defects,
        "stress_concentration": K_t,
        "damage_threshold_reduction": influence,
        "monte_carlo_expected_distance": mc_expected_d,
        "theoretical_expected_distance": theoretical_d,
        "mc_relative_error": abs(mc_expected_d - theoretical_d) / max(theoretical_d, 1.0e-15),
        "defect_area_fraction": float(np.sum(np.pi * radii ** 2))
        / max((domain_x[1] - domain_x[0]) * (domain_y[1] - domain_y[0]), 1.0e-15),
    }
