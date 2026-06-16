"""
damage_grid.py — Structured grid generation around crack tips.

Adapts annular / Fibonacci grid generation (seed 008_annulus_grid) and
signed-distance geometry description (seed 305_dist_plot) for constructing
body-fitted grids in the neighbourhood of a propagating crack tip.

The grid combines:
  (a) A background Cartesian mesh covering the full domain.
  (b) A locally refined annular sub-grid around the crack tip, generated
      via a Fibonacci-spiral mapping that clusters points near the singularity.
  (c) Signed-distance level-set fields φ(x,y) and ψ(x,y) that implicitly
      represent the crack surface and the crack front.
"""

import math
import numpy as np
from typing import Tuple, Dict, Optional
from config import SimulationConfig, CrackConfig


# ===================================================================
# Signed distance functions  (seed 305_dist_plot)
# ===================================================================

def signed_distance_circle(px: np.ndarray, py: np.ndarray,
                           cx: float, cy: float, r: float) -> np.ndarray:
    """φ(x,y) = sqrt((x-cx)² + (y-cy)²) - r.
    Negative inside, positive outside, zero on the boundary."""
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2) - r


def signed_distance_segment(px: np.ndarray, py: np.ndarray,
                            x1: float, y1: float,
                            x2: float, y2: float) -> np.ndarray:
    """Signed distance to the line segment from (x1,y1) to (x2,y2).
    Uses the projection-based formulation:
        d = |PA × PB| / |AB|   with sign from normal direction."""
    dx_seg, dy_seg = x2 - x1, y2 - y1
    seg_len = math.sqrt(dx_seg ** 2 + dy_seg ** 2)
    if seg_len < 1.0e-30:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    # Parametric coordinate of projection onto segment
    t_param = ((px - x1) * dx_seg + (py - y1) * dy_seg) / (seg_len ** 2)
    t_param = np.clip(t_param, 0.0, 1.0)
    proj_x = x1 + t_param * dx_seg
    proj_y = y1 + t_param * dy_seg
    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    # Sign: positive above the crack line (convention)
    normal_x, normal_y = -dy_seg / seg_len, dx_seg / seg_len
    side = (px - x1) * normal_x + (py - y1) * normal_y
    return np.where(side >= 0.0, dist, -dist)


def signed_distance_union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Boolean union of two SDFs:  d_union = min(d1, d2)."""
    return np.minimum(d1, d2)


def signed_distance_intersection(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Boolean intersection:  d_inter = max(d1, d2)."""
    return np.maximum(d1, d2)


def signed_distance_difference(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Boolean difference (d1 \ d2):  d_diff = max(d1, -d2)."""
    return np.maximum(d1, -d2)


# ===================================================================
# Crack level-set fields
# ===================================================================

def compute_crack_level_sets(x: np.ndarray, y: np.ndarray,
                             crack: CrackConfig) -> Dict[str, np.ndarray]:
    """Compute the two level-set fields for an edge crack.

    φ(x,y) — signed distance to the crack line (normal to crack).
    ψ(x,y) — signed distance to the crack tip (tangential).

    The crack is defined by its tip position, length, and angle.
    """
    cos_a = math.cos(crack.angle)
    sin_a = math.sin(crack.angle)
    tail_x = crack.tip_x - crack.length * cos_a
    tail_y = crack.tip_y - crack.length * sin_a

    # φ: signed distance to the crack line (infinite line through tail→tip)
    phi = signed_distance_segment(x, y, tail_x, tail_y,
                                  crack.tip_x, crack.tip_y)
    # Enforce: only negative / zero *behind* the crack tip
    # Project onto crack direction; points ahead of tip get φ = +large
    local_x = (x - crack.tip_x) * cos_a + (y - crack.tip_y) * sin_a
    phi = np.where(local_x > 0.0, np.sqrt(local_x ** 2 + phi ** 2), phi)

    # ψ: signed distance to the crack tip
    psi = np.sqrt((x - crack.tip_x) ** 2 + (y - crack.tip_y) ** 2)
    psi = np.where(local_x >= 0.0, psi, -psi)

    return {"phi": phi, "psi": psi}


# ===================================================================
# Background Cartesian grid
# ===================================================================

def build_cartesian_grid(cfg: SimulationConfig) -> Dict[str, np.ndarray]:
    """Create a uniform Cartesian grid with metric information.

    Returns dict with:
      x, y         — 2D coordinate arrays  (ny, nx)
      dx, dy       — grid spacings
      hx, hy       — 1D coordinate vectors
      area_element — dA for each cell  (ny, nx)
    """
    nx, ny = cfg.numerical.nx, cfg.numerical.ny
    x0, x1 = cfg.numerical.domain_x
    y0, y1 = cfg.numerical.domain_y

    hx = np.linspace(x0, x1, nx)
    hy = np.linspace(y0, y1, ny)
    dx = hx[1] - hx[0] if nx > 1 else 1.0
    dy = hy[1] - hy[0] if ny > 1 else 1.0

    x2d, y2d = np.meshgrid(hx, hy)
    area = np.full_like(x2d, dx * dy)

    return {
        "x": x2d, "y": y2d,
        "dx": dx, "dy": dy,
        "hx": hx, "hy": hy,
        "area_element": area,
        "nx": nx, "ny": ny,
    }


# ===================================================================
# Fibonacci annular grid around crack tip  (seed 008_annulus_grid)
# ===================================================================

def fibonacci_annular_grid(cx: float, cy: float,
                           r_inner: float, r_outer: float,
                           n_radial: int, n_angular: int,
                           stretch: float = 1.5
                           ) -> Dict[str, np.ndarray]:
    """Generate a quasi-structured annular grid using the Fibonacci spiral.

    The golden ratio φ_g = (1+√5)/2 controls the angular increment.
    Radial points are stretched toward the inner radius using:
        r_i = r_inner + (r_outer - r_inner) * (i/(N-1))^stretch

    This maps naturally to the crack-tip singularity zone where
    the stress field σ ~ K_I / sqrt(2πr) requires high resolution
    near r → 0.

    Returns dict with  x, y, r, theta, jacobian  arrays.
    """
    phi_g = (1.0 + math.sqrt(5.0)) / 2.0   # golden ratio
    golden_angle = 2.0 * math.pi / (phi_g ** 2)   # ≈ 137.508°

    points_x, points_y, points_r, points_theta = [], [], [], []

    for j in range(n_angular):
        theta_offset = j * golden_angle
        for i in range(n_radial):
            # Stretched radial coordinate
            frac = i / max(n_radial - 1, 1)
            r_i = r_inner + (r_outer - r_inner) * (frac ** stretch)
            theta_i = theta_offset + 2.0 * math.pi * i / max(n_radial, 1)
            px = cx + r_i * math.cos(theta_i)
            py = cy + r_i * math.sin(theta_i)
            points_x.append(px)
            points_y.append(py)
            points_r.append(r_i)
            points_theta.append(theta_i)

    x_arr = np.array(points_x)
    y_arr = np.array(points_y)
    r_arr = np.array(points_r)
    theta_arr = np.array(points_theta)

    # Jacobian  J = r  (for polar mapping)
    jacobian = r_arr.copy()

    return {
        "x": x_arr, "y": y_arr,
        "r": r_arr, "theta": theta_arr,
        "jacobian": jacobian,
        "n_points": len(x_arr),
    }


# ===================================================================
# Composite grid: background + crack-tip refinement
# ===================================================================

def build_damage_grid(cfg: SimulationConfig) -> Dict:
    """Build the composite grid for damage simulation.

    Combines a Cartesian background with a Fibonacci annular refinement
    around the crack tip. Returns a unified grid dictionary.
    """
    # Background grid
    bg = build_cartesian_grid(cfg)

    # Crack-tip annular grid
    ann = fibonacci_annular_grid(
        cx=cfg.crack.tip_x, cy=cfg.crack.tip_y,
        r_inner=cfg.numerical.annulus_inner_radius,
        r_outer=cfg.numerical.annulus_outer_radius,
        n_radial=cfg.numerical.annulus_points_radial,
        n_angular=cfg.numerical.annulus_points_angular,
        stretch=1.5,
    )

    # Level sets on the background grid
    ls = compute_crack_level_sets(bg["x"], bg["y"], cfg.crack)

    # Compute distance-to-tip field on background grid (for refinement mask)
    dist_to_tip = np.sqrt(
        (bg["x"] - cfg.crack.tip_x) ** 2 +
        (bg["y"] - cfg.crack.tip_y) ** 2
    )

    # Refinement weight: w = 1 / (1 + (r / r_char)^2)
    r_char = cfg.material.characteristic_length
    refinement_weight = 1.0 / (1.0 + (dist_to_tip / r_char) ** 2)

    # Enrichment indicator: Heaviside enrichment H(x) = sign(φ)
    heaviside_enrichment = np.sign(ls["phi"])

    # Crack-tip asymptotic enrichment functions (Westergaard)
    # For mode-I:  {sqrt(r)*sin(θ/2), sqrt(r)*cos(θ/2),
    #               sqrt(r)*sin(θ/2)*sin(θ), sqrt(r)*cos(θ/2)*cos(θ)}
    r_tip = np.maximum(dist_to_tip, 1.0e-15)
    theta_tip = np.arctan2(bg["y"] - cfg.crack.tip_y,
                           bg["x"] - cfg.crack.tip_x)
    sqrt_r = np.sqrt(r_tip)
    enrichment_funcs = {
        "F1": sqrt_r * np.sin(theta_tip / 2.0),
        "F2": sqrt_r * np.cos(theta_tip / 2.0),
        "F3": sqrt_r * np.sin(theta_tip / 2.0) * np.sin(theta_tip),
        "F4": sqrt_r * np.cos(theta_tip / 2.0) * np.cos(theta_tip),
    }

    return {
        "background": bg,
        "annular": ann,
        "level_sets": ls,
        "dist_to_tip": dist_to_tip,
        "refinement_weight": refinement_weight,
        "heaviside_enrichment": heaviside_enrichment,
        "enrichment_funcs": enrichment_funcs,
    }


# ===================================================================
# Grid quality metrics
# ===================================================================

def compute_grid_quality(grid: Dict) -> Dict[str, float]:
    """Evaluate grid quality metrics.

    Returns:
      min_spacing  — smallest distance between adjacent points
      max_aspect   — maximum aspect ratio of cells
      skewness     — max deviation from orthogonality (degrees)
    """
    bg = grid["background"]
    dx, dy = bg["dx"], bg["dy"]

    min_spacing = min(dx, dy)
    max_aspect = max(dx / dy, dy / dx) if dy > 0 else 1.0

    # Skewness: for uniform Cartesian grid this is 0
    skewness = 0.0

    ann = grid["annular"]
    if ann["n_points"] > 0:
        r_min = ann["r"].min()
        if r_min < min_spacing:
            min_spacing = r_min

    return {
        "min_spacing": float(min_spacing),
        "max_aspect_ratio": float(max_aspect),
        "skewness_deg": float(skewness),
        "n_background": int(bg["nx"] * bg["ny"]),
        "n_annular": int(ann["n_points"]),
    }
