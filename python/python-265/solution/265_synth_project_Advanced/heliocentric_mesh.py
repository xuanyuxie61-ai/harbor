# -*- coding: utf-8 -*-
"""
heliocentric_mesh.py
--------------------
Mesh generation and boundary handling for heliospheric cosmic-ray
transport computations.

The heliospheric domain is  r in [r_sun + eps, r_HP]  with  r_HP  the
heliopause distance (~ 120 AU).  Because the relevant scales range from
~0.1 AU (inner acceleration region) to ~100 AU (modulation boundary),
a **logarithmic radial mesh** is the natural choice:

    r_i = r_min * exp( i * ln(r_max / r_min) / (N_r - 1) ),
        i = 0, ..., N_r - 1.

The pitch-angle cosine mu = cos(theta) is discretised uniformly on
(-1 + eps, 1 - eps) to avoid the coordinate singularities at
mu = +/- 1 (where the focusing term diverges in 1/mu).

Functions also compute cell volumes, areas, and boundary masks.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

import cosmic_ray_physics as crp


# =====================================================================
# Logarithmic radial mesh
# =====================================================================
def logarithmic_radial_mesh(r_min: float = 0.05 * crp.AU,
                            r_max: float = 120.0 * crp.AU,
                            Nr: int = 128) -> np.ndarray:
    """Return a logarithmically-spaced radial grid of size Nr."""
    if Nr < 3:
        raise ValueError("logarithmic_radial_mesh: Nr >= 3 required.")
    if r_min <= 0.0 or r_max <= r_min:
        raise ValueError("logarithmic_radial_mesh: invalid r_min, r_max.")
    return np.exp(np.linspace(math.log(r_min), math.log(r_max), Nr))


# =====================================================================
# Uniform pitch-angle mesh
# =====================================================================
def pitch_angle_mesh(Nmu: int = 32, eps: float = 5.0e-3) -> np.ndarray:
    """Return mu-grid on (-1+eps, 1-eps), size Nmu."""
    if Nmu < 3:
        raise ValueError("pitch_angle_mesh: Nmu >= 3 required.")
    return np.linspace(-1.0 + eps, 1.0 - eps, Nmu)


# =====================================================================
# Rigidity grid (logarithmic, in V)
# =====================================================================
def rigidity_mesh(R_min_GV: float = 0.1,
                  R_max_GV: float = 1.0e3,
                  NR: int = 32) -> np.ndarray:
    """Return a logarithmic rigidity grid in volts (input in GV)."""
    return np.exp(np.linspace(math.log(R_min_GV * 1.0e9),
                              math.log(R_max_GV * 1.0e9), NR))


# =====================================================================
# Cell metrics
# =====================================================================
def radial_cell_metrics(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dr[i], r_face[i+1/2]) of length  N-1  and  N+1."""
    dr = np.diff(r)
    r_face = np.zeros(r.size + 1)
    r_face[1:-1] = 0.5 * (r[:-1] + r[1:])
    r_face[0] = r[0] - 0.5 * dr[0]
    r_face[-1] = r[-1] + 0.5 * dr[-1]
    return dr, r_face


def cell_volume_spherical(r_face: np.ndarray, dmu: float) -> np.ndarray:
    """Return the spherical-shell cell volume  (4 pi / 3)(r_{i+1/2}^3 -
    r_{i-1/2}^3) * dmu / 2  (integrated over mu).
    """
    return (4.0 * math.pi / 3.0) * (r_face[1:] ** 3 - r_face[:-1] ** 3) * (
        dmu / 2.0)


# =====================================================================
# Boundary masks
# =====================================================================
def boundary_mask(Nr: int, Nmu: int,
                  dirichlet_inner: bool = True,
                  dirichlet_outer: bool = True) -> np.ndarray:
    """Return a boolean mask (Nr, Nmu) with True on Dirichlet boundary
    nodes.  Interior nodes have False.
    """
    mask = np.zeros((Nr, Nmu), dtype=bool)
    if dirichlet_inner:
        mask[0, :] = True
    if dirichlet_outer:
        mask[-1, :] = True
    # pitch-angle endpoints are reflecting
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


# =====================================================================
# Inner / outer boundary conditions
# =====================================================================
def apply_inner_boundary(f: np.ndarray, f_inner: np.ndarray) -> None:
    """Replace the inner boundary  f[0, :]  with  f_inner.

    The inner boundary typically carries the unmodulated LIS or an
    injection spectrum at low altitude.
    """
    f[0, :] = f_inner


def apply_outer_boundary(f: np.ndarray, f_outer: np.ndarray) -> None:
    """Replace the outer boundary (heliopause) with  f_outer.

    Usually  f_outer = 0  for GCR modulation problems (free escape).
    """
    f[-1, :] = f_outer


def apply_pitch_angle_reflection(f: np.ndarray) -> None:
    """Enforce  f(r, mu = -1) = f(r, mu = +1)  (isotropy at the
    singular points) -- zero-flux reflection.
    """
    f[:, 0] = f[:, 1]
    f[:, -1] = f[:, -2]


def apply_all_boundaries(f: np.ndarray, f_inner: np.ndarray,
                         f_outer: np.ndarray) -> None:
    apply_inner_boundary(f, f_inner)
    apply_outer_boundary(f, f_outer)
    apply_pitch_angle_reflection(f)


# =====================================================================
# Restriction / prolongation for multigrid
# =====================================================================
def restrict_radial(f_fine: np.ndarray) -> np.ndarray:
    """Full-weighting restriction on the radial coordinate.

    For N_fine = 2 N_coarse - 1,  f_c[i] = (1/4) f_{2i-1}
                                      + (1/2) f_{2i}
                                      + (1/4) f_{2i+1}.
    """
    Nf = f_fine.shape[0]
    Nc = (Nf + 1) // 2
    Nmu = f_fine.shape[1]
    f_c = np.zeros((Nc, Nmu))
    for i in range(Nc):
        j = 2 * i
        if j == 0:
            f_c[i, :] = 0.75 * f_fine[0, :] + 0.25 * f_fine[1, :]
        elif j == Nf - 1:
            f_c[i, :] = 0.75 * f_fine[-1, :] + 0.25 * f_fine[-2, :]
        else:
            f_c[i, :] = 0.25 * f_fine[j - 1, :] \
                + 0.50 * f_fine[j, :] \
                + 0.25 * f_fine[j + 1, :]
    return f_c


def prolongate_radial(f_coarse: np.ndarray, Nr_fine: int) -> np.ndarray:
    """Linear prolongation (injection + averaging) on the radial
    coordinate.
    """
    Nc = f_coarse.shape[0]
    Nmu = f_coarse.shape[1]
    f_f = np.zeros((Nr_fine, Nmu))
    for i in range(Nr_fine):
        ic = i // 2
        if i % 2 == 0:
            f_f[i, :] = f_coarse[min(ic, Nc - 1), :]
        else:
            cL = min(ic, Nc - 1)
            cR = min(ic + 1, Nc - 1)
            f_f[i, :] = 0.5 * (f_coarse[cL, :] + f_coarse[cR, :])
    return f_f


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    r = logarithmic_radial_mesh(Nr=32)
    mu = pitch_angle_mesh(Nmu=16)
    print(f"[heliocentric_mesh] r grid:  Nr = {r.size}, "
          f"min = {r[0] / crp.AU:.3f} AU, max = {r[-1] / crp.AU:.1f} AU")
    print(f"[heliocentric_mesh] mu grid: Nmu = {mu.size}, "
          f"min = {mu[0]:+.4f}, max = {mu[-1]:+.4f}")
    dr, r_face = radial_cell_metrics(r)
    vol = cell_volume_spherical(r_face, mu[1] - mu[0])
    print(f"[heliocentric_mesh] first 3 cell volumes (m^3): {vol[:3]}")
    mask = boundary_mask(r.size, mu.size)
    print(f"[heliocentric_mesh] boundary node count: {int(mask.sum())}")


if __name__ == "__main__":
    _demo()
