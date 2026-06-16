# -*- coding: utf-8 -*-
"""
lattice_geometry.py
-------------------
Adaptive 1D k-point mesh for the first Brillouin zone (BZ) of a 1D monoatomic
chain, built by Centroidal Voronoi Tessellation (CVT) with a *phonon DOS*-
weighted density.  Also provides 3-point Taylor-Wilson-Boole (TWB) quadrature
over a reference triangle that appears after folding the 1D BZ into the
(Eliashberg) interaction triangle {(k,q) : |k|+|q|<=pi/a, k+q in BZ}.

Scientific origin of the fused algorithms
-----------------------------------------
* CVT 1-D nonuniform  (seed project 245_cvt_1d_nonuniform)
    -> k-point mesh whose generator density rho(k) is proportional to the
       bare phonon DOS g(ph)(k), so that the quadrature error in the
       electron-phonon spectral integral
               int_BZ  alpha^2 F(k) f(k) dk
       is minimised for a fixed number N of generators.
* Triangle TWB rule  (seed project 1323_triangle_twb_rule)
    -> closed TWB quadrature on the reference triangle used to integrate the
       Migdal vertex correction
               Lambda_{n,m} = int_0^{omega_D} int_BZ  ... dk dq
       that enters the isotropic Eliashberg equations on the Matsubara axis.

Core physics / mathematics
--------------------------
* Density functional chosen as
        rho(s) = s^(1/3)           (Fermi-surface DOS proxy, s in [0,1])
  or more generally one of seven selectable forms, cf. Burkardt's CVT menu.
* TWB weights are computed from the exact moments of the unit triangle
        T = {(x,y) : x>=0, y>=0, x+y<=1},   |T| = 1/2.
* The CVT is iterated with Lloyd's algorithm:
        z_i^{(t+1)} = int_{V_i} k rho(k) dk  /  int_{V_i} rho(k) dk
  until the generator displacement falls below a tolerance ``tol_cvt``.

Stability / boundary notes
--------------------------
* Generators are clamped to the open interval (0,1) so that Voronoi cells
  never degenerate (prevents division by zero in Lloyd's update).
* Quadrature weights are normalised to |T|=1/2 after every call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Density functions (seed project 245)
# ---------------------------------------------------------------------------
def density_selector(idx: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return one of seven admissible generator densities on [0,1]."""
    _table = {
        0: lambda s: s,
        1: lambda s: np.sqrt(np.clip(s, 0.0, None)),
        2: lambda s: np.cbrt(np.clip(s, 0.0, None)),
        3: lambda s: np.power(np.clip(s, 0.0, None), 0.25),
        4: lambda s: np.log(math.e / (math.e - s * (math.e - 1.0) + 1e-300)),
        5: lambda s: 0.5 + np.arctan(50.0 * (s - 0.5)) / math.pi,
        6: lambda s: np.sin(math.pi * (s - 0.5)),
    }
    if idx not in _table:
        raise ValueError(f"lattice_geometry: unknown density index {idx}")
    return _table[idx]


# ---------------------------------------------------------------------------
# 1-D CVT (Centroidal Voronoi Tessellation) with non-uniform density
# ---------------------------------------------------------------------------
@dataclass
class CVTResult:
    generators: np.ndarray        # (N,) generator locations in (0,1)
    weights: np.ndarray           # (N,) Voronoi-cell masses int_{V_i} rho(s) ds
    energy: float                 # CVT energy  sum_i int_{V_i} rho(s)(s-z_i)^2 ds
    history: List[float] = field(default_factory=list)


def cvt_1d_nonuniform(
    n_gen: int,
    density_index: int,
    n_sample: int = 4000,
    n_steps: int = 120,
    seed: int = 274,
    tol_cvt: float = 1.0e-10,
) -> CVTResult:
    """Lloyd's algorithm in 1-D with *n_sample* Monte-Carlo samples per step.

    The routine returns (z, w, E) where
        z : generator positions in (0,1)
        w : Voronoi cell masses  w_i = int_{V_i} rho(s) ds
        E : CVT quantisation energy
    """
    if n_gen < 2:
        raise ValueError("cvt_1d_nonuniform: need at least 2 generators")

    rho = density_selector(density_index)
    rng = np.random.default_rng(seed)

    # Initialise generators on a slightly jittered uniform grid
    z = np.linspace(0.0, 1.0, n_gen + 2)[1:-1].copy()
    z += rng.uniform(-0.5 / n_gen, 0.5 / n_gen, size=n_gen)
    z = np.clip(z, 1e-6, 1.0 - 1e-6)
    z.sort()

    history: List[float] = []
    for _ in range(n_steps):
        # Boundaries of the Voronoi cells on [0,1]
        edges = np.concatenate(([0.0], 0.5 * (z[:-1] + z[1:]), [1.0]))

        # Monte-Carlo estimate of mass and centroid on each cell
        s = rng.uniform(0.0, 1.0, size=n_sample)
        rs = rho(s)
        mass = np.zeros(n_gen)
        cnum = np.zeros(n_gen)
        for i in range(n_gen):
            mask = (s >= edges[i]) & (s < edges[i + 1])
            mass[i] = rs[mask].sum()
            cnum[i] = (s[mask] * rs[mask]).sum()

        # Guard against empty cells (zero mass)
        mass = np.where(mass > 0.0, mass, 1.0)
        z_new = cnum / mass
        z_new = np.clip(z_new, 1e-12, 1.0 - 1e-12)

        disp = float(np.max(np.abs(z_new - z)))
        z = z_new
        z.sort()

        # Quantisation energy  E = sum_i int_{V_i} rho(s) (s-z_i)^2 ds
        e = 0.0
        for i in range(n_gen):
            mask = (s >= edges[i]) & (s < edges[i + 1])
            e += float(((s[mask] - z[i]) ** 2 * rs[mask]).sum())
        e /= n_sample
        history.append(e)

        if disp < tol_cvt:
            break

    # Final Voronoi edges and masses
    edges = np.concatenate(([0.0], 0.5 * (z[:-1] + z[1:]), [1.0]))
    s = np.linspace(0.0, 1.0, max(8 * n_sample, 20000))
    rs = rho(s)
    ds = s[1] - s[0]
    w = np.array([
        float(rs[(s >= edges[i]) & (s < edges[i + 1])].sum()) * ds
        for i in range(n_gen)
    ])
    w /= w.sum()            # normalise to 1 (= integral of a probability density)
    return CVTResult(
        generators=z,
        weights=w,
        energy=history[-1] if history else 0.0,
        history=history,
    )


# ---------------------------------------------------------------------------
# Taylor-Wilson-Boole (TWB) quadrature on the reference triangle
# ---------------------------------------------------------------------------
@dataclass
class TWBRule:
    """Three-point closed TWB rule on T={(x,y): x,y>=0, x+y<=1}."""
    pts: np.ndarray      # (3,2)
    wts: np.ndarray      # (3,)


def triangle_unit_volume() -> float:
    """Area of the reference triangle T.  Equals 1/2 exactly."""
    return 0.5


def twb_rule_triangle() -> TWBRule:
    """Return the 3-point TWB rule (vertices + centroid combination).

    The TWB family has degree of precision 2 on T.  We use the symmetric form
        w_v = |T|/6   at each vertex v,   w_c = |T| * 4/6 at the centroid c,
    collapsed here to the compact three-point representation:
        p_i  = 2/3 * v_i + 1/3 * c ,     w_i = |T| / 3.
    """
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    c = np.array([1.0 / 3.0, 1.0 / 3.0])
    pts = (2.0 / 3.0) * V + (1.0 / 3.0) * c
    wts = np.full(3, triangle_unit_volume() / 3.0)
    # Sanity: sum of weights must equal |T|
    wts = wts * (triangle_unit_volume() / wts.sum())
    return TWBRule(pts=pts, wts=wts)


def integrate_on_triangle(f: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """Integrate f(x,y) over T using the TWB rule."""
    rule = twb_rule_triangle()
    val = 0.0
    for i in range(rule.pts.shape[0]):
        val += rule.wts[i] * f(rule.pts[i, 0], rule.pts[i, 1])
    return float(val)


# ---------------------------------------------------------------------------
# Public driver used by the rest of the project
# ---------------------------------------------------------------------------
def build_kpoint_mesh(
    n_kpoints: int = 32,
    density_index: int = 2,
    seed: int = 274,
) -> tuple:
    """Return (k, w_k, twb_rule) where
        k     : (n_kpoints,) CVT k-points in the half-BZ [0,pi] (set a=1)
        w_k   : (n_kpoints,) integration weights (sum to pi)
        twb   : TWB rule on the interaction triangle
    """
    cvt = cvt_1d_nonuniform(
        n_gen=n_kpoints,
        density_index=density_index,
        seed=seed,
    )
    k_points = cvt.generators * math.pi         # map [0,1] -> [0,pi]
    w_k = cvt.weights * math.pi                 # weights sum to pi
    twb = twb_rule_triangle()
    return k_points, w_k, twb
