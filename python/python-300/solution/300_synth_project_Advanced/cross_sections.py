"""
cross_sections.py
=================
ENDF-like multigroup cross-section library with **piecewise-linear 2-D
interpolation** on (energy, temperature) grids.

The core algorithm is adapted from the `pwl_interp_2d` routine (Burkardt)
which evaluates a function tabulated on a rectangular grid by linear
interpolation inside each rectangle.  We extend it to:
  (i)  handle the strongly energy-dependent neutron cross-section surfaces
       sigma_{r,g}(T) that arise from Doppler broadening, and
  (ii) provide stratified sampling of the uncertainty envelope inspired by
       the `image_sample` routine (again Burkardt).

Two reaction families are supported:
  * absorption / total / scattering (always positive);
  * the threshold reaction  7Li(n,n't)4He  which turns on at E_th = 2.47 MeV.

The interpolation stencil is the canonical one:

      z(u,v) = (1-u)(1-v) z_00 + u(1-v) z_10
             + (1-u)v     z_01 + u v     z_11

with u, v in [0,1] the local normalised coordinates inside the rectangle
whose lower-left corner holds z_00.

Adapted from seed projects:
    * 927_pwl_interp_2d  -> pwl_interp_2d_eval
    * 585_image_sample   -> stratified_uncertainty_sampler
"""

from __future__ import annotations
import math
from typing import Callable, List, Optional, Sequence, Tuple

import physics_constants as pc


# ---------------------------------------------------------------------------
# Utility: clamp and find_interval (Burkardt's r8vec_bracket)
# ---------------------------------------------------------------------------
def find_interval(grid: Sequence[float], x: float) -> int:
    """Return index i such that grid[i] <= x < grid[i+1].

    Clamps to the last valid interval when x lies outside the grid.  This is
    the 1-D bracketing routine used as the inner loop of pwl_interp_2d.
    """
    n = len(grid)
    if n < 2:
        raise ValueError("grid must contain at least two nodes")
    if x <= grid[0]:
        return 0
    if x >= grid[n - 1]:
        return n - 2
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# 2-D piecewise-linear interpolation (Burkardt's pwl_interp_2d)
# ---------------------------------------------------------------------------
def pwl_interp_2d_eval(
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    z_table: Sequence[Sequence[float]],
    xq: float,
    yq: float,
) -> float:
    """Evaluate a 2-D piecewise-linear interpolant at a single point.

    Parameters
    ----------
    x_grid : monotonically increasing x nodes, length nx.
    y_grid : monotonically increasing y nodes, length ny.
    z_table : ny-by-nx nested list, z_table[j][i] is the value at
              (x_grid[i], y_grid[j]).
    xq, yq : query point.

    Returns
    -------
    The linearly interpolated value at (xq, yq).
    """
    nx = len(x_grid)
    ny = len(y_grid)
    if nx < 2 or ny < 2:
        raise ValueError("grids must each have length >= 2")
    if len(z_table) != ny or any(len(row) != nx for row in z_table):
        raise ValueError("z_table shape is inconsistent with grids")

    i = find_interval(x_grid, xq)
    j = find_interval(y_grid, yq)

    x0, x1 = x_grid[i], x_grid[i + 1]
    y0, y1 = y_grid[j], y_grid[j + 1]
    dx = x1 - x0 if x1 != x0 else 1.0
    dy = y1 - y0 if y1 != y0 else 1.0
    u = (xq - x0) / dx
    v = (yq - y0) / dy
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))

    z00 = z_table[j][i]
    z10 = z_table[j][i + 1]
    z01 = z_table[j + 1][i]
    z11 = z_table[j + 1][i + 1]
    return (
        (1.0 - u) * (1.0 - v) * z00
        + u * (1.0 - v) * z10
        + (1.0 - u) * v * z01
        + u * v * z11
    )


# ---------------------------------------------------------------------------
# Doppler-broadened cross-section surfaces
# ---------------------------------------------------------------------------
# The Doppler broadening of a single resonance follows the single-level
# Breit-Wigner form with the thermal kernel folded in.  For the *effective*
# multigroup cross section at temperature T we adopt the analytic
# approximation (Ade et al., 1969):
#
#   sigma_a^eff(T) = sigma_a^eff(T0) * sqrt(T0 / T)
#                   * [ 1 + (Gamma_gamma / (2 E_r)) * (sqrt(T) - sqrt(T0)) ]
#
# where E_r is the resonance energy and Gamma_gamma the capture width.
# This captures the 1/v + sqrt(T) behaviour relevant to Li-6(n,t).
# ---------------------------------------------------------------------------
def doppler_scale(E_r_mev: float, Gamma_eV: float,
                  T0_K: float, T_K: float) -> float:
    """Return the ratio sigma_eff(T) / sigma_eff(T0) for one resonance.

    The scaling uses the classic narrow-resonance Doppler formula with an
    additional sqrt(T) correction from the Gaussian thermal kernel:

        R(T) = sqrt(T0 / T) * (1 + alpha * (sqrt(T) - sqrt(T0)))
        alpha = Gamma_gamma / (2 * E_r)
    """
    if T0_K <= 0.0 or T_K <= 0.0 or E_r_mev <= 0.0:
        return 1.0
    alpha = (Gamma_eV * 1.0e-6) / (2.0 * E_r_mev)   # dimensionless
    ratio = math.sqrt(T0_K / T_K) * (
        1.0 + alpha * (math.sqrt(T_K) - math.sqrt(T0_K))
    )
    return max(ratio, 1.0e-6)


# ---------------------------------------------------------------------------
# Li-6(n,t)4He threshold-free cross-section surface
# ---------------------------------------------------------------------------
# At thermal energies sigma_{n,t}^{Li6} ~ 940 barn (1/v); we represent the
# 14 fast groups with a smooth interpolation surface over (E, T).
# ---------------------------------------------------------------------------
def _build_li6_nt_surface(
    T_grid_K: Sequence[float],
) -> Tuple[List[float], List[float], List[List[float]]]:
    """Return (E_grid, T_grid, sigma_table) for 6Li(n,t)4He.

    The table shape is (len(T_grid), len(E_grid)).
    """
    E_grid = list(pc.GROUP_BOUNDS_MEV[:-1])          # upper edges, high->low
    # We use the geometric-mean energy per group as representative abscissa.
    E_rep = [
        math.sqrt(pc.GROUP_BOUNDS_MEV[g] *
                  max(pc.GROUP_BOUNDS_MEV[g + 1], pc.EPS_NUMERICAL))
        for g in range(pc.N_GROUPS)
    ]
    table: List[List[float]] = []
    T0 = 300.0
    for T in T_grid_K:
        row: List[float] = []
        for g, Eg in enumerate(E_rep):
            # 1/v reference
            sigma_ref = 940.0 * math.sqrt(2.53e-8 / max(Eg, pc.EPS_NUMERICAL))
            sigma_ref = min(sigma_ref, 5000.0)
            sigma_ref = max(sigma_ref, 1.0e-3)
            scale = doppler_scale(E_r_mev=Eg, Gamma_eV=3.0,
                                   T0_K=T0, T_K=T)
            row.append(sigma_ref * scale)
        table.append(row)
    return E_rep, list(T_grid_K), table


def _build_li7_total_surface(
    T_grid_K: Sequence[float],
) -> Tuple[List[float], List[float], List[List[float]]]:
    """Return (E_grid, T_grid, sigma_table) for 7Li total.

    Includes the (n,n't) threshold turn-on at E_th = 2.47 MeV.  The
    cross-section follows the empirical form

        sigma_{n,n't}(E) = sigma_0 * (1 - E_th / E)^2 * H(E - E_th)

    with H the Heaviside step and sigma_0 = 0.50 barn (ENDF fit).
    """
    E_rep = [
        math.sqrt(pc.GROUP_BOUNDS_MEV[g] *
                  max(pc.GROUP_BOUNDS_MEV[g + 1], pc.EPS_NUMERICAL))
        for g in range(pc.N_GROUPS)
    ]
    E_th = 2.468
    sigma_0 = 0.50
    table: List[List[float]] = []
    for T in T_grid_K:
        row: List[float] = []
        for Eg in E_rep:
            # baseline total (smooth, slowly decreasing)
            base = 2.10 + 0.05 * math.log10(max(Eg, pc.EPS_NUMERICAL) + 1.0)
            # threshold contribution
            if Eg > E_th:
                thr = sigma_0 * (1.0 - E_th / Eg) ** 2
            else:
                thr = 0.0
            # mild Doppler smearing near threshold
            kT_mev = pc.K_BOLTZMANN_MEV * T
            smear = 4.0 * math.sqrt(kT_mev * E_th) / max(E_th, pc.EPS_NUMERICAL)
            thr *= (1.0 + 0.5 * smear)
            row.append(max(base + thr, 1.0e-3))
        table.append(row)
    return E_rep, list(T_grid_K), table


# ---------------------------------------------------------------------------
# MultigroupCrossSection library
# ---------------------------------------------------------------------------
class MultigroupCrossSection:
    """Tabulated multigroup cross-section library with PWL interp.

    Parameters
    ----------
    material : str
        One of 'li2o', 'lipb', 'eurofer', 'li6', 'li7'.
    T_grid_K : sequence of float, optional
        Temperature grid (K) for the interpolation table.
    """

    SUPPORTED: Tuple[str, ...] = ('li2o', 'lipb', 'eurofer', 'li6', 'li7')

    def __init__(
        self,
        material: str,
        T_grid_K: Optional[Sequence[float]] = None,
    ) -> None:
        if material not in self.SUPPORTED:
            raise ValueError(f"Unsupported material {material!r}")
        self.material = material
        self.T_grid = list(T_grid_K) if T_grid_K is not None else [
            300.0, 600.0, 900.0, 1200.0, 1500.0
        ]
        if material == 'li6':
            E, T, tab = _build_li6_nt_surface(self.T_grid)
            self.reaction = 'li6_nt'
        elif material == 'li7':
            E, T, tab = _build_li7_total_surface(self.T_grid)
            self.reaction = 'li7_total'
        elif material == 'li2o':
            # Build by superposition of Li6 and Li7
            E, T, tab_li6 = _build_li6_nt_surface(self.T_grid)
            _, _, tab_li7 = _build_li7_total_surface(self.T_grid)
            tab = [
                [pc.ATOM_DENSITY_LI * (pc.FRAC_LI6 * tab_li6[j][g]
                                        + pc.FRAC_LI7 * tab_li7[j][g])
                 + pc.ATOM_DENSITY_O * 3.5
                 for g in range(pc.N_GROUPS)]
                for j in range(len(T))
            ]
            self.reaction = 'li2o_total_macro'
        elif material == 'lipb':
            E, T, tab_li7 = _build_li7_total_surface(self.T_grid)
            tab = [
                [pc.ATOM_DENSITY_LIPB * pc.FRAC_LI_IN_LIPB * tab_li7[j][g]
                 + pc.ATOM_DENSITY_LIPB * pc.FRAC_PB * 5.2
                 for g in range(pc.N_GROUPS)]
                for j in range(len(T))
            ]
            self.reaction = 'lipb_total_macro'
        else:
            # Eurofer: smooth, structure-steel-like total XS
            E_rep = [
                math.sqrt(pc.GROUP_BOUNDS_MEV[g] *
                          max(pc.GROUP_BOUNDS_MEV[g + 1], pc.EPS_NUMERICAL))
                for g in range(pc.N_GROUPS)
            ]
            E, T = E_rep, self.T_grid
            tab = [
                [pc.ATOM_DENSITY_EUROFER * (3.0 + 0.05 * math.log10(Eg + 1.0))
                 for Eg in E_rep]
                for _ in T
            ]
            self.reaction = 'eurofer_total_macro'
        self.E_grid = E
        self.T_grid = T
        self.table = tab

    # ------------------------------------------------------------------
    def sigma(self, group: int, temperature_K: float) -> float:
        """Return sigma_g(T) in barn (microscopic) or cm^{-1} (macroscopic).

        For the pure-isotope libraries ('li6', 'li7') the result is in barn;
        for compound materials it is already macroscopic (cm^{-1}).
        """
        if not 0 <= group < pc.N_GROUPS:
            raise ValueError("group index out of range")
        E = self.E_grid[group]
        return pwl_interp_2d_eval(self.E_grid, self.T_grid, self.table, E,
                                   temperature_K)

    # ------------------------------------------------------------------
    def spectrum(self, temperature_K: float) -> List[float]:
        """Return sigma_g(T) for all g, as a length-N_GROUPS list."""
        return [self.sigma(g, temperature_K) for g in range(pc.N_GROUPS)]


# ---------------------------------------------------------------------------
# Stratified uncertainty sampler (from 585_image_sample)
# ---------------------------------------------------------------------------
def stratified_sample_cross_section(
    lib: MultigroupCrossSection,
    group: int,
    T_center_K: float,
    T_sigma_K: float,
    n_samples: int,
    seed: int = 1729,
) -> Tuple[float, float]:
    """Estimate mean and std of sigma_g(T) for T ~ N(T_center, T_sigma).

    The unit-interval [0,1] is split into n_samples equal strata and a
    Halton-style antithetic pair is drawn inside each stratum.  The
    cumulative normal is inverted via the rational approximation of
    Abramowitz & Stegun (26.2.23).

    Returns
    -------
    mean, std of the sampled sigma_g values.
    """
    if n_samples <= 0:
        return 0.0, 0.0

    def norm_ppf(p: float) -> float:
        # Rational approx for p in (0,1); handle tails safely.
        if p <= 0.0:
            return -8.0
        if p >= 1.0:
            return 8.0
        if p < 0.5:
            t = math.sqrt(-2.0 * math.log(p))
            s = -(2.515517 + t * (0.802853 + 0.010328 * t)) / (
                1.0 + t * (1.432788 + t * (0.189269 + 0.001308 * t)))
            return s - t
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
            s = (2.515517 + t * (0.802853 + 0.010328 * t)) / (
                1.0 + t * (1.432788 + t * (0.189269 + 0.001308 * t)))
            return t - s

    values: List[float] = []
    # Deterministic LCG for reproducibility
    state = (seed * 2654435761) & 0xFFFFFFFF
    for k in range(n_samples):
        # Stratum centre + jitter
        u = (k + 0.5) / n_samples
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        jitter = (state / 0xFFFFFFFF - 0.5) / n_samples
        p = max(1.0e-6, min(1.0 - 1.0e-6, u + jitter))
        z = norm_ppf(p)
        T_sample = T_center_K + T_sigma_K * z
        T_sample = max(100.0, T_sample)
        values.append(lib.sigma(group, T_sample))
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var)
