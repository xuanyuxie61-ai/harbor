"""
grid_manager.py
===============
Cylindrical (R, phi, z) shearing-box grid with optional logarithmic
radial spacing and adaptive refinement level selection by integer
bisection.

The module combines three ideas drawn from the seed projects:

    * The node/element data layout and file I/O conventions of the
      fd_to_tec tool (351_fd_to_tec) -- we keep explicit metric arrays
      (cell volumes, face areas) that are the structured-grid analogue
      of the unstructured FEM connectivity tables in ffmatlib (425).
    * The integer-bisection root finder (095) repurposed as an h/p
      refinement decision: given a target normalised MRI wavelength
      lambda_mri / H and a resolution floor/ceiling, bisect for the
      coarsest integer cell count that still resolves lambda_mri with
      at least N_cells_per_lambda points.
    * The parallel-Monte-Carlo envelope of the high-card simulation
      (533_high_card_parfor) is used to estimate, via sampling, the
      probability that the selected grid actually captures the local
      MRI growth rate within a prescribed tolerance.

The coordinate system is a Cartesian shearing box centred at (R0, phi0,
z=0), so that

    x  =  R - R0                 (radial,     outflow or periodic-shear)
    y  =  R0 (phi - phi0)        (azimuthal,  periodic)
    z  =  z                      (vertical,   reflecting)

and the differential rotation profile is linearised as

    v_y(x) = - q Omega0 x

with q = 3/2 for a Keplerian disk.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np

import physical_constants as pc


# ---------------------------------------------------------------------------
#                 Grid data class (analogous to fem_read output)
# ---------------------------------------------------------------------------
@dataclass
class ShearingBoxGrid:
    """Structured 3-D shearing-box grid with metric coefficients."""
    Nx: int
    Ny: int
    Nz: int
    Lx: float                # code units (H)
    Ly: float                # code units (H)
    Lz: float                # code units (H)
    x: np.ndarray = field(default_factory=lambda: np.zeros(1))   # (Nx+1,)
    y: np.ndarray = field(default_factory=lambda: np.zeros(1))   # (Ny+1,)
    z: np.ndarray = field(default_factory=lambda: np.zeros(1))   # (Nz+1,)
    xc: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (Nx,)
    yc: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (Ny,)
    zc: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (Nz,)
    dx: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (Nx,)
    dy: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (Ny,)
    dz: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (Nz,)
    volume: np.ndarray = field(default_factory=lambda: np.zeros(1))  # scalar
    refinement_level: int = 0
    cells_per_mri: float = 0.0


# ---------------------------------------------------------------------------
#                      Core grid constructor
# ---------------------------------------------------------------------------
def build_grid(Nx: int, Ny: int, Nz: int,
               Lx: float, Ly: float, Lz: float,
               radial_log: bool = False,
               refinement_level: int = 0) -> ShearingBoxGrid:
    """Construct a shearing-box grid with uniform or log-stretched x-spacing.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of cells in each direction.
    Lx, Ly, Lz : float
        Domain extents in code units (H).
    radial_log : bool
        If True, use a geometric stretching in x (useful for extending
        radial reach while keeping mid-plane resolution high).
    refinement_level : int
        Arbitrary integer tag; the bisection routine below writes here.
    """
    if Nx < 4 or Ny < 4 or Nz < 2:
        raise ValueError(
            f"grid_manager: resolution ({Nx},{Ny},{Nz}) below minimum (4,4,2)"
        )
    if Lx <= 0.0 or Ly <= 0.0 or Lz <= 0.0:
        raise ValueError("grid_manager: domain extents must be positive")

    g = ShearingBoxGrid(Nx=Nx, Ny=Ny, Nz=Nz,
                        Lx=Lx, Ly=Ly, Lz=Lz,
                        refinement_level=refinement_level)

    # ---- x direction (radial) ----
    if radial_log:
        # Geometric stretching factor: ratio of outermost to innermost dx
        stretch = 1.5
        ratio = stretch ** (1.0 / (Nx - 1)) if Nx > 1 else 1.0
        # Build node positions with a telescoping series
        dx0 = Lx * (ratio - 1.0) / (ratio**Nx - 1.0)
        dx_arr = dx0 * ratio ** np.arange(Nx)
        # Rescale so that sum(dx) = Lx
        dx_arr *= Lx / dx_arr.sum()
        x_node = np.concatenate(([0.0], np.cumsum(dx_arr))) - Lx / 2.0
        g.x = x_node
        g.dx = dx_arr
        g.xc = 0.5 * (x_node[:-1] + x_node[1:])
    else:
        g.x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx + 1)
        g.dx = np.full(Nx, Lx / Nx)
        g.xc = 0.5 * (g.x[:-1] + g.x[1:])

    # ---- y direction (azimuthal, strictly uniform & periodic) ----
    g.y  = np.linspace(-Ly / 2.0, Ly / 2.0, Ny + 1)
    g.dy = np.full(Ny, Ly / Ny)
    g.yc = 0.5 * (g.y[:-1] + g.y[1:])

    # ---- z direction (vertical, uniform, centred on mid-plane) ----
    g.z  = np.linspace(-Lz / 2.0, Lz / 2.0, Nz + 1)
    g.dz = np.full(Nz, Lz / Nz)
    g.zc = 0.5 * (g.z[:-1] + g.z[1:])

    # ---- scalar cell volume (Cartesian, dx * dy * dz for uniform) ----
    # For log-stretched x, volume is per-column; we store the mean here.
    g.volume = float(np.mean(g.dx)) * float(np.mean(g.dy)) * float(np.mean(g.dz))

    # ---- cells per MRI wavelength (informational) ----
    scales = pc.derived_scales()
    lam_mri = pc.mri_most_unstable_wavelength() / scales["H"]  # code units
    dx_min = float(np.min(g.dx))
    g.cells_per_mri = lam_mri / dx_min if dx_min > 0 else 0.0
    return g


# ---------------------------------------------------------------------------
#        Adaptive refinement level by integer bisection (from 095)
# ---------------------------------------------------------------------------
def bisect_refinement_level(target_cells_per_mri: float,
                            min_N: int = 8,
                            max_N: int = 512,
                            Ly: float = 4.0,
                            Lz: float = 1.0,
                            tolerance: float = 1.0e-3) -> Tuple[int, int]:
    """Bisect for the coarsest integer cell count N giving at least the
    requested number of cells per MRI wavelength in every direction.

    This is a direct adaptation of the integer bisection algorithm
    (095_bisection_integer): instead of a sign-change on a real-valued
    function we maintain a monotone integer residual

        F(N) = floor(N * dx_min / lambda_mri) - target_cells_per_mri

    and seek the smallest N with F(N) >= 0.

    Returns
    -------
    N_best : int
        The chosen cubic resolution (Nx = Ny = N_best, Nz = N_best // 4).
    level : int
        log2(N_best / min_N) rounded to the nearest integer.
    """
    if min_N < 4:
        min_N = 4
    if max_N < min_N:
        raise ValueError("grid_manager: max_N must be >= min_N")

    scales = pc.derived_scales()
    lam_mri_code = pc.mri_most_unstable_wavelength() / scales["H"]

    def residual(N: int) -> int:
        dx = Ly / N  # uniform azimuthal spacing
        return int(math.floor(N * dx / lam_mri_code)) - int(target_cells_per_mri)

    # Ensure sign change: F(min_N) typically negative, F(max_N) positive
    fa = residual(min_N)
    fb = residual(max_N)
    if fa >= 0:
        return min_N, 0
    if fb < 0:
        return max_N, int(round(math.log2(max_N / min_N)))

    a, b = min_N, max_N
    while abs(b - a) > 1:
        c = (a + b) // 2
        fc = residual(c)
        if fc == 0:
            # Walk back to the coarsest N that still satisfies the target
            while c > a and residual(c - 1) == 0:
                c -= 1
            level = int(round(math.log2(c / min_N))) if c > min_N else 0
            return c, level
        elif fc < 0:
            a = c
        else:
            b = c

    # Pick whichever of a, b satisfies the target with smaller N
    if residual(a) >= 0:
        best = a
    else:
        best = b
    level = int(round(math.log2(best / min_N))) if best > min_N else 0
    return best, level


# ---------------------------------------------------------------------------
#        Monte-Carlo estimate of grid adequacy (from 533_high_card)
# ---------------------------------------------------------------------------
def estimate_resolution_confidence(N: int, trials: int = 200,
                                   growth_tolerance: float = 0.1) -> float:
    """Monte-Carlo estimate of the probability that a uniform grid with N
    cells resolves the MRI growth rate to within ``growth_tolerance``.

    For each trial we draw a random realisation of the discretisation
    error (modelled as Gaussian with variance proportional to 1/N^p,
    p = 4 for a 4th-order compact scheme) and check whether the
    normalised error is smaller than the tolerance.
    """
    scales = pc.derived_scales()
    lam_mri_code = pc.mri_most_unstable_wavelength() / scales["H"]
    dx = scales["H"] * (pc.get("Ly_over_H") / N)
    # Truncation error amplitude ~ (dx / lambda_mri)^p with p=4
    err_amp = (dx / lam_mri_code) ** 4
    rng = np.random.default_rng(pc.get("seed") + N)
    errs = rng.normal(0.0, err_amp, size=trials)
    return float(np.mean(np.abs(errs) < growth_tolerance))


# ---------------------------------------------------------------------------
#        Minimum cell volume (robustness guard)
# ---------------------------------------------------------------------------
def min_cell_volume(g: ShearingBoxGrid) -> float:
    """Return the smallest cell volume in code units (H^3).

    This is used to detect pathological stretching and to floor the
    timestep denominator in the CFL computation.
    """
    return float(np.min(g.dx) * np.min(g.dy) * np.min(g.dz))


def aspect_ratio(g: ShearingBoxGrid) -> float:
    """Return max(dx,dy,dz) / min(dx,dy,dz); large values indicate that
    the grid may need rebalancing for stability."""
    d_all = np.concatenate([g.dx, g.dy, g.dz])
    return float(np.max(d_all) / max(np.min(d_all), 1.0e-30))
