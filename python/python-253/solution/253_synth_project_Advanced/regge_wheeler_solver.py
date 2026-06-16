"""
regge_wheeler_solver.py — Time-domain solver for the 1+1 Regge-Wheeler equation.

Solves  Psi_tt - Psi_{r*r*} + V(r) Psi = S(t, r*)
on a uniform tortoise-coordinate grid with:
  * high-order centred finite differences in space,
  * Störmer-Verlet (leapfrog) time integration,
  * outgoing Sommerfeld boundary conditions at both ends,
  * optional Gaussian initial-data pulse,
  * optional source term for black-hole excitation.

The scheme is second-order accurate in time; spatial accuracy is set by the
user-supplied FD order p in {2, 4, 6, 8, 10}.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Callable, Optional

from high_order_fd import apply_d2, apply_d1
from stability_analyzer import cfl_limit


# ---------------------------------------------------------------------------
#  Initial data: Gaussian pulse centred at r*_0
# ---------------------------------------------------------------------------
def gaussian_pulse(rstar: np.ndarray,
                   rstar_0: float,
                   width: float,
                   amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Psi, Psi_t) at t = 0 for a right-moving Gaussian pulse.

    The initial data is
        Psi(0, r*) = A exp( -(r* - r*_0)^2 / (2 w^2) ),
        Psi_t(0, r*) = - Psi_{r*}   (purely outgoing to the right).
    """
    xi = (rstar - rstar_0) / width
    Psi = amplitude * np.exp(-0.5 * xi ** 2)
    dPsi_drstar = -xi / width * Psi   # outgoing (right-moving) time derivative
    return Psi, dPsi_drstar


def bipartite_gaussian(rstar: np.ndarray,
                       rstar_left: float,
                       rstar_right: float,
                       width: float,
                       amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Superposition of two counter-propagating Gaussians for scattering studies."""
    xi_L = (rstar - rstar_left) / width
    xi_R = (rstar - rstar_right) / width
    Psi_L = amplitude * 0.5 * np.exp(-0.5 * xi_L ** 2)
    Psi_R = amplitude * 0.5 * np.exp(-0.5 * xi_R ** 2)
    Psi = Psi_L + Psi_R
    dPsi = (xi_L / width) * Psi_L - (xi_R / width) * Psi_R
    return Psi, dPsi


# ---------------------------------------------------------------------------
#  Boundary conditions
# ---------------------------------------------------------------------------
def sommerfeld_rhs(u: np.ndarray, dudt: np.ndarray,
                   dr: float, V_boundary: float) -> Tuple[float, float]:
    """Outgoing Sommerfeld condition  u_t +/- u_r + O(1/r) u = 0  at endpoints.

    Returns the corrected time derivatives at the two boundary points.
    """
    # left boundary: purely left-moving  u_t = + u_r
    dudt_left = (u[1] - u[0]) / dr
    # right boundary: purely right-moving u_t = - u_r
    dudt_right = -(u[-1] - u[-2]) / dr
    return dudt_left, dudt_right


# ---------------------------------------------------------------------------
#  Source term: Gaussian burst in time
# ---------------------------------------------------------------------------
def gaussian_source(t: float,
                    rstar: np.ndarray,
                    t0: float,
                    width_t: float,
                    rstar_0: float,
                    width_r: float,
                    amplitude: float = 1.0) -> np.ndarray:
    """Localized source S(t, r*) = A exp(-(t-t0)^2/(2 w_t^2)) exp(-(r*-r*0)^2/(2 w_r^2))."""
    time_part = amplitude * np.exp(-0.5 * ((t - t0) / width_t) ** 2)
    space_part = np.exp(-0.5 * ((rstar - rstar_0) / width_r) ** 2)
    return time_part * space_part


# ---------------------------------------------------------------------------
#  Main time-stepping loop
# ---------------------------------------------------------------------------
def solve_regge_wheeler(rstar: np.ndarray,
                        V: np.ndarray,
                        dr: float,
                        t_final: float,
                        cfl_factor: float = 0.5,
                        fd_order: int = 4,
                        initial_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        source: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
                        verbose: bool = True) -> dict:
    """Time-evolve  Psi_tt = Psi_{r*r*} - V Psi + S.

    Parameters
    ----------
    rstar     : 1D array of tortoise coordinates (uniform).
    V         : 1D array, Regge-Wheeler potential sampled on the grid.
    dr        : grid spacing in r_*.
    t_final   : final integration time (geometric units).
    cfl_factor: fraction of the CFL limit used for dt  (0 < cfl_factor < 1).
    fd_order  : spatial FD order (2, 4, 6, 8, 10).
    initial_data : (Psi0, dPsi0_dt); defaults to centred Gaussian.
    source    : callable (t, rstar) -> S(t, r*); None disables.
    verbose   : print progress every 10%.

    Returns
    -------
    result : dict with keys
        "t"      - (Nt,) time array
        "rstar"  - (Nx,) grid
        "Psi"    - (Nt, Nx) field values
        "dt"     - time step used
        "cfl_max": theoretical CFL limit
        "status" : "ok" | "unstable"
    """
    Nx = rstar.shape[0]
    cfl_max = cfl_limit(fd_order)
    dt = cfl_factor * cfl_max * dr
    if not np.isfinite(cfl_max) or dt <= 0.0:
        raise ValueError(f"Invalid CFL: cfl_max={cfl_max}, dt={dt}")
    Nt = int(np.ceil(t_final / dt)) + 1
    if verbose:
        print(f"[RW] grid: Nx={Nx}, dr={dr:.4e}, dt={dt:.4e}, "
              f"Nt={Nt}, cfl_max={cfl_max:.4f}, cfl_used={cfl_factor}")

    # initial data
    if initial_data is None:
        rstar_mid = 0.5 * (rstar[0] + rstar[-1])
        width = 2.0 * dr * 5.0
        Psi, dPsi = gaussian_pulse(rstar, rstar_mid, width, 1.0)
    else:
        Psi, dPsi = initial_data

    # Störmer-Verlet first half-step using the initial acceleration
    accel = _acceleration(Psi, V, dr, fd_order, 0.0, source)
    Psi_prev = Psi - dt * dPsi + 0.5 * dt ** 2 * accel

    t_arr = np.zeros(Nt)
    Psi_arr = np.zeros((Nt, Nx))
    t_arr[0] = 0.0
    Psi_arr[0] = Psi
    status = "ok"

    n_print = max(1, Nt // 10)
    for n in range(1, Nt):
        t_now = n * dt
        t_arr[n] = t_now
        accel = _acceleration(Psi, V, dr, fd_order, t_now, source)
        Psi_new = 2.0 * Psi - Psi_prev + dt ** 2 * accel
        # Sommerfeld update at the boundaries
        dudt_L, dudt_R = sommerfeld_rhs(Psi_new, np.zeros(Nx), dr, 0.0)
        Psi_new[0] = Psi[1] + (dr - dt) / (dr + dt) * (Psi_new[1] - Psi[0])
        Psi_new[-1] = Psi[-2] + (dr - dt) / (dr + dt) * (Psi_new[-2] - Psi[-1])
        # detect blow-up
        amp = float(np.max(np.abs(Psi_new)))
        if not np.isfinite(amp) or amp > 1.0e8:
            status = "unstable"
            if verbose:
                print(f"[RW] instability detected at step {n}, |Psi|={amp:.3e}")
            Psi_arr[n] = Psi_new
            break
        Psi_arr[n] = Psi_new
        Psi_prev, Psi = Psi, Psi_new
        if verbose and (n % n_print == 0):
            print(f"[RW] step {n:6d} / {Nt}  t = {t_now:10.4f}  max|Psi| = {amp:.3e}")

    return {"t": t_arr[: n + 1],
            "rstar": rstar,
            "Psi": Psi_arr[: n + 1],
            "dt": dt,
            "cfl_max": cfl_max,
            "status": status}


def _acceleration(Psi: np.ndarray,
                  V: np.ndarray,
                  dr: float,
                  fd_order: int,
                  t: float,
                  source) -> np.ndarray:
    """Compute the RHS  Psi_{r*r*} - V Psi + S."""
    D2Psi = apply_d2(Psi, dr, fd_order, bc="extrapolate")
    rhs = D2Psi - V * Psi
    if source is not None:
        rhs = rhs + source(t, np.zeros_like(Psi))  # source supplies its own r*
    return rhs


# ---------------------------------------------------------------------------
#  Waveform extraction at a given radius
# ---------------------------------------------------------------------------
def extract_waveform(result: dict,
                     rstar_ext: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate Psi(t, r*_ext) from the solution array.

    Returns
    -------
    t : (Nt,) time array
    h : (Nt,) strain at the extraction radius
    """
    rstar = result["rstar"]
    Psi = result["Psi"]
    t = result["t"]
    # find bracketing indices
    idx = int(np.searchsorted(rstar, rstar_ext))
    idx = np.clip(idx, 1, len(rstar) - 1)
    x = (rstar_ext - rstar[idx - 1]) / (rstar[idx] - rstar[idx - 1] + 1.0e-300)
    h = (1.0 - x) * Psi[:, idx - 1] + x * Psi[:, idx]
    return t, h
