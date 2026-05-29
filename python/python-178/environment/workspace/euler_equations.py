"""
euler_equations.py
==================
Compressible Euler equations of gas dynamics in conservative form.
Core physical model for the DG conservation law solver.

The 3D Euler equations:
    dU/dt + dF/dx + dG/dy + dH/dz = 0

where
    U = [rho, rho*u, rho*v, rho*w, E]^T
    F = [rho*u, rho*u^2+p, rho*u*v, rho*u*w, (E+p)*u]^T
    G = [rho*v, rho*u*v, rho*v^2+p, rho*v*w, (E+p)*v]^T
    H = [rho*w, rho*u*w, rho*v*w, rho*w^2+p, (E+p)*w]^T

with equation of state:
    p = (gamma - 1) * (E - 0.5*rho*(u^2+v^2+w^2))

and speed of sound:
    c = sqrt(gamma * p / rho)
"""

import numpy as np
from typing import Tuple

GAMMA = 1.4


def primitive_to_conservative(rho: float, u: float, v: float, w: float, p: float) -> np.ndarray:
    """Convert primitive variables to conservative vector."""
    rho = max(float(rho), 1e-14)
    p = max(float(p), 1e-14)
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
    return np.array([rho, rho * u, rho * v, rho * w, E], dtype=np.float64)


def conservative_to_primitive(U: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Convert conservative vector to primitive variables with robust clipping."""
    U = np.asarray(U, dtype=np.float64)
    rho = float(U[0])
    if rho < 1e-14 or not np.isfinite(rho):
        rho = 1e-14
    u = float(U[1]) / rho if np.isfinite(U[1]) else 0.0
    v = float(U[2]) / rho if np.isfinite(U[2]) else 0.0
    w = float(U[3]) / rho if np.isfinite(U[3]) else 0.0
    E = float(U[4])
    if not np.isfinite(E):
        E = 1e-14
    ke = 0.5 * rho * (u * u + v * v + w * w)
    p = (GAMMA - 1.0) * max(E - ke, 1e-14)
    if not np.isfinite(p):
        p = 1e-14
    return rho, u, v, w, p


def pressure(U: np.ndarray) -> float:
    """Compute pressure from conservative variables."""
    rho, u, v, w, p = conservative_to_primitive(U)
    return p


def speed_of_sound(U: np.ndarray) -> float:
    """Compute speed of sound."""
    rho, u, v, w, p = conservative_to_primitive(U)
    return np.sqrt(GAMMA * p / max(rho, 1e-14))


def flux_x(U: np.ndarray) -> np.ndarray:
    """Flux in x-direction."""
    rho, u, v, w, p = conservative_to_primitive(U)
    return np.array([
        rho * u,
        rho * u * u + p,
        rho * u * v,
        rho * u * w,
        (rho * 0.5 * (u * u + v * v + w * w) + p / (GAMMA - 1.0) + p) * u
    ], dtype=np.float64)


def flux_y(U: np.ndarray) -> np.ndarray:
    """Flux in y-direction."""
    rho, u, v, w, p = conservative_to_primitive(U)
    return np.array([
        rho * v,
        rho * u * v,
        rho * v * v + p,
        rho * v * w,
        (rho * 0.5 * (u * u + v * v + w * w) + p / (GAMMA - 1.0) + p) * v
    ], dtype=np.float64)


def flux_z(U: np.ndarray) -> np.ndarray:
    """Flux in z-direction."""
    rho, u, v, w, p = conservative_to_primitive(U)
    return np.array([
        rho * w,
        rho * u * w,
        rho * v * w,
        rho * w * w + p,
        (rho * 0.5 * (u * u + v * v + w * w) + p / (GAMMA - 1.0) + p) * w
    ], dtype=np.float64)


def flux_dot_n(U: np.ndarray, nx: float, ny: float, nz: float) -> np.ndarray:
    """Normal flux F·n = Fx*nx + Fy*ny + Fz*nz."""
    return nx * flux_x(U) + ny * flux_y(U) + nz * flux_z(U)


def roe_average(Ul: np.ndarray, Ur: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Compute Roe-averaged quantities between left and right states.
    """
    rhol, ul, vl, wl, pl = conservative_to_primitive(Ul)
    rhor, ur, vr, wr, pr = conservative_to_primitive(Ur)
    sqrt_rhol = np.sqrt(max(rhol, 1e-14))
    sqrt_rhor = np.sqrt(max(rhor, 1e-14))
    denom = sqrt_rhol + sqrt_rhor
    if denom < 1e-14:
        return 1e-14, 0.0, 0.0, 0.0, 1e-14
    rho_roe = sqrt_rhol * sqrt_rhor
    u_roe = (sqrt_rhol * ul + sqrt_rhor * ur) / denom
    v_roe = (sqrt_rhol * vl + sqrt_rhor * vr) / denom
    w_roe = (sqrt_rhol * wl + sqrt_rhor * wr) / denom
    hl = (Ul[4] + pl) / max(rhol, 1e-14)
    hr = (Ur[4] + pr) / max(rhor, 1e-14)
    h_roe = (sqrt_rhol * hl + sqrt_rhor * hr) / denom
    q2_roe = u_roe * u_roe + v_roe * v_roe + w_roe * w_roe
    c_roe_sq = (GAMMA - 1.0) * max(h_roe - 0.5 * q2_roe, 1e-14)
    c_roe = np.sqrt(c_roe_sq)
    return float(rho_roe), float(u_roe), float(v_roe), float(w_roe), float(c_roe)


def roe_flux(Ul: np.ndarray, Ur: np.ndarray,
             nx: float, ny: float, nz: float) -> np.ndarray:
    """
    Roe approximate Riemann solver.
    F_hat = 0.5*(F(Ul)+F(Ur)) - 0.5*|A_roe|*(Ur - Ul)
    """
    Fl = flux_dot_n(Ul, nx, ny, nz)
    Fr = flux_dot_n(Ur, nx, ny, nz)
    rho_roe, u_roe, v_roe, w_roe, c_roe = roe_average(Ul, Ur)
    un_roe = u_roe * nx + v_roe * ny + w_roe * nz
    lam1 = abs(un_roe - c_roe)
    lam2 = abs(un_roe)
    lam3 = abs(un_roe + c_roe)
    max_lam = max(lam1, lam2, lam3)
    flux = 0.5 * (Fl + Fr) - 0.5 * max_lam * (Ur - Ul)
    return flux


def rusanov_flux(Ul: np.ndarray, Ur: np.ndarray,
                 nx: float, ny: float, nz: float) -> np.ndarray:
    """
    Rusanov (local Lax-Friedrichs) numerical flux.
    F_hat = 0.5*(F(Ul)+F(Ur)) - 0.5*S_max*(Ur - Ul)
    where S_max = max(|un - c|, |un + c|) over left and right states.
    """
    # TODO: Implement Rusanov numerical flux for 3D Euler equations.
    # Hint: compute left/right physical fluxes, estimate maximum wave speed,
    # then combine using the Lax-Friedrichs formula.
    raise NotImplementedError("Hole 1: rusanov_flux is not implemented.")


# ---------------------------------------------------------------------------
# Manufactured solutions for verification
# ---------------------------------------------------------------------------

def manufactured_solution_3d(x: float, y: float, z: float, t: float) -> np.ndarray:
    """
    Smooth manufactured solution for convergence testing.
    Small-amplitude perturbation around constant state to ensure positivity.
    rho = 1 + 0.05*sin(pi*(x+y+z - 2*t))
    u = 0.2, v = 0.2, w = 0.2
    p = 1 + 0.05*sin(pi*(x+y+z - 2*t))
    """
    s = np.sin(np.pi * (x + y + z - 2.0 * t))
    rho = 1.0 + 0.05 * s
    u = 0.2
    v = 0.2
    w = 0.2
    p = 1.0 + 0.05 * s
    return primitive_to_conservative(rho, u, v, w, p)


def manufactured_source_3d(x: float, y: float, z: float, t: float) -> np.ndarray:
    """
    Exact source term for manufactured solution via analytic differentiation.
    For rho = 1 + a*sin(k*(x+y+z - vt)), u=v=w=const, p = 1 + a*sin(...)
    U_t + div(F) = S,  so S = U_t + div(F)
    """
    a = 0.05
    k = np.pi
    v_wave = 2.0
    u0 = 0.2
    v0 = 0.2
    w0 = 0.2
    c = np.cos(k * (x + y + z - v_wave * t))
    akc = a * k * c
    V2 = u0 * u0 + v0 * v0 + w0 * w0
    S0 = akc * (u0 + v0 + w0 - v_wave)
    S1 = akc * (u0 * u0 + u0 * v0 + u0 * w0 + 1.0 - v_wave * u0)
    S2 = akc * (u0 * v0 + v0 * v0 + v0 * w0 + 1.0 - v_wave * v0)
    S3 = akc * (u0 * w0 + v0 * w0 + w0 * w0 + 1.0 - v_wave * w0)
    E_coeff = 1.0 / (GAMMA - 1.0) + 0.5 * V2
    S4 = akc * (E_coeff * (u0 + v0 + w0 - v_wave) + (u0 + v0 + w0))
    return np.array([S0, S1, S2, S3, S4], dtype=np.float64)


# ---------------------------------------------------------------------------
# Scalar advection test (for simple validation)
# ---------------------------------------------------------------------------

def scalar_advection_solution_3d(x: float, y: float, z: float, t: float,
                                  ax: float = 1.0, ay: float = 0.5, az: float = 0.25) -> float:
    """Smooth Gaussian pulse for scalar advection."""
    x0 = 0.5 - ax * t
    y0 = 0.5 - ay * t
    z0 = 0.5 - az * t
    r2 = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
    return np.exp(-20.0 * r2)
