# -*- coding: utf-8 -*-
"""
stability_analysis.py
---------------------
Stability analysis for the numerical schemes used to solve the
focused Parker transport equation.

Modules contained
-----------------
1. ``explicit_cfl_timestep``       -- CFL-based time step for explicit
   schemes (diffusion + advection + pitch-angle terms).
2. ``von_neumann_amplification``   -- amplification factor  G(k) for
   constant-coefficient advection-diffusion.
3. ``spectral_radius_advection``   -- spectral radius of the discrete
   advection operator with various stencils.
4. ``dispersion_error``            -- numerical dispersion relation
   vs. exact.
5. ``matrix_stability_test``       -- build the full spatial operator
   as a sparse matrix and compute its spectral radius.
6. ``diffusion_stability_limit``   -- classical dt <= dx^2 / (2 kappa).
7. ``combined_cfl_diffusion``      -- combined advective + diffusive
   restriction.
8. ``critical_rigidity``           -- for a given grid and solar wind,
   the maximum rigidity for which an explicit scheme is stable.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Tuple


# =====================================================================
# 1. CFL time step
# =====================================================================
def explicit_cfl_timestep(r: np.ndarray, mu: np.ndarray,
                          Krr: np.ndarray,
                          Dmumu: np.ndarray,
                          streaming_speed_max: float,
                          safety: float = 0.4) -> float:
    """Return the largest stable explicit time step.

    The CFL conditions are

        dt <= min_i (dr_i / |mu v + V_sw|)              (advection)
        dt <= min_i (dr_i^2 / (2 kappa_rr_i))          (diffusion)
        dt <= min_j (dmu_j^2 / (2 D_mumu_j))           (pitch angle)

    The safety factor ``safety`` is multiplied onto the minimum.
    """
    dr_min = float(np.min(np.diff(r)))
    dmu_min = float(np.min(np.diff(np.abs(mu)))) if mu.size > 1 else 1.0
    vadv = max(abs(streaming_speed_max), 1.0)
    dt_adv = dr_min / vadv
    # diffusion in r
    kmax = float(np.max(Krr))
    dt_diffr = (dr_min ** 2) / max(2.0 * kmax, 1.0e-30)
    # diffusion in mu
    dmu_max = float(np.max(np.abs(Dmumu)))
    dt_diffmu = (dmu_min ** 2) / max(2.0 * dmu_max, 1.0e-30)
    return safety * min(dt_adv, dt_diffr, dt_diffmu)


# =====================================================================
# 2. Von Neumann amplification factor
# =====================================================================
def von_neumann_amplification(k: np.ndarray, dx: float,
                              v_adv: float, kappa: float,
                              dt: float,
                              scheme: str = "upwind1") -> np.ndarray:
    """Return the complex amplification factor  G(k) for constant-
    coefficient advection-diffusion

        u_t + v u_x = kappa u_xx

    under a given spatial scheme and forward Euler in time.

    Supported schemes: ``'upwind1'``, ``'central2'``, ``'compact4'``,
    ``'weno5'`` (linearised).
    """
    if scheme == "upwind1":
        if v_adv >= 0:
            ux_hat = (1.0 - np.exp(-1j * k * dx)) / dx
        else:
            ux_hat = (np.exp(1j * k * dx) - 1.0) / dx
    elif scheme == "central2":
        ux_hat = 1j * np.sin(k * dx) / dx
    elif scheme == "compact4":
        # Padé: (1/6 e^{-i k dx} + 2/3 + 1/6 e^{i k dx}) (i k)_num =
        # i sin(k dx) / dx  =>  (i k)_num = i sin(k dx) / (dx (2/3 + cos(k dx)/3))
        denom = 2.0 / 3.0 + (2.0 / 6.0) * np.cos(k * dx)
        ux_hat = 1j * np.sin(k * dx) / (dx * denom)
    elif scheme == "weno5":
        # linearised WENO-5 reduces to 4th-order central
        ux_hat = (1j / (12.0 * dx)) * (
            -np.sin(2.0 * k * dx) + 8.0 * np.sin(k * dx))
    else:
        raise ValueError(f"unknown scheme '{scheme}'")
    uxx_hat = -(k ** 2)          # diffusion is always centred
    G = 1.0 + dt * (-v_adv * ux_hat + kappa * uxx_hat)
    return G


# =====================================================================
# 3. Spectral radius of the advection operator
# =====================================================================
def spectral_radius_advection(N: int, dx: float, v: float,
                              scheme: str = "upwind1") -> float:
    """Return the spectral radius (maximum |eigenvalue|) of the
    discrete first-derivative operator with chosen scheme.
    """
    if N < 4:
        return 0.0
    # build matrix
    A = np.zeros((N, N))
    for i in range(1, N - 1):
        if scheme == "upwind1":
            if v >= 0:
                A[i, i] = -v / dx
                A[i, i - 1] = v / dx
            else:
                A[i, i] = v / dx
                A[i, i + 1] = -v / dx
        elif scheme == "central2":
            A[i, i + 1] = -v * 0.5 / dx
            A[i, i - 1] = v * 0.5 / dx
        elif scheme == "compact4":
            # approximate with central2 spectral radius
            A[i, i + 1] = -v * 0.5 / dx
            A[i, i - 1] = v * 0.5 / dx
    eig = np.linalg.eigvals(A)
    return float(np.max(np.abs(eig)))


# =====================================================================
# 4. Dispersion error
# =====================================================================
def dispersion_error(k: np.ndarray, dx: float, v: float,
                     scheme: str = "central2") -> Tuple[np.ndarray, np.ndarray]:
    """Return (phase error, group error) of the discrete operator.

    The modified wavenumber  k'  is defined via the discrete operator
    so that  f'(x) ~ i k' f_hat e^{i k x}.  The phase velocity is
    v_phase = omega / k,  group velocity v_g = d omega / dk.
    """
    if scheme == "central2":
        k_prime = np.sin(k * dx) / dx
    elif scheme == "compact4":
        denom = 2.0 / 3.0 + (2.0 / 6.0) * np.cos(k * dx)
        k_prime = np.sin(k * dx) / (dx * denom)
    elif scheme == "weno5":
        k_prime = (-np.sin(2.0 * k * dx) + 8.0 * np.sin(k * dx)) / (12.0 * dx)
    else:
        k_prime = (1.0 - np.cos(k * dx)) / dx  # upwind1 modulus
    phase_err = np.where(np.abs(k) > 1.0e-12,
                         v * k_prime / (v * k) - 1.0, 0.0)
    # group error (numerical derivative of k')
    dk = k[1] - k[0] if k.size > 1 else 1.0
    dkp_dk = np.gradient(k_prime, dk)
    group_err = np.where(np.abs(k) > 1.0e-12, dkp_dk - 1.0, 0.0)
    return phase_err, group_err


# =====================================================================
# 5. Matrix stability test
# =====================================================================
def matrix_stability_test(r: np.ndarray, mu: np.ndarray,
                          Krr: np.ndarray,
                          Dmumu: np.ndarray,
                          V_stream: float,
                          scheme: str = "upwind1") -> dict:
    """Build the full spatial operator  L  as a dense matrix and
    compute the spectral radius, trace, and minimum real eigenvalue.

    The operator acts on  f  flattened as  f[i_r + Nr * i_mu].
    The scheme is stable with forward Euler iff
        dt <= -2 / Re(lambda_max)
    for any eigenvalue  lambda  with Re(lambda) < 0.
    """
    Nr = r.size
    Nmu = mu.size
    N = Nr * Nmu
    L = np.zeros((N, N))

    def idx(i, j):
        return i + Nr * j

    for j in range(Nmu):
        for i in range(1, Nr - 1):
            drL = r[i] - r[i - 1]
            drR = r[i + 1] - r[i]
            kface_L = 0.5 * (Krr[i - 1, j] + Krr[i, j])
            kface_R = 0.5 * (Krr[i, j] + Krr[i + 1, j])
            # diffusion
            L[idx(i, j), idx(i - 1, j)] += kface_L / (drL * (drL + drR) / 2.0)
            L[idx(i, j), idx(i + 1, j)] += kface_R / (drR * (drL + drR) / 2.0)
            L[idx(i, j), idx(i, j)] -= (kface_L / drL + kface_R / drR) / (
                (drL + drR) / 2.0)
            # advection
            if scheme == "upwind1":
                if V_stream >= 0:
                    L[idx(i, j), idx(i, j)] -= V_stream / drL
                    L[idx(i, j), idx(i - 1, j)] += V_stream / drL
                else:
                    L[idx(i, j), idx(i, j)] += V_stream / drR
                    L[idx(i, j), idx(i + 1, j)] -= V_stream / drR

    for i in range(1, Nr - 1):
        for j in range(1, Nmu - 1):
            dmuL = mu[j] - mu[j - 1]
            dmuR = mu[j + 1] - mu[j]
            dface_L = 0.5 * (Dmumu[i, j - 1] + Dmumu[i, j])
            dface_R = 0.5 * (Dmumu[i, j] + Dmumu[i, j + 1])
            L[idx(i, j), idx(i, j - 1)] += dface_L / (dmuL * (dmuL + dmuR) / 2.0)
            L[idx(i, j), idx(i, j + 1)] += dface_R / (dmuR * (dmuL + dmuR) / 2.0)
            L[idx(i, j), idx(i, j)] -= (dface_L / dmuL + dface_R / dmuR) / (
                (dmuL + dmuR) / 2.0)

    eig = np.linalg.eigvals(L)
    reig = np.real(eig)
    return {
        "spectral_radius": float(np.max(np.abs(eig))),
        "max_real": float(np.max(reig)),
        "min_real": float(np.min(reig)),
        "trace": float(np.sum(reig)),
        "max_dt_euler": float(-2.0 / max(np.max(reig), 1.0e-30))
        if np.max(reig) > 0.0 else float("inf"),
    }


# =====================================================================
# 6. Pure diffusion stability
# =====================================================================
def diffusion_stability_limit(dx: float, kappa: float,
                              scheme: str = "euler") -> float:
    """Classical stability limit for forward Euler with diffusion.

        dt <= dx^2 / (2 kappa)      (1-D)
        dt <= dx^2 / (4 kappa)      (2-D square grid)

    ``scheme='euler'`` returns the 1-D result; ``'rk4'`` returns the
    larger limit  dt <= 2.785 dx^2 / kappa  for RK4.
    """
    if scheme == "euler":
        return (dx * dx) / max(2.0 * kappa, 1.0e-30)
    if scheme == "rk4":
        return 2.785 * (dx * dx) / max(kappa, 1.0e-30)
    raise ValueError(f"unknown scheme '{scheme}'")


# =====================================================================
# 7. Combined advective-diffusive restriction
# =====================================================================
def combined_cfl_diffusion(dx: float, v: float, kappa: float,
                           safety: float = 0.8) -> float:
    """Combined advective + diffusive limit

        dt <= 2 kappa / v^2   (cell Peclet)  and  dt <= dx^2 / (2 kappa)
    """
    Pe = abs(v) * dx / max(kappa, 1.0e-30)
    if Pe < 2.0:
        dt_adv = dx / max(abs(v), 1.0)
        dt_dif = (dx * dx) / max(2.0 * kappa, 1.0e-30)
        return safety * min(dt_adv, dt_dif)
    return safety * (dx * dx) / max(2.0 * kappa, 1.0e-30)


# =====================================================================
# 8. Critical rigidity
# =====================================================================
def critical_rigidity(dx: float, V_sw: float, v_particle: float,
                      B: float, safety: float = 0.4) -> float:
    """Return the maximum rigidity R_c such that the explicit scheme
    is stable for particles of speed  v_particle  in the given grid
    with solar wind  V_sw  and magnetic field  B.

    kappa_|| ~ (v r_L / 3) (B / delta B)^2  =>  kappa ~ R.

    From the diffusion limit  dt = dx^2 / (2 kappa) = dt_adv = dx / V
    we get  R_c ~ dx V / (2 xi c / (3 B))  with xi = 1/3.
    """
    import cosmic_ray_physics as crp
    k_coeff = crp.c_light / (9.0 * B) if B > 0.0 else 1.0e30
    return safety * dx * max(V_sw, v_particle) / max(k_coeff, 1.0e-30)


# =====================================================================
# Self-contained demo
# =====================================================================
def _demo() -> None:
    print("[stability_analysis] von Neumann amplification factor for "
          "CFL = 0.8:")
    k = np.linspace(0.0, math.pi, 64)
    dx = 1.0
    v = 1.0
    kappa = 0.1
    dt = 0.8 * dx / abs(v)
    for sch in ("upwind1", "central2", "compact4", "weno5"):
        G = von_neumann_amplification(k, dx, v, kappa, dt, scheme=sch)
        print(f"  scheme={sch:9s}  max|G| = {np.max(np.abs(G)):.4f}")
    print("[stability_analysis] spectral radius of advection operator:")
    for sch in ("upwind1", "central2"):
        rho = spectral_radius_advection(64, 0.01, 1.0, scheme=sch)
        print(f"  scheme={sch:9s}  rho = {rho:.3e}")


if __name__ == "__main__":
    _demo()
