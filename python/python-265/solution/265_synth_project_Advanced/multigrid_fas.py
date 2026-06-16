# -*- coding: utf-8 -*-
"""
multigrid_fas.py
----------------
Full Approximation Scheme (FAS) multigrid solver for the *steady*
focused transport equation  L f = Q.

At each grid level  l  (l = 0 is finest, l = L is coarsest), we
perform nu_1 pre-smoothing and nu_2 post-smoothing steps of point
Gauss-Seidel (red-black) / weighted Jacobi, then:

  - Restrict the residual   r = Q - L f   to the next coarser grid;
  - On the coarser grid solve for the correction  e;
  - Prolongate  e  and update  f.

The restriction is full-weighting; the prolongation is linear
injection.  Grid coarsening is by a factor of 2 in the radial
coordinate (the mu direction is unchanged).

Reference:
    Brandt, A. (1977). Multi-level adaptive solutions to boundary-
    value problems. Mathematics of Computation 31, 333-390.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, List, Optional, Tuple

import heliocentric_mesh as hm


# =====================================================================
# Smoothers
# =====================================================================
def weighted_jacobi(Lf_func: Callable[[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray], np.ndarray],
                    f: np.ndarray,
                    Q: np.ndarray,
                    r: np.ndarray,
                    mu: np.ndarray,
                    omega: float = 2.0 / 3.0) -> np.ndarray:
    """Perform one weighted Jacobi smoothing iteration.

    Given the current  f, solve  a_i f_i = Q_i - (off-diagonal terms)
    and set  f_i = omega f_i^{new} + (1 - omega) f_i^{old}.
    """
    Nr, Nmu = f.shape
    dmu_arr = np.diff(mu)
    f_new = f.copy()
    for i in range(1, Nr - 1):
        drL = r[i] - r[i - 1]
        drR = r[i + 1] - r[i]
        for j in range(1, Nmu - 1):
            dmuL = dmu_arr[j - 1]
            dmuR = dmu_arr[j]
            aL = 1.0 / (drL * (drL + drR) / 2.0)
            aR = 1.0 / (drR * (drL + drR) / 2.0)
            bL = 1.0 / (dmuL * (dmuL + dmuR) / 2.0)
            bR = 1.0 / (dmuR * (dmuL + dmuR) / 2.0)
            diag = aL + aR + bL + bR
            if diag <= 0.0:
                continue
            rhs = Q[i, j] - (aL * f[i - 1, j] + aR * f[i + 1, j]
                             + bL * f[i, j - 1] + bR * f[i, j + 1])
            f_new[i, j] = omega * rhs / diag + (1.0 - omega) * f[i, j]
    return f_new


def gauss_seidel_sweep(Lf_func, f, Q, r, mu,
                       colour: str = "red",
                       kappa_r: np.ndarray = None) -> np.ndarray:
    """Gauss-Seidel red-black sweep.

    The 'colour' determines which (i + j) parity is updated.
    If ``kappa_r`` is provided, uses the physical diffusion operator
    (1/r^2) d/dr[r^2 kappa df/dr] + d^2 f/dmu^2.
    """
    Nr, Nmu = f.shape
    dmu_arr = np.diff(mu)
    if kappa_r is None:
        kappa_r = np.ones(Nr)
    for i in range(1, Nr - 1):
        drL = r[i] - r[i - 1]
        drR = r[i + 1] - r[i]
        r2 = r[i] * r[i]
        kL = 0.5 * (kappa_r[i - 1] + kappa_r[i])
        kR = 0.5 * (kappa_r[i] + kappa_r[i + 1])
        aL = kL / (drL * (drL + drR) / 2.0 * r2)
        aR = kR / (drR * (drL + drR) / 2.0 * r2)
        for j in range(1, Nmu - 1):
            dmuL = dmu_arr[j - 1]
            dmuR = dmu_arr[j]
            bL = 1.0 / (dmuL * (dmuL + dmuR) / 2.0)
            bR = 1.0 / (dmuR * (dmuL + dmuR) / 2.0)
            diag = aL + aR + bL + bR
            if diag <= 0.0:
                continue
            rhs = Q[i, j] - (aL * f[i - 1, j] + aR * f[i + 1, j]
                             + bL * f[i, j - 1] + bR * f[i, j + 1])
            f[i, j] = rhs / diag
    return f


# =====================================================================
# Laplacian-like discrete operator (used as residual driver)
# =====================================================================
def discrete_operator(f: np.ndarray, r: np.ndarray, mu: np.ndarray,
                      kappa_r: np.ndarray) -> np.ndarray:
    """Return the action of the discrete diffusion operator

        L f = (1/r^2) d/dr[ r^2 kappa(r) df/dr ] + d/dmu[ D_mumu df/dmu ]

    with D_mumu approximated as constant = 1 on the mu grid (the true
    D_mumu profile is folded into the smoother via the actual problem
    operator; this simplified operator is used for convergence tests).
    """
    Nr, Nmu = f.shape
    out = np.zeros_like(f)
    dr = np.diff(r)
    dmu_arr = np.diff(mu)
    for i in range(1, Nr - 1):
        drL = dr[i - 1]
        drR = dr[i]
        kL = 0.5 * (kappa_r[i - 1] + kappa_r[i])
        kR = 0.5 * (kappa_r[i] + kappa_r[i + 1])
        r2 = r[i] * r[i]
        dfr = (kR * (f[i + 1, :] - f[i, :]) / drR
               - kL * (f[i, :] - f[i - 1, :]) / drL)
        out[i, :] += dfr / (r2 * (drL + drR) * 0.5)
    for j in range(1, Nmu - 1):
        dmuL = dmu_arr[j - 1]
        dmuR = dmu_arr[j]
        out[:, j] += ((f[:, j + 1] - f[:, j]) / dmuR
                      - (f[:, j] - f[:, j - 1]) / dmuL) / (
                          (dmuL + dmuR) * 0.5)
    return out


# =====================================================================
# FAS V-cycle
# =====================================================================
class FASMultigrid:
    """Full Approximation Scheme multigrid for  L f = Q.

    The hierarchy is built by dyadic coarsening of the radial grid,
    keeping the mu grid fixed.
    """

    def __init__(self, r: np.ndarray, mu: np.ndarray,
                 kappa_r: np.ndarray, Q: np.ndarray,
                 nu1: int = 2, nu2: int = 2,
                 max_cycles: int = 40, tol: float = 1.0e-8):
        if r.size < 5:
            raise ValueError("FASMultigrid: need Nr >= 5.")
        self.r_fine = r.copy()
        self.mu = mu.copy()
        self.kappa_r = kappa_r.copy()
        self.Q = Q.copy()
        self.Nr = r.size
        self.Nmu = mu.size
        self.nu1 = nu1
        self.nu2 = nu2
        self.max_cycles = max_cycles
        self.tol = tol
        # build hierarchy
        self.levels: List[dict] = []
        self._build_hierarchy()

    # -----------------------------------------------------------------
    def _build_hierarchy(self) -> None:
        r = self.r_fine.copy()
        kappa = self.kappa_r.copy()
        Q = self.Q.copy()
        while r.size >= 5:
            level = dict(r=r.copy(), mu=self.mu.copy(),
                         kappa=kappa.copy(), Q=Q.copy(),
                         Nr=r.size, Nmu=self.mu.size,
                         dmu=self.mu[1:] - self.mu[:-1])
            self.levels.append(level)
            if r.size % 2 == 0:
                r = r[::2]
            else:
                r = np.concatenate([r[::2], [r[-1]]])
            # coarsen kappa, Q by averaging
            kappa_c = np.zeros(r.size)
            Q_c = np.zeros((r.size, self.Nmu))
            for i in range(r.size):
                j = 2 * i
                if j < kappa.size:
                    kappa_c[i] = kappa[j]
                    Q_c[i, :] = Q[j, :]
            kappa = kappa_c
            Q = Q_c
        # coarsest: just copy
        self.levels.append(dict(r=r.copy(), mu=self.mu.copy(),
                                kappa=kappa.copy(), Q=Q.copy(),
                                Nr=r.size, Nmu=self.Nmu,
                                dmu=self.mu[1:] - self.mu[:-1]))

    # -----------------------------------------------------------------
    def vcycle(self, f: np.ndarray) -> np.ndarray:
        """Perform one V-cycle starting from the finest grid."""
        return self._vcycle_at(f, level=0)

    # -----------------------------------------------------------------
    def _vcycle_at(self, f: np.ndarray, level: int) -> np.ndarray:
        lev = self.levels[level]
        r = lev["r"]
        mu = lev["mu"]
        dmu = lev["dmu"]
        Q = lev["Q"]
        kappa = lev["kappa"]
        if level == len(self.levels) - 1:
            # coarsest: many smoothing steps
            for _ in range(50):
                f = gauss_seidel_sweep(lambda *a: 0, f, Q, r, mu, "red",
                                       kappa_r=kappa)
                f = gauss_seidel_sweep(lambda *a: 0, f, Q, r, mu, "black",
                                       kappa_r=kappa)
            return f
        # pre-smoothing
        for _ in range(self.nu1):
            f = gauss_seidel_sweep(lambda *a: 0, f, Q, r, mu, "red",
                                   kappa_r=kappa)
            f = gauss_seidel_sweep(lambda *a: 0, f, Q, r, mu, "black",
                                   kappa_r=kappa)
        # residual
        Lf = discrete_operator(f, r, mu, kappa)
        residual = Q - Lf
        # restrict
        r_c = self.levels[level + 1]["r"]
        Q_c = self.levels[level + 1]["Q"]
        res_c = hm.restrict_radial(residual)
        f_c = hm.restrict_radial(f)
        # coarse-grid RHS = L f_c + residual_c
        Lf_c = discrete_operator(f_c, r_c, mu,
                                 self.levels[level + 1]["kappa"])
        Q_coarse = Lf_c + res_c
        self.levels[level + 1]["Q"] = Q_coarse
        # recursive V-cycle
        e_c = self._vcycle_at(f_c.copy(), level + 1)
        # error
        e = e_c - f_c
        # prolongate correction
        e_fine = hm.prolongate_radial(e, f.shape[0])
        f = f + e_fine
        # post-smoothing
        for _ in range(self.nu2):
            f = gauss_seidel_sweep(lambda *a: 0, f, Q, r, mu, "red",
                                   kappa_r=kappa)
            f = gauss_seidel_sweep(lambda *a: 0, f, Q, r, mu, "black",
                                   kappa_r=kappa)
        return f

    # -----------------------------------------------------------------
    def solve(self) -> Tuple[np.ndarray, dict]:
        """Solve and return (f, diagnostics)."""
        f = np.zeros((self.Nr, self.Nmu))
        # linear initial guess between boundary values
        for j in range(self.Nmu):
            f[:, j] = np.linspace(self.Q[0, j], self.Q[-1, j], self.Nr)
        hm.apply_all_boundaries(f, self.Q[0, :], self.Q[-1, :])
        residual_history = []
        for cyc in range(self.max_cycles):
            f = self.vcycle(f)
            hm.apply_all_boundaries(f, self.Q[0, :], self.Q[-1, :])
            Lf = discrete_operator(f, self.r_fine, self.mu, self.kappa_r)
            res = float(np.max(np.abs(self.Q - Lf)))
            residual_history.append(res)
            if res < self.tol:
                return f, {"converged": True, "cycles": cyc + 1,
                           "residual_history": residual_history}
        return f, {"converged": False, "cycles": self.max_cycles,
                   "residual_history": residual_history}


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    import cosmic_ray_physics as crp
    r = hm.logarithmic_radial_mesh(Nr=32)
    mu = hm.pitch_angle_mesh(Nmu=16)
    kappa = 1.0e22 * (r / crp.AU) ** 0.3
    Q = np.zeros((r.size, mu.size))
    # source at r = 1 AU
    src = np.exp(-((r - crp.AU) ** 2) / (0.1 * crp.AU) ** 2)
    for j in range(mu.size):
        Q[:, j] += src * 1.0e-20
    Q[0, :] = 1.0
    Q[-1, :] = 0.0
    mg = FASMultigrid(r, mu, kappa, Q, nu1=2, nu2=2,
                      max_cycles=30, tol=1.0e-5)
    f, info = mg.solve()
    print(f"[multigrid_fas] V-cycle: {info}")
    print(f"[multigrid_fas] f at 1 AU (central mu): {f[16, 8]:.3e}")


if __name__ == "__main__":
    _demo()
