
import numpy as np
from scipy.special import jv
from typing import Dict, Tuple

from linear_algebra_core import (
    R83TMatrix, build_womersley_tridiagonal, thomas_algorithm,
    r83t_cg_solve
)
from quadrature_rules import gauss_legendre_quadrature
from stochastic_diffusion import einstein_viscosity_correction






class WomersleySolver:
    def __init__(self, radius: float = 0.005,
                 kinematic_viscosity: float = 3.3e-6,
                 blood_density: float = 1060.0,
                 n_radial: int = 100,
                 heart_rate_bpm: float = 72.0):
        self.R = radius
        self.nu = kinematic_viscosity
        self.rho = blood_density
        self.n_r = n_radial
        self.HR = heart_rate_bpm


        self.r = np.linspace(0, self.R, n_radial)
        self.dr = self.r[1] - self.r[0]


        self.omega = 2.0 * np.pi * self.HR / 60.0
        self.alpha = self.R * np.sqrt(self.omega / self.nu)


        self.wss_history = []
        self.time_history = []

    def pressure_gradient(self, t: float,
                          base_gradient: float = 100.0,
                          pulsatile_amp: float = 80.0) -> float:
        T = 60.0 / self.HR
        phase = 2.0 * np.pi * t / T

        return base_gradient + pulsatile_amp * np.cos(phase) + 0.3 * pulsatile_amp * np.cos(2.0 * phase)

    def time_step(self, u_old: np.ndarray, dt: float, t: float,
                  use_thomas: bool = True) -> np.ndarray:







        raise NotImplementedError("Hole 2: WomersleySolver.time_step 待实现")

    def solve_steady_state(self, max_iter: int = 10000,
                           dt: float = 1e-4,
                           tol: float = 1e-8) -> np.ndarray:

        dpdx = 100.0
        u = np.zeros(self.n_r)
        mu = self.nu * self.rho
        for _ in range(max_iter):
            u_new = u.copy()

            for j in range(1, self.n_r - 1):
                r_j = self.r[j]

                laplacian = (u[j-1] - 2*u[j] + u[j+1]) / self.dr**2 + \
                            (u[j+1] - u[j-1]) / (2 * r_j * self.dr)
                u_new[j] = u[j] + dt * (dpdx / self.rho + self.nu * laplacian)

            u_new[0] = u_new[1]
            u_new[-1] = 0.0

            if np.linalg.norm(u_new - u, ord=np.inf) < tol:
                return u_new
            u = u_new
        return u

    def solve_pulsatile(self, n_cardiac_cycles: float = 2.0,
                        n_steps_per_cycle: int = 200,
                        dt: float = None) -> Dict:
        T = 60.0 / self.HR
        if dt is None:
            dt = T / n_steps_per_cycle

        total_steps = int(n_cardiac_cycles * n_steps_per_cycle)
        u = np.zeros(self.n_r)


        warmup_steps = n_steps_per_cycle
        for i in range(warmup_steps):
            t = i * dt
            u = self.time_step(u, dt, t)

        self.wss_history = []
        self.time_history = []
        velocity_snapshots = []

        for i in range(total_steps):
            t = i * dt
            u = self.time_step(u, dt, t)



            wss = abs(self.nu * self.rho * (u[-1] - u[-2]) / self.dr)
            self.wss_history.append(float(wss))
            self.time_history.append(float(t))

            if i % (n_steps_per_cycle // 4) == 0:
                velocity_snapshots.append(u.copy())

        return {
            "velocity_final": u,
            "velocity_snapshots": velocity_snapshots,
            "wss_history": np.array(self.wss_history),
            "time_history": np.array(self.time_history),
            "radial_grid": self.r,
            "alpha": self.alpha
        }

    def womersley_exact_solution(self, t: float, n_harmonics: int = 3) -> np.ndarray:
        u_exact = np.zeros(self.n_r, dtype=complex)
        A1 = -80.0

        for n in range(1, n_harmonics + 1):
            omega_n = n * self.omega
            alpha_n = self.R * np.sqrt(omega_n / self.nu)
            z = alpha_n * ((-1j) ** 1.5)
            z_wall = z


            j0_wall = jv(0, z_wall)
            if abs(j0_wall) < 1e-15:
                j0_wall = 1e-15

            coeff = 1j * A1 / (self.rho * omega_n * n)
            for j, rj in enumerate(self.r):
                zr = z * rj / self.R
                j0_r = jv(0, zr)
                u_exact[j] += coeff * (1.0 - j0_r / j0_wall) * np.exp(1j * omega_n * t)

        return np.real(u_exact)






def compute_tawss(wss_history: np.ndarray, time_history: np.ndarray) -> float:
    if len(wss_history) < 2:
        return 0.0

    tawss = np.trapezoid(np.abs(wss_history), time_history)
    T = time_history[-1] - time_history[0]
    return float(tawss / (T + 1e-15))


def compute_osi(wss_history: np.ndarray, time_history: np.ndarray) -> float:





    raise NotImplementedError("Hole 3: compute_osi 待实现")


def compute_wss_gradient(wss_history: np.ndarray, time_history: np.ndarray) -> float:
    if len(wss_history) < 3:
        return 0.0
    dt = np.diff(time_history)
    d_wss = np.diff(wss_history)
    d_tau_dt = d_wss / (dt + 1e-15)

    T = time_history[-1] - time_history[0]
    wssg = np.sqrt(np.mean(d_tau_dt ** 2))
    return float(wssg)


def relative_resistance_index(wss_max: float, wss_min: float) -> float:
    denom = wss_max + wss_min + 1e-15
    return float((wss_max - wss_min) / denom)






def generate_wss_report(solver: WomersleySolver, result: Dict) -> Dict:
    wss_hist = result["wss_history"]
    time_hist = result["time_history"]

    if len(wss_hist) == 0:
        return {"error": "No WSS data"}

    tawss = compute_tawss(wss_hist, time_hist)
    osi = compute_osi(wss_hist, time_hist)
    wssg = compute_wss_gradient(wss_hist, time_hist)
    wss_max = float(np.max(wss_hist))
    wss_min = float(np.min(wss_hist))
    rri = relative_resistance_index(wss_max, wss_min)


    def wss_abs_interp(t):

        return np.interp(t, time_hist, np.abs(wss_hist))

    T = time_hist[-1] - time_hist[0]
    if T > 0:
        tawss_gl = gauss_legendre_quadrature(wss_abs_interp, time_hist[0], time_hist[-1], n=64) / T
    else:
        tawss_gl = tawss

    return {
        "TAWSS_Pa": tawss,
        "TAWSS_GaussLegendre_Pa": float(tawss_gl),
        "OSI": osi,
        "WSSG_Pa_s": wssg,
        "WSS_max_Pa": wss_max,
        "WSS_min_Pa": wss_min,
        "RRI": rri,
        "Womersley_alpha": float(solver.alpha),
        "physiological_score": _overall_physiological_score(tawss, osi)
    }


def _overall_physiological_score(tawss: float, osi: float) -> float:
    score = 1.0
    if tawss < 0.5:
        score -= 0.3
    elif tawss > 7.0:
        score -= 0.3
    if osi > 0.15:
        score -= 0.3
    return max(score, 0.0)
