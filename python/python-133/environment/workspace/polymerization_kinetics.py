
import numpy as np
from typing import Tuple, Dict, Callable, Optional


class PolymerizationParameters:

    def __init__(self,
                 kd: float = 1.0e-4,
                 ki: float = 1.0e3,
                 kp: float = 3.0e3,
                 ktc: float = 1.0e7,
                 ktd: float = 1.0e7,
                 ktr: float = 1.0e-1,
                 f: float = 0.6,
                 M0: float = 8.0,
                 I0: float = 1.0e-2,
                 S0: float = 0.0,
                 T: float = 333.15,
                 Ea_p: float = 32.0e3,
                 Ea_t: float = 8.0e3,
                 R_gas: float = 8.314,
                 T_ref: float = 333.15,
                 t0: float = 0.0,
                 tstop: float = 7200.0):
        self.kd = float(kd)
        self.ki = float(ki)
        self.kp = float(kp)
        self.ktc = float(ktc)
        self.ktd = float(ktd)
        self.ktr = float(ktr)
        self.f = float(f)
        self.M0 = float(M0)
        self.I0 = float(I0)
        self.S0 = float(S0)
        self.T = float(T)
        self.Ea_p = float(Ea_p)
        self.Ea_t = float(Ea_t)
        self.R_gas = float(R_gas)
        self.T_ref = float(T_ref)
        self.t0 = float(t0)
        self.tstop = float(tstop)
        self._validate()

    def _validate(self) -> None:
        assert self.kd > 0.0, "kd must be positive"
        assert self.kp > 0.0, "kp must be positive"
        assert self.f > 0.0 and self.f <= 1.0, "f must be in (0,1]"
        assert self.M0 > 0.0, "M0 must be positive"
        assert self.I0 >= 0.0, "I0 must be non-negative"
        assert self.T > 0.0, "T must be positive"
        assert self.tstop > self.t0, "tstop must exceed t0"

    def temperature_correction(self, rate_const: float, Ea: float) -> float:
        return rate_const * np.exp(-Ea / self.R_gas * (1.0 / self.T - 1.0 / self.T_ref))

    def effective_rate_constants(self) -> Dict[str, float]:
        kp_eff = self.temperature_correction(self.kp, self.Ea_p)
        ki_eff = self.temperature_correction(self.ki, self.Ea_p * 0.8)
        ktc_eff = self.temperature_correction(self.ktc, self.Ea_t)
        ktd_eff = self.temperature_correction(self.ktd, self.Ea_t)
        ktr_eff = self.temperature_correction(self.ktr, self.Ea_p * 1.1)
        return {
            'kd': self.kd,
            'ki': ki_eff,
            'kp': kp_eff,
            'ktc': ktc_eff,
            'ktd': ktd_eff,
            'ktr': ktr_eff,
        }


def polymerization_deriv(t: float, y: np.ndarray, params: PolymerizationParameters) -> np.ndarray:
    if y.ndim != 1:
        y = y.flatten()


    M, I_conc, S = y[0], y[1], y[2]
    lam0, lam1, lam2 = y[3], y[4], y[5]
    mu0, mu1, mu2 = y[6], y[7], y[8]


    k = params.effective_rate_constants()
    kd, ki, kp = k['kd'], k['ki'], k['kp']
    ktc, ktd, ktr = k['ktc'], k['ktd'], k['ktr']
    kt = ktc + ktd
    f = params.f



    denom = ki * M + ktr * S + 1.0e-12
    R = 2.0 * f * kd * I_conc / denom


    M = max(M, 1.0e-15)
    I_conc = max(I_conc, 1.0e-15)
    S = max(S, 0.0)
    lam0 = max(lam0, 1.0e-15)


    dMdt = -(kp * lam0 + ki * R) * M


    dIdt = -kd * I_conc


    dSdt = -ktr * lam0 * S


    dlam0dt = 2.0 * f * kd * I_conc - kt * lam0 ** 2
    dlam1dt = ki * M * R + kp * M * lam0 - kt * lam0 * lam1 - ktr * S * lam1 + ktr * S * lam0
    dlam2dt = ki * M * R + kp * M * (2.0 * lam1 + lam0) - kt * lam0 * lam2 - ktr * S * lam2 + ktr * S * lam0








    raise NotImplementedError("Hole 1: 请实现死链矩方程与 dydt 构造")


def polymerization_initial_state(params: PolymerizationParameters) -> np.ndarray:
    y0 = np.array([
        params.M0,
        params.I0,
        params.S0,
        1.0e-12,
        1.0e-12,
        1.0e-12,
        1.0e-12,
        1.0e-12,
        1.0e-12,
    ])
    return y0


def rk45_step(yprime: Callable, t: float, y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
        [3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0],
        [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0],
        [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0,
         44275.0 / 110592.0, 253.0 / 4096.0]
    ], dtype=float)

    b5 = np.array([37.0 / 378.0, 0.0, 250.0 / 621.0,
                   125.0 / 594.0, 0.0, 512.0 / 1771.0], dtype=float)
    b4 = np.array([2825.0 / 27648.0, 0.0, 18575.0 / 48384.0,
                   13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0], dtype=float)
    c = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0], dtype=float)

    k = np.zeros((6, y.size), dtype=float)
    k[0, :] = dt * yprime(t + c[0] * dt, y)
    for i in range(1, 6):
        yi = y.copy()
        for j in range(i):
            yi += a[i, j] * k[j, :]
        k[i, :] = dt * yprime(t + c[i] * dt, yi)

    y5 = y + np.dot(b5, k)
    y4 = y + np.dot(b4, k)
    error = np.abs(y5 - y4)
    return y5, error


def integrate_polymerization(params: PolymerizationParameters,
                             n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    t0 = params.t0
    tstop = params.tstop
    y0 = polymerization_initial_state(params)

    dt_initial = (tstop - t0) / n_steps
    t_vec = np.zeros(n_steps + 1)
    y_mat = np.zeros((n_steps + 1, y0.size))
    t_vec[0] = t0
    y_mat[0, :] = y0

    t = t0
    y = y0.copy()
    dt = dt_initial
    step = 0


    atol = 1.0e-8
    rtol = 1.0e-6
    safety = 0.9
    min_dt = 1.0e-6
    max_dt = (tstop - t0) / 10.0

    while t < tstop and step < n_steps:
        dt = min(dt, tstop - t)

        def yprime(tau, yy):
            return polymerization_deriv(tau, yy, params)

        y_next, err = rk45_step(yprime, t, y, dt)


        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        if err_norm <= 1.0 or dt <= min_dt * 1.01:
            t += dt
            step += 1
            y = y_next.copy()

            y = np.maximum(y, 1.0e-15)
            if step <= n_steps:
                t_vec[step] = t
                y_mat[step, :] = y


            if err_norm > 0.0:
                dt = min(max_dt, safety * dt * err_norm ** (-0.2))
        else:

            dt = max(min_dt, safety * dt * err_norm ** (-0.25))


    if step < n_steps:
        y_mat[step + 1:, :] = y_mat[step, :]
        t_vec[step + 1:] = tstop

    return t_vec, y_mat


def compute_conversion_and_pdi(t_vec: np.ndarray,
                               y_mat: np.ndarray,
                               params: PolymerizationParameters) -> Dict[str, np.ndarray]:
    M = y_mat[:, 0]
    lam0, lam1, lam2 = y_mat[:, 3], y_mat[:, 4], y_mat[:, 5]
    mu0, mu1, mu2 = y_mat[:, 6], y_mat[:, 7], y_mat[:, 8]

    M0_monomer = 104.15
    conversion = (params.M0 - M) / params.M0
    conversion = np.clip(conversion, 0.0, 1.0)

    total_active = lam0 + mu0
    total_active = np.where(total_active < 1.0e-14, 1.0e-14, total_active)

    DP_n = (lam1 + mu1) / total_active
    DP_w = (lam2 + mu2) / np.where(lam1 + mu1 < 1.0e-14, 1.0e-14, lam1 + mu1)
    PDI = DP_w / np.where(DP_n < 1.0e-14, 1.0e-14, DP_n)

    Mn = DP_n * M0_monomer
    Mw = DP_w * M0_monomer

    return {
        't': t_vec,
        'conversion': conversion,
        'DP_n': DP_n,
        'DP_w': DP_w,
        'PDI': PDI,
        'Mn': Mn,
        'Mw': Mw,
        'M': M,
        'lam0': lam0,
        'mu0': mu0,
    }


def exact_solution_batch(t_array: np.ndarray, params: PolymerizationParameters) -> np.ndarray:
    t = np.asarray(t_array)
    kd = params.kd
    kt = params.ktc + params.ktd
    f = params.f
    I0 = params.I0
    kp_eff = params.effective_rate_constants()['kp']

    A = np.sqrt(2.0 * f * kd * I0 / kt) * (2.0 / kd)
    integral = A * (1.0 - np.exp(-kd * t / 2.0))
    M_approx = params.M0 * np.exp(-kp_eff * integral)
    return M_approx
