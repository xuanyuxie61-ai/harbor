
import numpy as np
from typing import Tuple, Optional


class MultiferroicMaterialParams:

    def __init__(self, temperature: float = 300.0):
        self.T = temperature
        self.Tc = 1103.0
        self.Tn = 643.0
        self.alpha0 = 1.0e5
        self.alpha1 = self.alpha0 * (self.T - self.Tc)
        self.alpha11 = 7.0e8
        self.alpha12 = 2.0e8
        self.beta0 = 5.0e3
        self.beta1 = self.beta0 * (self.T - self.Tn)
        self.beta11 = 1.0e6
        self.beta12 = 3.0e5
        self.gamma = 2.0e-3
        self.g11 = 2.0e-10
        self.g12 = 1.0e-10
        self.g44 = 1.0e-10
        self.A11 = 5.0e-12
        self.A12 = 2.0e-12
        self.eta = 1.0e-15
        self.sigma = 0.05

    def validate(self):
        assert np.isfinite(self.alpha1), "alpha1 必须为有限值"
        assert np.isfinite(self.beta1), "beta1 必须为有限值"
        assert self.alpha11 > 0, "alpha11 必须为正（保证铁电相稳定）"
        assert self.beta11 > 0, "beta11 必须为正"


def hermite_probabilist(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if n < 0:
        raise ValueError("n 必须非负")
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return x.copy()
    H_prev2 = np.ones_like(x, dtype=float)
    H_prev1 = x.copy()
    for j in range(2, n + 1):
        H_curr = x * H_prev1 - (j - 1) * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


def normalized_hermite_probabilist(n: int, x: np.ndarray) -> np.ndarray:
    He = hermite_probabilist(n, x)
    norm = np.sqrt(np.math.factorial(n) * np.sqrt(2.0 * np.pi))
    return He / norm


def landau_free_energy_density(P: np.ndarray, M: np.ndarray,
                                dPdx: np.ndarray, dPdy: np.ndarray,
                                dMdx: np.ndarray, dMdy: np.ndarray,
                                params: MultiferroicMaterialParams) -> float:
    Px, Py = float(P[0]), float(P[1])
    Mx, My = float(M[0]), float(M[1])


    P2 = Px * Px + Py * Py
    fP = (params.alpha1 * P2
          + params.alpha11 * P2 * P2
          + params.alpha12 * Px * Px * Py * Py)


    M2 = Mx * Mx + My * My
    fM = (params.beta1 * M2
          + params.beta11 * M2 * M2
          + params.beta12 * Mx * Mx * My * My)


    fc = params.gamma * (Px * My - Py * Mx) ** 2


    f_grad_P = (0.5 * params.g11 * (dPdx[0] ** 2 + dPdy[1] ** 2)
                + 0.5 * params.g12 * (dPdx[1] ** 2 + dPdy[0] ** 2)
                + params.g44 * dPdx[1] * dPdy[0])


    f_grad_M = (0.5 * params.A11 * (dMdx[0] ** 2 + dMdy[1] ** 2)
                + 0.5 * params.A12 * (dMdx[1] ** 2 + dMdy[0] ** 2))


    f_cross = params.eta * (dPdx[0] * dMdy[1] - dPdy[1] * dMdx[0])

    f_total = fP + fM + fc + f_grad_P + f_grad_M + f_cross


    if not np.isfinite(f_total):
        return 1e20
    return f_total


def variational_derivative_P(P: np.ndarray, M: np.ndarray,
                              lapP: np.ndarray, params: MultiferroicMaterialParams) -> np.ndarray:






    pass


def variational_derivative_M(P: np.ndarray, M: np.ndarray,
                              lapM: np.ndarray, params: MultiferroicMaterialParams) -> np.ndarray:






    pass


def thermal_fluctuation_correction(P: np.ndarray, M: np.ndarray,
                                    params: MultiferroicMaterialParams,
                                    max_hermite_order: int = 6) -> float:
    kB = 1.380649e-23
    Px, Py = P[0], P[1]
    Mx, My = M[0], M[1]


    H = np.zeros((4, 4), dtype=float)
    P2 = Px * Px + Py * Py
    M2 = Mx * Mx + My * My


    H[0, 0] = params.alpha1 + 12.0 * params.alpha11 * Px * Px + 4.0 * params.alpha11 * Py * Py + 2.0 * params.alpha12 * Py * Py
    H[1, 1] = params.alpha1 + 12.0 * params.alpha11 * Py * Py + 4.0 * params.alpha11 * Px * Px + 2.0 * params.alpha12 * Px * Px
    H[0, 1] = H[1, 0] = 8.0 * params.alpha11 * Px * Py + 4.0 * params.alpha12 * Px * Py


    H[2, 2] = params.beta1 + 12.0 * params.beta11 * Mx * Mx + 4.0 * params.beta11 * My * My + 2.0 * params.beta12 * My * My
    H[3, 3] = params.beta1 + 12.0 * params.beta11 * My * My + 4.0 * params.beta11 * Mx * Mx + 2.0 * params.beta12 * Mx * Mx
    H[2, 3] = H[3, 2] = 8.0 * params.beta11 * Mx * My + 4.0 * params.beta12 * Mx * My


    cross = Px * My - Py * Mx
    H[0, 2] = H[2, 0] = 2.0 * params.gamma * cross * (-Py) + 2.0 * params.gamma * My * My
    H[0, 3] = H[3, 0] = 2.0 * params.gamma * cross * Px + 2.0 * params.gamma * My * (-Mx)
    H[1, 2] = H[2, 1] = 2.0 * params.gamma * cross * My + 2.0 * params.gamma * (-Mx) * (-Py)
    H[1, 3] = H[3, 1] = 2.0 * params.gamma * cross * (-Mx) + 2.0 * params.gamma * (-Mx) * Px


    H += np.eye(4) * 1e-10

    eigvals = np.linalg.eigvalsh(H)

    eigvals = np.clip(eigvals, 1e-20, None)


    delta_f = 0.5 * kB * params.T * np.sum(np.log(eigvals / 1.0))
    if not np.isfinite(delta_f):
        delta_f = 0.0
    return delta_f
