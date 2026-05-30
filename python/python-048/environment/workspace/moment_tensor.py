
import numpy as np
from typing import Tuple


def vandermonde_matrix(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size != n:
        raise ValueError("x 长度必须等于 n")
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0 and abs(x[j]) < 1.0e-14:
                V[i, j] = 1.0
            else:
                V[i, j] = x[j] ** i
    return V


def caesar_cycle_matrix_3d() -> np.ndarray:
    return np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]])


def apply_caesar_rotation(M: np.ndarray, k: int = 1) -> np.ndarray:
    if M.shape != (3, 3):
        raise ValueError("M 必须是 3x3 矩阵")
    P = caesar_cycle_matrix_3d()

    Pk = np.linalg.matrix_power(P, k % 3)
    return Pk @ M @ Pk.T


class MomentTensor:

    def __init__(self, M: np.ndarray):
        M = np.asarray(M, dtype=float)
        if M.shape != (3, 3):
            raise ValueError("矩张量必须是 3x3 矩阵")

        self.M = M

    @classmethod
    def from_strike_dip_rake(cls, strike_deg: float, dip_deg: float,
                              rake_deg: float, M0: float = 1.0e12):
        sigma = np.deg2rad(strike_deg)
        delta = np.deg2rad(dip_deg)
        lam = np.deg2rad(rake_deg)

        s = np.array([
            np.cos(lam) * np.cos(delta) * np.cos(sigma) + np.sin(lam) * np.sin(sigma),
            np.cos(lam) * np.cos(delta) * np.sin(sigma) - np.sin(lam) * np.cos(sigma),
            -np.cos(lam) * np.sin(delta)
        ])
        n = np.array([
            -np.sin(delta) * np.cos(sigma),
            -np.sin(delta) * np.sin(sigma),
            -np.cos(delta)
        ])
        M = M0 * (np.outer(s, n) + np.outer(n, s))
        return cls(M)

    @property
    def eigenvalues(self) -> np.ndarray:
        w = np.linalg.eigvalsh(self.M)
        return np.sort(w)[::-1]

    @property
    def seismic_moment(self) -> float:

        raise NotImplementedError("Hole 3: 请实现地震矩公式")

    @property
    def moment_magnitude(self) -> float:
        M0 = self.seismic_moment
        if M0 <= 0:
            return -np.inf
        return (2.0 / 3.0) * np.log10(M0) - 6.07

    def radiation_pattern_p(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        gamma1 = np.sin(theta) * np.cos(phi)
        gamma2 = np.sin(theta) * np.sin(phi)
        gamma3 = np.cos(theta)

        A = (gamma1 ** 2 * self.M[0, 0]
             + gamma2 ** 2 * self.M[1, 1]
             + gamma3 ** 2 * self.M[2, 2]
             + 2.0 * gamma1 * gamma2 * self.M[0, 1]
             + 2.0 * gamma1 * gamma3 * self.M[0, 2]
             + 2.0 * gamma2 * gamma3 * self.M[1, 2])
        return A

    def radiation_pattern_s(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        gamma1 = np.sin(theta) * np.cos(phi)
        gamma2 = np.sin(theta) * np.sin(phi)
        gamma3 = np.cos(theta)


        Mgk = self.M @ np.array([gamma1, gamma2, gamma3])

        Mproj = (gamma1 * Mgk[0] + gamma2 * Mgk[1] + gamma3 * Mgk[2])

        S1 = Mgk[0] - gamma1 * Mproj
        S2 = Mgk[1] - gamma2 * Mproj
        S3 = Mgk[2] - gamma3 * Mproj




        e_th1 = np.cos(theta) * np.cos(phi)
        e_th2 = np.cos(theta) * np.sin(phi)
        e_th3 = -np.sin(theta)
        e_ph1 = -np.sin(phi)
        e_ph2 = np.cos(phi)
        e_ph3 = 0.0

        Asv = S1 * e_th1 + S2 * e_th2 + S3 * e_th3
        Ash = S1 * e_ph1 + S2 * e_ph2 + S3 * e_ph3
        return Asv, Ash

    def interpolate_radiation_vandermonde(self, n_samples: int = 8) -> np.ndarray:
        phi_samples = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        theta_fixed = np.full_like(phi_samples, np.pi / 2.0)
        A_samples = self.radiation_pattern_p(theta_fixed, phi_samples)



        V = vandermonde_matrix(n_samples, phi_samples)

        coeffs = np.linalg.lstsq(V, A_samples, rcond=None)[0]
        return coeffs
