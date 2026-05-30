
import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple, Optional


class MatsubaraGrid:

    def __init__(self, beta: float, n_max: int, fermionic: bool = True):
        if beta <= 0:
            raise ValueError("beta > 0 required")
        if n_max < 0:
            raise ValueError("n_max >= 0 required")
        self.beta = beta
        self.n_max = n_max
        self.fermionic = fermionic
        if fermionic:
            self.omega_n = np.array([(2 * n + 1) * np.pi / beta for n in range(-n_max, n_max + 1)])
        else:
            self.omega_n = np.array([2 * n * np.pi / beta for n in range(-n_max, n_max + 1)])


def build_matsubara_green(nsites: int, K: np.ndarray, mu: float, beta: float, U: float,
                          n_max: int, sigma: int) -> np.ndarray:
    if n_max < 0:
        raise ValueError("n_max >= 0")
    nw = 2 * n_max + 1
    G0 = np.zeros((nw, nsites, nsites), dtype=np.complex128)
    I = np.eye(nsites)
    mg = MatsubaraGrid(beta, n_max, fermionic=True)
    for idx, wn in enumerate(mg.omega_n):
        M = (1j * wn + mu) * I - K
        G0[idx] = np.linalg.inv(M)
    return G0


def dft_time_to_frequency(g_tau: np.ndarray, beta: float, fermionic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if len(g_tau) < 2:
        raise ValueError("g_tau 长度必须 >= 2")
    ntau = len(g_tau)
    tau = np.linspace(0, beta, ntau)
    dtau = tau[1] - tau[0]


    if fermionic:
        factor = np.exp(1j * np.pi * tau / beta)
        g_shifted = g_tau * factor
    else:
        g_shifted = g_tau

    g_freq = fft(g_shifted) * dtau

    n_max = ntau // 2
    omega = np.array([(2 * n + 1) * np.pi / beta for n in range(-n_max, n_max + 1)])
    if fermionic:
        omega = np.array([(2 * n + 1) * np.pi / beta for n in range(-n_max, n_max + 1)])
    else:
        omega = np.array([2 * n * np.pi / beta for n in range(-n_max, n_max + 1)])

    g_freq = np.fft.fftshift(g_freq)
    return omega, g_freq


def dft_frequency_to_time(g_omega: np.ndarray, omega: np.ndarray, beta: float, ntau: int) -> np.ndarray:
    if ntau < 2:
        raise ValueError("ntau >= 2")
    tau = np.linspace(0, beta, ntau)
    g_tau = np.zeros(ntau, dtype=np.complex128)
    for idx, t in enumerate(tau):
        g_tau[idx] = np.sum(np.exp(-1j * omega * t) * g_omega) / beta
    return g_tau.real






def newton_divided_differences(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    n = len(xd)
    if n != len(yd):
        raise ValueError("xd 与 yd 长度不一致")
    if len(np.unique(xd)) != n:
        raise ValueError("xd 节点必须互异")
    dif = yd.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            dif[i] = (dif[i] - dif[i - 1]) / (xd[i] - xd[i - j])
    return dif


def evaluate_divided_difference(xd: np.ndarray, dif: np.ndarray, xv: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd).ravel()
    dif = np.asarray(dif).ravel()
    xv = np.asarray(xv).ravel()
    n = len(dif)
    yv = np.full_like(xv, dif[n - 1], dtype=np.float64)
    for i in range(n - 2, -1, -1):
        yv = dif[i] + (xv - xd[i]) * yv
    return yv






def shepard_interp_1d(xd: np.ndarray, yd: np.ndarray, p: float, xi: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    xi = np.asarray(xi).ravel()
    nd = len(xd)
    if nd != len(yd):
        raise ValueError("xd 与 yd 长度不一致")
    if p < 0:
        raise ValueError("p >= 0 required")
    yi = np.zeros(len(xi))
    for i in range(len(xi)):
        if p == 0.0:
            w = np.ones(nd) / nd
        else:
            dist = np.abs(xi[i] - xd)

            if np.any(dist == 0.0):
                yi[i] = yd[np.argmin(dist)]
                continue
            w = 1.0 / dist ** p
            w = w / np.sum(w)
        yi[i] = np.dot(w, yd)
    return yi






def lebesgue_function(n: int, x: np.ndarray, xfun: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    xfun = np.asarray(xfun).ravel()
    if len(x) != n:
        raise ValueError("len(x) 必须等于 n")
    lfun = np.zeros(len(xfun))
    for j in range(n):

        lj = np.ones(len(xfun))
        for k in range(n):
            if k != j:
                denom = x[j] - x[k]
                if abs(denom) < 1e-14:
                    denom = 1e-14
                lj *= (xfun - x[k]) / denom
        lfun += np.abs(lj)
    return lfun


def lebesgue_constant_estimate(n: int, x: np.ndarray, xfun: np.ndarray) -> float:
    lfun = lebesgue_function(n, x, xfun)
    return float(np.max(lfun))


def dyson_equation(G0: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    nw, N, _ = G0.shape
    G = np.zeros_like(G0)
    for w in range(nw):
        M = np.linalg.inv(G0[w]) - Sigma[w]
        G[w] = np.linalg.inv(M)
    return G


if __name__ == "__main__":
    xd = np.array([0.0, 1.0, 2.0, 3.0])
    yd = np.array([1.0, 2.0, 1.5, 0.5])
    dif = newton_divided_differences(xd, yd)
    xv = np.linspace(0, 3, 20)
    yv = evaluate_divided_difference(xd, dif, xv)
    print("Newton interpolation sample:", yv[:3])
    yi = shepard_interp_1d(xd, yd, 2.0, xv)
    print("Shepard interpolation sample:", yi[:3])
    lmax = lebesgue_constant_estimate(len(xd), xd, xv)
    print("Lebesgue constant:", lmax)
