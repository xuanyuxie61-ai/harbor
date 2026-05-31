"""
matsubara_green.py

基于 divdif (310_divdif)、shepard_interp_1d (1071_shepard_interp_1d)
与 lebesgue (658_lebesgue) 的 Matsubara 格林函数处理模块。

虚时 (Matsubara) 频率:
    ω_n = (2n+1)π/β   (费米子)
    ν_n = 2nπ/β       (玻色子)

本模块提供:
1. Matsubara 频率格点生成
2. 离散虚时到 Matsubara 频率的 Fourier 变换
3. 基于 Newton 分差插值 (divdif) 的自能插值
4. 基于 Shepard 插值的谱函数重构
5. Lebesgue 常数估计用于判断插值稳定性
"""

import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple, Optional


class MatsubaraGrid:
    """Matsubara 频率网格。"""

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
    """
    构造非相互作用 Matsubara 格林函数:
        G_0(iω_n) = [(iω_n + μ)I - K]^{-1}。
    
    返回:
        G0: 形状 (2*n_max+1, nsites, nsites)，复数数组
    """
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
    """
    将虚时格林函数 g(τ) 通过离散 Fourier 变换映射到 Matsubara 频率。
    
    g(iω_n) = ∫_0^β dτ e^{iω_n τ} g(τ)
    
    采用 Simpson 积分或 FFT。
    """
    if len(g_tau) < 2:
        raise ValueError("g_tau 长度必须 >= 2")
    ntau = len(g_tau)
    tau = np.linspace(0, beta, ntau)
    dtau = tau[1] - tau[0]
    # 使用 FFT 近似
    # 对费米子，需要处理 β-周期性中的反对称因子
    if fermionic:
        factor = np.exp(1j * np.pi * tau / beta)
        g_shifted = g_tau * factor
    else:
        g_shifted = g_tau
    # FFT
    g_freq = fft(g_shifted) * dtau
    # 频率对应关系
    n_max = ntau // 2
    omega = np.array([(2 * n + 1) * np.pi / beta for n in range(-n_max, n_max + 1)])
    if fermionic:
        omega = np.array([(2 * n + 1) * np.pi / beta for n in range(-n_max, n_max + 1)])
    else:
        omega = np.array([2 * n * np.pi / beta for n in range(-n_max, n_max + 1)])
    # 重排频率顺序
    g_freq = np.fft.fftshift(g_freq)
    return omega, g_freq


def dft_frequency_to_time(g_omega: np.ndarray, omega: np.ndarray, beta: float, ntau: int) -> np.ndarray:
    """
    从 Matsubara 频率反变换回虚时。
    
    g(τ) = (1/β) Σ_n e^{-iω_n τ} g(iω_n)
    """
    if ntau < 2:
        raise ValueError("ntau >= 2")
    tau = np.linspace(0, beta, ntau)
    g_tau = np.zeros(ntau, dtype=np.complex128)
    for idx, t in enumerate(tau):
        g_tau[idx] = np.sum(np.exp(-1j * omega * t) * g_omega) / beta
    return g_tau.real


# ---------------------------------------------------------------------------
# divdif: Newton 分差插值
# ---------------------------------------------------------------------------

def newton_divided_differences(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    """
    计算 Newton 分差表。
    
    输入:
        xd: 插值节点 (要求互异)
        yd: 节点函数值
    返回:
        dif: 分差系数数组，其中 dif[i] = f[x_0, ..., x_i]
    """
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
    """
    用 Horner 法则求值 Newton 分差多项式。
    
    P(x) = dif[0] + (x-x0)(dif[1] + (x-x1)(dif[2] + ...))
    """
    xd = np.asarray(xd).ravel()
    dif = np.asarray(dif).ravel()
    xv = np.asarray(xv).ravel()
    n = len(dif)
    yv = np.full_like(xv, dif[n - 1], dtype=np.float64)
    for i in range(n - 2, -1, -1):
        yv = dif[i] + (xv - xd[i]) * yv
    return yv


# ---------------------------------------------------------------------------
# shepard_interp_1d: Shepard 插值
# ---------------------------------------------------------------------------

def shepard_interp_1d(xd: np.ndarray, yd: np.ndarray, p: float, xi: np.ndarray) -> np.ndarray:
    """
    一维 Shepard (逆距离加权) 插值。
    
    参数:
        xd: 数据点
        yd: 数据值
        p: 幂指数 (p>0)
        xi: 插值点
    
    返回:
        yi: 插值结果
    """
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
            # 精确命中数据点
            if np.any(dist == 0.0):
                yi[i] = yd[np.argmin(dist)]
                continue
            w = 1.0 / dist ** p
            w = w / np.sum(w)
        yi[i] = np.dot(w, yd)
    return yi


# ---------------------------------------------------------------------------
# lebesgue: Lebesgue 常数估计
# ---------------------------------------------------------------------------

def lebesgue_function(n: int, x: np.ndarray, xfun: np.ndarray) -> np.ndarray:
    """
    计算 Lebesgue 函数 L(x) = Σ_{j=0}^{n-1} |l_j(x)|，
    其中 l_j 为 Lagrange 基多项式。
    """
    x = np.asarray(x).ravel()
    xfun = np.asarray(xfun).ravel()
    if len(x) != n:
        raise ValueError("len(x) 必须等于 n")
    lfun = np.zeros(len(xfun))
    for j in range(n):
        # 构造第 j 个 Lagrange 基
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
    """估计插值点集 x 的 Lebesgue 常数。"""
    lfun = lebesgue_function(n, x, xfun)
    return float(np.max(lfun))


def dyson_equation(G0: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Dyson 方程: G = [G0^{-1} - Σ]^{-1}。
    
    参数:
        G0: 非相互作用格林函数，形状 (nw, N, N)
        Sigma: 自能，形状 (nw, N, N)
    
    返回:
        G: 完全格林函数
    """
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
