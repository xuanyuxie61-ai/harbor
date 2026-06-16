"""
stability_analysis.py  —  数值稳定性与色散关系分析
================================================

科学来源种子:
  - 1429_zero_itp / zero_itp.m
    直接使用其 ITP (Interpolate-Truncate-Project) 求根算法,
    用于求解数值色散关系的根 (临界 CFL 数、中性稳定曲线)。
    ITP 结合了割线法的超线性收敛与二分法的有界保证。
  - 902_power_method / power_method.m, power_method2.m
    直接使用其幂法与 power_method2 的复数特征值扩展,
    用于估计放大矩阵的谱半径 ρ(G),判定 von Neumann 稳定性:
        稳定 ⟺ ρ(G) ≤ 1 + O(Δt)

物理背景:
  对 PM 时间离散 (leapfrog),应用 von Neumann 分析到线性化
  Vlasov-Poisson 系统,得到放大矩阵 G(k, Δt):
      | δ^{n+1} |     | 1     iα  |  | δ^n |
      | v^{n+1} |  =  | iβ    1   |  | v^n |
  其中 α, β 依赖于 k·v_th Δt 与 4πG ρ̄ / k²。
  稳定性条件 (Ostriker & Gunn 1969):
      Δt < 2 / √(4πG ρ̄) · 1/|sin(kh/2)|
  本模块:
    1. 构造放大矩阵 G(k)
    2. 用 power_method2 计算谱半径 ρ(k)
    3. 用 zero_itp 求 ρ(k) = 1 的根 → 临界 Δt_crit
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable


# ---------------------------------------------------------------------------- #
#                  ITP 求根算法 (来自 zero_itp.m)
# ---------------------------------------------------------------------------- #
def zero_itp(f: Callable[[float], float], a: float, b: float,
             epsi: float = 1e-10, k1: float = 0.05,
             k2: float = 2.0, n0: int = 1,
             max_iter: int = 200) -> Tuple[float, float, int]:
    """
    ITP (Interpolate-Truncate-Project) 求根算法。
    寻找 f(x) = 0 在 [a, b] 内的根,假定 f(a) · f(b) < 0。

    算法步骤 (每轮):
      1. 截断: 缩小搜索区间至 min(2^(-n_h+n_0)·(b-a)/2, ε)
      2. 插值: 用割线法估计 x_hat
      3. 投影: x_itp = clip(x_hat, x_1/2 - δ, x_1/2 + δ)
      4. 更新区间
    保证 worst-case 收敛: n_h + n_0 步,其中
        n_h = ceil(log2((b-a)/(2 ε)))

    Returns
    -------
    z  : 根的估计
    fz : f(z)
    calls : 函数调用次数
    """
    if b < a:
        a, b = b, a
    ya = f(a)
    yb = f(b)
    if ya * yb > 0:
        raise ValueError(f"zero_itp: f(a)={ya}, f(b)={yb} 同号,无法求根")
    if ya > 0:
        s = -1.0
        ya, yb = -ya, -yb
        f_signed = lambda x: -f(x)
    else:
        s = 1.0
        f_signed = f
    nh = int(np.ceil(np.log2((b - a) / (2.0 * epsi))))
    nmax = nh + n0
    calls = 2
    n = 0
    a_k, b_k = a, b
    ya_k, yb_k = ya, yb
    while (b_k - a_k) > 2.0 * epsi and n < max_iter:
        # 中点:
        x_half = 0.5 * (a_k + b_k)
        # 截断长度 δ:
        delta = k1 * (b_k - a_k) ** k2
        # 割线插值:
        if abs(yb_k - ya_k) < 1e-300:
            x_hat = x_half
        else:
            x_hat = (a_k * yb_k - b_k * ya_k) / (yb_k - ya_k)
        # 投影:
        x_itp = max(a_k + delta, min(x_hat, b_k - delta))
        x_itp = min(max(x_itp, x_half - delta), x_half + delta)
        # 保证在区间内:
        x_itp = max(a_k, min(b_k, x_itp))
        fx = f_signed(x_itp)
        calls += 1
        if fx > 0:
            b_k = x_itp
            yb_k = fx
        elif fx < 0:
            a_k = x_itp
            ya_k = fx
        else:
            return x_itp, 0.0, calls
        n += 1
        if n >= nmax:
            break
    z = 0.5 * (a_k + b_k)
    return z, f(z), calls


# ---------------------------------------------------------------------------- #
#             幂法 (power_method.m + power_method2.m)
# ---------------------------------------------------------------------------- #
def power_method(A: NDArray, x0: NDArray, it_max: int = 1000,
                 tol: float = 1e-10) -> Tuple[complex, NDArray, int]:
    """
    标准幂法估计矩阵 A 的主特征值。
    对应 power_method.m。

    Returns
    -------
    lambda : complex  主特征值估计
    v      : ndarray   对应特征向量
    it_num : int       迭代次数
    """
    A = np.asarray(A, dtype=complex)
    x = x0.astype(complex).copy()
    x /= np.linalg.norm(x)
    it = 0
    lam_old = 0.0 + 0j
    for it in range(1, it_max + 1):
        y = A @ x
        lam = np.vdot(x, y)  # Rayleigh quotient
        y_norm = np.linalg.norm(y)
        if y_norm < 1e-30:
            return lam, x, it
        x = y / y_norm
        if abs(lam - lam_old) < tol * max(1.0, abs(lam)):
            return lam, x, it
        lam_old = lam
    return lam, x, it


def power_method2(A: NDArray, x_init: NDArray, it_max: int = 1000,
                  tol: float = 1e-10) -> Tuple[complex, NDArray, int]:
    """
    复数特征值幂法 (power_method2.m 复现):
    通过 3-term 递推同时估计共轭特征值对:
        x_{n+1} = α x_n + β y_n
    其中 α, β 由内积 π_xy, π_yy, π_xz, ... 决定。
    若 α² + 4β < 0, 得到复数特征值 λ = (-α ± i√(4β+α²))/2。
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    x = x_init.astype(complex).copy()
    pi_xx = np.vdot(x, x).real
    x = x / pi_xx
    y = A @ x
    pi_xy = np.vdot(x, y)
    pi_yy = np.vdot(y, y)
    for it in range(1, it_max + 1):
        if pi_yy - pi_xy * np.conj(pi_xy) < tol * tol * pi_yy:
            return complex(pi_xy), y / np.sqrt(pi_yy), it
        z = A @ y
        pi_xz = np.vdot(x, z)
        pi_yz = np.vdot(y, z)
        pi_zz = np.vdot(z, z)
        denom = pi_yy - pi_xy * np.conj(pi_xy)
        if abs(denom) < 1e-30:
            break
        alpha = -(pi_yz - pi_xy * pi_xz) / denom
        beta = (pi_xy * pi_yz - pi_yy * pi_xz) / denom
        gamma = pi_zz + abs(alpha) ** 2 * pi_yy + abs(beta) ** 2 \
            + 2.0 * (alpha * pi_yz + beta * pi_xz + alpha * beta * pi_xy)
        if abs(gamma) < tol * tol * pi_zz and alpha ** 2 < -4.0 * beta:
            lam_real = -alpha / 2.0
            lam_imag = np.sqrt(max(0.0, -4.0 * beta - alpha ** 2)) / 2.0
            lam = lam_real + 1j * lam_imag
            v = (lam * y - z) / np.sqrt(
                max(1e-30, beta * pi_yy + alpha * pi_yz + pi_zz))
            return complex(lam), v, it
        x = y / np.sqrt(max(1e-30, pi_yy))
        y = z / np.sqrt(max(1e-30, pi_yy))
        pi_xy = pi_yz / max(1e-30, pi_yy)
        pi_yy = pi_zz / max(1e-30, pi_yy)
    return complex(pi_xy), y / np.sqrt(max(1e-30, pi_yy)), it


# ---------------------------------------------------------------------------- #
#                  放大矩阵构造
# ---------------------------------------------------------------------------- #
def amplification_matrix_leapfrog(k_mag: float, dt: float,
                                  rho_bar: float, a: float) -> NDArray:
    """
    Leapfrog 时间离散的 2x2 放大矩阵 (线性化 Vlasov-Poisson):
        G = | 1           i α(k)  |
            | i β(k)      1       |
    其中:
        α(k) = -4πG ρ̄ a² / k² · Δt · sin(kh/2)² / (kh/2)²
        β(k) = k² · Δt · a
    简化:使用无量纲形式,设 4πG ρ̄ = 3/2 · H0² Ω_m,
         kh/2 为网格 Nyquist 参数。
    """
    G_H2_Om = 1.5 * 0.308 * (100.0 ** 2)  # 4πG ρ̄ 近似
    kappa = G_H2_Om / max(k_mag ** 2, 1e-10)
    alpha = -kappa * (a ** 2) * dt
    beta = (k_mag ** 2) * dt * a
    G = np.array([[1.0 + 0j, 1j * alpha],
                  [1j * beta, 1.0 + 0j]], dtype=complex)
    return G


def spectral_radius_leapfrog(k_mag: float, dt: float,
                              rho_bar: float = 1.0, a: float = 1.0) -> float:
    """计算给定 k, dt 下 leapfrog 放大矩阵的谱半径。"""
    G = amplification_matrix_leapfrog(k_mag, dt, rho_bar, a)
    lam, _, _ = power_method(G, np.array([1.0, 0.0], dtype=complex),
                              it_max=50, tol=1e-12)
    return float(abs(lam))


# ---------------------------------------------------------------------------- #
#                  临界 CFL 数求解
# ---------------------------------------------------------------------------- #
def critical_cfl_number(k_mag: float, rho_bar: float = 1.0,
                        a: float = 1.0, tol: float = 1e-8) -> float:
    """
    求 Δt_crit,使得 max eigenvalue = 1:
        ρ(G(k, Δt_crit)) = 1
    使用 ITP 算法在 [1e-4, 10] 区间求根。
    """
    def objective(dt):
        return spectral_radius_leapfrog(k_mag, dt, rho_bar, a) - 1.0
    # 检查端点:
    f_low = objective(1e-4)
    f_high = objective(10.0)
    if f_low * f_high > 0:
        # 无根; 返回保守估计
        return 0.5 / max(k_mag, 1e-10)
    dt_crit, _, calls = zero_itp(objective, 1e-4, 10.0, epsi=tol)
    return float(dt_crit)


# ---------------------------------------------------------------------------- #
#                  完整 von Neumann 稳定性扫描
# ---------------------------------------------------------------------------- #
def von_neumann_stability_scan(k_values: NDArray, dt_values: NDArray,
                                rho_bar: float = 1.0,
                                a: float = 1.0) -> NDArray:
    """
    在 (k, Δt) 网格上扫描谱半径,返回 (n_k, n_dt) 数组。
    """
    n_k = k_values.size
    n_dt = dt_values.size
    rho = np.zeros((n_k, n_dt))
    for i, k in enumerate(k_values):
        for j, dt in enumerate(dt_values):
            rho[i, j] = spectral_radius_leapfrog(k, dt, rho_bar, a)
    return rho


# ---------------------------------------------------------------------------- #
#              空间离散稳定性 (有限差分阶数影响)
# ---------------------------------------------------------------------------- #
def fd_symbol_amplification(p: int, kh: NDArray) -> NDArray:
    """
    2p 阶中心差分的符号放大因子:
        σ(θ) = Σ_{m=-p}^p s_m e^{i m θ}  / h²
    其中 θ = k h 为无量纲波数, s_m 为 FD 模板。
    稳定性要求 Im(σ) = 0 且 Re(σ) ≤ 0 (对热传导类问题)。
    """
    from finite_difference import fd_stencil_2nd
    s = fd_stencil_2nd(p)
    sigma = np.zeros_like(kh, dtype=complex)
    for m in range(-p, p + 1):
        sigma += s[m + p] * np.exp(1j * m * kh)
    return sigma
