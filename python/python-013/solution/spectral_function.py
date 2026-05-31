"""
spectral_function.py

谱函数计算与解析延拓模块。

在 Matsubara 频率上获得的格林函数 G(iω_n) 需解析延拓到实轴
才能得到可观测量:
    G(ω+i0^+) = ∫ dω' A(ω') / (ω - ω' + i0^+)

其中 A(ω) = -(1/π) Im G(ω+i0^+) 为谱函数，满足求和规则:
    ∫ dω A(ω) = 1。

本模块提供:
1. Padé 近似解析延拓
2. 最大熵方法 (MaxEnt) 的简化实现
3. 谱矩计算
4. 谱函数与自能的关系
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Padé 近似解析延拓
# ---------------------------------------------------------------------------

def pade_approximant(z_points: np.ndarray, g_points: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    """
    对角 Padé 近似 [N/N]:
        G(z) ≈ P_N(z) / Q_N(z)
    
    使用线性方程组直接求解系数:
        P_N(z_i) = G(z_i) Q_N(z_i)
    其中 P_N(z) = p_0 + p_1 z + ... + p_N z^N
          Q_N(z) = 1 + q_1 z + ... + q_N z^N
    
    参数:
        z_points: Matsubara 频率点 (复数)
        g_points: 对应格林函数值 (复数)
        z_eval: 待求值的实频率点 (复数，通常 ω + iη)
    
    返回:
        g_eval: Padé 近似值
    """
    n = len(z_points)
    if n != len(g_points):
        raise ValueError("z_points 与 g_points 长度不一致")
    if n < 2:
        raise ValueError("至少需要 2 个点")
    # 使用 [N/2, N/2] 型，N = n - 1
    N = n // 2
    if 2 * N + 1 > n:
        N = (n - 1) // 2
    # 构造线性系统 A x = b
    # x = [p_0, ..., p_N, q_1, ..., q_N]
    # 方程: Σ_{k=0}^N p_k z_i^k - G(z_i) Σ_{k=1}^N q_k z_i^k = G(z_i)
    m = 2 * N + 1
    A = np.zeros((m, m), dtype=np.complex128)
    b = np.zeros(m, dtype=np.complex128)
    for i in range(m):
        zi = z_points[i]
        gi = g_points[i]
        # P_N 系数
        for k in range(N + 1):
            A[i, k] = zi ** k
        # Q_N 系数 (不含常数项 1)
        for k in range(1, N + 1):
            A[i, N + k] = -gi * zi ** k
        b[i] = gi
    # 求解最小二乘 (允许超定)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    p = x[:N + 1]
    q = np.concatenate([[1.0], x[N + 1:]])
    # 求值
    g_eval = np.zeros(len(z_eval), dtype=np.complex128)
    for k, z in enumerate(z_eval):
        num = np.polyval(p[::-1], z)
        den = np.polyval(q[::-1], z)
        if abs(den) < 1e-14:
            den = 1e-14
        g_eval[k] = num / den
    g_eval = np.where(np.isfinite(g_eval), g_eval, 0.0)
    return g_eval


def pade_spectral_function(omega_n: np.ndarray, g_iw: np.ndarray,
                           omega_real: np.ndarray, eta: float = 0.05) -> np.ndarray:
    """
    使用 Padé 近似计算实频率谱函数 A(ω)。
    
    参数:
        omega_n: Matsubara 频率 (正数部分即可)
        g_iw: G(iω_n) 值
        omega_real: 实频率网格
        eta: 展宽
    
    返回:
        A(ω): 谱函数
    """
    # 构造对称的 Matsubara 点 (正+负)
    z_full = np.concatenate([-omega_n[::-1], omega_n])
    g_full = np.concatenate([g_iw.conj()[::-1], g_iw])
    z_eval = omega_real + 1j * eta
    g_real = pade_approximant(z_full, g_full, z_eval)
    A = -g_real.imag / np.pi
    A = np.where(A > 0, A, 0.0)
    return A


# ---------------------------------------------------------------------------
# 最大熵方法 (MaxEnt) 简化实现
# ---------------------------------------------------------------------------

def maxent_spectral_function(omega_n: np.ndarray, g_iw: np.ndarray,
                             omega_real: np.ndarray, default_model: Optional[np.ndarray] = None,
                             alpha: float = 1.0) -> np.ndarray:
    """
    简化版最大熵解析延拓:
        最小化 Q = χ^2 / 2 - α S
    
    其中 χ^2 = Σ |G(iω_n) - G_{model}(iω_n)|^2 / σ_n^2
          S = ∫ dω [A(ω) - D(ω) - A(ω) log(A(ω)/D(ω))]   (相对熵)
    
    参数:
        omega_n: 正 Matsubara 频率
        g_iw: G(iω_n)
        omega_real: 实频率网格
        default_model: 默认模型 D(ω)，若 None 则用常数
        alpha: 熵权重
    
    返回:
        A(ω)
    """
    n_omega = len(omega_real)
    domega = omega_real[1] - omega_real[0] if n_omega > 1 else 1.0
    if default_model is None:
        default_model = np.ones(n_omega) / (n_omega * domega)
    default_model = np.abs(default_model)
    default_model = default_model / np.trapezoid(default_model, omega_real)
    
    # 构造核函数 K_{n,m} = 1 / (iω_n - ω_m)
    K = np.zeros((len(omega_n), n_omega), dtype=np.complex128)
    for n, wn in enumerate(omega_n):
        for m, w in enumerate(omega_real):
            K[n, m] = 1.0 / (1j * wn - w)
    
    def objective(A):
        A = np.abs(A)
        # 模型格林函数
        g_model = K @ A * domega
        chi2 = 0.5 * np.sum(np.abs(g_iw - g_model) ** 2)
        # 相对熵 (简化)
        ratio = A / (default_model + 1e-14)
        ratio = np.where(ratio > 1e-14, ratio, 1e-14)
        S = np.sum(A * np.log(ratio)) * domega
        return chi2 - alpha * S
    
    # 约束优化: A >= 0, ∫ A dω = 1
    from scipy.optimize import minimize
    A0 = default_model.copy()
    bounds = [(0.0, None) for _ in range(n_omega)]
    # 等式约束
    def eq_con(A):
        return np.trapezoid(A, omega_real) - 1.0
    
    cons = {"type": "eq", "fun": eq_con}
    result = minimize(objective, A0, method="SLSQP", bounds=bounds, constraints=cons,
                      options={"maxiter": 500, "ftol": 1e-8})
    A_opt = np.abs(result.x)
    # 再次归一化
    norm = np.trapezoid(A_opt, omega_real)
    if norm > 0:
        A_opt /= norm
    return A_opt


# ---------------------------------------------------------------------------
# 谱矩计算
# ---------------------------------------------------------------------------

def spectral_moments(A: np.ndarray, omega: np.ndarray, max_moment: int = 4) -> dict:
    """
    计算谱函数的矩:
        M_n = ∫ dω ω^n A(ω)
    
    对于 Hubbard 模型，前几个矩有精确值:
        M_0 = 1
        M_1 = ε_k + Σ(∞)
        M_2 = (ε_k + Σ(∞))^2 + Σ'(∞)
    """
    if max_moment < 0:
        raise ValueError("max_moment >= 0")
    moments = {}
    domega = omega[1] - omega[0] if len(omega) > 1 else 1.0
    for n in range(max_moment + 1):
        M = np.trapezoid(omega ** n * A, omega)
        moments[f"M_{n}"] = float(M)
    return moments


def self_energy_from_greens_function(omega: np.ndarray, g: np.ndarray, epsilon_k: float) -> np.ndarray:
    """
    从完全格林函数提取自能:
        Σ(ω) = ω - ε_k - 1/G(ω)
    """
    g = np.where(np.abs(g) > 1e-14, g, 1e-14)
    return omega - epsilon_k - 1.0 / g


def kramers_kronig_relation(imag_part: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Kramers-Kronig 关系:
        Re[f(ω)] = (1/π) P ∫ dω' Im[f(ω')] / (ω' - ω)
    
    用数值积分实现主值积分。
    """
    n = len(omega)
    real_part = np.zeros(n)
    domega = omega[1] - omega[0]
    for i in range(n):
        integrand = imag_part / (omega - omega[i])
        # 排除奇点
        mask = np.abs(omega - omega[i]) > 1e-10
        if np.any(mask):
            real_part[i] = np.trapezoid(integrand[mask], omega[mask]) / np.pi
    return real_part


if __name__ == "__main__":
    # 测试 Padé 近似
    omega_n = np.array([1.0, 3.0, 5.0, 7.0, 9.0]) * np.pi
    g_iw = 1.0 / (1j * omega_n + 0.5)
    omega_real = np.linspace(-5, 5, 100)
    A = pade_spectral_function(omega_n, g_iw, omega_real, eta=0.1)
    print(f"Spectral sum rule: {np.trapezoid(A, omega_real):.6f} (expect ~1.0)")
