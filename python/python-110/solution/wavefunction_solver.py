"""
wavefunction_solver.py - 量子点波函数求解与插值模块

本模块融合原项目 460_gegenbauer_cc（Gegenbauer 正交多项式数值积分）
与 592_interp_equal（Newton 插值）的核心算法，用于：
1. 使用特殊函数展开求解球对称量子点的径向波函数
2. 通过 Newton 差商插值在非均匀网格上重构波函数与势场

核心物理模型：
    球坐标下径向薛定谔方程（s 波）：
        - (hbar^2 / 2m*) d^2u/dr^2 + [V(r) + l(l+1)hbar^2/(2m*r^2)] u = E u
    其中 u(r) = r R(r)。
"""

import numpy as np
from typing import Callable, Tuple, List
from utils import validate_array_1d, tridiagonal_solve


def gegenbauer_weight(x: np.ndarray, lam: float) -> np.ndarray:
    """
    计算 Gegenbauer 权函数：
        w(x) = (1 - x^2)^(lambda - 1/2)
    
    用于球谐展开与角向积分。
    """
    x = validate_array_1d(x, "x")
    if lam <= -0.5:
        raise ValueError("lambda must be > -0.5")
    # 数值稳定性：在 |x| -> 1 时截断
    val = 1.0 - x ** 2
    val = np.where(val < 1e-14, 1e-14, val)
    return val ** (lam - 0.5)


def chebyshev_even_coefficients(n: int, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    计算函数 f 的偶次 Chebyshev 系数（基于 Clenshaw-Curtis 节点）：
    
        x_j = cos(j * pi / n),  j = 0, ..., n
    
    偶次系数：
        a_{2r} = (2/n) * [ 0.5 f(1) + sum_{j=1}^{n-1} f(x_j) cos(2r j pi / n) + 0.5 (-1)^r f(-1) ]
    
    返回 a2[0 : s+1]，其中 a2[rh] 对应 a_{2*rh}。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    s = n // 2
    sigma = n % 2
    a2 = np.zeros(s + 1, dtype=float)
    for r in range(0, 2 * s + 1, 2):
        rh = r // 2
        total = 0.5 * float(f(np.array([1.0]))[0])
        j_pts = np.arange(1, n)
        xj = np.cos(j_pts * np.pi / n)
        fj = f(xj)
        total += np.sum(fj * np.cos(r * j_pts * np.pi / n))
        total += 0.5 * ((-1.0) ** r) * float(f(np.array([-1.0]))[0])
        a2[rh] = (2.0 / n) * total
    return a2


def gegenbauer_cc_quadrature(n: int, lam: float, f: Callable[[np.ndarray], np.ndarray]) -> float:
    """
    Gegenbauer-Clenshaw-Curtis 数值积分：
    
        I = integral_{-1}^{+1} (1 - x^2)^(lambda - 1/2) f(x) dx
    
    算法步骤（Hunter & Smith, 2005）：
        1. 计算 f 的偶次 Chebyshev 系数 a_{2r}
        2. 递推计算 u：
            u_s = 0.5 (sigma + 1) a_{2s}
            u_{rh} = (rh - lam)/(rh + lam + 1) * u_{rh+1} + a_{2rh},  rh = s-1, ..., 1
            u_0 = -lam/(lam+1) * u_1 + 0.5 a_0
        3. I = Gamma(lam + 1/2) * sqrt(pi) / Gamma(lam + 1) * u_0
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if lam <= -0.5:
        raise ValueError("lambda must be > -0.5")
    a2 = chebyshev_even_coefficients(n, f)
    s = n // 2
    sigma = n % 2
    rh = s
    u = 0.5 * (sigma + 1.0) * a2[rh]
    for rh in range(s - 1, 0, -1):
        u = ((rh - lam) / (rh + lam + 1.0)) * u + a2[rh]
    u = -lam * u / (lam + 1.0) + 0.5 * a2[0]
    # 使用 scipy.special.gamma 的近似（通过 log-gamma 避免溢出）
    from math import lgamma, exp, sqrt, pi
    gamma_ratio = exp(lgamma(lam + 0.5) + 0.5 * np.log(np.pi) - lgamma(lam + 1.0))
    value = gamma_ratio * u
    return float(value)


def divided_differences(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    """
    计算 Newton 差商表（源自 interp_equal 的 divdif 函数）：
    
        dd_0 = y_0
        dd_1 = (y_1 - y_0) / (x_1 - x_0)
        dd_2 = (dd_1^{(1)} - dd_1^{(0)}) / (x_2 - x_0)
        ...
    """
    xd = validate_array_1d(xd, "xd")
    yd = validate_array_1d(yd, "yd")
    n = xd.size
    if yd.size != n:
        raise ValueError("xd and yd must have same length")
    dd = np.array(yd, dtype=float)
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            denom = xd[j] - xd[j - i]
            if abs(denom) < 1e-14:
                denom = 1e-14
            dd[j] = (dd[j] - dd[j - 1]) / denom
    return dd


def newton_interpolation_eval(xd: np.ndarray, dd: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """
    利用 Newton 差商形式在 xp 处求插值多项式的值：
    
        P(x) = dd_0 + dd_1 (x - x_0) + dd_2 (x - x_0)(x - x_1) + ...
    
    源自 interp_equal 的 interp 函数。
    """
    xd = validate_array_1d(xd, "xd")
    dd = validate_array_1d(dd, "dd")
    xp = np.asarray(xp, dtype=float)
    nd = xd.size
    if dd.size != nd:
        raise ValueError("dd must have same length as xd")
    yp = dd[nd - 1] * np.ones_like(xp)
    for i in range(nd - 2, -1, -1):
        yp = dd[i] + (xp - xd[i]) * yp
    return yp


def interpolate_potential(
    x_nodes: np.ndarray, V_nodes: np.ndarray, x_fine: np.ndarray
) -> np.ndarray:
    """
    使用 Newton 插值在细网格上重构势场分布。
    """
    x_nodes = validate_array_1d(x_nodes, "x_nodes")
    V_nodes = validate_array_1d(V_nodes, "V_nodes")
    dd = divided_differences(x_nodes, V_nodes)
    return newton_interpolation_eval(x_nodes, dd, x_fine)


def radial_wavefunction_shooting(
    r_grid: np.ndarray,
    V: np.ndarray,
    m_star_ratio: float,
    E_guess: float,
    l: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    使用打靶法（shooting method）求解球对称径向薛定谔方程。
    
    方程：
        - (hbar^2 / 2m*) u''(r) + [V(r) + l(l+1)hbar^2/(2m*r^2)] u(r) = E u(r)
    
    边界条件：
        u(0) = 0,   u(R_max) = 0
    
    采用 Numerov 方法进行稳定积分：
        u_{n+1} = [2(1 - 5h^2 k_n^2 / 12) u_n - (1 + h^2 k_{n-1}^2 / 12) u_{n-1}] / (1 + h^2 k_{n+1}^2 / 12)
    
    其中 k^2(r) = (2m*/hbar^2) [E - V_eff(r)]，V_eff = V + l(l+1)hbar^2/(2m*r^2)。
    """
    r_grid = validate_array_1d(r_grid, "r_grid")
    V = validate_array_1d(V, "V")
    if r_grid.size != V.size:
        raise ValueError("r_grid and V must have same size")
    n = r_grid.size
    if n < 4:
        raise ValueError("Need at least 4 grid points")
    h = float(r_grid[1] - r_grid[0])
    if abs(h) < 1e-20:
        raise ValueError("Grid spacing too small")

    H_BAR = 1.054571817e-34
    M_E = 9.10938356e-31
    m_star = m_star_ratio * M_E
    prefactor = 2.0 * m_star / (H_BAR ** 2)

    # 有效势
    V_eff = V.copy()
    if l > 0:
        centrifugal = np.zeros_like(r_grid)
        # 避免 r=0 除零
        r_safe = np.where(r_grid < 1e-18, 1e-18, r_grid)
        centrifugal = l * (l + 1) * (H_BAR ** 2) / (2.0 * m_star * r_safe ** 2)
        V_eff += centrifugal

    k2 = prefactor * (E_guess - V_eff)

    # Numerov 积分
    u = np.zeros(n, dtype=float)
    u[0] = 0.0
    u[1] = 1e-6  # 非零初始值

    h2 = h ** 2
    fac = h2 / 12.0
    for i in range(1, n - 1):
        denom = 1.0 + fac * k2[i + 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        u[i + 1] = (
            2.0 * (1.0 - 5.0 * fac * k2[i]) * u[i]
            - (1.0 + fac * k2[i - 1]) * u[i - 1]
        ) / denom

    # 归一化（兼容旧版 numpy 的 trapz）
    try:
        norm = np.sqrt(np.trapezoid(u ** 2, r_grid))
    except AttributeError:
        norm = np.sqrt(np.trapz(u ** 2, r_grid))
    if norm > 1e-20:
        u /= norm
    else:
        u[:] = 0.0

    # 计算边界斜率偏差作为残差
    residual = u[-1]
    return u, residual


def solve_radial_wavefunctions(
    r_grid: np.ndarray,
    V: np.ndarray,
    m_star_ratio: float,
    num_states: int = 3,
    l: int = 0,
    E_min: float = 0.0,
    E_max: float = 2.0,
) -> List[dict]:
    """
    在能量区间 [E_min, E_max]（单位：eV）内搜索前 num_states 个束缚态波函数。
    采用变步长扫描+符号变化检测定位本征能量。
    """
    EV_TO_J = 1.602176634e-19
    energies_eV = np.linspace(E_min, E_max, 200)
    residuals = []
    for E_eV in energies_eV:
        _, res = radial_wavefunction_shooting(r_grid, V, m_star_ratio, E_eV * EV_TO_J, l)
        residuals.append(res)
    residuals = np.array(residuals)

    # 检测残差符号变化（对应本征值）
    states = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i + 1] < 0:
            E_low = energies_eV[i]
            E_high = energies_eV[i + 1]
            # 二分法 refine
            for _ in range(20):
                E_mid = 0.5 * (E_low + E_high)
                _, res_mid = radial_wavefunction_shooting(r_grid, V, m_star_ratio, E_mid * EV_TO_J, l)
                if residuals[i] * res_mid < 0:
                    E_high = E_mid
                else:
                    E_low = E_mid
            E_eig = 0.5 * (E_low + E_high)
            u_eig, _ = radial_wavefunction_shooting(r_grid, V, m_star_ratio, E_eig * EV_TO_J, l)
            states.append({
                "energy_eV": float(E_eig),
                "energy_J": float(E_eig * EV_TO_J),
                "wavefunction": u_eig,
                "r_grid": r_grid,
            })
            if len(states) >= num_states:
                break
    return states
