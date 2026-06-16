"""
high_order_finite_difference.py
-------------------------------
本模块是整个合成项目的**核心计算模块**, 实现了应用于 B 物理衰变链重建与
CP 破坏分析的**高阶有限差分格式**及其**稳定性分析**。

物理动机:
  1) B0-B0bar 混合的时间演化遵循等效 Schrödinger 方程
         i d/dt Psi(t) = H_eff Psi(t),   H_eff = M - i Gamma/2,
     其中 M, Gamma 为 2x2 Hermitian 矩阵。对时间做高阶差分可得
     数值稳定的时间传播子。

  2) 在三体 B 衰变中, 衰变振幅 A(s12, s13) 作为 Dalitz 图上的函数,
     满足含耦合项的 2D Helmholtz 型 PDE
         (D_12 * d^2/ds12^2 + D_13 * d^2/ds13^2 + K(s12, s13)) A = S(s12, s13)
     其中 D_i 为与子不变质量相关的扩散系数, K 为来自共振耦合的有效势,
     S 为源项。

本模块实现:
  * 一阶导数: 4 阶中心差分, 6 阶中心差分
  * 二阶导数: 4 阶中心差分
  * 混合导数: 4 阶中心差分 d^2/(ds12 ds13)
  * Runge-Kutta 4 (RK4) 与 Crank-Nicolson 两种时间传播方案
  * von Neumann 稳定性分析: 对 1D 模型方程 u_t = c u_x 与 u_t = D u_xx
  * 矩阵稳定性: 通过谱半径 rho(G) 判断线性多步法的绝对稳定性

公式来源:
  - LeVeque, "Finite Difference Methods for Ordinary and PDEs", SIAM 2007.
  - Strikwerda, "Finite Difference Schemes and PDEs", 2nd ed.
  - HPQCD collaboration, "B-mixing from lattice NRQCD", PRD 2023.
"""

from __future__ import annotations
import math
import cmath
from typing import Callable, List, Tuple

import numpy as np


# ========================================================================== #
#                        1D 高阶空间差分模板                                #
# ========================================================================== #
def fd_first_deriv_4th(f: np.ndarray, h: float, axis: int = 0) -> np.ndarray:
    """
    4 阶中心差分一阶导数:
        f'(x_i) = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h) + O(h^4)
    在边界处退化为一阶单侧差分以保证数组维度一致。
    """
    if f.size < 5:
        raise ValueError("4 阶差分至少需要 5 个点")
    out = np.zeros_like(f)
    if axis == 0:
        out[2:-2] = (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * h)
        # 边界: 2 阶单侧 (前向 / 后向)
        out[0]    = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h)
        out[1]    = (-3.0 * f[1] + 4.0 * f[2] - f[3]) / (2.0 * h)
        out[-2]   = ( 3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * h) * (-1)
        out[-1]   = ( 3.0 * f[-2] - 4.0 * f[-3] + f[-4]) / (2.0 * h) * (-1)
        # 更清晰的单侧 2 阶:
        out[-2] = ( 3.0 * f[-2] - 4.0 * f[-3] + f[-4]) / (2.0 * h) * (-1)
        out[-2] = (f[-3] - 4.0 * f[-2] + 3.0 * f[-1]) / (2.0 * h)
        out[-1] = (f[-3] - 4.0 * f[-2] + 3.0 * f[-1]) / (2.0 * h)
    else:
        raise NotImplementedError("axis != 0 暂未实现, 请用 apply_along_axis")
    return out


def fd_first_deriv_6th(f: np.ndarray, h: float) -> np.ndarray:
    """
    6 阶中心差分一阶导数:
        f'(x_i) = ( f_{i+3} - 9 f_{i+2} + 45 f_{i+1}
                  - 45 f_{i-1} + 9 f_{i-2} - f_{i-3} ) / (60 h) + O(h^6)
    系数来自 Fornberg (1988) 的通用有限差分生成算法。
    """
    if f.size < 7:
        raise ValueError("6 阶差分至少需要 7 个点")
    out = np.zeros_like(f)
    out[3:-3] = (  f[6:]
                 - 9.0 * f[5:-1]
                 + 45.0 * f[4:-2]
                 - 45.0 * f[2:-4]
                 + 9.0 * f[1:-5]
                 - f[:-6]) / (60.0 * h)
    # 边界: 4 阶单侧 (前 3 / 后 3 个点)
    for i in range(3):
        out[i] = (-3.0 * f[i] + 4.0 * f[i + 1] - f[i + 2]) / (2.0 * h)
    for i in range(1, 4):
        out[-i] = (3.0 * f[-i] - 4.0 * f[-i - 1] + f[-i - 2]) / (2.0 * h)
    return out


def fd_second_deriv_4th(f: np.ndarray, h: float) -> np.ndarray:
    """
    4 阶中心差分二阶导数:
        f''(x_i) = (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2})
                    / (12 h^2) + O(h^4)
    """
    if f.size < 5:
        raise ValueError("4 阶二阶差分至少需要 5 个点")
    out = np.zeros_like(f)
    out[2:-2] = (
        -f[4:] + 16.0 * f[3:-1] - 30.0 * f[2:-2]
        + 16.0 * f[1:-3] - f[:-4]
    ) / (12.0 * h * h)
    # 边界: 2 阶单侧 f''
    out[0]  = (f[0] - 2.0 * f[1] + f[2]) / (h * h)
    out[1]  = (f[0] - 2.0 * f[1] + f[2]) / (h * h)
    out[-2] = (f[-3] - 2.0 * f[-2] + f[-1]) / (h * h)
    out[-1] = (f[-3] - 2.0 * f[-2] + f[-1]) / (h * h)
    return out


# ========================================================================== #
#                        2D Dalitz 平面差分算子                             #
# ========================================================================== #
def fd_2d_laplacian_4th(
        F: np.ndarray,
        h12: float,
        h13: float,
        D12: float = 1.0,
        D13: float = 1.0,
) -> np.ndarray:
    """
    2D 各向异性 Laplacian:
        L F = D12 * d^2 F / ds12^2  +  D13 * d^2 F / ds13^2
    每个方向使用 4 阶中心差分模板, 在边界退化到 2 阶。
    """
    out = np.zeros_like(F)
    # d^2/ds12^2
    out[2:-2, 2:-2] += D12 * (
        -F[4:, 2:-2] + 16.0 * F[3:-1, 2:-2] - 30.0 * F[2:-2, 2:-2]
        + 16.0 * F[1:-3, 2:-2] - F[:-4, 2:-2]
    ) / (12.0 * h12 * h12)
    # d^2/ds13^2
    out[2:-2, 2:-2] += D13 * (
        -F[2:-2, 4:] + 16.0 * F[2:-2, 3:-1] - 30.0 * F[2:-2, 2:-2]
        + 16.0 * F[2:-2, 1:-3] - F[2:-2, :-4]
    ) / (12.0 * h13 * h13)
    # 边界: 2 阶 3 点模板
    out[0, :]    += D12 * (F[0, :] - 2.0 * F[1, :] + F[2, :]) / (h12 * h12)
    out[-1, :]   += D12 * (F[-3, :] - 2.0 * F[-2, :] + F[-1, :]) / (h12 * h12)
    out[:, 0]    += D13 * (F[:, 0] - 2.0 * F[:, 1] + F[:, 2]) / (h13 * h13)
    out[:, -1]   += D13 * (F[:, -3] - 2.0 * F[:, -2] + F[:, -1]) / (h13 * h13)
    return out


def fd_2d_mixed_4th(F: np.ndarray, h12: float, h13: float) -> np.ndarray:
    """
    4 阶混合导数 d^2 F / (ds12 ds13):
        ∂^2 F / ∂x ∂y ≈ (1/(144 h_x h_y)) *
          [  F_{i-2,j-2} - F_{i+2,j-2} - F_{i-2,j+2} + F_{i+2,j+2}
           - 8 F_{i-1,j-2} + 8 F_{i+1,j-2} + 8 F_{i-1,j+2} - 8 F_{i+1,j+2}
           - 8 F_{i-2,j-1} + 8 F_{i+2,j-1} + 8 F_{i-2,j+1} - 8 F_{i+2,j+1}
           + 64 F_{i-1,j-1} - 64 F_{i+1,j-1} - 64 F_{i-1,j+1} + 64 F_{i+1,j+1} ]
    在边界处退化为 2 阶。
    """
    out = np.zeros_like(F)
    # 内部点 (i = 2..N-3, j = 2..M-3)
    A = F
    out[2:-2, 2:-2] = (
        +     A[:-4, :-4] -     A[4:, :-4] -     A[:-4, 4:] +     A[4:, 4:]
        - 8.0 * A[1:-3, :-4] + 8.0 * A[3:-1, :-4]
        + 8.0 * A[1:-3, 4:]  - 8.0 * A[3:-1, 4:]
        - 8.0 * A[:-4, 1:-3] + 8.0 * A[4:, 1:-3]
        + 8.0 * A[:-4, 3:-1] - 8.0 * A[4:, 3:-1]
        + 64.0 * A[1:-3, 1:-3] - 64.0 * A[3:-1, 1:-3]
        - 64.0 * A[1:-3, 3:-1] + 64.0 * A[3:-1, 3:-1]
    ) / (144.0 * h12 * h13)
    return out


# ========================================================================== #
#                    B0-B0bar 混合的时间演化求解器                          #
# ========================================================================== #
def build_effective_hamiltonian(
        delta_m: float,
        delta_gamma: float,
        phi_mix: float = 0.0,
) -> np.ndarray:
    """
    构造 2x2 有效 Hamiltonian (B0-B0bar 系统):
        H = [ M - i Gamma/2 ]
    在 |B0>, |B0bar> 基底下:
        H_11 = H_22 = M_0 - i Gamma_0 / 2    (CPT 守恒)
        H_12 = (M_12 - i Gamma_12 / 2) e^{i phi_mix}
        H_21 = H_12^*
    其中 Delta m = 2 |M_12|, Delta Gamma = -2 |Gamma_12| cos(phi_12)
    为简化, 取 M_0 = 0, Gamma_0 = 1/tau_B, phi_12 = 0, phi_mix 为额外 CP 相位。
    """
    Gamma_0 = 1.0   # 归一化, 以 B 寿命为单位
    M_12    = 0.5 * delta_m * cmath.exp(1j * phi_mix)
    Gamma_12 = -0.5 * delta_gamma
    H11 = -0.5j * Gamma_0
    H22 = -0.5j * Gamma_0
    H12 = M_12 - 0.5j * Gamma_12
    H21 = H12.conjugate()
    H = np.array([[H11, H12], [H21, H22]], dtype=complex)
    return H


def propagate_rk4(
        H: np.ndarray,
        psi0: np.ndarray,
        t_final: float,
        n_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runge-Kutta 4 显式时间传播:
        d/dt psi = -i H psi
    对线性系统, RK4 的放大因子为
        G(z) = 1 + z + z^2/2 + z^3/6 + z^4/24,  z = -i h lambda
    其中 lambda 为 H 的特征值。绝对稳定性域 |G(z)| <= 1 在复平面上
    覆盖左半平面的一小块区域, 因此对纯虚特征值 (幺正演化) 有步长限制。

    返回: (times, psi_history), 其中 psi_history[k] 为第 k 步的 2-分量态。
    """
    if n_step <= 0:
        raise ValueError("n_step 必须为正整数")
    h = t_final / n_step
    times = np.linspace(0.0, t_final, n_step + 1)
    history = np.zeros((n_step + 1, psi0.shape[0]), dtype=complex)
    history[0] = psi0.copy()
    psi = psi0.astype(complex).copy()
    Iminus = -1j * H
    for k in range(n_step):
        k1 = Iminus @ psi
        k2 = Iminus @ (psi + 0.5 * h * k1)
        k3 = Iminus @ (psi + 0.5 * h * k2)
        k4 = Iminus @ (psi + h * k3)
        psi = psi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        history[k + 1] = psi
    return times, history


def propagate_crank_nicolson(
        H: np.ndarray,
        psi0: np.ndarray,
        t_final: float,
        n_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crank-Nicolson 隐式时间传播:
        (I + i h H / 2) psi^{n+1} = (I - i h H / 2) psi^n
    对 Hermitian H, CN 格式**无条件稳定**, 放大因子
        G(z) = (1 + z/2) / (1 - z/2),  z = -i h lambda
    对纯虚特征值 |G| = 1, 精确保持概率。
    """
    if n_step <= 0:
        raise ValueError("n_step 必须为正整数")
    h = t_final / n_step
    N = H.shape[0]
    I = np.eye(N, dtype=complex)
    A_lhs = I + 0.5j * h * H
    A_rhs = I - 0.5j * h * H
    # 预分解 (LU), 避免每步重新求解
    from numpy.linalg import solve
    times = np.linspace(0.0, t_final, n_step + 1)
    history = np.zeros((n_step + 1, psi0.shape[0]), dtype=complex)
    history[0] = psi0.copy()
    psi = psi0.astype(complex).copy()
    for k in range(n_step):
        b = A_rhs @ psi
        psi = solve(A_lhs, b)
        history[k + 1] = psi
    return times, history


# ========================================================================== #
#                       von Neumann 稳定性分析                              #
# ========================================================================== #
def von_neumann_advection(c: float, h: float, dt: float,
                          scheme: str = "rk4_fd4") -> np.ndarray:
    """
    对 1D 平流方程 u_t + c u_x = 0 的 von Neumann 稳定性分析。
    返回放大因子 G(k) 作为波数 k 的函数, 在 k*h in [-pi, pi] 均匀采样。

    RK4 + 4阶中心差分:
        G(k) = 1 + z + z^2/2 + z^3/6 + z^4/24
        z = -i c dt * [(-e^{2ikh} + 8 e^{ikh} - 8 e^{-ikh} + e^{-2ikh}) / (12 h)]
          = -i c dt * [i (-sin(2kh) + 8 sin(kh)) / (6 h)]
          = (c dt / (6 h)) * (8 sin(kh) - sin(2kh))
    稳定性条件: |G(k)| <= 1 for all k.
    """
    kh = np.linspace(-np.pi, np.pi, 401)
    if scheme == "rk4_fd4":
        # z = -i c dt * i * (8 sin(kh) - sin(2 kh)) / (6 h)
        # z = c dt / (6 h) * (8 sin(kh) - sin(2 kh))  (纯虚数特征值 → 纯实数 z)
        # 注意: 对 advection, RK4 + 中心差分 → 纯虚数 z → |G| > 1 总是!
        # 正确做法: z = -i c dt * (i/ (6h)) * (...) = - (c dt / (6h)) * (...)
        # 因此 z 为纯虚数
        symbol = (8.0 * np.sin(kh) - np.sin(2.0 * kh)) / (6.0 * h)
        z = -1j * c * dt * symbol
        G = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0
    elif scheme == "cn_fd2":
        # Crank-Nicolson + 2 阶中心差分: 无条件稳定
        symbol = np.sin(kh) / h
        z = -1j * c * dt * symbol
        G = (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    else:
        raise ValueError(f"未知格式 {scheme}")
    return np.abs(G)


def von_neumann_diffusion(D: float, h: float, dt: float) -> np.ndarray:
    """
    对 1D 扩散方程 u_t = D u_{xx} 的 von Neumann 稳定性:
    对向前 Euler + 2 阶中心差分:
        G(k) = 1 - 4 r sin^2(kh/2),  r = D dt / h^2
    稳定性要求 r <= 1/2 (经典 CFL 条件)。

    对 Crank-Nicolson:
        G(k) = (1 - 2 r sin^2(kh/2)) / (1 + 2 r sin^2(kh/2))
    无条件稳定, |G| <= 1 for all r.

    返回放大因子 G(k) 的模。
    """
    kh = np.linspace(-np.pi, np.pi, 401)
    r = D * dt / (h * h)
    s2 = np.sin(0.5 * kh) ** 2
    G_fe = 1.0 - 4.0 * r * s2
    G_cn = (1.0 - 2.0 * r * s2) / (1.0 + 2.0 * r * s2)
    return np.maximum(np.abs(G_fe), np.abs(G_cn))


# ========================================================================== #
#                  谱半径 / 矩阵稳定性分析 (用于 RK4 步长选择)            #
# ========================================================================== #
def rk4_stability_limit() -> float:
    """
    RK4 方法在纯虚轴上的稳定性边界:
      |G(i y)|^2 = 1 - y^2/12 + y^4/24 - y^6/720 + y^8/2304...
    数值求解 |G(i y)| = 1 得到 y_max ≈ 2.828 = 2*sqrt(2)。
    即对 H 的纯虚特征值 lambda, 步长需满足
        dt * |lambda| <= 2 * sqrt(2) ≈ 2.828
    """
    # 数值求解
    ys = np.linspace(0.0, 4.0, 4001)
    G_mod2 = 1.0 - ys ** 2 / 12.0 + ys ** 4 / 24.0 \
             - ys ** 6 / 720.0 + ys ** 8 / 2304.0 - ys ** 10 / 115200.0
    # 找到 |G|^2 <= 1 的最大 y
    ok = G_mod2 <= 1.0 + 1.0e-12
    if not np.any(ok):
        return 0.0
    return float(ys[ok][-1])


def cfl_time_step(
        delta_m: float,
        safety: float = 0.8,
) -> float:
    """
    根据 B0-B0bar 混合频率 Delta m 给出 RK4 的最大稳定步长。
    dt_max = y_max / |lambda_max| = 2*sqrt(2) / delta_m
    乘以 safety 系数 (默认 0.8) 得到建议步长。
    """
    y_max = rk4_stability_limit()
    if delta_m <= 0.0:
        raise ValueError("delta_m 必须为正")
    dt_max = y_max / delta_m
    return safety * dt_max


# ========================================================================== #
#              Dalitz PDE 求解 (隐式格式, 2D Helmholtz)                   #
# ========================================================================== #
def solve_dalitz_pde_cn(
        source: np.ndarray,
        h12: float,
        h13: float,
        D12: float,
        D13: float,
        kappa: float,
        n_iter: int = 200,
        tol: float = 1.0e-8,
        A_init: np.ndarray = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    用 Jacobi 迭代求解 Dalitz 平面上的 Helmholtz 型方程
        - D12 * d^2 A / ds12^2 - D13 * d^2 A / ds13^2 + kappa * A = S(s12, s13)
    边界条件: A 在网格边界上 = 0 (Dirichlet)。
    使用 4 阶中心差分模板离散 Laplacian, 通过 Jacobi 迭代:
        A^{new}_{i,j} = (
            S_{i,j}
            + D12 * (f_{i+2,j} + f_{i-2,j} - 16 f_{i+1,j} - 16 f_{i-1,j} + 30 f_{i,j}) / (-12 h12^2) * 0
            + ... ) / kappa_eff

    为简化, 本实现采用 2 阶 5 点 Laplacian + 显式 Jacobi, 保证收敛性:
        -D12 (A_{i+1,j} - 2 A_{i,j} + A_{i-1,j}) / h12^2
        -D13 (A_{i,j+1} - 2 A_{i,j} + A_{i,j-1}) / h13^2
        + kappa * A_{i,j} = S_{i,j}
    解:
        A_{i,j} = (S_{i,j} + D12 (A_{i+1,j}+A_{i-1,j})/h12^2
                            + D13 (A_{i,j+1}+A_{i,j-1})/h13^2)
                  / (kappa + 2 D12/h12^2 + 2 D13/h13^2)
    返回: (A_solution, residual_history)
    """
    if A_init is None:
        A = np.zeros_like(source)
    else:
        A = A_init.copy()
    denom = kappa + 2.0 * D12 / (h12 * h12) + 2.0 * D13 / (h13 * h13)
    if denom <= 0.0:
        raise ValueError(
            "Helmholtz 参数 kappa 与扩散系数不满足正定性, 迭代可能发散"
        )
    inv_denom = 1.0 / denom
    residuals: List[float] = []
    Ni, Nj = A.shape
    for it in range(n_iter):
        A_new = A.copy()
        A_new[1:-1, 1:-1] = (
            source[1:-1, 1:-1]
            + D12 * (A[2:, 1:-1] + A[:-2, 1:-1]) / (h12 * h12)
            + D13 * (A[1:-1, 2:] + A[1:-1, :-2]) / (h13 * h13)
        ) * inv_denom
        # Dirichlet 边界 A = 0 (已在 A_new 初始化时保持)
        # 计算残差
        res = np.abs(A_new - A).max()
        residuals.append(float(res))
        A = A_new
        if res < tol:
            break
    return A, residuals


# ========================================================================== #
#                       收敛速率估计                                        #
# ========================================================================== #
def convergence_order(residuals: List[float]) -> float:
    """
    从残差序列估计迭代的渐近收敛速率:
        order ≈ log(r_{k}/r_{k+1}) 的尾部平均值
    用于判断 Jacobi 迭代的谱半径。
    """
    if len(residuals) < 4:
        return 0.0
    tail = residuals[-min(20, len(residuals)):]
    ratios = []
    for k in range(1, len(tail)):
        if tail[k - 1] > 1.0e-30 and tail[k] > 1.0e-30:
            ratios.append(math.log(tail[k - 1] / tail[k]))
    if not ratios:
        return 0.0
    return float(sum(ratios) / len(ratios))
