"""
sheath_vlasov_splitting.py
==========================
Vlasov 方程算子分裂求解器。

本模块融合种子项目 1015_sheffieldquantum_qsim 的量子模拟
算子分裂技术 (Trotter-Suzuki splitting)，用于求解
等离子体鞘层的 Vlasov 方程。

物理背景：
    动理学鞘层模型由 Vlasov-Poisson 方程组描述：
        ∂f/∂t + v ∂f/∂x + (qE/m) ∂f/∂v = 0
        ∂²φ/∂x² = -(1/ε₀) ∫ q f dv

    使用 Strang 分裂 (时间二阶)：
        f(t+Δt) ≈ T_v(Δt/2) T_x(Δt) T_v(Δt/2) f(t)
    其中：
        T_x: 空间平流 exp(Δt v ∂/∂x)
        T_v: 速度空间加速 exp(Δt a ∂/∂v)

    量子模拟启发：
        Trotter 分解：exp(-iHt) ≈ [exp(-iH₁t/n) exp(-iH₂t/n)]^n
        与 Strang 分裂等价
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, Optional
import math


def strang_splitting_step(
    f: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    E_field: np.ndarray,
    dt: float,
    q_over_m: float = -1.0,
) -> np.ndarray:
    """
    Strang 分裂单步
    （源自种子项目 1015_sheffieldquantum_qsim 的 Trotter 分解思想）

    分裂格式：
        1. 半步速度空间平流: f* = T_v(dt/2) f^n
        2. 全步空间平流:     f** = T_x(dt) f*
        3. 半步速度空间平流: f^{n+1} = T_v(dt/2) f**

    空间/速度平流使用半 Lagrange 方法（三次插值）

    参数：
        f: shape (Nx, Nv) 分布函数
        v: shape (Nv,) 速度网格
        x: shape (Nx,) 空间网格
        E_field: shape (Nx,) 电场
        dt: 时间步长
        q_over_m: 荷质比

    返回：
        f_new: shape (Nx, Nv) 更新后的分布函数
    """
    Nx, Nv = f.shape
    dx = x[1] - x[0] if Nx > 1 else 1.0
    dv = v[1] - v[0] if Nv > 1 else 1.0

    # 步骤1: 半步 T_v(dt/2)
    f = _velocity_advection(f, v, E_field, q_over_m, 0.5 * dt)

    # 步骤2: 全步 T_x(dt)
    f = _spatial_advection(f, x, v, dt)

    # 步骤3: 半步 T_v(dt/2)
    f = _velocity_advection(f, v, E_field, q_over_m, 0.5 * dt)

    # 确保非负
    f = np.maximum(f, 0.0)

    return f


def _velocity_advection(
    f: np.ndarray,
    v: np.ndarray,
    E_field: np.ndarray,
    q_over_m: float,
    dt: float,
) -> np.ndarray:
    """
    速度空间平流 (半 Lagrange)

    ∂f/∂t + a(x) ∂f/∂v = 0
    其中 a(x) = (q/m) E(x)

    特征线法：
        f(x, v, t+dt) = f(x, v - a*dt, t)
    """
    Nx, Nv = f.shape
    dv = v[1] - v[0] if Nv > 1 else 1.0

    f_new = np.zeros_like(f)

    for i in range(Nx):
        a = q_over_m * E_field[i]
        # 回溯位置
        v_back = v - a * dt

        # 三次插值
        f_new[i, :] = _cubic_interpolate(f[i, :], v, v_back)

    return f_new


def _spatial_advection(
    f: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    空间平流 (半 Lagrange)

    ∂f/∂t + v ∂f/∂x = 0

    特征线法：
        f(x, v, t+dt) = f(x - v*dt, v, t)
    """
    Nx, Nv = f.shape

    f_new = np.zeros_like(f)

    for j in range(Nv):
        # 回溯位置
        x_back = x - v[j] * dt

        # 对每个速度进行空间插值
        for i in range(Nx):
            f_new[i, j] = _cubic_interpolate_1d(
                f[:, j], x, x_back[i]
            )

    return f_new


def _cubic_interpolate(
    y: np.ndarray,
    x_grid: np.ndarray,
    x_target: np.ndarray,
) -> np.ndarray:
    """三次 Hermite 插值"""
    n = len(x_grid)
    y_out = np.zeros_like(x_target, dtype=float)

    for k, xt in enumerate(x_target):
        # 找区间
        idx = np.searchsorted(x_grid, xt) - 1
        idx = max(0, min(idx, n - 2))

        # 局部坐标
        dx = x_grid[idx + 1] - x_grid[idx]
        if dx < 1e-30:
            y_out[k] = y[idx]
            continue

        t = (xt - x_grid[idx]) / dx
        t = max(0.0, min(1.0, t))

        # Catmull-Rom 插值
        i0 = max(0, idx - 1)
        i1 = idx
        i2 = min(n - 1, idx + 1)
        i3 = min(n - 1, idx + 2)

        y_out[k] = _cubic_hermite(y[i0], y[i1], y[i2], y[i3], t)

    return y_out


def _cubic_interpolate_1d(
    y: np.ndarray,
    x_grid: np.ndarray,
    x_target: float,
) -> float:
    """一维三次插值"""
    n = len(x_grid)
    idx = np.searchsorted(x_grid, x_target) - 1
    idx = max(0, min(idx, n - 2))

    dx = x_grid[idx + 1] - x_grid[idx]
    if dx < 1e-30:
        return y[idx]

    t = (x_target - x_grid[idx]) / dx
    t = max(0.0, min(1.0, t))

    i0 = max(0, idx - 1)
    i1 = idx
    i2 = min(n - 1, idx + 1)
    i3 = min(n - 1, idx + 2)

    return _cubic_hermite(y[i0], y[i1], y[i2], y[i3], t)


def _cubic_hermite(y0: float, y1: float, y2: float, y3: float, t: float) -> float:
    """Catmull-Rom 三次 Hermite 插值"""
    m1 = 0.5 * (y2 - y0)
    m2 = 0.5 * (y3 - y1)

    t2 = t * t
    t3 = t2 * t

    h1 = 2 * t3 - 3 * t2 + 1
    h2 = t3 - 2 * t2 + t
    h3 = -2 * t3 + 3 * t2
    h4 = t3 - t2

    return h1 * y1 + h2 * m1 + h3 * y2 + h4 * m2


def krylov_matrix_exp(
    A: np.ndarray,
    v: np.ndarray,
    t: float,
    m: int = 20,
) -> np.ndarray:
    """
    Krylov 子空间矩阵指数
    （源自种子项目 1015 的量子态传播技术）

    计算 exp(tA) v 使用 Lanczos/Arnoldi 方法：
        exp(tA) v ≈ ||v|| V_m exp(t H_m) e_1

    其中 H_m 为 m×m 上 Hessenberg 矩阵，
    V_m 为 Krylov 基

    参数：
        A: shape (n, n) 矩阵
        v: shape (n,) 向量
        t: 时间参数
        m: Krylov 子空间维度

    返回：
        w: shape (n,) ≈ exp(tA) v
    """
    n = len(v)
    m = min(m, n)

    # Arnoldi 过程
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    beta = np.linalg.norm(v)
    if beta < 1e-30:
        return np.zeros_like(v)

    V[:, 0] = v / beta

    for j in range(m):
        w = A @ V[:, j]

        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w -= H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] < 1e-14:
            m = j + 1
            break

        V[:, j + 1] = w / H[j + 1, j]

    # 计算 exp(t H_m)
    H_m = H[:m, :m]
    exp_H = la.expm(t * H_m)

    # 组合
    e1 = np.zeros(m)
    e1[0] = 1.0
    y_m = exp_H @ e1

    w = beta * (V[:, :m] @ y_m)
    return w


def vlasov_poisson_solve(
    Nx: int,
    Nv: int,
    L_x: float,
    v_max: float,
    n_steps: int,
    dt: float,
    f_init: Optional[np.ndarray] = None,
) -> dict:
    """
    Vlasov-Poisson 方程时间推进

    完整算法：
        1. 初始化分布函数 f(x, v, 0)
        2. 对于每个时间步：
            a. 计算电荷密度 ρ(x) = ∫ f dv - n_0
            b. 求解 Poisson 方程 ∂²φ/∂x² = -ρ/ε₀
            c. 计算电场 E = -∂φ/∂x
            d. Strang 分裂更新 f

    参数：
        Nx: 空间网格点数
        Nv: 速度网格点数
        L_x: 计算域长度
        v_max: 速度范围 [-v_max, v_max]
        n_steps: 时间步数
        dt: 时间步长
        f_init: 初始分布函数 (可选)

    返回：
        结果字典
    """
    x = np.linspace(0, L_x, Nx)
    v = np.linspace(-v_max, v_max, Nv)
    dx = x[1] - x[0]
    dv = v[1] - v[0]

    # 初始化
    if f_init is not None:
        f = f_init.copy()
    else:
        # Maxwell 分布 + 小扰动
        v_th = 1.0
        f = np.zeros((Nx, Nv))
        for i in range(Nx):
            f[i, :] = (1.0 / (math.sqrt(2 * math.pi) * v_th)
                       * np.exp(-0.5 * (v / v_th)**2))
        # Landau 阻尼测试扰动
        alpha = 0.01
        k_mode = 2 * math.pi / L_x
        for i in range(Nx):
            f[i, :] *= (1.0 + alpha * math.cos(k_mode * x[i]))

    # 存储
    energy_history = []
    density_history = []

    for step in range(n_steps):
        # 计算密度
        density = np.sum(f, axis=1) * dv

        # 求解 Poisson (简单 FFT 方法)
        rho = density - 1.0  # 准中性背景
        rho_hat = np.fft.rfft(rho)
        k_values = np.fft.rfftfreq(Nx, d=dx) * 2 * math.pi

        phi_hat = np.zeros_like(rho_hat)
        for k_idx in range(1, len(k_values)):
            k = k_values[k_idx]
            if abs(k) > 1e-10:
                phi_hat[k_idx] = rho_hat[k_idx] / (k**2)
        phi = np.fft.irfft(phi_hat, n=Nx)

        # 电场
        E = -np.gradient(phi) / dx

        # Strang 分裂更新
        f = strang_splitting_step(f, v, x, E, dt, q_over_m=-1.0)

        # 场能量
        E_energy = 0.5 * np.sum(E**2) * dx
        energy_history.append(E_energy)
        density_history.append(density.copy())

    return {
        'x': x,
        'v': v,
        'f_final': f,
        'energy_history': energy_history,
        'density_final': density_history[-1] if density_history else np.zeros(Nx),
    }
