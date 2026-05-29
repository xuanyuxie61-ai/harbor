"""
les_core.py
================================================================================
大涡模拟核心方程模块 —— 基于种子项目 790_navier_stokes_mesh3d

本模块实现过滤后的不可压 Navier-Stokes 方程（LES 控制方程），
包括速度-压力耦合、对流-扩散算子与投影步求解。

核心物理公式
--------------------------------------------------------------------------------
过滤不可压 Navier-Stokes 方程：

    ∂ū_i/∂t + ū_j ∂ū_i/∂x_j = -(1/ρ) ∂p̄/∂x_i - ∂τ_{ij}^{sgs}/∂x_j + ν ∇²ū_i + f_i

    ∂ū_i/∂x_i = 0

其中 ū_i 为过滤速度，τ_{ij}^{sgs} = ū_iū_j - ū_i ū_j 为亚格子应力张量。

在谱-有限元混合框架中：
- 水平方向（x, y）：谱展开（球谐或 Fourier）
- 垂直方向（z）：有限元离散

能量方程（位温）：
    ∂θ̄/∂t + ū_j ∂θ̄/∂x_j = - ∂q_j^{sgs}/∂x_j + κ_θ ∇²θ̄ + Q

其中 q_j^{sgs} = ū_jθ - ū_j θ̄ 为亚格子热通量。
"""

import numpy as np


def divergence(u, v, w, dx, dy, dz):
    """
    计算速度散度 ∂u/∂x + ∂v/∂y + ∂w/∂z（中心差分）。

    参数
    ----------
    u, v, w : np.ndarray, shape (nx, ny, nz)
    dx, dy, dz : float

    返回
    -------
    div : np.ndarray
    """
    nx, ny, nz = u.shape
    div = np.zeros_like(u)

    # 内部点
    div[1:-1, 1:-1, 1:-1] = (
        (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx) +
        (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
        (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)
    )

    return div


def laplacian_3d(phi, dx, dy, dz):
    """
    计算三维标量场的 Laplacian（7点 stencil）。

    参数
    ----------
    phi : np.ndarray, shape (nx, ny, nz)
    dx, dy, dz : float

    返回
    -------
    lap : np.ndarray
    """
    lap = np.zeros_like(phi)

    lap[1:-1, 1:-1, 1:-1] = (
        (phi[2:, 1:-1, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / dx**2 +
        (phi[1:-1, 2:, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, :-2, 1:-1]) / dy**2 +
        (phi[1:-1, 1:-1, 2:] - 2 * phi[1:-1, 1:-1, 1:-1] + phi[1:-1, 1:-1, :-2]) / dz**2
    )

    return lap


def convection_term(u, v, w, dx, dy, dz):
    """
    计算对流项 u_j ∂u_i/∂x_j（守恒形式）。

    参数
    ----------
    u, v, w : np.ndarray, shape (nx, ny, nz)

    返回
    -------
    conv_u, conv_v, conv_w : np.ndarray
    """
    def grad_central(f, axis, h):
        df = np.zeros_like(f)
        slc_p = [slice(None)] * 3
        slc_m = [slice(None)] * 3
        slc_p[axis] = slice(2, None)
        slc_m[axis] = slice(None, -2)
        df[tuple(slc_p)] = (f[tuple(slc_p)] - f[tuple(slc_m)]) / (2 * h)
        return df

    dudx = grad_central(u, 0, dx)
    dudy = grad_central(u, 1, dy)
    dudz = grad_central(u, 2, dz)

    dvdx = grad_central(v, 0, dx)
    dvdy = grad_central(v, 1, dy)
    dvdz = grad_central(v, 2, dz)

    dwdx = grad_central(w, 0, dx)
    dwdy = grad_central(w, 1, dy)
    dwdz = grad_central(w, 2, dz)

    # 守恒形式：u ∂u/∂x + v ∂u/∂y + w ∂u/∂z
    conv_u = u * dudx + v * dudy + w * dudz
    conv_v = u * dvdx + v * dvdy + w * dvdz
    conv_w = u * dwdx + v * dwdy + w * dwdz

    return conv_u, conv_v, conv_w


def solve_poisson_fft(rhs, dx, dy, dz):
    """
    使用 3D FFT 求解周期边界泊松方程 ∇²p = rhs。

    参数
    ----------
    rhs : np.ndarray, shape (nx, ny, nz)
    dx, dy, dz : float

    返回
    -------
    p : np.ndarray
    """
    nx, ny, nz = rhs.shape
    rhs_hat = np.fft.fftn(rhs)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # 避免除零，零模对应压力常数

    p_hat = rhs_hat / k2
    p = np.fft.ifftn(p_hat).real
    return p


def projection_step(u_star, v_star, w_star, dx, dy, dz, dt, rho=1.0,
                    max_iter=100, tol=1e-8):
    """
    投影法速度修正步：求解压力泊松方程并修正速度使其无散。

    参数
    ----------
    u_star, v_star, w_star : np.ndarray
        中间速度场
    dx, dy, dz, dt : float
    rho : float
        密度
    max_iter, tol : int, float
        迭代求解参数（保留用于兼容，实际使用 FFT）

    返回
    -------
    u, v, w, p : np.ndarray
        修正后的速度与压力
    converged : bool
    """
    # === HOLE 3 BEGIN ===
    # 此处应实现投影法速度修正步的核心计算：
    # 1. 计算速度散度 div_u = divergence(u_star, v_star, w_star, dx, dy, dz)
    # 2. 构建泊松方程右端项 rhs = (rho / dt) * div_u
    # 3. 调用 solve_poisson_fft 求解压力场 p
    # 4. 计算压力梯度 grad_p（使用前向差分）
    # 5. 修正速度: u = u* - (dt/ρ) * grad_p
    # 返回: u, v, w, p, converged (bool)
    # 注意：此处的实现必须与 main.py (HOLE 2) 的调用方式协同修复
    raise NotImplementedError("HOLE 3: 请实现投影法速度修正步")
    # === HOLE 3 END ===


def initialize_turbulent_field(nx, ny, nz, dx, dy, dz, u_mean=5.0, v_mean=0.0,
                                turbulence_intensity=0.1, theta_mean=300.0,
                                theta_gradient=0.003):
    """
    初始化具有边界层特征的速度场与温度场。

    参数
    ----------
    nx, ny, nz : int
        网格点数
    dx, dy, dz : float
        网格间距
    u_mean : float
        平均风速（m/s）
    v_mean : float
        横向风速（m/s）
    turbulence_intensity : float
        湍流强度
    theta_mean : float
        地表位温（K）
    theta_gradient : float
        位温垂直梯度（K/m）

    返回
    -------
    u, v, w, theta : np.ndarray
    """
    np.random.seed(42)

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 对数风剖面（中性边界层）
    z0 = 0.1  # 粗糙度长度
    kappa = 0.4
    u_star = u_mean * kappa / np.log((nz * dz) / z0)

    # 避免 z=0 处的对数奇点
    z_safe = np.maximum(Z, z0 * 1.1)
    u_profile = (u_star / kappa) * np.log(z_safe / z0)

    # 归一化到平均风速
    u_profile = u_profile * (u_mean / np.mean(u_profile))

    # 添加随机湍流脉动
    u = u_profile + turbulence_intensity * u_mean * np.random.randn(nx, ny, nz)
    v = v_mean + turbulence_intensity * u_mean * 0.5 * np.random.randn(nx, ny, nz)
    w = turbulence_intensity * u_mean * 0.3 * np.random.randn(nx, ny, nz)

    # 位温：近地表恒定，上部稳定层结
    theta = theta_mean + theta_gradient * Z

    return u, v, w, theta
