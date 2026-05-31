"""
spatial_ecology.py
渔业空间生态学模块：海洋流场、立体投影与扩散-对流模型

整合算法：
1. 无散度二维速度场生成（基于 continuity_exact）
2. 球面立体投影（基于 sphere_stereograph_display）

核心科学问题：
模拟海洋环境流场对鱼卵、仔稚鱼被动输运的影响，
以及渔业资源在球面地球表面的空间分布。

数学公式：
1. 流函数：
   Φ(Z) = (1 - cos(C π Z)) (1 - Z)^2
   速度场：
   U(X,Y) =  10 ∂/∂Y [Φ(X)Φ(Y)]
   V(X,Y) = -10 ∂/∂X [Φ(X)Φ(Y)]
   该构造保证 ∇·V = ∂U/∂X + ∂V/∂Y = 0

2. 立体投影（从南极 S=(0,0,-1) 到平面 Z=1）：
   正向：q_1 = 2p_1/(1+p_3),  q_2 = 2p_2/(1+p_3),  q_3 = 1
   逆向：p = (4e_1, 4e_2, 4-||e||^2) / (4+||e||^2)
   其中 e = (e_1, e_2) 为平面坐标

3. 对流-扩散方程（被动标量输运）：
   ∂C/∂t + u·∇C = D ∇^2 C - λ C
   C: 鱼卵/幼鱼浓度, D: 扩散系数, λ: 沉降/死亡率
"""

import numpy as np
from utils import NumericalConfig, safe_divide


# ============================================================================
# Divergence-Free Velocity Field
# ============================================================================

def streamfunction_phi(Z, C_param):
    """
    流函数的基本构建块
    Φ(Z) = (1 - cos(C π Z)) (1 - Z)^2
    """
    return (1.0 - np.cos(C_param * np.pi * Z)) * ((1.0 - Z) ** 2)


def streamfunction_dphi(Z, C_param):
    """
    Φ(Z) 的导数
    dΦ/dZ = Cπ sin(CπZ) (1-Z)^2 - 2(1-cos(CπZ))(1-Z)
    """
    term1 = C_param * np.pi * np.sin(C_param * np.pi * Z) * ((1.0 - Z) ** 2)
    term2 = -2.0 * (1.0 - np.cos(C_param * np.pi * Z)) * (1.0 - Z)
    return term1 + term2


def divergence_free_velocity(n, X, Y, C_param):
    """
    生成无散度二维速度场

    U(X,Y) =  10 * Φ(X) * dΦ/dY
    V(X,Y) = -10 * Φ(Y) * dΦ/dX

    数学上：
    U =  10 ∂/∂Y [Φ(X)Φ(Y)] =  10 Φ(X) Φ'(Y)
    V = -10 ∂/∂X [Φ(X)Φ(Y)] = -10 Φ(Y) Φ'(X)
    ∇·V = ∂U/∂X + ∂V/∂Y = 10 Φ'(X)Φ'(Y) - 10 Φ'(Y)Φ'(X) = 0

    Parameters
    ----------
    n : int
        评估点数
    X, Y : ndarray
        评估点坐标，长度 n，范围 [0,1]
    C_param : float
        流函数参数，建议 0 < C < 2π

    Returns
    -------
    U, V : ndarray
        速度分量
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = np.clip(X, 0.0, 1.0)
    Y = np.clip(Y, 0.0, 1.0)

    phi_X = streamfunction_phi(X, C_param)
    phi_Y = streamfunction_phi(Y, C_param)
    dphi_X = streamfunction_dphi(X, C_param)
    dphi_Y = streamfunction_dphi(Y, C_param)

    U = 10.0 * phi_X * dphi_Y
    V = -10.0 * phi_Y * dphi_X
    return U, V


def compute_divergence(X, Y, U, V, dx, dy):
    """
    数值计算速度场的散度，用于验证无散度条件
    div V ≈ (U[i+1,j] - U[i-1,j])/(2dx) + (V[i,j+1] - V[i,j-1])/(2dy)
    """
    dUdx = np.gradient(U, dx, axis=0)
    dVdy = np.gradient(V, dy, axis=1)
    return dUdx + dVdy


# ============================================================================
# Stereographic Projection
# ============================================================================

def sphere_stereograph(points_sphere):
    """
    球面立体投影：从单位球面投影到 Z=1 平面

    投影中心为南极 S=(0,0,-1)
    对于球面上点 P=(p1,p2,p3)，投影点 Q 满足：
    Q = S + t(P-S)，且 q3 = 1
    解得 t = 2/(1+p3)，因此：
    q1 = 2p1/(1+p3), q2 = 2p2/(1+p3), q3 = 1

    Parameters
    ----------
    points_sphere : ndarray, shape (N, 3)
        单位球面上的点

    Returns
    -------
    points_plane : ndarray, shape (N, 3)
        投影平面上的点（z=1）
    """
    points_sphere = np.asarray(points_sphere, dtype=float)
    if points_sphere.ndim == 1:
        points_sphere = points_sphere.reshape(1, -1)

    p1 = points_sphere[:, 0]
    p2 = points_sphere[:, 1]
    p3 = points_sphere[:, 2]

    denom = 1.0 + p3
    denom = np.where(np.abs(denom) < NumericalConfig.EPS, NumericalConfig.EPS, denom)

    q1 = 2.0 * p1 / denom
    q2 = 2.0 * p2 / denom
    q3 = np.ones_like(q1)

    return np.column_stack([q1, q2, q3])


def sphere_stereograph_inverse(points_plane):
    """
    立体投影逆变换：从平面 Z=1 映射回单位球面

    对于平面上点 e=(e1,e2,1)，球面上对应点 P 满足：
    P = (4e1, 4e2, 4 - ||e||^2) / (4 + ||e||^2)

    Parameters
    ----------
    points_plane : ndarray, shape (N, 3)
        平面上的点

    Returns
    -------
    points_sphere : ndarray, shape (N, 3)
        单位球面上的点
    """
    points_plane = np.asarray(points_plane, dtype=float)
    if points_plane.ndim == 1:
        points_plane = points_plane.reshape(1, -1)

    e1 = points_plane[:, 0]
    e2 = points_plane[:, 1]

    norm_sq = e1 ** 2 + e2 ** 2
    denom = 4.0 + norm_sq

    p1 = 4.0 * e1 / denom
    p2 = 4.0 * e2 / denom
    p3 = (4.0 - norm_sq) / denom

    return np.column_stack([p1, p2, p3])


def icosahedron_vertices():
    """
    正二十面体的 12 个顶点坐标（位于单位球面上）

    利用黄金比例 φ = (1+√5)/2：
    顶点形式为 (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) 的归一化
    """
    phi = 0.5 * (1.0 + np.sqrt(5.0))

    vertices = np.array([
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
        [-phi, 0.0, -1.0]
    ], dtype=float)

    # 归一化到单位球面
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms


# ============================================================================
# Advection-Diffusion Model for Larval Dispersal
# ============================================================================

def advection_diffusion_2d_step(C, U, V, D, dx, dy, dt, lambda_mortality=0.0):
    """
    执行一个时间步的二维对流-扩散方程

    方程：∂C/∂t + U ∂C/∂x + V ∂C/∂y = D (∂²C/∂x² + ∂²C/∂y²) - λ C

    使用一阶迎风（Upwind）格式处理对流项，中心差分处理扩散项：
    - U>0: ∂C/∂x ≈ (C[i,j] - C[i-1,j]) / dx
    - U<0: ∂C/∂x ≈ (C[i+1,j] - C[i,j]) / dx

    该格式在 CFL 条件下稳定：dt <= min(dx/|U|, dy/|V|)

    Parameters
    ----------
    C : ndarray, shape (nx, ny)
        当前浓度场
    U, V : ndarray, shape (nx, ny)
        速度场
    D : float
        扩散系数
    dx, dy : float
        空间步长
    dt : float
        时间步长
    lambda_mortality : float
        死亡率

    Returns
    -------
    C_new : ndarray
        下一时刻浓度场
    """
    nx, ny = C.shape
    C_new = C.copy()

    # CFL 稳定性检查与自动调整
    u_max = np.max(np.abs(U))
    v_max = np.max(np.abs(V))
    cfl_limit = min(dx / (u_max + NumericalConfig.EPS),
                    dy / (v_max + NumericalConfig.EPS))
    diff_limit = 0.5 / (D * (1.0 / dx ** 2 + 1.0 / dy ** 2) + NumericalConfig.EPS)
    dt_safe = min(cfl_limit, diff_limit, dt)
    if dt_safe < dt:
        dt = dt_safe

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # 对流项：一阶迎风格式（无条件稳定，满足 CFL 即可）
            if U[i, j] >= 0:
                adv_x = U[i, j] * (C[i, j] - C[i - 1, j]) / dx
            else:
                adv_x = U[i, j] * (C[i + 1, j] - C[i, j]) / dx

            if V[i, j] >= 0:
                adv_y = V[i, j] * (C[i, j] - C[i, j - 1]) / dy
            else:
                adv_y = V[i, j] * (C[i, j + 1] - C[i, j]) / dy

            # 扩散项：中心差分
            diff_x = (C[i + 1, j] - 2.0 * C[i, j] + C[i - 1, j]) / (dx ** 2)
            diff_y = (C[i, j + 1] - 2.0 * C[i, j] + C[i, j - 1]) / (dy ** 2)

            C_new[i, j] = C[i, j] - dt * (adv_x + adv_y) \
                          + dt * D * (diff_x + diff_y) \
                          - dt * lambda_mortality * C[i, j]

    # 零梯度边界条件（开放边界）
    C_new[0, :] = C_new[1, :]
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]

    return C_new


def simulate_larval_dispersal(nx, ny, Lx, Ly, C0_center, C0_sigma,
                               U, V, D, T_total, dt, lambda_mortality=0.0):
    """
    模拟鱼卵/仔稚鱼的被动扩散过程

    Parameters
    ----------
    nx, ny : int
        网格数
    Lx, Ly : float
        域尺寸
    C0_center : tuple
        初始高斯分布中心 (cx, cy)
    C0_sigma : float
        初始高斯分布标准差
    U, V : ndarray
        速度场（可随时间变化，这里假设稳态）
    D : float
        扩散系数
    T_total : float
        总模拟时间
    dt : float
        时间步长
    lambda_mortality : float
        死亡率

    Returns
    -------
    C_final : ndarray
        最终浓度分布
    times : list
        时间序列
    C_history : list
        浓度场历史（每隔一定步数保存）
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # 初始条件：高斯分布
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    cx, cy = C0_center
    C = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * C0_sigma ** 2))

    n_steps = int(T_total / dt)
    C_history = []
    times = []

    for step in range(n_steps):
        C = advection_diffusion_2d_step(C, U, V, D, dx, dy, dt, lambda_mortality)
        if step % max(1, n_steps // 10) == 0:
            C_history.append(C.copy())
            times.append(step * dt)

    return C, times, C_history
