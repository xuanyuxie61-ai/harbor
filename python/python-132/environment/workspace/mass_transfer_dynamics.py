"""
mass_transfer_dynamics.py
=========================
精馏塔传质动力学与ODE系统求解模块。

本模块融合以下种子项目：
- RK45 Runge-Kutta 积分（源自项目 1037_rk45）
- 三体问题ODE（源自项目 1260_three_body_ode）
- Langford 非线性ODE（源自项目 645_langford_ode）
- Lorenz96 混沌ODE（源自项目 703_lorenz96_ode）

科学背景
--------
精馏塔动态传质可用组分物料平衡描述。对于第 j 块理论塔板：

    dM_j x_{i,j} / dt = L_{j-1} x_{i,j-1} + V_{j+1} y_{i,j+1}
                        - L_j x_{i,j} - V_j y_{i,j} + F_j z_{i,j}

其中 M_j 为持液量，L 为液相流量，V 为汽相流量，F 为进料流量。

本模块将多组分传质耦合类比为"多体动力学"：
- 三体ODE思想映射为三组分Maxwell-Stefan扩散方程；
- Langford非线性动力学刻画局部湍流混合；
- Lorenz96混沌系统模拟塔内对流混沌；
- RK45提供高精度时间积分与误差估计。

Maxwell-Stefan 扩散方程（三组分）：
    Σ_j (x_i N_j - x_j N_i) / (c_t D_{ij}) = -∇x_i

可改写为矩阵形式：
    [B] {N} = -c_t {∇x}

其中 B_{ii} = Σ_{j≠i} x_j / D_{ij}, B_{ij} = -x_i / D_{ij}

Langford 系统用于局部混合子模型：
    dx/dt = (z - b)x - d y
    dy/dt = d x + (z - b) y
    dz/dt = c + a z - z³/3 - (x²+y²)(1+e z) + f z x³

Lorenz96 系统用于大尺度对流模拟（N=40）：
    dy_i/dt = (y_{i+1} - y_{i-2}) y_{i-1} - y_i + F
"""

import numpy as np
from utils import ensure_positive, clip_with_warning


# ---------------------------------------------------------------------------
# RK45 Runge-Kutta 积分（源自项目 1037_rk45）
# ---------------------------------------------------------------------------

def rk45_integrate(yprime, tspan, y0, n_steps, projection=None):
    """
    显式 Runge-Kutta 4/5 阶积分，同时返回每步误差估计。
    支持可选的投影函数，在每步后对解进行裁剪/投影。

    Parameters
    ----------
    yprime : callable
        导数函数 yprime(t, y) -> ndarray。
    tspan : tuple
        (t0, t_end)。
    y0 : ndarray
        初始条件。
    n_steps : int
        步数。
    projection : callable, optional
        投影函数 projection(y) -> y_projected。

    Returns
    -------
    t : ndarray
        时间序列。
    y : ndarray
        解序列。
    e : ndarray
        误差估计。
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    t = np.zeros(n_steps + 1, dtype=float)
    y = np.zeros((n_steps + 1, m), dtype=float)
    e = np.zeros((n_steps + 1, m), dtype=float)

    dt = (tspan[1] - tspan[0]) / n_steps
    t[0] = tspan[0]
    y[0, :] = y0
    if projection is not None:
        y[0, :] = projection(y[0, :])
    e[0, :] = 0.0

    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0]
    ], dtype=float)
    b = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0], dtype=float)
    c = np.array([0.0, 0.25, 3.0/8.0, 12.0/13.0, 1.0, 0.5], dtype=float)
    d = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0], dtype=float)

    for i in range(n_steps):
        k1 = dt * yprime(t[i] + c[0] * dt, y[i, :])
        k2 = dt * yprime(t[i] + c[1] * dt, y[i, :] + a[1, 0] * k1)
        k3 = dt * yprime(t[i] + c[2] * dt, y[i, :] + a[2, 0] * k1 + a[2, 1] * k2)
        k4 = dt * yprime(t[i] + c[3] * dt, y[i, :] + a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3)
        k5 = dt * yprime(t[i] + c[4] * dt, y[i, :] + a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4)
        k6 = dt * yprime(t[i] + c[5] * dt, y[i, :] + a[5, 0] * k1 + a[5, 1] * k2 + a[5, 2] * k3 + a[5, 3] * k4 + a[5, 4] * k5)

        y4 = y[i, :] + d[0] * k1 + d[1] * k2 + d[2] * k3 + d[3] * k4 + d[4] * k5 + d[5] * k6
        y5 = y[i, :] + b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6

        t[i + 1] = t[i] + dt
        y[i + 1, :] = y5
        if projection is not None:
            y[i + 1, :] = projection(y[i + 1, :])
        e[i + 1, :] = np.abs(y5 - y4)

    return t, y, e


# ---------------------------------------------------------------------------
# 三组分 Maxwell-Stefan 扩散（类比三体ODE，源自项目 1260_three_body_ode）
# ---------------------------------------------------------------------------

def maxwell_stefan_diffusion(y, D_matrix, c_total):
    """
    三组分 Maxwell-Stefan 扩散方程右端项。

    状态向量 y = [x1, x2, x3, N1, N2, N3]（组成与通量）。
    将组分间的交互作用类比为三体引力相互作用：
        d(x_i)/dt 由通量梯度驱动
        d(N_i)/dt 由组成梯度与交互阻力驱动

    Parameters
    ----------
    y : ndarray, shape (6,)
        状态向量。
    D_matrix : ndarray, shape (3, 3)
        二元扩散系数矩阵 [m²/s]。
    c_total : float
        总摩尔浓度 [mol/m³]。

    Returns
    -------
    dydt : ndarray, shape (6,)
        时间导数。
    """
    y = np.asarray(y, dtype=float)
    x = y[0:3]
    N = y[3:6]

    # 归一化并裁剪组成
    x = np.clip(x, 1e-6, 1.0)
    xsum = np.sum(x)
    if xsum > 1e-12:
        x = x / xsum
    else:
        x = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    # 组成变化由通量驱动（带阻尼）
    dxdt = N / max(c_total, 1.0)
    dxdt = np.clip(dxdt, -0.1, 0.1)

    # 通量变化由交互阻力驱动（类比三体引力，但加饱和限制）
    dNdt = np.zeros(3, dtype=float)
    for i in range(3):
        force = 0.0
        for j in range(3):
            if i != j:
                Dij = max(D_matrix[i, j], 1e-15)
                diff = x[j] - x[i]
                # 使用带饱和的交互项，避免奇点
                dist = np.abs(diff) + 1e-3
                force += diff / (dist * Dij)
        dNdt[i] = np.clip(force * c_total * 1e-4, -1.0, 1.0)

    return np.concatenate([dxdt, dNdt])


def simulate_three_component_diffusion(y0, D_matrix, c_total, tspan, n_steps):
    """
    模拟三组分传质扩散过程。

    Returns
    -------
    t, y, e : ndarray
        时间、解、误差估计。
    """
    def deriv(t, y):
        return maxwell_stefan_diffusion(y, D_matrix, c_total)
    return rk45_integrate(deriv, tspan, y0, n_steps)


# ---------------------------------------------------------------------------
# Langford 局部混合子模型（源自项目 645_langford_ode）
# ---------------------------------------------------------------------------

def langford_deriv(t, xyz, a=3.0, b=1.5, c=1.0, d=0.5, e=0.2, f=0.1):
    """
    Langford 非线性ODE右端项，用于刻画塔板上局部湍流混合。

    dx/dt = (z - b) x - d y
    dy/dt = d x + (z - b) y
    dz/dt = c + a z - z³/3 - (x²+y²)(1+e z) + f z x³

    Parameters
    ----------
    t : float
        时间。
    xyz : ndarray, shape (3,)
        [x, y, z]。
    a, b, c, d, e, f : float
        模型参数。

    Returns
    -------
    dxyzdt : ndarray, shape (3,)
        导数。
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[0], xyz[1], xyz[2]

    dxdt = (z - b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - z ** 3 / 3.0 - (x ** 2 + y ** 2) * (1.0 + e * z) + f * z * x ** 3

    return np.array([dxdt, dydt, dzdt], dtype=float)


def simulate_langford_mixing(xyz0, tspan, n_steps, a=3.0, b=1.5, c=1.0, d=0.5, e=0.2, f=0.1):
    """
    模拟 Langford 局部混合动力学。
    """
    def deriv(t, y):
        return langford_deriv(t, y, a, b, c, d, e, f)
    return rk45_integrate(deriv, tspan, xyz0, n_steps)


# ---------------------------------------------------------------------------
# Lorenz96 大尺度对流模型（源自项目 703_lorenz96_ode）
# ---------------------------------------------------------------------------

def lorenz96_deriv(t, y, n=40, force=8.0):
    """
    Lorenz96 混沌ODE，用于模拟塔内大尺度对流的混沌特性。

    dy_i/dt = (y_{i+1} - y_{i-2}) y_{i-1} - y_i + F

    Parameters
    ----------
    t : float
        时间。
    y : ndarray, shape (n,)
        状态向量（环状连接）。
    n : int
        维度数（对应塔板数或空间离散数）。
    force : float
        外力参数。

    Returns
    -------
    dydt : ndarray, shape (n,)
        导数。
    """
    y = np.asarray(y, dtype=float)
    if len(y) != n:
        n = len(y)

    dydt = np.zeros(n, dtype=float)
    for i in range(n):
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        ip1 = (i + 1) % n
        dydt[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + force

    return dydt


def simulate_lorenz96_convection(y0, tspan, n_steps, force=8.0):
    """
    模拟 Lorenz96 对流混沌。
    """
    n = len(y0)

    def deriv(t, y):
        return lorenz96_deriv(t, y, n, force)

    return rk45_integrate(deriv, tspan, y0, n_steps)


# ---------------------------------------------------------------------------
# 精馏塔动态物料平衡（综合ODE系统）
# ---------------------------------------------------------------------------

def distillation_column_deriv(t, state, n_trays, nc, F, z_feed, q_feed,
                               L, V, holdup, alpha_rel, tray_eff):
    """
    精馏塔动态物料平衡右端项。

    对于第 j 块塔板（j=0..n_trays-1，0为再沸器，n_trays-1为冷凝器）：

        d(M_j x_{i,j})/dt = L_{j-1} x_{i,j-1} + V_{j+1} y_{i,j+1}
                           - L_j x_{i,j} - V_j y_{i,j} + F_j z_{i,j}

    其中 y_{i,j} = E_j * (K_{i,j} x_{i,j}) + (1 - E_j) y_{i,j+1}
    E_j 为 Murphree 汽相效率。

    状态向量排列：state[j*nc + i] = x_{i,j}

    Parameters
    ----------
    t : float
        时间 [s]。
    state : ndarray, shape (n_trays * nc,)
        组成状态。
    n_trays : int
        塔板数。
    nc : int
        组分数。
    F : ndarray, shape (n_trays,)
        各板进料流量 [mol/s]。
    z_feed : ndarray, shape (n_trays, nc)
        进料组成。
    q_feed : ndarray, shape (n_trays,)
        进料热状态参数。
    L : ndarray, shape (n_trays,)
        液相流量 [mol/s]。
    V : ndarray, shape (n_trays,)
        汽相流量 [mol/s]。
    holdup : ndarray, shape (n_trays,)
        持液量 [mol]。
    alpha_rel : ndarray, shape (nc,)
        相对挥发度。
    tray_eff : ndarray, shape (n_trays,)
        Murphree 效率。

    Returns
    -------
    dstate : ndarray
        时间导数。
    """
    state = np.asarray(state, dtype=float)
    dstate = np.zeros_like(state)

    for j in range(n_trays):
        x_j = state[j * nc:(j + 1) * nc].copy()
        x_j = np.clip(x_j, 1e-12, 1.0)
        x_j = x_j / np.sum(x_j)

        # TODO [Hole 2]: 实现平衡汽相组成 y_eq、实际汽相组成 y_j 的计算。
        # 科学背景：
        #   1. alpha_rel 是从 vle_thermodynamics.py 传入的相对挥发度（由 K_i / K_ref 计算）。
        #   2. 平衡汽相组成 y_eq* = K_eq * x_j。当前代码中 alpha_rel 被直接当作 K_eq 使用。
        #   3. Murphree 汽相效率：y_j = E_j * y_eq* + (1 - E_j) * y_{j+1}
        #   4. 需考虑上下板边界（再沸器 j=0 无上板来液，冷凝器 j=n-1 无下板来汽）。
        # 要求：
        #   - 正确利用 alpha_rel 计算各板平衡汽相组成
        #   - 正确应用 Murphree 效率得到实际汽相组成
        #   - 与 Hole 1 的 K 值计算保持数据一致性
        #   - 对组成进行裁剪和归一化，防止数值不稳定
        K_j = None  # 待修复
        y_eq = None  # 待修复
        y_j = None  # 待修复
        y_jp1 = None  # 待修复
        # --------------------------------------------------

        # 物料平衡
        L_jm1 = float(L[j - 1]) if j > 0 else 0.0
        x_jm1 = state[(j - 1) * nc:j * nc].copy() if j > 0 else x_j.copy()
        if j > 0:
            x_jm1 = np.clip(x_jm1, 1e-12, 1.0)
            x_jm1 = x_jm1 / np.sum(x_jm1)

        V_jp1 = float(V[j + 1]) if j < n_trays - 1 else 0.0
        y_jp1_in = y_jp1.copy()

        F_j = float(F[j])
        z_j = z_feed[j, :].copy()
        z_j = np.clip(z_j, 1e-12, 1.0)
        z_j = z_j / (np.sum(z_j) + 1e-15)

        # 混合进料液相/汽相分配
        qf = float(np.clip(q_feed[j], 0.0, 1.0))
        L_feed = qf * F_j
        V_feed = (1.0 - qf) * F_j

        M_j = max(float(holdup[j]), 1e-6)

        for i in range(nc):
            dxdt = (
                L_jm1 * x_jm1[i] + V_jp1 * y_jp1_in[i]
                + L_feed * z_j[i] + V_feed * z_j[i]
                - float(L[j]) * x_j[i] - float(V[j]) * y_j[i]
            ) / M_j
            # 边界处理：防止组成超出 [0,1]
            if x_j[i] <= 1e-8 and dxdt < 0:
                dxdt = 0.0
            if x_j[i] >= 1.0 - 1e-8 and dxdt > 0:
                dxdt = 0.0
            dstate[j * nc + i] = dxdt

    return dstate


def simulate_distillation_dynamics(n_trays, nc, F, z_feed, q_feed, L, V,
                                    holdup, alpha_rel, tray_eff,
                                    x0, tspan, n_steps):
    """
    模拟精馏塔动态传质过程。

    Returns
    -------
    t, y, e : ndarray
        时间、组成状态、误差估计。
    composition_profiles : ndarray, shape (n_steps+1, n_trays, nc)
        各时刻各板组成。
    """
    def deriv(t, y):
        return distillation_column_deriv(
            t, y, n_trays, nc, F, z_feed, q_feed,
            L, V, holdup, alpha_rel, tray_eff
        )

    def projection(y):
        """将每块板的组成投影到 [0,1] 并归一化。"""
        y = np.asarray(y, dtype=float)
        for j in range(n_trays):
            xj = y[j * nc:(j + 1) * nc]
            xj = np.clip(xj, 1e-12, 1.0)
            s = np.sum(xj)
            if s > 1e-12:
                y[j * nc:(j + 1) * nc] = xj / s
        return y

    t, y, e = rk45_integrate(deriv, tspan, x0, n_steps, projection=projection)

    composition_profiles = y.reshape((n_steps + 1, n_trays, nc))
    return t, y, e, composition_profiles
