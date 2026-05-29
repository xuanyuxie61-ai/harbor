"""
dynamics_solver.py
动力学时间积分模块

融合种子项目:
- 138_cauchy_method: Cauchy(theta)单步法（隐式-显式混合）
- 1059_sawtooth_ode: 锯齿波驱动ODE
- 472_glycolysis_ode: 糖酵解化学ODE（化学-力学耦合）

科学应用: Cosserat杆动力学方程的时间积分、周期性驱动、化学耦合
"""

import numpy as np
from typing import Callable, Tuple, Optional


def cauchy_theta_method(f: Callable[[float, np.ndarray], np.ndarray],
                       tspan: Tuple[float, float],
                       y0: np.ndarray,
                       n: int,
                       theta: float = 0.5,
                       it_max: int = 20,
                       tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cauchy(theta)单步法 — 基于种子项目138_cauchy_method

    格式:
        预测步（隐式）:
            y_m = y_i + theta*dt*f(t_i + theta*dt, y_m)   [不动点迭代]
        校正步（显式）:
            y_{i+1} = (1/theta)*y_m + (1 - 1/theta)*y_i

    theta=0:  显式Euler（退化，此处不支持）
    theta=0.5: 二阶精度的梯形法则等价形式
    theta=1:  隐式Euler

    参数:
        f: ODE右端函数 f(t, y)
        tspan: (t0, tf)
        y0: 初始条件
        n: 时间步数
        theta: 方法参数
        it_max: 不动点迭代最大次数
        tol: 不动点迭代容差
    """
    if theta <= 0 or theta > 1:
        raise ValueError("theta must be in (0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")

    t0, tf = tspan
    dt = (tf - t0) / n
    m = len(y0)

    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, m))
    y[0] = y0

    for i in range(n):
        ti = t[i]
        yi = y[i]
        tm = ti + theta * dt

        # 预测步: 不动点迭代求解 y_m = yi + theta*dt*f(tm, y_m)
        ym = yi.copy()  # 初始猜测
        for _ in range(it_max):
            ym_new = yi + theta * dt * f(tm, ym)
            if np.linalg.norm(ym_new - ym) < tol:
                ym = ym_new
                break
            ym = ym_new

        # 校正步
        y[i + 1] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * yi

    return t, y


def low_storage_rk4(f: Callable[[float, np.ndarray], np.ndarray],
                   tspan: Tuple[float, float],
                   y0: np.ndarray,
                   n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    5级4阶低存储Runge-Kutta — 基于种子项目273_dg1d_heat

    系数（Kennedy-Carpenter低存储格式）:
        a = [0.0, -567301805773/1357537059087, -2404267990393/2016746695238,
             -3550918686646/2091501179385, -1275806237668/842570457699]
        b = [1432997174477/9575080441755, 5161836677717/13612068292357,
             1720146321549/2090206949498, 3134564353537/4481467310338,
             2277821191437/14882151754819]
        c = [0.0, 1432997174477/9575080441755, 2526269341429/6820363962896,
             2006345519317/3224310063776, 2802321613138/2924317926251]

    每步仅需2个寄存器
    """
    t0, tf = tspan
    dt = (tf - t0) / n
    m = len(y0)

    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, m))
    y[0] = y0

    # RK系数
    a = np.array([0.0,
                  -567301805773.0 / 1357537059087.0,
                  -2404267990393.0 / 2016746695238.0,
                  -3550918686646.0 / 2091501179385.0,
                  -1275806237668.0 / 842570457699.0])
    b = np.array([1432997174477.0 / 9575080441755.0,
                  5161836677717.0 / 13612068292357.0,
                  1720146321549.0 / 2090206949498.0,
                  3134564353537.0 / 4481467310338.0,
                  2277821191437.0 / 14882151754819.0])
    c = np.array([0.0,
                  1432997174477.0 / 9575080441755.0,
                  2526269341429.0 / 6820363962896.0,
                  2006345519317.0 / 3224310063776.0,
                  2802321613138.0 / 2924317926251.0])

    for i in range(n):
        ti = t[i]
        yi = y[i]
        res = np.zeros(m)
        dy = np.zeros(m)

        for s in range(5):
            ts = ti + c[s] * dt
            ys = yi + a[s] * dy
            res = f(ts, ys)
            dy = b[s] * dt * res + dy

        y[i + 1] = yi + dy

    return t, y


def sawtooth_driver(t: float, omega: float = 1.0) -> float:
    """
    锯齿波驱动 — 基于种子项目1059_sawtooth_ode

    f(t) = mod(t + omega*pi, 2*omega*pi) - omega*pi

    周期: T = 2*omega*pi
    幅值: omega*pi
    """
    T = 2.0 * omega * np.pi
    phase = t + omega * np.pi
    f = np.mod(phase, T) - omega * np.pi
    return f


def driven_harmonic_oscillator(t: float, y: np.ndarray,
                               omega0: float = 1.0,
                               zeta: float = 0.1,
                               omega_drive: float = 1.0) -> np.ndarray:
    """
    锯齿波驱动的阻尼谐振子 — 基于种子项目1059_sawtooth_ode

    方程:
        u' = v
        v' = -omega0^2 * u - 2*zeta*omega0*v + sawtooth_driver(t, omega_drive)

    参数:
        omega0: 固有频率
        zeta: 阻尼比
        omega_drive: 驱动频率
    """
    u, v = y
    forcing = sawtooth_driver(t, omega_drive)
    dudt = v
    dvdt = -omega0 ** 2 * u - 2.0 * zeta * omega0 * v + forcing
    return np.array([dudt, dvdt])


def cosserat_dynamics_rhs(t: float, state: np.ndarray,
                          L: float, Ns: int,
                          E: float, G: float, A: float,
                          Ixx: float, Iyy: float, J: float,
                          rho: float, F_ext: Optional[Callable] = None,
                          chemical_state: Optional[np.ndarray] = None,
                          chemo_params: Optional[dict] = None) -> np.ndarray:
    """
    1D软体Cosserat杆动力学右端项 — 稳定化的Euler-Bernoulli梁动力学

    简化模型: 平面问题 (r_x, r_y, theta_z)
    每节点3DOF: [u_x, u_y, theta]

    方程:
        rho*A*u_ddot = EA*u''_x + f_x
        rho*A*v_ddot = -EI*v''''_y + f_y    (横向弯曲)
        rho*I*theta_ddot = EI*theta'' + tau

    使用稳定的有限差分离散
    """
    dof_per_node = 3  # ux, uy, theta
    n_nodes = Ns + 1
    total_dof = n_nodes * dof_per_node

    if len(state) != 2 * total_dof:
        raise ValueError(f"state length {len(state)} != {2*total_dof}")

    q = state[:total_dof]
    qdot = state[total_dof:]

    ds = L / Ns
    if ds < 1e-14:
        raise ValueError("ds too small")

    # 质量矩阵（对角化 lumped mass）
    M_diag = np.zeros(total_dof)
    for i in range(n_nodes):
        base = i * dof_per_node
        M_diag[base] = rho * A
        M_diag[base + 1] = rho * A
        M_diag[base + 2] = rho * (Ixx + Iyy)

    # 化学耦合有效模量
    E_eff = E
    if chemical_state is not None and chemo_params is not None:
        from hyperelastic_law import chemo_mechanical_coupling
        E0 = chemo_params.get('E0', E)
        gamma = chemo_params.get('gamma', 0.0)
        beta_chem = chemo_params.get('beta_chem', 0.0)
        E_eff = chemo_mechanical_coupling(chemical_state, 0.0, E0, gamma, beta_chem)

    EA_eff = E_eff * A
    EI = E_eff * max(Ixx, Iyy)

    # 构建刚度作用 K*q （逐点差分，避免组装病态矩阵）
    Kq = np.zeros(total_dof)

    # 轴向: -EA * d²u/ds² （中心差分）
    for i in range(n_nodes):
        base = i * dof_per_node
        if i == 0:
            # 固支边界: u=0
            Kq[base] = q[base] * 1.0e12  # 大惩罚
        elif i == n_nodes - 1:
            Kq[base] = EA_eff * (q[base] - q[base - dof_per_node]) / ds ** 2
        else:
            Kq[base] = -EA_eff * (q[base + dof_per_node] - 2.0 * q[base] + q[base - dof_per_node]) / ds ** 2

    # 横向弯曲: EI * d⁴v/ds⁴ （中心差分）
    # 使用稳定化系数避免高频振荡
    stab_factor = 0.1
    for i in range(n_nodes):
        base = i * dof_per_node + 1
        if i == 0 or i == 1:
            # 固支: v=0, v'=0
            Kq[base] = q[base] * 1.0e6
        elif i == n_nodes - 1 or i == n_nodes - 2:
            # 自由端: 使用降阶稳定差分
            if i == n_nodes - 2:
                Kq[base] = EI * (q[base - 2*dof_per_node] - 2.0*q[base - dof_per_node] + q[base]) / ds ** 4
            else:
                Kq[base] = EI * (q[base] - q[base - dof_per_node]) / ds ** 4
        else:
            Kq[base] = EI * (q[base - 2*dof_per_node] - 4.0*q[base - dof_per_node]
                             + 6.0*q[base] - 4.0*q[base + dof_per_node]
                             + q[base + 2*dof_per_node]) / ds ** 4
            # 添加数值耗散稳定化
            Kq[base] += stab_factor * EI * q[base] / ds ** 4

    # 扭转: -GJ * d²theta/ds²
    GJ = G * J
    for i in range(n_nodes):
        base = i * dof_per_node + 2
        if i == 0:
            Kq[base] = q[base] * 1.0e12  # 固支
        elif i == n_nodes - 1:
            Kq[base] = GJ * (q[base] - q[base - dof_per_node]) / ds ** 2
        else:
            Kq[base] = -GJ * (q[base + dof_per_node] - 2.0 * q[base] + q[base - dof_per_node]) / ds ** 2

    # 阻尼（Rayleigh）
    alpha_ray = 0.5
    beta_ray = 0.001
    Cqdot = alpha_ray * M_diag * qdot + beta_ray * Kq * 0.0  # 只保留质量比例阻尼

    # 外力
    F = np.zeros(total_dof)
    if F_ext is not None:
        for i in range(n_nodes):
            s = i * ds
            f = F_ext(t, s)
            base = i * dof_per_node
            F[base:base + 3] = f

    # 右端项
    rhs = F - Cqdot - Kq

    # 解出加速度（对角质量矩阵，直接除）
    qddot = rhs / M_diag
    qddot = np.where(np.abs(M_diag) < 1e-14, 0.0, qddot)

    # 组装状态导数
    dstate = np.zeros(len(state))
    dstate[:total_dof] = qdot
    dstate[total_dof:] = qddot
    return dstate


def integrate_cosserat_dynamics(tspan: Tuple[float, float],
                                q0: np.ndarray, qdot0: np.ndarray,
                                Ns: int, L: float,
                                material_params: dict,
                                n_steps: int = 500,
                                method: str = 'rk4') -> Tuple[np.ndarray, np.ndarray]:
    """
    积分Cosserat杆动力学方程
    """
    state0 = np.concatenate([q0, qdot0])

    def rhs(t, s):
        return cosserat_dynamics_rhs(
            t, s, L, Ns,
            material_params['E'], material_params['G'],
            material_params['A'], material_params['Ixx'],
            material_params['Iyy'], material_params['J'],
            material_params['rho']
        )

    if method == 'rk4':
        return low_storage_rk4(rhs, tspan, state0, n_steps)
    elif method == 'cauchy':
        return cauchy_theta_method(rhs, tspan, state0, n_steps, theta=0.5)
    else:
        raise ValueError(f"Unknown method: {method}")
