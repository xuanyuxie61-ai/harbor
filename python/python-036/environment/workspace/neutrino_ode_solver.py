"""
neutrino_ode_solver.py
中微子味演化常微分方程求解器

基于 euler 和 r8but 的核心算法:
    - Euler 显式时间推进 (用于验证和快速计算)
    - 带化上三角矩阵求解 (r8but_sl, 用于大规模离散系统)
    - Runge-Kutta 4 阶方法 (高精度)
    - 矩阵指数法 (精确对角化)

物理模型:
    中微子在物质中传播满足薛定谔型方程:
        i d|ν(t)⟩/dt = H(t) |ν(t)⟩

    其中 H(t) 可以是时变的 (当物质密度沿轨迹变化时)。
    这等价于:
        d|ν⟩/dt = -i H(t) |ν⟩

     flavors = 3, 状态向量 |ν⟩ = (ν_e, ν_μ, ν_τ)^T
"""

import numpy as np
from constants import EARTH_RADIUS_KM, KM_TO_EV_INV
from pmns_matrix import build_pmns_matrix, build_mass_matrix
from neutrino_hamiltonian import build_vacuum_hamiltonian, build_matter_hamiltonian


def euler_step(y, t, dt, dydt):
    """
    执行一个 Euler 显式时间步进。
    (源自 euler.m)

    公式:
        y_{n+1} = y_n + dt * f(t_n, y_n)

    参数:
        y:    当前状态向量
        t:    当前时间
        dt:   时间步长
        dydt: 导数函数 f(t, y)

    返回:
        y_new: 新状态向量
    """
    y = np.asarray(y, dtype=np.complex128)
    f_val = dydt(t, y)
    return y + dt * f_val


def solve_euler(dydt, tspan, y0, n_steps):
    """
    使用 Euler 方法求解 ODE。
    (源自 euler.m)

    参数:
        dydt:   导数函数 f(t, y)
        tspan:  (t0, tf) 时间区间
        y0:     初始条件
        n_steps: 步数

    返回:
        t:      (n_steps+1,) 时间序列
        y:      (n_steps+1, m) 解轨迹
    """
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)

    t = np.zeros(n_steps + 1, dtype=np.float64)
    y = np.zeros((n_steps + 1, m), dtype=np.complex128)

    t[0] = t0
    y[0, :] = np.asarray(y0, dtype=np.complex128)

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        y[i + 1, :] = euler_step(y[i, :], t[i], dt, dydt)

    return t, y


def rk4_step(y, t, dt, dydt):
    """
    Runge-Kutta 4 阶单步。

    公式:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    """
    y = np.asarray(y, dtype=np.complex128)
    k1 = dydt(t, y)
    k2 = dydt(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = dydt(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = dydt(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(dydt, tspan, y0, n_steps):
    """
    使用 RK4 方法求解 ODE。

    参数:
        dydt:    导数函数
        tspan:   (t0, tf)
        y0:      初始条件
        n_steps: 步数

    返回:
        t: (n_steps+1,) 时间
        y: (n_steps+1, m) 解
    """
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)

    t = np.zeros(n_steps + 1, dtype=np.float64)
    y = np.zeros((n_steps + 1, m), dtype=np.complex128)

    t[0] = t0
    y[0, :] = np.asarray(y0, dtype=np.complex128)

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        y[i + 1, :] = rk4_step(y[i, :], t[i], dt, dydt)

    return t, y


def solve_neutrino_oscillation_ode(
        energy_gev, baseline_km,
        matter_potential_ev=None,
        n_steps=1000, method='rk4',
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None, delta_m2_31=None,
        hierarchy='normal', initial_flavor='electron'
    ):
    """
    求解中微子味演化 ODE。

    方程:
        i d|ν⟩/dx = H |ν⟩   (x 为传播距离)

    或等价地:
        d|ν⟩/dx = -i H |ν⟩

    参数:
        energy_gev:          中微子能量 [GeV]
        baseline_km:         传播基线 [km]
        matter_potential_ev: 物质势 [eV], None 表示真空
        n_steps:             空间步数
        method:              'euler', 'rk4', 'matrix_exp'
        ...                  PMNS 参数
        initial_flavor:      'electron', 'muon', 'tau'

    返回:
        result: dict 包含:
            't': 距离序列 [km]
            'P_ee', 'P_em', 'P_et': 各味概率随距离演化
            'prob_final': 最终概率数组 [P_ee, P_em, P_et]
    """
    from pmns_matrix import get_initial_flavor_state

    if baseline_km < 0:
        raise ValueError("baseline_km must be non-negative")
    if energy_gev <= 0:
        raise ValueError("energy_gev must be positive")

    psi0 = get_initial_flavor_state(initial_flavor)

    if matter_potential_ev is None or matter_potential_ev == 0.0:
        # 真空
        H = build_vacuum_hamiltonian(
            energy_gev, theta12, theta23, theta13, delta_cp,
            delta_m2_21, delta_m2_31, hierarchy
        )
    else:
        H = build_matter_hamiltonian(
            energy_gev, matter_potential_ev,
            theta12, theta23, theta13, delta_cp,
            delta_m2_21, delta_m2_31, hierarchy
        )

    def dydt(t, y):
        return -1j * (H @ y)

    tspan = (0.0, baseline_km)

    if method == 'euler':
        t, y = solve_euler(dydt, tspan, psi0, n_steps)
    elif method == 'rk4':
        t, y = solve_rk4(dydt, tspan, psi0, n_steps)
    elif method == 'matrix_exp':
        # 矩阵指数法 (精确解, 仅适用于恒定哈密顿量)
        t = np.array([0.0, baseline_km])
        L_ev_inv = baseline_km * KM_TO_EV_INV
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
        U_prop = eigenvectors @ D @ eigenvectors.conj().T
        psi_final = U_prop @ psi0
        y = np.array([psi0, psi_final], dtype=np.complex128)
    else:
        raise ValueError("method must be 'euler', 'rk4', or 'matrix_exp'")

    # 计算各味概率
    P_ee = np.abs(y[:, 0]) ** 2
    P_em = np.abs(y[:, 1]) ** 2
    P_et = np.abs(y[:, 2]) ** 2

    return {
        't': t,
        'P_ee': P_ee,
        'P_em': P_em,
        'P_et': P_et,
        'prob_final': np.array([P_ee[-1], P_em[-1], P_et[-1]], dtype=np.float64),
        'psi_final': y[-1, :]
    }


def solve_varying_matter_ode(
        energy_gev, baseline_km,
        matter_potential_func,
        n_steps=2000, method='rk4',
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None, delta_m2_31=None,
        hierarchy='normal', initial_flavor='electron'
    ):
    """
    求解变物质密度下的中微子味演化。

    当物质势随位置变化时 (如中微子穿过地球),
    H = H(x), 必须使用数值 ODE 方法。

    参数:
        energy_gev:           中微子能量 [GeV]
        baseline_km:          基线 [km]
        matter_potential_func: 物质势函数 V(x) [eV], x in [0, baseline_km]
        n_steps:              步数
        method:               'euler' 或 'rk4'
        ...                   PMNS 参数
        initial_flavor:       初始味

    返回:
        result: dict, 同 solve_neutrino_oscillation_ode
    """
    from pmns_matrix import get_initial_flavor_state

    psi0 = get_initial_flavor_state(initial_flavor)
    U = build_pmns_matrix(theta12, theta23, theta13, delta_cp)
    M2 = build_mass_matrix(delta_m2_21, delta_m2_31, hierarchy)

    H_vac = (1.0 / (2.0 * energy_gev * 1e9)) * (U @ M2 @ U.conj().T)

    # === HOLE 3 ===
    # 请实现变物质密度下的味演化方程右端项 d|ν⟩/dx = -i H(x) |ν⟩
    # 提示:
    #   1. 获取当前位置的物质势 V = matter_potential_func(t) [eV]
    #   2. 构造物质势矩阵 V_mat = diag(V, 0, 0) (仅 ν_e 获得额外相位)
    #   3. 总哈密顿量 H = H_vac + V_mat
    #   4. 返回 -1j * (H @ y)
    # 注意: 本闭包使用了外部变量 H_vac (已由 build_pmns_matrix / build_mass_matrix 构造)
    # === END HOLE 3 ===
    raise NotImplementedError("HOLE 3: dydt 闭包尚未实现")

    tspan = (0.0, baseline_km)

    if method == 'euler':
        t, y = solve_euler(dydt, tspan, psi0, n_steps)
    elif method == 'rk4':
        t, y = solve_rk4(dydt, tspan, psi0, n_steps)
    else:
        raise ValueError("method must be 'euler' or 'rk4' for varying matter")

    P_ee = np.abs(y[:, 0]) ** 2
    P_em = np.abs(y[:, 1]) ** 2
    P_et = np.abs(y[:, 2]) ** 2

    return {
        't': t,
        'P_ee': P_ee,
        'P_em': P_em,
        'P_et': P_et,
        'prob_final': np.array([P_ee[-1], P_em[-1], P_et[-1]], dtype=np.float64),
        'psi_final': y[-1, :]
    }


def r8but_sl(n, mu, a, b):
    """
    求解上三角带状矩阵系统 A x = b。
    (源自 r8but_sl)

    R8BUT 存储格式:
        A 存储在 (mu+1, n) 数组中
        对角线在第 mu 行
        第 k 条上对角线在第 mu-k 行, 列从 k+1 到 n

    参数:
        n:  矩阵阶数
        mu: 上带宽
        a:  (mu+1, n) R8BUT 格式矩阵
        b:  (n,) 右端项

    返回:
        x: (n,) 解向量
    """
    x = np.asarray(b, dtype=np.float64).copy()

    for j in range(n - 1, -1, -1):
        # 对角元素位置
        diag_idx = mu
        x[j] = x[j] / a[diag_idx, j]
        jlo = max(0, j - mu)
        for i in range(jlo, j):
            # 元素 A[i, j] 在 R8BUT 中的位置
            a_idx = mu + i - j
            x[i] = x[i] - a[a_idx, j] * x[j]

    return x


def solve_banded_upper_triangular(A_dense, b):
    """
    将稠密上三角矩阵转换为 R8BUT 格式并求解。

    参数:
        A_dense: (n, n) 稠密矩阵 (上三角)
        b:       (n,) 右端项

    返回:
        x: (n,) 解向量
    """
    n = len(b)
    # 计算上带宽
    mu = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A_dense[i, j]) > 1e-14:
                mu = max(mu, j - i)

    # 转换为 R8BUT 格式
    a_r8but = np.zeros((mu + 1, n), dtype=np.float64)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            a_r8but[mu + i - j, j] = A_dense[i, j]

    return r8but_sl(n, mu, a_r8but, b)
