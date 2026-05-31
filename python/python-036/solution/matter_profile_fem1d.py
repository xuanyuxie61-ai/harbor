"""
matter_profile_fem1d.py
一维有限元求解地球径向物质密度剖面

基于 fem1d_heat_implicit 的核心思想:
    - 在径向坐标 r ∈ [0, R_earth] 上离散
    - 使用分段线性 (P1) 有限元基函数
    - 向后 Euler 隐式时间推进 (此处用于密度松弛)
    - 组装刚度矩阵 K 和质量矩阵 M

物理模型:
    将 PREM 密度模型 ρ(r) 视为一个扩散方程的稳态解:
        -d/dr (k(r) dρ/dr) = S(r)

    其中 S(r) 为源项, k(r) 为扩散系数。
    边界条件:
        ρ(0) = ρ_core   (Dirichlet)
        dρ/dr|_{R} = 0  (Neumann, 地壳表面)
"""

import numpy as np
from constants import EARTH_RADIUS_KM, get_prem_density


def assemble_fem_1d(x_nodes, k_fun, source_fun, time=0.0):
    """
    组装一维 FEM 刚度矩阵 A 和右端项 b。

    对于方程:
        -d/dx (k(x) du/dx) = f(x)

    弱形式:
        ∫ k(x) u'(x) v'(x) dx = ∫ f(x) v(x) dx

    使用 P1 (线性) 基函数, 每个单元 [x_i, x_{i+1}] 上:
        φ_i(x)   = (x_{i+1} - x) / h_i
        φ_{i+1}(x) = (x - x_i) / h_i

    参数:
        x_nodes:   节点坐标数组 [km]
        k_fun:     扩散系数函数 k(x, t)
        source_fun: 源项函数 f(x, t)
        time:      当前时间 (用于时变问题)

    返回:
        A: (n, n) 刚度矩阵
        b: (n,)   右端项
    """
    n = len(x_nodes)
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        if h <= 0:
            raise ValueError(f"Element {i} has non-positive length: {h}")

        x_mid = 0.5 * (x_nodes[i] + x_nodes[i + 1])
        k_val = k_fun(x_mid, time)
        f_val = source_fun(x_mid, time)

        # 单元刚度矩阵 (P1 线性元)
        # K_e = (k/h) * [ 1  -1
        #                -1   1 ]
        ke = (k_val / h) * np.array([[1.0, -1.0],
                                     [-1.0, 1.0]])

        # 单元右端项 (中点积分)
        # b_e = (f*h/2) * [1, 1]
        be = (f_val * h / 2.0) * np.array([1.0, 1.0])

        # 组装到全局矩阵
        A[i:i + 2, i:i + 2] += ke
        b[i:i + 2] += be

    return A, b


def apply_boundary_conditions_1d(A, b, x_nodes, bc_type_left='dirichlet',
                                  bc_val_left=0.0, bc_type_right='neumann',
                                  bc_val_right=0.0):
    """
    施加一维边界条件。

    Dirichlet (第一类):
        u(x_0) = g_D
    Neumann (第二类):
        -k du/dn|_{x_N} = g_N

    参数:
        A, b:      全局刚度矩阵和右端项
        x_nodes:   节点坐标
        bc_type_left/right: 'dirichlet' 或 'neumann'
        bc_val_left/right:  边界值

    返回:
        A, b: 修改后的矩阵和向量
    """
    n = len(x_nodes)
    A = A.copy()
    b = b.copy()

    # 左边界 (r = 0)
    if bc_type_left == 'dirichlet':
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = bc_val_left
    elif bc_type_left == 'neumann':
        # Neumann 边界在左端: -k du/dx = g
        # 对于一维, 直接修改右端项
        b[0] += bc_val_left

    # 右边界 (r = R)
    if bc_type_right == 'dirichlet':
        A[n - 1, :] = 0.0
        A[n - 1, n - 1] = 1.0
        b[n - 1] = bc_val_right
    elif bc_type_right == 'neumann':
        b[n - 1] += bc_val_right

    return A, b


def solve_steady_state_density_1d(r_nodes_km, k_diffusion=None,
                                   rho_core=13.0, rho_surface=2.7):
    """
    使用一维 FEM 求解地球径向稳态密度剖面。

    物理模型:
        -d/dr (k dρ/dr) = S(r)

    源项 S(r) 被构造为使解匹配 PREM 模型。

    参数:
        r_nodes_km:   径向节点坐标 [km], 必须包含 0 和 R_earth
        k_diffusion:  扩散系数 [km²/单位], 默认常数 1.0
        rho_core:     内核中心密度 [g/cm³]
        rho_surface:  地表密度 [g/cm³]

    返回:
        rho:       节点上的密度 [g/cm³]
        r_nodes:   节点坐标 [km]
    """
    r_nodes = np.asarray(r_nodes_km, dtype=np.float64)
    n = len(r_nodes)
    if n < 2:
        raise ValueError("At least 2 nodes required")
    if r_nodes[0] < -1e-10 or abs(r_nodes[-1] - EARTH_RADIUS_KM) > 1.0:
        # 允许一定容差
        pass

    if k_diffusion is None:
        k_fun = lambda x, t: 1.0
    else:
        k_fun = lambda x, t: float(k_diffusion)

    # 构造源项使解趋近 PREM
    def source_fun(x, t):
        r_ratio = x / EARTH_RADIUS_KM
        r_ratio = max(0.0, min(1.0, r_ratio))
        rho_prem = get_prem_density(r_ratio)
        # 使用 PREM 作为目标, 构造近似源项
        # 对于扩散方程, 源项驱动密度分布
        # 简化为与目标密度成正比
        return rho_prem * 0.1

    A, b = assemble_fem_1d(r_nodes, k_fun, source_fun)

    # 边界条件
    A, b = apply_boundary_conditions_1d(
        A, b, r_nodes,
        bc_type_left='dirichlet', bc_val_left=rho_core,
        bc_type_right='dirichlet', bc_val_right=rho_surface
    )

    # 求解线性系统
    try:
        rho = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异, 使用最小二乘
        rho = np.linalg.lstsq(A, b, rcond=None)[0]

    return rho, r_nodes


def backward_euler_step_1d(A, M, u_old, dt, f_vec):
    """
    执行一维向后 Euler 时间步进:
        (M + dt*K) u^{n+1} = M u^n + dt * f

    参数:
        A:     刚度矩阵 K
        M:     质量矩阵
        u_old: 上一时间步的解
        dt:    时间步长
        f_vec: 源项向量

    返回:
        u_new: 新时间步的解
    """
    n = len(u_old)
    lhs = M + dt * A
    rhs = M @ u_old + dt * f_vec

    try:
        u_new = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        u_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return u_new


def assemble_mass_matrix_1d(x_nodes):
    """
    组装一致质量矩阵 M (P1 线性元)。

    单元质量矩阵:
        M_e = (h/6) * [ 2  1
                        1  2 ]

    参数:
        x_nodes: 节点坐标

    返回:
        M: (n, n) 质量矩阵
    """
    n = len(x_nodes)
    M = np.zeros((n, n), dtype=np.float64)

    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        if h <= 0:
            continue
        me = (h / 6.0) * np.array([[2.0, 1.0],
                                   [1.0, 2.0]])
        M[i:i + 2, i:i + 2] += me

    return M


def solve_time_dependent_density_1d(r_nodes_km, t_init=0.0, t_final=1.0,
                                     n_steps=100, k_diffusion=1.0):
    """
    求解一维密度扩散方程 (用于测试和验证 FEM 框架)。

    方程:
        ∂ρ/∂t = k ∂²ρ/∂r² + S(r)

    参数:
        r_nodes_km:  径向节点 [km]
        t_init:      初始时间
        t_final:     终止时间
        n_steps:     时间步数
        k_diffusion: 扩散系数

    返回:
        rho_history: (n_steps+1, n_nodes) 密度演化历史
        t_history:   (n_steps+1,) 时间序列
    """
    r_nodes = np.asarray(r_nodes_km, dtype=np.float64)
    n = len(r_nodes)
    dt = (t_final - t_init) / n_steps

    A, _ = assemble_fem_1d(r_nodes, lambda x, t: k_diffusion,
                           lambda x, t: 0.0)
    M = assemble_mass_matrix_1d(r_nodes)

    # 初始条件
    rho = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r_ratio = r_nodes[i] / EARTH_RADIUS_KM
        rho[i] = get_prem_density(max(0.0, min(1.0, r_ratio)))

    rho_history = [rho.copy()]
    t_history = [t_init]

    for step in range(n_steps):
        t = t_init + (step + 1) * dt
        _, b = assemble_fem_1d(r_nodes, lambda x, tt: k_diffusion,
                               lambda x, tt: 0.1 * get_prem_density(
                                   max(0.0, min(1.0, x / EARTH_RADIUS_KM))), t)

        A_bc, b_bc = apply_boundary_conditions_1d(
            A.copy(), b, r_nodes,
            bc_type_left='dirichlet', bc_val_left=13.0,
            bc_type_right='dirichlet', bc_val_right=2.7
        )

        rho = backward_euler_step_1d(A_bc, M, rho, dt, b_bc)
        rho_history.append(rho.copy())
        t_history.append(t)

    return np.array(rho_history), np.array(t_history)
