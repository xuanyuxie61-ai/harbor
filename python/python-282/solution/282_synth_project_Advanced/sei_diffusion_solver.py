"""
sei_diffusion_solver.py
=======================

SEI 中 Li⁺ 反应-扩散-电迁移耦合方程的时间推进求解器。

融合种子项目：
    360_fd1d_heat_explicit：显式时间推进 + CFL 条件
    1402_wave_pde：波动方程双场时间演化

核心控制方程（Nernst-Planck + 反应源项）：
    ∂c/∂t = ∇·(D_eff ∇c) - ∇·(z F D_eff c / (RT) ∇φ) + R_sei(c, φ)

其中：
    c     — Li+ 浓度 [mol/m^3]
    D_eff — 有效扩散系数 [m^2/s]
    φ     — 电势 [V]
    R_sei — SEI 反应源项 [mol/(m^3 s)]

显式有限差分格式（FTCS）：
    c_i^{n+1} = c_i^n + dt * [ D_eff (c_{i-1}^n - 2c_i^n + c_{i+1}^n)/dx^2
                               + R_sei(c_i^n, φ_i^n) ]

CFL 稳定性条件：
    dt <= dx^2 / (2 D_eff)  （纯扩散限制）

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P
try:
    from . import high_order_fd as fd
except ImportError:
    import high_order_fd as fd
try:
    from . import butler_volmer_kinetics as bv
except ImportError:
    import butler_volmer_kinetics as bv


# ============================================================
#  初始条件与边界条件
# ============================================================

def initial_concentration_profile(x, c_bulk, c_surface):
    """
    构造 SEI 中 Li+ 浓度的初始分布。

    初始假设：线性浓度梯度从电极侧到电解液侧。

    Parameters
    ----------
    x : list[float]
        网格坐标 [m]。
    c_bulk : float
        电解液侧（x=L）浓度 [mol/m^3]。
    c_surface : float
        电极侧（x=0）浓度 [mol/m^3]。

    Returns
    -------
    c : list[float]
        初始浓度场。
    """
    n = len(x)
    if n < 2:
        return [c_bulk]
    L = x[-1] - x[0]
    if abs(L) < 1.0e-30:
        return [c_bulk] * n
    c = [c_surface + (c_bulk - c_surface) * (xi - x[0]) / L for xi in x]
    return c


def initial_potential_profile(x, phi_electrode, phi_electrolyte):
    """
    构造 SEI 中电势的初始分布（线性）。

    Parameters
    ----------
    x : list[float]
        网格坐标。
    phi_electrode : float
        电极侧电位 [V]。
    phi_electrolyte : float
        电解液侧电位 [V]。

    Returns
    -------
    phi : list[float]
        初始电势场。
    """
    n = len(x)
    if n < 2:
        return [phi_electrolyte]
    L = x[-1] - x[0]
    if abs(L) < 1.0e-30:
        return [phi_electrolyte] * n
    return [phi_electrode + (phi_electrolyte - phi_electrode) *
            (xi - x[0]) / L for xi in x]


def apply_boundary_conditions(c, bc_left_type, bc_left_value,
                               bc_right_type, bc_right_value):
    """
    施加浓度场的边界条件。

    Parameters
    ----------
    c : list[float]
        浓度场。
    bc_left_type, bc_right_type : str
        'dirichlet' 或 'neumann'。
    bc_left_value, bc_right_value : float
        边界值（Dirichlet: 浓度值, Neumann: 通量值）。

    Returns
    -------
    c : list[float]
        施加边界条件后的浓度场。
    """
    n = len(c)
    if n < 2:
        return c

    if bc_left_type == "dirichlet":
        c[0] = bc_left_value
    elif bc_left_type == "neumann":
        # 一阶前向差分: (c[1] - c[0])/dx = flux
        # => c[0] = c[1] - flux * dx
        c[0] = c[1] - bc_left_value * P.DX

    if bc_right_type == "dirichlet":
        c[n - 1] = bc_right_value
    elif bc_right_type == "neumann":
        c[n - 1] = c[n - 2] + bc_right_value * P.DX

    return c


# ============================================================
#  源项计算
# ============================================================

def compute_reaction_source(c, phi, dx):
    """
    计算 SEI 反应源项 R_sei(c, φ)。

    模型：
        R_sei = -k_sei * c * exp(-alpha_c * F * (phi - U_eq_sei) / (RT))
    表示 Li+ 消耗速率（SEI 形成消耗锂）。

    Parameters
    ----------
    c : list[float]
        浓度场 [mol/m^3]。
    phi : list[float]
        电势场 [V]。
    dx : float
        空间步长。

    Returns
    -------
    R : list[float]
        反应源项 [mol/(m^3 s)]。
    """
    n = len(c)
    R = [0.0] * n
    vt = P.V_THERMAL

    for i in range(n):
        ci = max(c[i], 0.0)  # 浓度非负
        eta_local = phi[i] - P.U_EQ_SEI
        # 数值安全的指数
        arg = -P.ALPHA_C_SEI * eta_local / vt
        arg = max(-80.0, min(80.0, arg))
        R[i] = -P.K0_SEI * ci * math.exp(arg)

    return R


# ============================================================
#  显式时间步进（参考 fd1d_heat_explicit）
# ============================================================

def explicit_time_step(c_old, phi, dx, dt, d_eff, scheme="2nd_order"):
    """
    单步显式 FTCS 时间推进（参考 360_fd1d_heat_explicit.m）。

    数学形式：
        c_i^{n+1} = c_i^n + dt * [ D_eff * Laplacian(c^n)_i
                                    + R(c^n, phi^n)_i ]

    CFL 稳定性条件（必须满足）：
        CFL = D_eff * dt / dx^2 < 0.5

    Parameters
    ----------
    c_old : list[float]
        当前时间步的浓度场。
    phi : list[float]
        电势场。
    dx : float
        空间步长。
    dt : float
        时间步长。
    d_eff : float
        有效扩散系数。
    scheme : str
        差分格式: '2nd_order' 或 '4th_order'。

    Returns
    -------
    c_new : list[float]
        下一时间步的浓度场。
    cfl_actual : float
        实际 CFL 数。
    """
    n = len(c_old)
    c_new = list(c_old)

    # 计算 Laplacian
    lap = fd.compute_laplacian(c_old, dx, scheme=scheme)

    # 计算反应源项
    R = compute_reaction_source(c_old, phi, dx)

    # FTCS 推进（内部节点）
    for i in range(1, n - 1):
        c_new[i] = c_old[i] + dt * (d_eff * lap[i] + R[i])

    # 浓度非负约束（数值鲁棒性）
    c_new = [max(ci, 0.0) for ci in c_new]

    cfl_actual = d_eff * dt / (dx * dx)
    return c_new, cfl_actual


# ============================================================
#  波动方程耦合：SEI 应力演化（参考 1402_wave_pde）
# ============================================================

def stress_wave_step(u_disp, v_vel, dx, dt, c_wave):
    """
    SEI 应力波传播的单步时间推进（参考 1402_wave_pde）。

    波动方程：
        ∂²u/∂t² = c² ∂²u/∂x²
    化为一阶系统：
        ∂u/∂t = v
        ∂v/∂t = c² ∂²u/∂x²

    其中 c 为 SEI 中弹性波速。

    Parameters
    ----------
    u_disp : list[float]
        位移场 [m]。
    v_vel : list[float]
        速度场 [m/s]。
    dx : float
        空间步长。
    dt : float
        时间步长。
    c_wave : float
        波速 [m/s]。

    Returns
    -------
    u_new, v_new : list[float]
        下一时间步的位移和速度。
    cfl_wave : float
        波动方程 CFL 数。
    """
    n = len(u_disp)
    lap_u = fd.laplacian_2nd_order(u_disp, dx, periodic=False)

    u_new = list(u_disp)
    v_new = list(v_vel)

    cfl_wave = c_wave * dt / dx

    # 时间推进（leapfrog-like）
    for i in range(1, n - 1):
        u_new[i] = u_disp[i] + dt * v_vel[i]
        v_new[i] = v_vel[i] + dt * c_wave * c_wave * lap_u[i]

    # 边界：固定位移（SEI 粘附于电极）
    u_new[0] = 0.0
    u_new[n - 1] = 0.0
    v_new[0] = 0.0
    v_new[n - 1] = 0.0

    return u_new, v_new, cfl_wave


# ============================================================
#  完整仿真流程
# ============================================================

def run_diffusion_simulation(n_steps=None, fd_scheme="2nd_order",
                               coupled_wave=False):
    """
    运行 SEI Li+ 扩散-反应耦合仿真。

    Parameters
    ----------
    n_steps : int or None
        时间步数，None 使用默认值。
    fd_scheme : str
        差分格式。
    coupled_wave : bool
        是否耦合应力波。

    Returns
    -------
    dict
        仿真结果（浓度场历史、CFL、质量守恒等）。
    """
    if n_steps is None:
        n_steps = P.N_STEPS

    x, dx = [i * P.DX for i in range(P.N_GRID)], P.DX
    dt = P.DT_EXPLICIT

    # 初始场
    c = initial_concentration_profile(x, P.C_LI_INIT, P.C_LI_INIT * 0.8)
    phi = initial_potential_profile(x, 0.0, P.U_EQ_SEI)

    # 波动场（可选）
    if coupled_wave:
        u_disp = [0.0] * P.N_GRID
        v_vel = [0.0] * P.N_GRID

    # 时间推进
    c_history = [list(c)]
    cfl_history = []
    mass_history = []

    for step in range(n_steps):
        c, cfl = explicit_time_step(c, phi, dx, dt, P.D_LI_SEI,
                                     scheme=fd_scheme)

        # 施加边界条件
        c = apply_boundary_conditions(
            c,
            bc_left_type="neumann", bc_left_value=0.0,  # 绝缘
            bc_right_type="dirichlet", bc_right_value=P.C_LI_INIT
        )

        cfl_history.append(cfl)
        # 总质量（浓度积分）
        mass = sum(c) * dx
        mass_history.append(mass)

        # 每 10 步记录一次
        if step % 10 == 0 or step == n_steps - 1:
            c_history.append(list(c))

        # 耦合应力波
        if coupled_wave:
            u_disp, v_vel, cfl_w = stress_wave_step(
                u_disp, v_vel, dx, dt * 0.1, P.C_WAVE_SEI
            )

    return {
        "x": x,
        "c_final": c,
        "c_history": c_history,
        "cfl_history": cfl_history,
        "mass_history": mass_history,
        "phi": phi,
    }


if __name__ == "__main__":
    result = run_diffusion_simulation(n_steps=50, fd_scheme="2nd_order")
    print(f"[sei_diffusion] 仿真完成")
    print(f"  最终浓度范围: [{min(result['c_final']):.4f}, "
          f"{max(result['c_final']):.4f}] mol/m^3")
    print(f"  最终 CFL 数: {result['cfl_history'][-1]:.4f}")
    print(f"  初始质量: {result['mass_history'][0]:.6e}")
    print(f"  最终质量: {result['mass_history'][-1]:.6e}")
