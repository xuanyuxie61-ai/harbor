"""
vlasov_maxwell.py - Vlasov-Maxwell 方程求解器

本模块实现 1D1V Vlasov-Maxwell 方程组的数值求解,
这是描述激光等离子体相互作用最基本的动力学模型。

Vlasov-Maxwell 方程组:

  1. Vlasov 方程 (电子分布函数演化):
     df/dt + v * df/dx - (e/m_e)(E + v x B) * df/dv = C[f]

     在 1D1V 简化 (x, v_x, E_x, E_y, B_z) 中:
         df/dt + v_x * df/dx - (e/m_e) * (E_x + v_y * B_z) * df/dv_x = C[f]

  2. Maxwell 方程:
     dE_x/dt = -(1/eps_0) * J_x + (激光源项)
     dE_y/dt = c^2 * dB_z/dx - (1/eps_0) * J_y
     dB_z/dt = -dE_y/dx

     其中电流密度:
         J_x = -e * integral v_x * f dv_x
         J_y = -e * integral v_y * f dv_x

  3. 泊松方程 (高斯定律):
     dE_x/dx = (rho - rho_0) / eps_0
     其中 rho = -e * integral f dv_x

数值方法:

  空间离散: 高阶有限差分 (见 fd_operators.py)
  速度离散: 谱方法或高阶有限差分
  时间积分: 5 阶低存储 Runge-Kutta (RK45-LS)

  DG (Discontinuous Galerkin) 方法:
    在单元接口使用数值通量 (Lax-Friedrichs):
        F_hat = {F} - (alpha/2) * [[u]]
    其中 {F} = (F^+ + F^-)/2 为平均通量,
    [[u]] = u^+ - u^- 为跳跃,
    alpha = max|df/dv| 为局部最大特征速度。

  RK45 低存储格式 (5 级):
    使用 Butcher 表的 4 阶 5 级显式 Runge-Kutta:
      u^(1) = u^n
      u^(2) = u^(1) + dt * a_21 * L(u^(1))
      u^(3) = u^(2) + dt * a_32 * L(u^(2))
      u^(4) = u^(3) + dt * a_43 * L(u^(3))
      u^(5) = u^(4) + dt * a_54 * L(u^(4))
      u^{n+1} = u^(5) + dt * b_5 * L(u^(5))
"""

import numpy as np
from fd_operators import apply_first_derivative, apply_second_derivative


# RK45 低存储 Butcher 系数 (4 阶 5 级)
_RK45_A = [0.0, 0.39175222657471, 0.58632491897783, 0.47454236302687, 0.93404087898582]
_RK45_B = [
    [0.0],
    [0.39175222657471],
    [-0.07206434765828, 0.65838926663611],
    [0.10260404689682, -0.27146982582468, 0.64294812143873],
    [-0.16380384076049, 0.57926405527598, -1.21910404742266, 1.38631100971228],
]
_RK45_C = [0.0, 0.39175222657471, 0.58632491897783, 0.47454236302687, 0.93404087898582]
_RK45_BFINAL = [-0.03717369225409, 0.22382196000290, -0.53452826649782, 1.38590775017321, -0.03802777142420]


def compute_current_density(f, v, dv):
    """从分布函数计算电流密度。

    J(x) = -e * integral v * f(x, v) dv

    在归一化单位中 (e=1):
        J(x) = -sum_j v_j * f(x, v_j) * dv

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        分布函数
    v : ndarray (N_v,)
        速度网格
    dv : float
        速度步长

    Returns
    -------
    J : ndarray (N_x,)
        电流密度
    """
    return -np.sum(f * v[np.newaxis, :], axis=1) * dv


def compute_charge_density(f, dv, n_background=None):
    """从分布函数计算电荷密度。

    rho(x) = -e * integral f(x, v) dv + rho_background

    在归一化单位中:
        rho(x) = -sum_j f(x, v_j) * dv + n_background(x)

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        分布函数
    dv : float
        速度步长
    n_background : ndarray (N_x,), optional
        背景离子密度 (准中性假设)

    Returns
    -------
    rho : ndarray (N_x,)
        净电荷密度
    """
    n_electron = np.sum(f, axis=1) * dv
    if n_background is None:
        n_background = n_electron  # 准中性
    rho = n_background - n_electron
    return rho


def vlasov_rhs_v_x_dfdx(f, v, dx, fd_order=4, bc='periodic'):
    """计算 Vlasov 方程中的空间对流项 -v * df/dx。

    对于每个速度 v_j:
        (-v_j * df/dx)|_j = -v_j * D_h f(x, v_j)

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        分布函数
    v : ndarray (N_v,)
        速度网格
    dx : float
        空间步长
    fd_order : int
        有限差分阶数
    bc : str
        边界条件

    Returns
    -------
    rhs : ndarray (N_x, N_v)
        对流项贡献
    """
    N_v = len(v)
    rhs = np.zeros_like(f)

    for j in range(N_v):
        dfdx = apply_first_derivative(f[:, j], dx, order=fd_order, bc=bc)
        rhs[:, j] = -v[j] * dfdx

    return rhs


def vlasov_rhs_E_dfdfv(f, E_field, v, dv, fd_order=4):
    """计算 Vlasov 方程中的电场加速项 -(e/m)*E*df/dv。

    对于每个空间位置 x_i:
        -(e/m)*E(x_i) * df/dv(x_i, v) = -E(x_i) * D_v f(x_i, v)

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        分布函数
    E_field : ndarray (N_x,)
        电场
    v : ndarray (N_v,)
        速度网格
    dv : float
        速度步长
    fd_order : int
        速度空间差分阶数

    Returns
    -------
    rhs : ndarray (N_x, N_v)
        电场加速项贡献
    """
    N_x, N_v = f.shape
    rhs = np.zeros_like(f)

    # 速度空间导数 (使用中心差分)
    for i in range(N_x):
        if N_v < 3:
            continue
        dfdv = np.zeros(N_v)
        p = fd_order // 2
        for j in range(p, N_v - p):
            if fd_order == 2:
                dfdv[j] = (f[i, j + 1] - f[i, j - 1]) / (2.0 * dv)
            elif fd_order == 4:
                dfdv[j] = (
                    -f[i, j + 2] + 8.0 * f[i, j + 1]
                    - 8.0 * f[i, j - 1] + f[i, j - 2]
                ) / (12.0 * dv)
            else:
                dfdv[j] = (f[i, min(j + 1, N_v - 1)] - f[i, max(j - 1, 0)]) / (2.0 * dv)

        # 边界
        if N_v > 2:
            dfdv[0] = dfdv[1]
            dfdv[-1] = dfdv[-2]

        rhs[i, :] = -E_field[i] * dfdv

    return rhs


def maxwell_rhs(E_y, B_z, J_y, dx, fd_order=4):
    """计算 Maxwell 方程的右端项。

    dE_y/dt = c^2 * dB_z/dx - J_y/eps_0
    dB_z/dt = -dE_y/dx

    在归一化单位中 (c=1, eps_0=1):
        dE_y/dt = dB_z/dx - J_y
        dB_z/dt = -dE_y/dx

    Parameters
    ----------
    E_y : ndarray (N_x,)
        横向电场
    B_z : ndarray (N_x,)
        磁场
    J_y : ndarray (N_x,)
        横向电流密度
    dx : float
        空间步长
    fd_order : int
        有限差分阶数

    Returns
    -------
    dE_y_dt : ndarray
        E_y 的时间导数
    dB_z_dt : ndarray
        B_z 的时间导数
    """
    dE_y_dt = apply_first_derivative(B_z, dx, order=fd_order) - J_y
    dB_z_dt = -apply_first_derivative(E_y, dx, order=fd_order)

    return dE_y_dt, dB_z_dt


def lax_friedrichs_flux(f_left, f_right, v_advect):
    """Lax-Friedrichs 数值通量。

    用于 Vlasov 方程中速度空间对流的 DG 方法。

    F_hat = (F(u_L) + F(u_R))/2 - (alpha/2) * (u_R - u_L)

    其中 F(u) = a * u 为通量函数,
    alpha = max|a| 为最大特征速度。

    Parameters
    ----------
    f_left : ndarray
        左侧状态
    f_right : ndarray
        右侧状态
    v_advect : float
        对流速度

    Returns
    -------
    flux : ndarray
        数值通量
    """
    alpha = abs(v_advect)
    return 0.5 * v_advect * (f_left + f_right) - 0.5 * alpha * (f_right - f_left)


def dg_vlasov_step(f, v, x, dx, dv, E_field, dt, fd_order=4):
    """执行一步 DG Vlasov 求解。

    组合空间对流和电场加速:
        df/dt = -v * df/dx - E * df/dv

    使用算子分裂: 先推进空间对流半步, 再推进电场加速半步。

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        当前分布函数
    v : ndarray (N_v,)
        速度网格
    x : ndarray (N_x,)
        空间网格
    dx : float
        空间步长
    dv : float
        速度步长
    E_field : ndarray (N_x,)
        电场
    dt : float
        时间步长
    fd_order : int
        差分阶数

    Returns
    -------
    f_new : ndarray (N_x, N_v)
        更新后的分布函数
    """
    # Strang 分裂: 半步空间 + 全步电场 + 半步空间
    # 半步空间对流
    rhs_x = vlasov_rhs_v_x_dfdx(f, v, dx, fd_order)
    f_half = f + 0.5 * dt * rhs_x
    f_half = np.maximum(f_half, 0.0)

    # 全步电场加速
    rhs_E = vlasov_rhs_E_dfdfv(f_half, E_field, v, dv, fd_order)
    f_new = f_half + dt * rhs_E
    f_new = np.maximum(f_new, 0.0)

    # 半步空间对流
    rhs_x2 = vlasov_rhs_v_x_dfdx(f_new, v, dx, fd_order)
    f_new = f_new + 0.5 * dt * rhs_x2
    f_new = np.maximum(f_new, 0.0)

    return f_new


def maxwell_step(E_y, B_z, J_y, dx, dt, fd_order=4):
    """执行一步 Maxwell 方程求解。

    使用 Leapfrog 格式 (电场和磁场交错更新):
        E_y^{n+1} = E_y^n + dt * (dB_z^n/dx - J_y^n)
        B_z^{n+1/2} = B_z^{n-1/2} - dt * dE_y^n/dx

    Parameters
    ----------
    E_y : ndarray
        横向电场
    B_z : ndarray
        磁场
    J_y : ndarray
        电流
    dx : float
        空间步长
    dt : float
        时间步长
    fd_order : int
        差分阶数

    Returns
    -------
    E_y_new : ndarray
        更新的电场
    B_z_new : ndarray
        更新的磁场
    """
    dE_y_dt, dB_z_dt = maxwell_rhs(E_y, B_z, J_y, dx, fd_order)

    E_y_new = E_y + dt * dE_y_dt
    B_z_new = B_z + dt * dB_z_dt

    return E_y_new, B_z_new


def run_vlasov_maxwell_simulation(config, n_steps=None):
    """运行完整的 Vlasov-Maxwell 仿真。

    Parameters
    ----------
    config : SimulationConfig
        仿真配置
    n_steps : int, optional
        运行步数。默认使用配置中的 N_t。

    Returns
    -------
    results : dict
        仿真结果
    """
    from distribution_sampler import initialize_phase_space

    if n_steps is None:
        n_steps = min(config.N_t, 200)  # 限制步数以保证效率

    x = config.x
    v = config.v
    dx = config.dx
    dv = config.dv

    # 初始化分布函数
    f, phase_info = initialize_phase_space(config)

    # 初始化场
    n_profile = config.plasma_density_profile(x)
    E_x = np.zeros(config.N_x)
    E_y = np.zeros(config.N_x)
    B_z = np.zeros(config.N_x)

    # 激光驱动 (施加在左边界)
    t = 0.0

    # 记录历史
    history_interval = max(1, n_steps // 50)
    times = []
    field_energies = []
    kinetic_energies = []
    total_energies = []

    for step in range(n_steps):
        t = step * config.dt

        # 计算电流
        J_x = compute_current_density(f, v, dv)
        J_y = np.zeros(config.N_x)  # 简化: 只考虑纵向

        # 激光源项
        envelope = config.laser_envelope(t)
        E_y[0] += envelope * np.sin(config.omega_0 * t) * 0.1

        # Vlasov 步
        f = dg_vlasov_step(f, v, x, dx, dv, E_x, config.dt, config.fd_order)

        # Maxwell 步
        E_y_new, B_z_new = maxwell_step(E_y, B_z, J_y, dx, config.dt, config.fd_order)
        E_y = E_y_new
        B_z = B_z_new

        # 更新纵向电场 (泊松方程)
        rho = compute_charge_density(f, dv, n_profile)
        # 简化求解: E_x = -d(potential)/dx, d^2(phi)/dx^2 = -rho
        # 使用直接积分
        E_x = np.cumsum(rho) * dx
        E_x -= np.mean(E_x)

        # 记录
        if step % history_interval == 0:
            times.append(t)
            fe = 0.5 * np.sum(E_y ** 2 + B_z ** 2 + E_x ** 2) * dx
            ke = 0.5 * np.sum(f * v[np.newaxis, :] ** 2) * dv * dx
            field_energies.append(fe)
            kinetic_energies.append(ke)
            total_energies.append(fe + ke)

    results = {
        'f_final': f,
        'E_x': E_x,
        'E_y': E_y,
        'B_z': B_z,
        'times': np.array(times),
        'field_energies': np.array(field_energies),
        'kinetic_energies': np.array(kinetic_energies),
        'total_energies': np.array(total_energies),
        'phase_space_info': phase_info,
    }

    return results
