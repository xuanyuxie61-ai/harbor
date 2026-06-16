"""
collision_operator.py - 碰撞算子 (扩散方程求解)

本模块实现电子-离子碰撞算子，基于扩散方程方法。

在激光等离子体相互作用中，碰撞过程影响:
  1. 逆韧致吸收 (激光能量转化为电子热能)
  2. 电子热传导
  3. 分布函数的热化 (趋向 Maxwell 分布)
  4. 等离子体电阻率

碰撞算子的 Fokker-Planck 形式:
    C[f] = nu * d/dv [v * f + (v_th^2/2) * df/dv]

  其中 nu 为碰撞频率, 第一项为摩擦 ( slowing-down), 第二项为扩散。

扩散方程近似:
    将碰撞算子简化为速度空间的扩散方程:
        df/dt = D_v * d^2f/dv^2
    其中 D_v = nu * v_th^2 / 2 为速度空间扩散系数。

  结合空间扩散 (热传导):
        df/dt = D_v * d^2f/dv^2 + D_x * d^2f/dx^2

  数值方法: 采用方法线 (Method of Lines),
  将 PDE 转化为 ODE 系统, 然后用显式或隐式方法推进。

Landau 碰撞频率:
    nu_ei = n_i * Z^2 * e^4 * ln(Lambda) / (4*pi*eps_0^2 * m_e^2 * v_th^3)

  其中 ln(Lambda) 为 Coulomb 对数:
    ln(Lambda) = ln(12*pi * n_e * lambda_D^3 / Z)

Spitzer 电阻率:
    eta_Sp = m_e * nu_ei / (n_e * e^2)
"""

import numpy as np


def coulomb_logarithm(n_e, T_e):
    """计算 Coulomb 对数 ln(Lambda)。

    Coulomb 对数来自对 Rutherford 散射截面积分:
        ln(Lambda) = ln(b_max / b_min)

    其中:
        b_max = lambda_D = sqrt(eps_0 * k_B * T_e / (n_e * e^2))  德拜屏蔽长度
        b_min = max(b_90, lambda_deBroglie)  最小碰撞参数
        b_90 = e^2 / (4*pi*eps_0 * k_B * T_e)  90度偏转距离
        lambda_dB = hbar / sqrt(2*m_e*k_B*T_e)  量子波长

    典型值: ln(Lambda) ~ 10-20 (弱耦合等离子体)

    Parameters
    ----------
    n_e : float
        电子数密度 [m^-3]
    T_e : float
        电子温度 [J]

    Returns
    -------
    ln_Lambda : float
        Coulomb 对数
    """
    pc = {
        'e': 1.602e-19,
        'm_e': 9.109e-31,
        'eps_0': 8.854e-12,
        'k_B': 1.381e-23,
        'hbar': 1.055e-34,
    }

    # 德拜长度
    lambda_D = np.sqrt(pc['eps_0'] * T_e / (n_e * pc['e'] ** 2))

    # 90度偏转距离
    b_90 = pc['e'] ** 2 / (4.0 * np.pi * pc['eps_0'] * T_e)

    # de Broglie 波长
    v_th = np.sqrt(T_e / pc['m_e'])
    lambda_dB = pc['hbar'] / (pc['m_e'] * v_th) if v_th > 0 else 0.0

    b_min = max(b_90, lambda_dB)

    if b_min > 0 and lambda_D > b_min:
        ln_Lambda = np.log(lambda_D / b_min)
    else:
        ln_Lambda = 10.0  # 典型值

    return max(ln_Lambda, 2.0)  # 下限


def electron_ion_collision_frequency(n_e, T_e, Z=1):
    """计算电子-离子碰撞频率。

    Landau 碰撞频率:
        nu_ei = n_i * Z^2 * e^4 * ln(Lambda) /
                (4*pi * eps_0^2 * m_e^2 * v_th^3)

    简化形式:
        nu_ei = 4*sqrt(2*pi) * n_e * Z * e^4 * ln(Lambda) /
                (3 * (4*pi*eps_0)^2 * sqrt(m_e) * (k_B*T_e)^{3/2})

    Parameters
    ----------
    n_e : float
        电子数密度 [m^-3]
    T_e : float
        电子温度 [J]
    Z : int
        离子电荷数

    Returns
    -------
    nu_ei : float
        碰撞频率 [s^-1]
    """
    pc = {
        'e': 1.602e-19,
        'm_e': 9.109e-31,
        'eps_0': 8.854e-12,
    }

    ln_Lambda = coulomb_logarithm(n_e, T_e)

    nu_ei = (
        4.0 * np.sqrt(2.0 * np.pi) * n_e * Z * pc['e'] ** 4 * ln_Lambda /
        (3.0 * (4.0 * np.pi * pc['eps_0']) ** 2 * np.sqrt(pc['m_e']) * T_e ** 1.5)
    )

    return nu_ei


def spitzer_resistivity(n_e, T_e, Z=1):
    """计算 Spitzer 电阻率。

    eta_Sp = m_e * nu_ei / (n_e * e^2)

    或等价地:
        eta_Sp = Z * e^2 * ln(Lambda) /
                 (3 * (2*pi)^{1/2} * eps_0^2 * sqrt(m_e) * (k_B*T_e)^{3/2})

    典型值 (日冕等离子体, T=1MK): eta ~ 1e-5 Ohm*m

    Parameters
    ----------
    n_e : float
        电子密度 [m^-3]
    T_e : float
        电子温度 [J]
    Z : int
        离子电荷数

    Returns
    -------
    eta : float
        Spitzer 电阻率 [Ohm*m]
    """
    m_e = 9.109e-31
    e = 1.602e-19

    nu_ei = electron_ion_collision_frequency(n_e, T_e, Z)
    eta = m_e * nu_ei / (n_e * e ** 2)
    return eta


def collision_operator_fokker_planck(f, v, dv, nu_ei, v_th):
    """Fokker-Planck 碰撞算子。

    C[f] = nu * d/dv [v * f + (v_th^2/2) * df/dv]

    第一项: 摩擦 (slowing-down), 使分布向低速移动
    第二项: 速度空间扩散, 使分布展宽

    平衡态: 当 C[f] = 0 时, f 为 Maxwell 分布:
        f_eq ~ exp(-v^2 / (2*v_th^2))

    离散化 (中心差分):
        C[f]_j = nu * {
            (v_{j+1}*f_{j+1} - v_{j-1}*f_{j-1}) / (2*dv)
            + (v_th^2/2) * (f_{j+1} - 2*f_j + f_{j-1}) / dv^2
        }

    Parameters
    ----------
    f : ndarray (N_v,)
        分布函数
    v : ndarray (N_v,)
        速度网格
    dv : float
        速度步长
    nu_ei : float
        碰撞频率
    v_th : float
        热速度

    Returns
    -------
    C_f : ndarray (N_v,)
        碰撞算子作用结果
    """
    N_v = len(f)
    C_f = np.zeros(N_v)

    D_v = 0.5 * v_th ** 2  # 速度空间扩散系数

    for j in range(1, N_v - 1):
        # 摩擦项: d/dv (v * f)
        friction = (v[j + 1] * f[j + 1] - v[j - 1] * f[j - 1]) / (2.0 * dv)
        # 扩散项: D_v * d^2f/dv^2
        diffusion = D_v * (f[j + 1] - 2.0 * f[j] + f[j - 1]) / dv ** 2

        C_f[j] = nu_ei * (friction + diffusion)

    # 边界: 零通量 (Neumann)
    C_f[0] = C_f[1]
    C_f[-1] = C_f[-2]

    return C_f


def solve_collisional_relaxation(f_init, v, dv, nu_ei, v_th, n_steps, dt_coll):
    """求解碰撞弛豫过程。

    时间推进: f^{n+1} = f^n + dt * C[f^n]

    显式 Euler 格式的稳定性条件:
        dt < dv^2 / (2 * D_v * nu_ei) = dv^2 / (nu_ei * v_th^2)

    Parameters
    ----------
    f_init : ndarray (N_v,)
        初始分布函数
    v : ndarray (N_v,)
        速度网格
    dv : float
        速度步长
    nu_ei : float
        碰撞频率
    v_th : float
        热速度
    n_steps : int
        时间步数
    dt_coll : float
        碰撞时间步长

    Returns
    -------
    f_final : ndarray (N_v,)
        弛豫后的分布函数
    history : dict
        弛豫历史
    """
    f = f_init.copy()
    N_v = len(v)

    # 稳定性检查
    D_v = 0.5 * v_th ** 2
    dt_max = dv ** 2 / (2.0 * D_v * max(nu_ei, 1.0e-10))
    if dt_coll > dt_max:
        dt_coll = 0.9 * dt_max  # 安全因子

    # 初始矩
    density_init = np.sum(f) * dv
    energy_init = np.sum(f * v ** 2) * dv

    # 弛豫历史
    densities = [np.sum(f) * dv]
    energies = [np.sum(f * v ** 2) * dv]
    temperatures = [energies[-1] / max(densities[-1], 1e-30)]
    entropies = [_compute_entropy(f, dv)]

    for step in range(n_steps):
        C_f = collision_operator_fokker_planck(f, v, dv, nu_ei, v_th)

        # 确保非负
        f_new = f + dt_coll * C_f
        f = np.maximum(f_new, 0.0)

        if (step + 1) % max(1, n_steps // 20) == 0:
            densities.append(np.sum(f) * dv)
            energies.append(np.sum(f * v ** 2) * dv)
            temperatures.append(energies[-1] / max(densities[-1], 1e-30))
            entropies.append(_compute_entropy(f, dv))

    history = {
        'densities': np.array(densities),
        'energies': np.array(energies),
        'temperatures': np.array(temperatures),
        'entropies': np.array(entropies),
        'density_conservation': abs(densities[-1] - density_init) / max(abs(density_init), 1e-30),
        'energy_conservation': abs(energies[-1] - energy_init) / max(abs(energy_init), 1e-30),
    }

    return f, history


def _compute_entropy(f, dv):
    """计算分布函数的 Boltzmann 熵。

    S = -integral f * ln(f) dv

    碰撞应使熵单调增加 (H 定理)。

    Parameters
    ----------
    f : ndarray
        分布函数
    dv : float
        速度步长

    Returns
    -------
    entropy : float
        熵值
    """
    f_safe = np.maximum(f, 1.0e-30)
    return -np.sum(f_safe * np.log(f_safe)) * dv


def inverse_bremsstrahlung_absorption(n_e, T_e, omega_L, Z=1):
    """计算逆韧致吸收率。

    逆韧致吸收是激光能量通过碰撞转化为电子热能的过程:
        Q_IB = nu_ei * eps_0 * |E|^2 * omega_p^2 / (omega_L^2 + nu_ei^2)

    吸收系数 (单位长度):
        alpha_IB = nu_ei * omega_p^2 / (c * omega_L^2)
                   * 1 / sqrt(1 - omega_p^2/omega_L^2)

    Parameters
    ----------
    n_e : float
        电子密度
    T_e : float
        电子温度 [J]
    omega_L : float
        激光频率
    Z : int
        离子电荷数

    Returns
    -------
    absorption_rate : dict
        吸收率信息
    """
    e = 1.602e-19
    m_e = 9.109e-31
    eps_0 = 8.854e-12
    c = 2.998e8

    nu_ei = electron_ion_collision_frequency(n_e, T_e, Z)
    omega_p = np.sqrt(n_e * e ** 2 / (m_e * eps_0))

    # 吸收系数
    ratio = omega_p / omega_L
    if ratio >= 1.0:
        # 过密: 激光不能传播
        return {
            'nu_ei': nu_ei,
            'omega_p': omega_p,
            'alpha_IB': float('inf'),
            'propagates': False,
        }

    alpha_IB = nu_ei * omega_p ** 2 / (c * omega_L ** 2 * np.sqrt(1.0 - ratio ** 2))

    return {
        'nu_ei': nu_ei,
        'omega_p': omega_p,
        'alpha_IB': alpha_IB,
        'propagates': True,
        'ratio': ratio,
    }
