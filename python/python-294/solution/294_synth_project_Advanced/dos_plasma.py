"""
dos_plasma.py - 等离子体电磁模式态密度

本模块计算等离子体中电磁模式的态密度 (Density of States, DOS)。
态密度是分析激光能量耦合到等离子体模式的核心物理量。

物理背景:

  在等离子体中，电磁波的色散关系为:
      omega^2 = omega_p^2 + c^2 * k^2  (横波/光波)
      omega^2 = omega_p^2 + 3*k^2*v_th^2  (纵波/Langmuir 波)

  态密度定义为:
      D(omega) = sum_n delta(omega - omega_n)
  其中 omega_n 为第 n 个允许的模式频率。

  对于连续近似:
      D(omega) = L/(2*pi) * dk/domega

  对于横波: dk/domega = omega / (c^2 * k) = omega / (c * sqrt(omega^2 - omega_p^2))
      D_T(omega) = L * omega / (2*pi*c*sqrt(omega^2 - omega_p^2))  for omega > omega_p

  对于纵波 (Bohm-Gross): dk/domega = omega / (3*v_th^2*k)
      D_L(omega) = L * omega / (2*pi*3*v_th^2*k_L)

  在非均匀等离子体中，模式结构更复杂:
    - 存在截断面 (cutoff): omega = omega_p, 波不能传播
    - 存在共振面 (resonance): omega = omega_UH (上混合频率)
    - 存在表面模式 (surface modes)
    - 存在消逝模 (evanescent modes)

  态密度的计算通过求解色散关系的允许模式实现:
    1. 在频率空间搜索色散关系的根
    2. 对每个允许模式计算归一化常数
    3. 态密度 = sum_n |E_n|^2 * delta(omega - omega_n)

方法 (源自光子态密度计算):
  - 搜索色散函数 sign change 确定模式位置
  - Brent 方法精确求根
  - 模式归一化通过能量积分实现
"""

import numpy as np
from scipy.optimize import brentq


def epsilon_cold_plasma(omega, omega_p, nu_coll=0.0):
    """冷等离子体介电函数。

    无碰撞:
        eps(omega) = 1 - omega_p^2 / omega^2

    有碰撞 (Drude 模型):
        eps(omega) = 1 - omega_p^2 / (omega * (omega + i*nu))

    Parameters
    ----------
    omega : complex or float
        角频率
    omega_p : float
        等离子体频率
    nu_coll : float
        碰撞频率

    Returns
    -------
    eps : complex
        介电函数值
    """
    if nu_coll > 0:
        return 1.0 - omega_p ** 2 / (omega * (omega + 1j * nu_coll))
    else:
        return 1.0 - omega_p ** 2 / omega ** 2


def em_dispersion_relation(omega, k, omega_p):
    """电磁波色散关系残差。

    横波色散关系:
        D(omega, k) = omega^2 - omega_p^2 - c^2*k^2 = 0

    归一化形式 (c=1, omega_p 归一化):
        D = omega^2 - omega_p^2 - k^2

    Parameters
    ----------
    omega : float
        角频率
    k : float
        波数
    omega_p : float
        局部等离子体频率

    Returns
    -------
    residual : float
        色散关系残差
    """
    return omega ** 2 - omega_p ** 2 - k ** 2


def langmuir_dispersion(omega, k, omega_p, v_th=1.0):
    """Langmuir 波 (纵波) 色散关系。

    Bohm-Gross 色散关系:
        D(omega, k) = omega^2 - omega_p^2 - 3*k^2*v_th^2 = 0

    热修正到 更高阶:
        omega^2 = omega_p^2 + 3*k^2*v_th^2 + 9*k^4*v_th^4/omega_p^2 + ...

    Parameters
    ----------
    omega : float
        角频率
    k : float
        波数
    omega_p : float
        等离子体频率
    v_th : float
        热速度

    Returns
    -------
    residual : float
        色散关系残差
    """
    return omega ** 2 - omega_p ** 2 - 3.0 * k ** 2 * v_th ** 2


def find_allowed_modes_1d(omega_range, k, omega_p_profile, mode_type='em'):
    """在给定波数下搜索允许的电磁模式频率。

    在非均匀等离子体中，对于给定的 k, 色散关系:
        D(omega; x) = omega^2 - omega_p(x)^2 - k^2 = 0

    允许的频率取决于局部等离子体频率。通过扫描频率空间,
    找到色散函数的过零点来确定允许模式。

    Parameters
    ----------
    omega_range : ndarray
        搜索频率范围
    k : float
        波数
    omega_p_profile : ndarray
        空间依赖的等离子体频率
    mode_type : str
        模式类型 ('em' 或 'langmuir')

    Returns
    -------
    omega_modes : list
        允许的模式频率列表
    """
    omega_modes = []

    # 平均色散函数
    omega_p_avg = np.mean(omega_p_profile)

    if mode_type == 'em':
        def dispersion(omega):
            return em_dispersion_relation(omega, k, omega_p_avg)
    else:
        def dispersion(omega):
            return langmuir_dispersion(omega, k, omega_p_avg)

    # 搜索过零点
    disp_values = np.array([dispersion(w) for w in omega_range])
    sign_changes = np.where(np.diff(np.sign(disp_values)))[0]

    for idx in sign_changes:
        try:
            omega_root = brentq(
                dispersion,
                omega_range[idx],
                omega_range[idx + 1],
                xtol=1.0e-12
            )
            omega_modes.append(omega_root)
        except ValueError:
            pass

    return omega_modes


def compute_em_dos(omega_array, k_array, omega_p, nu_coll=0.01):
    """计算电磁模式的态密度。

    对于均匀等离子体中的横波:
        D_T(omega) = (L / 2*pi) * sum_k delta(omega - omega_T(k))

    其中 omega_T(k) = sqrt(omega_p^2 + k^2)

    用 Lorentzian 展宽的 delta 函数:
        delta_eps(x) = (1/pi) * eps / (x^2 + eps^2)

    Parameters
    ----------
    omega_array : ndarray
        频率数组
    k_array : ndarray
        波数数组
    omega_p : float
        等离子体频率
    nu_coll : float
        碰撞展宽参数

    Returns
    -------
    dos : ndarray
        态密度 D(omega)
    dos_em : ndarray
        电磁模式贡献
    """
    dos = np.zeros_like(omega_array)
    eps = max(nu_coll, 1.0e-6)

    for k in k_array:
        omega_k = np.sqrt(omega_p ** 2 + k ** 2)
        # Lorentzian 展宽的 delta 函数
        dos += (1.0 / np.pi) * eps / ((omega_array - omega_k) ** 2 + eps ** 2)

    # 归一化
    dk = k_array[1] - k_array[0] if len(k_array) > 1 else 1.0
    dos *= dk / (2.0 * np.pi)

    dos_em = dos.copy()
    return dos, dos_em


def compute_langmuir_dos(omega_array, k_array, omega_p, v_th=1.0, nu_coll=0.01):
    """计算 Langmuir 波模式的态密度。

    Bohm-Gross 色散: omega_L(k) = sqrt(omega_p^2 + 3*k^2*v_th^2)

    Parameters
    ----------
    omega_array : ndarray
        频率数组
    k_array : ndarray
        波数数组
    omega_p : float
        等离子体频率
    v_th : float
        热速度
    nu_coll : float
        碰撞展宽

    Returns
    -------
    dos : ndarray
        Langmuir 波态密度
    """
    dos = np.zeros_like(omega_array)
    eps = max(nu_coll, 1.0e-6)

    for k in k_array:
        omega_k = np.sqrt(omega_p ** 2 + 3.0 * k ** 2 * v_th ** 2)
        dos += (1.0 / np.pi) * eps / ((omega_array - omega_k) ** 2 + eps ** 2)

    dk = k_array[1] - k_array[0] if len(k_array) > 1 else 1.0
    dos *= dk / (2.0 * np.pi)

    return dos


def compute_total_dos(omega_array, k_array, omega_p, v_th=1.0, nu_coll=0.01):
    """计算总的态密度 (电磁 + Langmuir)。

    D_total(omega) = D_em(omega) + D_langmuir(omega)

    总态密度决定了等离子体可接受能量的模式数量,
    是计算激光能量吸收效率的基础。

    Parameters
    ----------
    omega_array : ndarray
        频率数组
    k_array : ndarray
        波数数组
    omega_p : float
        等离子体频率
    v_th : float
        热速度
    nu_coll : float
        碰撞展宽

    Returns
    -------
    dos_total : ndarray
        总态密度
    dos_em : ndarray
        电磁部分
    dos_langmuir : ndarray
        Langmuir 部分
    """
    dos_em = compute_em_dos(omega_array, k_array, omega_p, nu_coll)[0]
    dos_langmuir = compute_langmuir_dos(omega_array, k_array, omega_p, v_th, nu_coll)
    dos_total = dos_em + dos_langmuir

    return dos_total, dos_em, dos_langmuir


def dos_frequency_integral(dos, omega_array, weight_func=None):
    """对态密度进行频率积分。

    计算物理量:
        Q = integral D(omega) * W(omega) domega

    其中 W(omega) 为权重函数。常见应用:
      - W=1: 总模式数
      -W=omega: 总能量 (每个模式贡献 hbar*omega)
      -W=omega^2: 辐射功率

    Parameters
    ----------
    dos : ndarray
        态密度
    omega_array : ndarray
        频率数组
    weight_func : callable, optional
        权重函数 W(omega)。默认为 W=1。

    Returns
    -------
    integral : float
        积分结果
    """
    if weight_func is None:
        integrand = dos
    else:
        integrand = dos * weight_func(omega_array)

    return np.trapz(integrand, omega_array)


def plasma_dos_analysis(config):
    """执行完整的等离子体态密度分析。

    Parameters
    ----------
    config : SimulationConfig
        仿真配置

    Returns
    -------
    results : dict
        态密度分析结果
    """
    # 频率和波数网格
    omega_array = np.linspace(0.1, 3.0, 500)
    k_array = np.linspace(0.01, 5.0, 200)

    omega_p = np.sqrt(config.plasma_density_profile().max())

    # 计算各模式态密度
    dos_total, dos_em, dos_langmuir = compute_total_dos(
        omega_array, k_array, omega_p, v_th=1.0, nu_coll=0.01
    )

    # 频率积分
    total_modes = dos_frequency_integral(dos_total, omega_array)
    total_energy = dos_frequency_integral(dos_total, omega_array, lambda w: w)

    # 搜索特定 k 下的允许模式
    k_test = 1.0
    omega_p_profile = np.sqrt(config.plasma_density_profile())
    em_modes = find_allowed_modes_1d(omega_array, k_test, omega_p_profile, 'em')
    lm_modes = find_allowed_modes_1d(omega_array, k_test, omega_p_profile, 'langmuir')

    results = {
        'omega': omega_array,
        'k': k_array,
        'dos_total': dos_total,
        'dos_em': dos_em,
        'dos_langmuir': dos_langmuir,
        'total_modes': total_modes,
        'total_energy_weighted': total_energy,
        'em_modes_at_k1': em_modes,
        'langmuir_modes_at_k1': lm_modes,
        'omega_p_max': omega_p,
    }

    return results
