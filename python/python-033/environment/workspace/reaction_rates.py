"""
reaction_rates.py
r 过程核反应率计算模块

核心物理模型：
1. 中子捕获率 (n, gamma)：
   <sigma v>_{n,gamma} = (2 / sqrt(pi)) * (mu / (2kT))^{3/2} * integral_0^inf sigma(E) E exp(-E/kT) dE

2. 光致分解率 (gamma, n)：
   lambda_{gamma,n} = (2 / sqrt(pi)) * (m_n c^2 / (kT))^{3/2} * (G_f / G_i) * integral_0^inf sigma(E) E^2 exp(-E/kT) dE / (hbar^3 c^2)
   通过细致平衡：lambda_{gamma,n} = <sigma v>_{n,gamma} * (2m_n kT / hbar^2)^{3/2} * (G_f / G_i) * exp(-S_n/kT)

3. beta 衰变率：
   lambda_beta = ln(2) / T_{1/2}

4. 裂变率（对超重复元素）：
   lambda_fission = nu_f * n_n * sigma_f

温度依赖采用移位 Legendre 多项式展开（见 spectral_expansion.py）。
"""

import numpy as np
from spectral_expansion import spectral_expand_reaction_rate, spectral_evaluate_reaction_rate


# 物理常数
K_BOLTZMANN = 1.380649e-16  # erg/K
HBAR = 1.054571817e-27  # erg*s
AMU = 1.66053906660e-24  # g
C_LIGHT = 2.99792458e10  # cm/s
N_AVOGADRO = 6.02214076e23  # mol^{-1}


def reduced_mass(m1, m2):
    """计算约化质量 mu = m1*m2/(m1+m2)"""
    return m1 * m2 / (m1 + m2)


def neutron_capture_rate_mackeown(Z, A, T9, S_n, level_density_param=10.0):
    """
    基于 Hauser-Feshbach 统计模型的中子捕获率近似公式（Mackeown 公式）。

    参数:
        Z : int, 靶核质子数
        A : int, 靶核质量数
        T9 : float, 温度 (10^9 K)
        S_n : float, 中子分离能 (MeV)
        level_density_param : float, 能级密度参数 a (MeV^{-1})

    返回:
        rate : float, 热平均反应率 <sigma v> (cm^3/s)
    """
    T = T9 * 1e9  # K
    kT_MeV = K_BOLTZMANN * T * 6.241509e5  # 转换为 MeV (1 erg = 6.24e5 MeV)

    # Mackeown 近似：
    # <sigma v> ≈ C * (2J_f+1)/(2J_i+1) * exp(-S_n/kT) / (T9^{2/3})
    # 其中 C 为与核大小相关的常数
    # 这里使用简化的统计模型估计
    mu = reduced_mass(AMU, AMU)  # 中子-核系统约化质量（近似）
    thermal_wavelength = HBAR / np.sqrt(2.0 * np.pi * mu * K_BOLTZMANN * T)

    # 中子共振积分近似：
    # <sigma v> ≈ v_T * pi * lambda_bar^2 * (Gamma_n * Gamma_gamma / Gamma_total) * D^{-1}
    # 使用简化公式
    v_T = np.sqrt(2.0 * K_BOLTZMANN * T / AMU)
    de_broglie = HBAR / (AMU * v_T)

    # 统计因子（简化）
    g_factor = (A + 1.0) / A

    # 平均 S波 中子宽度与辐射宽度比
    gamma_ratio = 0.1 * (S_n / kT_MeV) ** 2

    # 能级间距 D ≈ exp(-2 sqrt(a*E*))
    E_star = S_n + kT_MeV
    level_spacing = np.exp(-2.0 * np.sqrt(level_density_param * E_star))

    # 捕获截面近似
    sigma_cap = 2.0 * np.pi * de_broglie ** 2 * g_factor * gamma_ratio / level_spacing
    rate = v_T * sigma_cap

    # 边界处理
    if np.isnan(rate) or np.isinf(rate) or rate < 0:
        rate = 1e-30
    return rate


def photodisintegration_rate(Z, A, T9, S_n, capture_rate):
    """
    通过细致平衡计算光致分解率。

    细致平衡关系：
        lambda_{gamma,n} = <sigma v>_{n,gamma} * (2 m_n kT / hbar^2)^{3/2} * (2J_i+1)/(2J_f+1) * exp(-S_n/kT)

    参数:
        Z, A : int
        T9 : float
        S_n : float, 中子分离能 (MeV)
        capture_rate : float, 中子捕获率

    返回:
        rate : float, 光致分解率 (s^{-1})
    """
    T = T9 * 1e9
    kT_MeV = K_BOLTZMANN * T * 6.241509e5

    # TODO [Hole 1]: 实现细致平衡公式计算光致分解率
    # 需要根据物理常数 AMU, K_BOLTZMANN, T, HBAR、中子分离能 S_n 和捕获率 capture_rate，
    # 依次计算：自旋统计因子 spin_factor、热德布罗意波长因子 thermal_factor，
    # 最终得到 rate = capture_rate * thermal_factor * spin_factor * exp(-S_n / kT_MeV)
    # 并进行数值边界处理（NaN/Inf/负值 -> 1e-30）
    raise NotImplementedError("Hole 1: photodisintegration_rate 核心公式待实现")


def beta_decay_rate(T_half):
    """
    beta 衰变率：lambda = ln(2) / T_{1/2}

    参数:
        T_half : float, 半衰期 (s)

    返回:
        rate : float, 衰变率 (s^{-1})
    """
    if T_half <= 0 or np.isnan(T_half) or np.isinf(T_half):
        return 1e-30
    return np.log(2.0) / T_half


def alpha_decay_rate(Z, A, Q_alpha):
    """
    基于 Gamow 因子的 alpha 衰变率近似。

    Gamow 因子：
        G = exp(-2 * integral_{r1}^{r2} sqrt(2mu(V(r)-E))/hbar dr)
    其中 V(r) = 2Z_d e^2 / r 为库仑势。

    简化公式（Geiger-Nuttall 定律）：
        log10(lambda) = a - b * Z / sqrt(Q_alpha)

    参数:
        Z, A : int
        Q_alpha : float, Q值 (MeV)

    返回:
        rate : float, alpha 衰变率 (s^{-1})
    """
    if Q_alpha <= 0.1:
        return 1e-30
    # Geiger-Nuttall 参数（简化）
    a_gn = -25.0
    b_gn = 1.5
    log_rate = a_gn - b_gn * Z / np.sqrt(Q_alpha)
    rate = 10.0 ** log_rate
    if np.isnan(rate) or rate < 0:
        rate = 1e-30
    return rate


def fission_rate(Z, A, n_n_density, T9):
    """
    中子诱发裂变率。

    参数:
        Z, A : int
        n_n_density : float, 中子数密度 (cm^{-3})
        T9 : float

    返回:
        rate : float, 裂变率 (s^{-1})
    """
    if Z < 90 or A < 230:
        return 0.0  # 低质量核素不考虑裂变
    # 裂变截面近似（Bohr-Wheeler）
    sigma_f = 1e-24 * max(0.0, (A - 220) / 20.0)  # barn -> cm^2
    rate = n_n_density * sigma_f
    return max(rate, 0.0)


def build_reaction_rate_table(nuclides, T9_range, S_n_table, T_half_table):
    """
    为核素集合构建温度依赖的反应率表。

    参数:
        nuclides : list of tuple, [(Z,N,A), ...]
        T9_range : ndarray, 温度范围 (10^9 K)
        S_n_table : dict, (Z,A) -> S_n (MeV)
        T_half_table : dict, (Z,A) -> T_{1/2} (s)

    返回:
        rates : dict, 包含 'capture', 'photodis', 'beta', 'alpha', 'fission'
    """
    rates = {
        'capture': {},
        'photodis': {},
        'beta': {},
        'alpha': {},
        'fission': {}
    }
    for z, n, a in nuclides:
        key = (z, a)
        S_n = S_n_table.get(key, 8.0)  # 默认 8 MeV
        T_half = T_half_table.get(key, 1.0)  # 默认 1 s

        cap_rates = []
        phot_rates = []
        for T9 in T9_range:
            cr = neutron_capture_rate_mackeown(z, a, T9, S_n)
            cap_rates.append(cr)
            pr = photodisintegration_rate(z, a, T9, S_n, cr)
            phot_rates.append(pr)

        rates['capture'][key] = np.array(cap_rates)
        rates['photodis'][key] = np.array(phot_rates)
        rates['beta'][key] = beta_decay_rate(T_half)
        rates['alpha'][key] = alpha_decay_rate(z, a, 5.0)  # 简化 Q_alpha
        rates['fission'][key] = 0.0

    return rates


def test_reaction_rates():
    """自包含测试"""
    T9 = 1.5
    cr = neutron_capture_rate_mackeown(26, 56, T9, 8.0)
    pr = photodisintegration_rate(26, 56, T9, 8.0, cr)
    br = beta_decay_rate(1.0)
    ar = alpha_decay_rate(92, 238, 4.5)
    fr = fission_rate(92, 238, 1e30, T9)
    print(f"[reaction_rates] n-capture rate = {cr:.3e} cm^3/s")
    print(f"[reaction_rates] photodis rate = {pr:.3e} s^{-1}")
    print(f"[reaction_rates] beta decay rate = {br:.3e} s^{-1}")
    print(f"[reaction_rates] alpha decay rate = {ar:.3e} s^{-1}")
    print(f"[reaction_rates] fission rate = {fr:.3e} s^{-1}")


if __name__ == "__main__":
    test_reaction_rates()
