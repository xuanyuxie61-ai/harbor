"""
s_matrix.py
=============
S-矩阵、相移与核反应截面计算模块

基于种子项目 777_monomial_value 的多维单项式求值思想
(用于多极展开与角分布计算) 以及 1188_svd_gray 的 SVD
低秩分析思想，本模块实现:
1. 各分波的 S-矩阵元与相移提取
2. 总截面、弹性截面、反应截面计算
3. 散射振幅的角分布展开
4. S-矩阵的 SVD 秩截断分析 (模拟复合核统计涨落)

核心公式
--------
总截面 (optical theorem):
    σ_tot = (π/k²) Σ_{l=0}^{∞} (2l+1) [1 - Re(S_l)]

反应截面:
    σ_react = (π/k²) Σ_{l=0}^{∞} (2l+1) [1 - |S_l|²]

弹性截面:
    σ_el = (π/k²) Σ_{l=0}^{∞} (2l+1) |1 - S_l|²

散射振幅 (Legendre 展开):
    f(θ) = (1/2ik) Σ_{l=0}^{∞} (2l+1) (S_l - 1) P_l(cos θ)

微分截面:
    dσ/dΩ = |f(θ)|²

SVD 秩截断分析:
    S = U Σ V^†
    保留前 R 个奇异值可得到低秩近似 S_R，
    用于分析统计涨落对 S-矩阵的影响。
"""

import numpy as np
from scipy.special import legendre, eval_legendre


def compute_cross_sections(S_matrix_dict, k, l_max):
    """
    从 S-矩阵字典计算各种截面。

    Parameters
    ----------
    S_matrix_dict : dict
        键为 (l, j) 元组，值为 S-矩阵元 (复数)。
    k : float
        波数 (fm^{-1})。
    l_max : int
        最大分波数。

    Returns
    -------
    result : dict
        包含 σ_tot, σ_react, σ_el, σ_abs (吸收截面) 等。
    """
    prefactor = np.pi / (k ** 2)

    sigma_tot = 0.0
    sigma_react = 0.0
    sigma_el = 0.0

    for l in range(l_max + 1):
        # 考虑自旋-轨道分裂: j = l ± 1/2 (l > 0)
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key not in S_matrix_dict:
                continue
            S_l = S_matrix_dict[key]
            gj = int(2 * j + 1)  # 简并度因子 (2j+1)

            # HOLE 3: 请实现截面求和公式（光学定理）
            # σ_tot   累加项: gj * (1 - Re(S_l))
            # σ_react 累加项: gj * (1 - |S_l|²)
            # σ_el    累加项: gj * |1 - S_l|²
            sigma_tot += 0.0    # 占位，不正确
            sigma_react += 0.0  # 占位，不正确
            sigma_el += 0.0     # 占位，不正确

    return {
        'sigma_total': prefactor * sigma_tot,
        'sigma_reaction': prefactor * sigma_react,
        'sigma_elastic': prefactor * sigma_el,
        'sigma_absorption': prefactor * sigma_react,  # 反应截面 = 吸收截面
    }


def scattering_amplitude(theta, S_matrix_dict, k, l_max):
    """
    计算散射振幅 f(θ)。

    f(θ) = (1 / 2ik) Σ_{l=0}^{l_max} (2l+1) (S_l - 1) P_l(cos θ)

    Parameters
    ----------
    theta : array_like
        散射角 (弧度)。
    S_matrix_dict : dict
        S-矩阵字典。
    k : float
        波数。
    l_max : int
        最大分波。

    Returns
    -------
    f_theta : ndarray
        复数散射振幅。
    """
    theta = np.asarray(theta, dtype=float)
    mu = np.cos(theta)
    f = np.zeros_like(mu, dtype=complex)
    prefactor = 1.0 / (2.0j * k)

    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key not in S_matrix_dict:
                continue
            S_l = S_matrix_dict[key]
            gj = int(2 * j + 1)
            P_l = eval_legendre(l, mu)
            f += gj * (S_l - 1.0) * P_l

    return prefactor * f


def differential_cross_section(theta, S_matrix_dict, k, l_max):
    """
    计算微分截面 dσ/dΩ = |f(θ)|² (单位: fm²/sr)。
    """
    f = scattering_amplitude(theta, S_matrix_dict, k, l_max)
    return np.abs(f) ** 2


def svd_analysis_smatrix(S_matrix_dict, l_max):
    """
    对 S-矩阵进行 SVD 低秩分析。

    基于种子项目 1188_svd_gray 的 SVD 低秩近似思想，
    构造 S-矩阵的角动量空间表示并进行秩截断分析。

    将 S-矩阵视为从入射道到出射道的线性映射:
        S_{lj} 矩阵的 SVD 可揭示反应系统的有效自由度数目。

    Parameters
    ----------
    S_matrix_dict : dict
        S-矩阵字典。
    l_max : int
        最大分波数。

    Returns
    -------
    result : dict
        包含奇异值、低秩近似误差等信息。
    """
    # 构建矩阵表示 (将 S-矩阵按 j 展开为方阵)
    dim = l_max + 1
    S_mat = np.zeros((dim, dim), dtype=complex)

    for l in range(dim):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for idx, j in enumerate(js):
            col = min(l + idx, dim - 1)
            key = (l, j)
            if key in S_matrix_dict:
                S_mat[l, col] = S_matrix_dict[key]

    # SVD 分解
    U, s, Vh = np.linalg.svd(S_mat, full_matrices=False)

    # 低秩近似误差分析
    total_energy = np.sum(s ** 2)
    cumulative = np.cumsum(s ** 2)
    relative_error = 1.0 - cumulative / total_energy

    # 找到捕获 99% 能量的秩
    rank_99 = np.searchsorted(cumulative / total_energy, 0.99) + 1

    return {
        'singular_values': s,
        'rank_99': rank_99,
        'relative_error': relative_error,
        'U': U,
        'Vh': Vh,
    }


def multipole_expansion_scattering(f_theta, theta, max_order):
    """
    将散射振幅进行多极展开 (Legendre 多项式展开)。

    基于种子项目 777_monomial_value 的多维单项式求值思想，
    将角分布分解为各阶多极矩:

    f(θ) = Σ_{λ=0}^{max_order} a_λ P_λ(cos θ)

    展开系数:
    a_λ = (2λ+1)/2 ∫_{-1}^{1} f(arccos μ) P_λ(μ) dμ

    Parameters
    ----------
    f_theta : callable
        接受 theta 数组并返回复数散射振幅的函数。
    theta : ndarray
        采样角度 (弧度)。
    max_order : int
        最大多极阶数。

    Returns
    -------
    coeffs : ndarray
        复数展开系数 a_λ。
    """
    mu = np.cos(theta)
    f_vals = f_theta(theta)
    coeffs = np.zeros(max_order + 1, dtype=complex)

    # 使用 Gauss-Legendre 风格数值积分
    # 简化：使用等距采样 Simpson 积分
    dmu = np.diff(mu)
    # 使用梯形法则
    for lam in range(max_order + 1):
        P_lam = eval_legendre(lam, mu)
        integrand = f_vals * P_lam
        # 梯形积分
        integral = np.trapezoid(integrand, mu)
        coeffs[lam] = (2.0 * lam + 1.0) / 2.0 * integral

    return coeffs


def transmission_coefficients(S_matrix_dict, l_max):
    """
    计算各分波的穿透系数 (transmission coefficient):

    T_l = 1 - |S_l|²

    这是 Hauser-Feshbach 理论中的关键输入量。
    """
    T = {}
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key in S_matrix_dict:
                T[key] = 1.0 - abs(S_matrix_dict[key]) ** 2
            else:
                T[key] = 0.0
    return T


def compound_formation_cross_section(params, T_dict, l_max):
    """
    计算复合核形成截面。

    σ_CF = (π/k²) Σ_{l,j} (2j+1) T_{lj}

    这是入射道形成复合核的总概率。
    """
    prefactor = np.pi / (params.k ** 2)
    sigma_cf = 0.0
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            key = (l, j)
            if key in T_dict:
                sigma_cf += (2 * j + 1) * T_dict[key]
    return prefactor * sigma_cf


if __name__ == "__main__":
    # 自检：使用模拟 S-矩阵
    k = 0.5
    l_max = 5
    S_dict = {}
    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            delta = 0.1 * (l + 1)
            eta = 0.9
            S_dict[(l, j)] = eta * np.exp(2.0j * delta)

    xs = compute_cross_sections(S_dict, k, l_max)
    print("截面 (fm²):", xs)

    theta = np.linspace(0.01, np.pi, 100)
    dsigma = differential_cross_section(theta, S_dict, k, l_max)
    print("微分截面范围:", dsigma.min(), dsigma.max())

    svd_res = svd_analysis_smatrix(S_dict, l_max)
    print("奇异值:", svd_res['singular_values'])
    print("99% 能量秩:", svd_res['rank_99'])
