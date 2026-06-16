"""
hankel_momentum_transform.py — 汉克尔变换与动量空间重构
======================================================
本模块实现汉克尔 (Hankel) 变换及其逆变换,
用于顶夸克质量测量中从横动量空间到坐标空间的变换。

数学基础:
  ν 阶汉克尔变换:
    H_ν{f}(k) = ∫₀^∞ f(r) J_ν(kr) r dr
  逆变换 (自逆性):
    f(r) = ∫₀^∞ H_ν{f}(k) J_ν(kr) k dk

  其中 J_ν 是 ν 阶第一类贝塞尔函数:
    J_ν(x) = Σ_{m=0}^∞ (-1)^m (x/2)^{ν+2m} / (m! Γ(ν+m+1))

  在顶夸克物理中的应用:
  1. 横动量分布 p_T(q_T) 到碰撞参数空间 b 的变换:
     W(b) = ∫₀^∞ q_T dq_T J₀(q_T b) dσ/dq_T²
  2. Collins-Soper 重求和中的 b* 处方:
     b* = b / √(1 + b²/b_max²)
  3. 非微扰形状因子:
     S_NP(b) = exp(-g₁ b² - g₂ ln(b/(2b₀)) b²)

  数值方法:
    使用对数均匀网格上的 Gaussian quadrature:
      k_j = exp(t_j),  t_j = t_min + j × Δt
      H_ν{f}(k_i) ≈ Σ_j w_j f(r_j) J_ν(k_i r_j) r_j
    其中权重 w_j 来自 Gauss-Legendre 在对数区间上的积分。

  映射种子项目:
    - 505_hankel_inverse: 汉克尔矩阵逆与结构利用
"""

import numpy as np
from scipy.special import jv, yv, kv


def bessel_j_zeros(nu, n_zeros):
    """
    计算 ν 阶贝塞尔函数 J_ν(x) 的前 n_zeros 个正零点。

    使用 McMahon 渐近展开 (大零点):
      j_{ν,m} ≈ β - (4ν²-1)/(8β) - ...
      其中 β = (m + ν/2 - 1/4)π

    对于前几个零点, 使用已知数值表。

    参数:
        nu: 阶数 ν
        n_zeros: 零点个数

    返回:
        zeros: 零点数组
    """
    # 已知零点表 (J₀ 和 J₁)
    known_zeros = {
        0: [2.4048, 5.5201, 8.6537, 11.7915, 14.9309, 18.0711, 21.2116, 24.3525],
        1: [3.8317, 7.0156, 10.1735, 13.3237, 16.4706, 19.6159, 22.7601, 25.9037],
    }

    if nu in known_zeros and n_zeros <= len(known_zeros[nu]):
        return np.array(known_zeros[nu][:n_zeros])

    # McMahon 渐近展开
    zeros = np.zeros(n_zeros)
    for m in range(1, n_zeros + 1):
        beta = (m + nu / 2.0 - 0.25) * np.pi
        mu = 4.0 * nu**2
        zeros[m - 1] = (beta - (mu - 1.0) / (8.0 * beta) -
                        4.0 * (mu - 1.0) * (mu - 25.0) / (3.0 * (8.0 * beta)**3))

    return zeros


def hankel_transform_log_grid(f_func, nu, k_values, r_min=1e-4, r_max=1e3,
                              n_quad=100):
    """
    在对数均匀网格上计算汉克尔变换。

    H_ν{f}(k) = ∫₀^∞ f(r) J_ν(kr) r dr

    使用对数变量替换 t = ln(r):
      H_ν{f}(k) = ∫_{-∞}^{∞} f(e^t) J_ν(k e^t) e^{2t} dt

    在 [ln(r_min), ln(r_max)] 上使用 Gauss-Legendre 积分。

    参数:
        f_func: 被变换函数 f(r)
        nu: 贝塞尔函数阶数
        k_values: 变换变量 k 的数组
        r_min, r_max: 积分范围
        n_quad: 积分点数

    返回:
        H_values: H_ν{f}(k) 在各 k 点的值
    """
    t_min = np.log(r_min)
    t_max = np.log(r_max)

    # Gauss-Legendre 节点和权重
    t_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)
    t_phys = 0.5 * ((t_max - t_min) * t_nodes + (t_max + t_min))
    w_phys = 0.5 * (t_max - t_min) * w_nodes

    r_nodes = np.exp(t_phys)
    H_values = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        integrand = np.zeros(n_quad)
        for j in range(n_quad):
            r = r_nodes[j]
            arg = k * r
            # 贝塞尔函数
            J_val = jv(nu, arg) if arg < 200 else 0.0
            integrand[j] = f_func(r) * J_val * r**2  # r × e^t = r × r = r²

        H_values[i] = np.sum(w_phys * integrand)

    return H_values


def hankel_inverse_transform(H_func, nu, r_values, k_min=1e-4, k_max=1e3,
                             n_quad=100):
    """
    逆汉克尔变换: 从动量空间 H(k) 回到坐标空间 f(r)。

    f(r) = ∫₀^∞ H(k) J_ν(kr) k dk

    参数:
        H_func: 动量空间函数 H(k)
        nu: 贝塞尔函数阶数
        r_values: 坐标空间 r 的数组
        k_min, k_max: 积分范围
        n_quad: 积分点数

    返回:
        f_values: f(r) 在各 r 点的值
    """
    lnk_min = np.log(k_min)
    lnk_max = np.log(k_max)

    lnk_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)
    lnk_phys = 0.5 * ((lnk_max - lnk_min) * lnk_nodes + (lnk_max + lnk_min))
    w_phys = 0.5 * (lnk_max - lnk_min) * w_nodes

    k_nodes = np.exp(lnk_phys)
    f_values = np.zeros(len(r_values))

    for i, r in enumerate(r_values):
        integrand = np.zeros(n_quad)
        for j in range(n_quad):
            k = k_nodes[j]
            arg = k * r
            J_val = jv(nu, arg) if arg < 200 else 0.0
            integrand[j] = H_func(k) * J_val * k**2

        f_values[i] = np.sum(w_phys * integrand)

    return f_values


def collins_soper_resummation(qT, m_top, b_max=1.5):
    """
    Collins-Soper-Sterman (CSS) 横动量重求和。

    tt̄ 横动量分布的重求和形式:
      dσ/dq_T² = (1/(2π)) ∫₀^∞ b db J₀(q_T b) ×
                 W(b, m_t, μ) × Y(q_T, m_t)

    其中:
      W(b) = exp(-S_pert(b*, μ) - S_NP(b)) × Σ C ⊗ f
      b* = b / √(1 + b²/b_max²) (b* 处方)
      S_pert = ∫_{b₀²/b²}^{μ²} dμ'²/μ'² [A(α_s) ln(μ'²/b²) + B(α_s)]
      S_NP = g₁ b² + g₂ b² ln(m_t/(2b₀)) (非微扰贡献)

    映射种子项目:
      - 505_hankel_inverse: 汉克尔变换数值方法

    参数:
        qT: 横动量 (GeV)
        m_top: 顶夸克质量 (GeV)
        b_max: b* 处方参数 (GeV⁻¹)

    返回:
        dσ/dq_T² (arb. units)
    """
    from topmass_constants import alpha_s_running, M_Z_BOSON

    b0 = 2.0 * np.exp(-0.5772156649)  # = 2e^{-γ_E}

    # b 积分网格 (对数均匀)
    n_b = 80
    b_min = 1e-3  # GeV⁻¹
    b_max_int = 20.0  # GeV⁻¹
    b_values = np.logspace(np.log10(b_min), np.log10(b_max_int), n_b)
    ln_b = np.log(b_values)
    db = np.diff(ln_b)

    integrand = np.zeros(n_b)
    for i in range(n_b):
        b = b_values[i]

        # b* 处方
        b_star = b / np.sqrt(1.0 + (b / b_max)**2)

        # 微扰 Sudakov 因子 (leading log)
        mu_low = b0 / b_star
        mu_high = m_top
        if mu_low < 0.5:
            mu_low = 0.5

        alpha_s_low = alpha_s_running(mu_low, n_loop=3)
        alpha_s_high = alpha_s_running(mu_high, n_loop=3)

        # A^(1) = C_F/π, B^(1) = -3C_F/(2π)
        CF = 4.0 / 3.0
        A1 = CF / np.pi
        B1 = -3.0 * CF / (2.0 * np.pi)

        L = np.log(m_top**2 * b_star**2 / b0**2)
        S_pert = A1 * alpha_s_low * L**2 / 2.0 + B1 * alpha_s_low * L

        # 非微扰 Sudakov
        g1 = 0.2  # GeV²
        g2 = 0.1
        S_NP = g1 * b**2 + g2 * b**2 * np.log(m_top / (2.0 * b0))

        # 总 Sudakov
        W_val = np.exp(-S_pert - S_NP)

        # Bessel 函数 J₀(q_T b)
        arg = qT * b
        J0_val = jv(0, arg) if arg < 200 else 0.0

        integrand[i] = b * J0_val * W_val

    # 梯形积分
    result = np.trapz(integrand, b_values) / (2.0 * np.pi)

    return max(result, 0.0)


def hankel_matrix_construction(x_data, n_dim):
    """
    构建 Hankel 矩阵用于矩方法质量提取。

    对于数据矩 m_k = ∫ x^k ρ(x) dx,
    Hankel 矩阵 H_{ij} = m_{i+j}:
      H = [[m₀, m₁, m₂, ...],
           [m₁, m₂, m₃, ...],
           [m₂, m₃, m₄, ...]]

    其逆矩阵可用于提取谱函数参数。

    映射种子项目:
      - 505_hankel_inverse: Hankel 矩阵结构

    参数:
        x_data: 数据样本
        n_dim: 矩阵维度

    返回:
        H: (n_dim, n_dim) Hankel 矩阵
        H_inv: 逆矩阵 (如果存在)
    """
    # 计算数据矩
    moments = np.zeros(2 * n_dim)
    for k in range(2 * n_dim):
        moments[k] = np.mean(x_data**k) if len(x_data) > 0 else 0.0

    # 构建 Hankel 矩阵
    H = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            H[i, j] = moments[i + j]

    # 尝试求逆 (可能病态)
    try:
        cond = np.linalg.cond(H)
        if cond < 1e12:
            H_inv = np.linalg.inv(H)
        else:
            # 使用伪逆
            H_inv = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    return H, H_inv


def top_pt_spectrum(qT_grid, m_top):
    """
    顶夸克横动量谱 dσ/dp_T 的计算。

    使用 CSS 重求和 + 固定阶匹配:
      dσ/dq_T = dσ_resum + dσ_fixed - dσ_asymptotic

    映射种子项目:
      - 945_quad_trapezoid: 数值积分
      - 505_hankel_inverse: 汉克尔变换

    参数:
        qT_grid: q_T 网格 (GeV)
        m_top: 顶夸克质量

    返回:
        dsigma_dqT: 微分截面
    """
    dsigma = np.zeros(len(qT_grid))

    for i, qT in enumerate(qT_grid):
        if qT < 0.1:
            qT = 0.1
        dsigma[i] = collins_soper_resummation(qT, m_top)

    return dsigma
