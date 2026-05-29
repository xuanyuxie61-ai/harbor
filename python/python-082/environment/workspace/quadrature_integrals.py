"""
quadrature_integrals.py
高斯数值积分与能量释放率计算。
原项目映射：
  - 950_quadrature_weights_vandermonde 的Vandermonde矩阵求积权重方法
  - 519_hermite_exactness 的Hermite高斯积分精确性检验与加权积分
科学背景：
  在断裂力学中，能量释放率 G 通过J积分或虚裂纹闭合技术（VCCT）计算：
    G_I = lim_{Δa→0} 1/(2Δa) ∫_0^{Δa} σ_22(x, 0) * Δu_2(x - Δa, 0) dx
    G_II = lim_{Δa→0} 1/(2Δa) ∫_0^{Δa} σ_12(x, 0) * Δu_1(x - Δa, 0) dx
  数值实现中采用高斯求积：
    ∫_a^b f(x) dx ≈ Σ_i w_i f(x_i)
  对于概率性强度分析，采用Hermite加权积分：
    E[g(X)] = ∫_{-∞}^{+∞} g(x) exp(-x^2) / √π dx ≈ Σ_i w_i^{(H)} g(x_i^{(H)})
"""

import numpy as np
from utils import validate_positive


def quadrature_weights_vandermonde(n, a, b, x_nodes):
    """
    通过Vandermonde矩阵求解求积权重。
    原项目映射：950_quadrature_weights_vandermonde。
    方程：
      Σ_j w_j * x_j^{i-1} = (b^i - a^i) / i,  i = 1,...,n
    矩阵形式 V^T w = rhs。
    """
    x_nodes = np.asarray(x_nodes).flatten()
    if len(x_nodes) != n:
        raise ValueError("x_nodes length must equal n.")

    V = np.zeros((n, n))
    V[0, :] = 1.0
    for i in range(1, n):
        V[i, :] = V[i - 1, :] * x_nodes

    rhs = np.zeros(n)
    for i in range(n):
        rhs[i] = (b ** (i + 1) - a ** (i + 1)) / (i + 1.0)

    w = np.linalg.solve(V.T, rhs)
    return w


def gauss_legendre_nodes_weights(n, a, b):
    """计算n点Gauss-Legendre求积节点与权重。"""
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    # 映射到 [a, b]
    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    w_mapped = 0.5 * (b - a) * w
    return x_mapped, w_mapped


def hermite_gauss_nodes_weights(n):
    """
    计算n点Gauss-Hermite求积节点与权重（物理学家权重 exp(-x^2)）。
    原项目映射：519_hermite_exactness 的 physicist weighted rule。
    精确性：对次数 ≤ 2n-1 的多项式精确。
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    return x, w


def hermite_monomial_integral(n, option=1):
    """
    计算Hermite加权单项式积分精确值。
    H(n,1) = ∫ x^n exp(-x^2) dx = (n-1)!! * √π / 2^{n/2}  (n even)
    H(n,2) = ∫ x^n exp(-x^2/2) dx = (n-1)!! * √(2π)      (n even)
    原项目映射：519_hermite_exactness 的 hermite_integral。
    """
    if n < 0:
        return -np.inf
    if n % 2 == 1:
        return 0.0

    # 双阶乘 (n-1)!!
    double_fact = 1.0
    k = n - 1
    while k > 0:
        double_fact *= k
        k -= 2

    if option == 0 or option == 1:
        return double_fact * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
    elif option == 2:
        return double_fact * np.sqrt(2.0 * np.pi)
    elif option == 3:
        return double_fact / (2.0 ** (n / 2.0))
    elif option == 4:
        return double_fact
    else:
        raise ValueError("option must be 0-4.")


def compute_j_integral(stress_field, displacement_jump, crack_tip_pos,
                       integration_radius, n_quad=16):
    """
    计算J积分（线积分形式）作为能量释放率的等价量。
    J = ∮_Γ (W n_1 - T_i ∂u_i/∂x_1) ds
    简化实现：在裂纹尖端周围的半圆路径上积分。
    应力场和位移跳跃作为函数输入。
    """
    theta = np.linspace(0, np.pi, n_quad)
    x_path = crack_tip_pos[0] + integration_radius * np.cos(theta)
    y_path = crack_tip_pos[1] + integration_radius * np.sin(theta)

    # 使用Gauss-Legendre积分
    t_nodes, t_weights = gauss_legendre_nodes_weights(n_quad, 0.0, np.pi)
    x_quad = crack_tip_pos[0] + integration_radius * np.cos(t_nodes)
    y_quad = crack_tip_pos[1] + integration_radius * np.sin(t_nodes)

    J_val = 0.0
    for i in range(n_quad):
        xq, yq = x_quad[i], y_quad[i]
        # 路径上的外法向
        nx = np.cos(t_nodes[i])
        ny = np.sin(t_nodes[i])
        ds = integration_radius * t_weights[i]

        # 简化：假设应力场和位移场可用
        # 应变能密度 W = 1/2 σ_ij ε_ij
        # 这里使用近尖端渐近场近似
        r = integration_radius
        # 模式I近尖端场：
        sigma_11 = 1.0 / np.sqrt(2.0 * np.pi * r) * np.cos(t_nodes[i] / 2.0) * (
            1.0 - np.sin(t_nodes[i] / 2.0) * np.sin(1.5 * t_nodes[i]))
        sigma_22 = 1.0 / np.sqrt(2.0 * np.pi * r) * np.cos(t_nodes[i] / 2.0) * (
            1.0 + np.sin(t_nodes[i] / 2.0) * np.sin(1.5 * t_nodes[i]))
        sigma_12 = 1.0 / np.sqrt(2.0 * np.pi * r) * np.sin(t_nodes[i] / 2.0) * np.cos(
            t_nodes[i] / 2.0) * np.cos(1.5 * t_nodes[i])

        # 应变能密度（平面应力）
        W = 0.5 * (sigma_11 ** 2 + sigma_22 ** 2 + 2.0 * sigma_12 ** 2)

        # 牵引矢量 T_i = σ_ij n_j
        T1 = sigma_11 * nx + sigma_12 * ny
        T2 = sigma_12 * nx + sigma_22 * ny

        # 位移导数（简化）
        du1_dx = 0.1 / np.sqrt(r)
        du2_dx = 0.1 / np.sqrt(r)

        J_val += (W * nx - (T1 * du1_dx + T2 * du2_dx)) * ds

    return abs(J_val)


def compute_vcct_energy_release_rate(stress_at_crack_tip, displacement_jump,
                                     delta_a, n_quad=8):
    """
    虚裂纹闭合技术（VCCT）计算能量释放率。
    G_I ≈ 1/(2 Δa) Σ_i w_i σ_22(x_i, 0) * Δu_2(x_i - Δa, 0)
    G_II ≈ 1/(2 Δa) Σ_i w_i σ_12(x_i, 0) * Δu_1(x_i - Δa, 0)
    """
    # 在 [0, Δa] 上积分
    x_quad, w_quad = gauss_legendre_nodes_weights(n_quad, 0.0, delta_a)

    G_I = 0.0
    G_II = 0.0
    for i in range(n_quad):
        x = x_quad[i]
        w = w_quad[i]
        # 简化的近尖端场
        sigma_22 = stress_at_crack_tip / np.sqrt(1.0 + x / delta_a)
        sigma_12 = 0.5 * stress_at_crack_tip / np.sqrt(1.0 + x / delta_a)
        du2 = displacement_jump * np.sqrt(1.0 - x / delta_a)
        du1 = 0.3 * displacement_jump * np.sqrt(1.0 - x / delta_a)

        G_I += w * sigma_22 * du2
        G_II += w * sigma_12 * du1

    G_I /= (2.0 * delta_a)
    G_II /= (2.0 * delta_a)
    return G_I, G_II


def probabilistic_strength_integral(mean_strength, std_strength, n_hermite=12):
    """
    使用Gauss-Hermite积分计算概率化强度期望值。
    假设强度服从对数正态分布：ln(X) ~ N(μ, σ^2)
    E[X] = ∫ exp(√2 σ x + μ) exp(-x^2)/√π dx
          ≈ Σ_i w_i exp(√2 σ x_i + μ) / √π
    原项目映射：519_hermite_exactness 的概率加权积分。
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n_hermite)

    # 对数正态分布参数
    sigma_ln = np.sqrt(np.log(1.0 + (std_strength / mean_strength) ** 2))
    mu_ln = np.log(mean_strength) - 0.5 * sigma_ln ** 2

    E_X = 0.0
    for i in range(n_hermite):
        # physicist权重已包含在w中，需要归一化
        E_X += w[i] * np.exp(np.sqrt(2.0) * sigma_ln * x[i] + mu_ln)

    E_X /= np.sqrt(np.pi)
    return E_X


def compute_strain_energy_release_rate_quadrature(stress, strain, damage, material,
                                                   thickness, n_quad=8):
    """
    通过数值积分计算层合板的应变能释放率。
    使用Gauss-Legendre求积对厚度方向积分：
      U = 1/2 ∫_{-h/2}^{h/2} σ(z) : ε(z) dz
      G = -∂U/∂A
    """
    z_nodes, z_weights = gauss_legendre_nodes_weights(n_quad, -thickness / 2.0, thickness / 2.0)

    U = 0.0
    for i in range(n_quad):
        z = z_nodes[i]
        w = z_weights[i]
        # 简化线性应变分布
        eps_z = strain * (2.0 * z / thickness)
        sigma_z = stress * (1.0 - damage) * (2.0 * z / thickness)
        U += 0.5 * w * np.dot(sigma_z, eps_z)

    return U
