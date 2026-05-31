"""
surface_integrals.py
气泡表面与体积的高阶数值积分

核心物理模型:
1. 气泡表面积（变形后）:
   S = ∫_0^{2π} ∫_0^π r² sinθ * sqrt(1 + (1/r²)(∂r/∂θ)² + (1/(r²sin²θ))(∂r/∂φ)²) dθ dφ

2. 气泡体积:
   V = ∫_0^{2π} ∫_0^π ∫_0^{r(θ,φ)} r'² sinθ dr' dθ dφ
     = (1/3) ∫_0^{2π} ∫_0^π r(θ,φ)³ sinθ dθ dφ

3. 表面张力做功:
   W_σ = σ * (S - S_0)

4. Gauss-Chebyshev Type 2 求积:
   ∫_a^b f(x) * sqrt((x-a)(b-x)) dx ≈ Σ_i w_i f(x_i)
   节点: x_i = (a+b)/2 + (b-a)/2 * cos(iπ/(n+1))
   权重: w_i = π/(n+1) * sin²(iπ/(n+1))

映射来源:
- 167_chebyshev2_rule: Gauss-Chebyshev 求积规则生成
- 231_cube_exactness: 3D Legendre 求积精确度测试
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import legendre as legendre_poly


def chebyshev2_nodes_weights(n, a=-1.0, b=1.0):
    """
    生成 Gauss-Chebyshev Type 2 求积节点与权重。
    对应 167_chebyshev2_rule 的核心算法。

    标准区间 [-1,1]:
      x_i = cos(i * π / (n + 1)),  i = 1, 2, ..., n
      w_i = π / (n + 1) * sin²(i * π / (n + 1))

    区间变换 [a,b]:
      x_i' = (a+b)/2 + (b-a)/2 * x_i
      w_i' = (b-a)/2 * w_i
    """
    i = np.arange(1, n + 1, dtype=float)
    x_std = np.cos(i * np.pi / (n + 1))
    w_std = (np.pi / (n + 1)) * np.sin(i * np.pi / (n + 1)) ** 2

    # 线性变换到 [a, b]
    shift = (a + b) / 2.0
    scale = (b - a) / 2.0
    x = shift + scale * x_std
    w = scale * w_std
    return x, w


def gauss_legendre_3d(nx, ny, nz, ax, bx, ay, by, az, bz):
    """
    生成三维 Gauss-Legendre 求积规则。
    对应 231_cube_exactness 的 3D 求积框架。

    参数:
        nx, ny, nz: 各方向节点数
        [ax,bx], [ay,by], [az,bz]: 各方向积分区间
    返回:
        x, y, z: 节点坐标数组（展平）
        w: 权重数组
    """
    x1d, wx = leggauss(nx)
    y1d, wy = leggauss(ny)
    z1d, wz = leggauss(nz)

    # 变换到物理区间
    x1d = 0.5 * (bx - ax) * x1d + 0.5 * (bx + ax)
    y1d = 0.5 * (by - ay) * y1d + 0.5 * (by + ay)
    z1d = 0.5 * (bz - az) * z1d + 0.5 * (bz + az)

    wx *= 0.5 * (bx - ax)
    wy *= 0.5 * (by - ay)
    wz *= 0.5 * (bz - az)

    X, Y, Z = np.meshgrid(x1d, y1d, z1d, indexing='ij')
    Wx, Wy, Wz = np.meshgrid(wx, wy, wz, indexing='ij')

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    w = (Wx * Wy * Wz).ravel()
    return x, y, z, w


def bubble_surface_area_quadrature(r_func, theta_nodes=32, phi_nodes=32):
    """
    使用 Gauss-Legendre / Chebyshev 混合求积计算气泡表面积。

    参数:
        r_func: 函数 r(θ, φ)，返回半径
        theta_nodes: θ 方向节点数
        phi_nodes: φ 方向节点数
    返回:
        area: 表面积 [m²]
    """
    # θ ∈ [0, π]，使用 Gauss-Legendre
    t_nodes, t_weights = leggauss(theta_nodes)
    theta = 0.5 * np.pi * (t_nodes + 1.0)
    w_theta = 0.5 * np.pi * t_weights

    # φ ∈ [0, 2π]，使用复合 Simpson 或 Chebyshev
    phi = np.linspace(0, 2 * np.pi, phi_nodes, endpoint=False)
    dphi = 2.0 * np.pi / phi_nodes

    area = 0.0
    for i, th in enumerate(theta):
        for j, ph in enumerate(phi):
            r = r_func(th, ph)
            if r <= 0:
                continue
            # 数值微分求 ∂r/∂θ 和 ∂r/∂φ
            h_theta = 1e-6
            h_phi = 1e-6
            r_plus_theta = r_func(th + h_theta, ph)
            r_minus_theta = r_func(max(th - h_theta, 0.0), ph)
            dr_dtheta = (r_plus_theta - r_minus_theta) / (2.0 * h_theta + 1e-15)

            r_plus_phi = r_func(th, ph + h_phi)
            r_minus_phi = r_func(th, ph - h_phi)
            dr_dphi = (r_plus_phi - r_minus_phi) / (2.0 * h_phi + 1e-15)

            sin_th = np.sin(th)
            metric = np.sqrt(1.0 + (dr_dtheta / (r + 1e-15)) ** 2 +
                             (dr_dphi / ((r + 1e-15) * sin_th + 1e-15)) ** 2)
            integrand = r ** 2 * sin_th * metric
            area += integrand * w_theta[i] * dphi

    return area


def bubble_volume_quadrature(r_func, theta_nodes=24, phi_nodes=24):
    """
    使用高阶求积计算气泡体积。
    V = (1/3) ∫_0^{2π} ∫_0^π r(θ,φ)³ sinθ dθ dφ
    """
    t_nodes, t_weights = leggauss(theta_nodes)
    theta = 0.5 * np.pi * (t_nodes + 1.0)
    w_theta = 0.5 * np.pi * t_weights

    phi = np.linspace(0, 2 * np.pi, phi_nodes, endpoint=False)
    dphi = 2.0 * np.pi / phi_nodes

    volume = 0.0
    for i, th in enumerate(theta):
        for j, ph in enumerate(phi):
            r = r_func(th, ph)
            r = max(r, 0.0)
            integrand = (r ** 3 / 3.0) * np.sin(th)
            volume += integrand * w_theta[i] * dphi

    return volume


def surface_tension_energy(r_func, sigma, theta_nodes=24, phi_nodes=24):
    """
    计算表面张力势能: E_σ = σ * S
    """
    S = bubble_surface_area_quadrature(r_func, theta_nodes, phi_nodes)
    return sigma * S


def kinetic_energy_integral(R, dRdt, rho, theta_nodes=16):
    """
    计算气泡周围液体的动能。
    不可压缩液体中球形气泡的动能:
    E_k = 2π ρ R³ (dR/dt)²

    非球形修正（一阶近似）:
    E_k ≈ 2π ρ R³ (dR/dt)² * [1 + Σ_{n=2}^N ((n+1)/2) * a_n²]
    """
    E_k = 2.0 * np.pi * rho * (R ** 3) * (dRdt ** 2)
    return E_k


def pressure_work_integral(p_in, p_out, r_func, theta_nodes=16, phi_nodes=16):
    """
    计算压力做功:
    W_p = ∫ (p_in - p_out) * dV = (p_in - p_out) * V
    """
    V = bubble_volume_quadrature(r_func, theta_nodes, phi_nodes)
    return (p_in - p_out) * V


def legendre_3d_exactness_test(n_points, max_degree=4):
    """
    测试 3D 求积规则对 Legendre 多项式单项式的精确度。
    对应 231_cube_exactness 的精确度测试框架。

    对 3D 区域 [-1,1]³，单项式 x^i y^j z^k 的精确积分:
    I = 8 / ((i+1)(j+1)(k+1))  （当 i,j,k 全为偶数时）
    I = 0 （其他情况）
    """
    x, y, z, w = gauss_legendre_3d(n_points, n_points, n_points, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    errors = []

    for tt in range(max_degree + 1):
        for k in range(tt + 1):
            for j in range(tt - k + 1):
                i = tt - j - k
                # 精确值
                if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                    exact = 8.0 / ((i + 1) * (j + 1) * (k + 1))
                else:
                    exact = 0.0
                # 数值积分
                v = (x ** i) * (y ** j) * (z ** k)
                approx = np.dot(w, v)
                if abs(exact) > 1e-15:
                    err = abs(approx - exact) / abs(exact)
                else:
                    err = abs(approx)
                errors.append((i, j, k, err))

    return errors


def chebyshev_surface_integral(f, a, b, n=32):
    """
    在气泡边界曲线（一维）上使用 Gauss-Chebyshev Type 2 求积。
    ∫_a^b f(s) * sqrt((s-a)(b-s)) ds
    这种权重函数自然出现在圆形/球形边界问题中。
    """
    x, w = chebyshev2_nodes_weights(n, a, b)
    fx = np.array([f(xi) for xi in x])
    return np.dot(w, fx)
