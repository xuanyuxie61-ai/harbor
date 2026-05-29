"""
thermal_quadrature.py
热力学积分高精度对称求积模块
融合来源：1151_square_symq_rule（正方形对称求积规则）

用于在反应区二维截面上计算热流积分、释热率空间平均等。
"""
import numpy as np
from combustion_utils import check_positive, check_nonnegative


def square_symq_rule(degree):
    r"""
    返回单位正方形 [-1,1]^2 上的对称求积规则。
    节点数与精度根据 degree 选择。

    规则基于 Gauss-Legendre 张量积的简化对称版本。
    对于 degree = k，使用 (k+1) x (k+1) 的 Gauss-Legendre 节点。

    返回:
        x, y: 节点坐标数组
        w:    对应权重数组
    """
    if not (0 <= degree <= 20):
        raise ValueError("degree must be in [0, 20]")

    n = degree + 1
    # 一维 Gauss-Legendre 节点与权重
    xi_1d, wi_1d = np.polynomial.legendre.leggauss(n)

    # 二维张量积
    x = np.zeros(n * n)
    y = np.zeros(n * n)
    w = np.zeros(n * n)
    idx = 0
    for i in range(n):
        for j in range(n):
            x[idx] = xi_1d[i]
            y[idx] = xi_1d[j]
            w[idx] = wi_1d[i] * wi_1d[j]
            idx += 1
    return x, y, w


def integrate_square(func, degree=5):
    r"""
    使用对称求积规则计算函数 func(x, y) 在 [-1,1]^2 上的积分。

        I ≈ Σ_i w_i * func(x_i, y_i)

    其中 func 接受两个标量或数组参数。
    """
    x, y, w = square_symq_rule(degree)
    vals = func(x, y)
    return np.sum(w * vals)


def integrate_thermal_source(lambda_field, T_field, rho_field,
                             dx, dy, degree=5,
                             A=1.0e8, Ea=8.314e4, Q=2.5e6,
                             R=8.314462618):
    r"""
    计算二维反应区释热率的空间积分:

        q_dot_total = ∫∫_Ω Q * rho * k(T) * (1-λ)^n  dx dy

    其中 k(T) = A * exp(-Ea/(R*T))。
    输入场量定义在均匀矩形网格上，通过数值求积计算。
    """
    check_positive(dx, "dx")
    check_positive(dy, "dy")
    lambda_field = np.asarray(lambda_field, dtype=float)
    T_field = np.asarray(T_field, dtype=float)
    rho_field = np.asarray(rho_field, dtype=float)

    if lambda_field.shape != T_field.shape or lambda_field.shape != rho_field.shape:
        raise ValueError("Field arrays must have the same shape")

    nx, ny = lambda_field.shape
    # 将场量插值到求积节点
    x_nodes, y_nodes, w_nodes = square_symq_rule(degree)

    total = 0.0
    for i in range(nx):
        for j in range(ny):
            # 单元中心坐标映射到 [-1,1]
            xc = (i + 0.5) * dx
            yc = (j + 0.5) * dy
            # 单元内求积（简化：直接取单元平均值乘以面积）
            lam = max(0.0, min(1.0, lambda_field[i, j]))
            T = max(T_field[i, j], 1.0e-6)
            rho = max(rho_field[i, j], 1.0e-9)
            k = A * np.exp(-Ea / (R * T))
            rate = rho * k * ((1.0 - lam) ** 1.0)
            cell_area = dx * dy
            total += Q * rate * cell_area

    return total


def average_temperature_profile(T_field, dx, dy, degree=5):
    r"""
    计算反应区加权平均温度:
        T_avg = (∫∫ T dx dy) / Area
    """
    nx, ny = T_field.shape
    area = nx * dx * ny * dy
    integral = np.sum(T_field) * dx * dy
    return integral / area if area > 0.0 else 0.0
