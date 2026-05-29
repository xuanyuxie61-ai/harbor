"""
validation_biharmonic.py
双调和方程验证模块

融合种子项目:
- 087_biharmonic_exact: 双调和方程精确解（3族测试函数）

科学应用: 软体机器人横截面/薄板弯曲变形的制造解验证
双调和方程 ∇⁴W = R 描述薄板弯曲，与软体材料的弯曲变形密切相关
"""

import numpy as np
from typing import Tuple


def biharmonic_w1(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    """
    双调和方程精确解族1 — 基于种子项目087_biharmonic_exact

    可分离双曲-三角函数形式:
        W = (a*cosh(gX) + b*sinh(gX) + c*X*cosh(gX) + d*X*sinh(gX))
            * (e*cos(gY) + f*sin(gY))
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    term_x = a * np.cosh(g * X) + b * np.sinh(g * X) + c * X * np.cosh(g * X) + d * X * np.sinh(g * X)
    term_y = e * np.cos(g * Y) + f * np.sin(g * Y)
    W = term_x * term_y
    return W


def biharmonic_r1(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    """
    双调和方程残差（右端项）族1

    R = W_xxxx + 2*W_xxyy + W_yyyy
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    # W_xxxx
    term_x_4 = (a * g ** 4 * np.cosh(g * X)
                + b * g ** 4 * np.sinh(g * X)
                + c * (4.0 * g ** 3 * np.sinh(g * X) + g ** 4 * X * np.cosh(g * X))
                + d * (4.0 * g ** 3 * np.cosh(g * X) + g ** 4 * X * np.sinh(g * X)))
    term_y = e * np.cos(g * Y) + f * np.sin(g * Y)
    w_xxxx = term_x_4 * term_y

    # W_yyyy
    term_x = a * np.cosh(g * X) + b * np.sinh(g * X) + c * X * np.cosh(g * X) + d * X * np.sinh(g * X)
    term_y_4 = g ** 4 * (e * np.cos(g * Y) + f * np.sin(g * Y))
    w_yyyy = term_x * term_y_4

    # W_xxyy = d^2/dx^2 (d^2W/dy^2)
    w_yy = -g ** 2 * term_x * term_y  # d^2W/dy^2
    term_x_2 = (a * g ** 2 * np.cosh(g * X)
                + b * g ** 2 * np.sinh(g * X)
                + c * (2.0 * g * np.sinh(g * X) + g ** 2 * X * np.cosh(g * X))
                + d * (2.0 * g * np.cosh(g * X) + g ** 2 * X * np.sinh(g * X)))
    w_xxyy = -g ** 2 * term_x_2 * term_y

    R = w_xxxx + 2.0 * w_xxyy + w_yyyy
    return R


def biharmonic_w2(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    """
    双调和方程精确解族2 — 三角-双曲函数互换

    W = (a*cos(gX) + b*sin(gX) + c*X*cos(gX) + d*X*sin(gX))
        * (e*cosh(gY) + f*sinh(gY))
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    term_x = a * np.cos(g * X) + b * np.sin(g * X) + c * X * np.cos(g * X) + d * X * np.sin(g * X)
    term_y = e * np.cosh(g * Y) + f * np.sinh(g * Y)
    W = term_x * term_y
    return W


def biharmonic_r2(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    """
    双调和方程残差族2
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    # 使用W2计算各项导数
    term_x = a * np.cos(g * X) + b * np.sin(g * X) + c * X * np.cos(g * X) + d * X * np.sin(g * X)
    term_y = e * np.cosh(g * Y) + f * np.sinh(g * Y)

    # W_xxxx
    term_x_4 = (a * g ** 4 * np.cos(g * X)
                + b * g ** 4 * np.sin(g * X)
                + c * (-4.0 * g ** 3 * np.sin(g * X) + g ** 4 * X * np.cos(g * X))
                + d * (4.0 * g ** 3 * np.cos(g * X) + g ** 4 * X * np.sin(g * X)))
    w_xxxx = term_x_4 * term_y

    # W_yyyy
    term_y_4 = g ** 4 * term_y
    w_yyyy = term_x * term_y_4

    # W_xxyy
    term_x_2 = (-a * g ** 2 * np.cos(g * X)
                - b * g ** 2 * np.sin(g * X)
                + c * (-2.0 * g * np.sin(g * X) - g ** 2 * X * np.cos(g * X))
                + d * (2.0 * g * np.cos(g * X) - g ** 2 * X * np.sin(g * X)))
    w_xxyy = g ** 2 * term_x_2 * term_y

    R = w_xxxx + 2.0 * w_xxyy + w_yyyy
    return R


def biharmonic_w3(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 1.0,
                  d: float = 1.0, e: float = 0.5, f: float = 0.5) -> np.ndarray:
    """
    双调和方程精确解族3 — 径向对数形式（在(e,f)处奇异）

    R = sqrt((X-e)^2 + (Y-f)^2)
    W = a*R^2*log(R) + b*R^2 + c*log(R) + d
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    dx = X - e
    dy = Y - f
    R = np.sqrt(dx ** 2 + dy ** 2)

    # 避免R=0处的奇异性
    R = np.where(R < 1e-10, 1e-10, R)

    W = a * R ** 2 * np.log(R) + b * R ** 2 + c * np.log(R) + d
    return W


def biharmonic_r3(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 1.0,
                  d: float = 1.0, e: float = 0.5, f: float = 0.5) -> np.ndarray:
    """
    双调和方程残差族3（径向形式）

    对于 W = a*R^2*log(R) + b*R^2 + c*log(R) + d
    双调和算子作用结果为:
        ∇⁴W = 8*a/R^2 + 16*b - 8*c/R^4
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    dx = X - e
    dy = Y - f
    R = np.sqrt(dx ** 2 + dy ** 2)
    R = np.where(R < 1e-10, 1e-10, R)

    R2 = R ** 2
    R4 = R ** 4

    residual = 8.0 * a / R2 + 16.0 * b - 8.0 * c / R4
    return residual


def verify_biharmonic_discretization(Nx: int = 32, Ny: int = 32) -> dict:
    """
    验证双调和方程离散化的精度

    在[-1,1]x[-1,1]上比较精确解和数值计算的残差
    """
    x = np.linspace(-1.0, 1.0, Nx)
    y = np.linspace(-1.0, 1.0, Ny)
    X, Y = np.meshgrid(x, y)

    # 选择测试函数族1
    W_exact = biharmonic_w1(X, Y)
    R_exact = biharmonic_r1(X, Y)

    # 数值计算双调和算子（5点中心差分）
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # W_xxxx (中心差分)
    W_xxxx = np.zeros_like(W_exact)
    W_xxxx[2:-2, 2:-2] = (W_exact[2:-2, :-4] - 4.0 * W_exact[2:-2, 1:-3]
                           + 6.0 * W_exact[2:-2, 2:-2]
                           - 4.0 * W_exact[2:-2, 3:-1]
                           + W_exact[2:-2, 4:]) / dx ** 4

    # W_yyyy
    W_yyyy = np.zeros_like(W_exact)
    W_yyyy[2:-2, 2:-2] = (W_exact[:-4, 2:-2] - 4.0 * W_exact[1:-3, 2:-2]
                           + 6.0 * W_exact[2:-2, 2:-2]
                           - 4.0 * W_exact[3:-1, 2:-2]
                           + W_exact[4:, 2:-2]) / dy ** 4

    # W_xxyy
    W_xxyy = np.zeros_like(W_exact)
    W_xxyy[1:-1, 1:-1] = ((W_exact[2:, 2:] - 2.0 * W_exact[2:, 1:-1] + W_exact[2:, :-2])
                           - 2.0 * (W_exact[1:-1, 2:] - 2.0 * W_exact[1:-1, 1:-1] + W_exact[1:-1, :-2])
                           + (W_exact[:-2, 2:] - 2.0 * W_exact[:-2, 1:-1] + W_exact[:-2, :-2])) / (dx ** 2 * dy ** 2)

    R_numerical = W_xxxx + 2.0 * W_xxyy + W_yyyy

    # 只在内部点比较
    interior = slice(2, -2)
    diff = np.abs(R_numerical[interior, interior] - R_exact[interior, interior])
    max_error = np.max(diff)
    l2_error = np.sqrt(np.mean(diff ** 2))

    return {
        'max_error': max_error,
        'l2_error': l2_error,
        'dx': dx,
        'dy': dy
    }


def plate_bending_energy(W: np.ndarray, D: float, dx: float, dy: float) -> float:
    """
    计算薄板弯曲应变能

    U = 0.5 * D * ∫∫ [ (∂²W/∂x² + ∂²W/∂y²)² - 2(1-ν)(∂²W/∂x² * ∂²W/∂y² - (∂²W/∂x∂y)²) ] dxdy

    D = E*h³ / (12*(1-ν²)) 为弯曲刚度
    """
    # 二阶导数的中心差分
    W_xx = np.zeros_like(W)
    W_yy = np.zeros_like(W)
    W_xy = np.zeros_like(W)

    W_xx[1:-1, 1:-1] = (W[1:-1, 2:] - 2.0 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dx ** 2
    W_yy[1:-1, 1:-1] = (W[2:, 1:-1] - 2.0 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dy ** 2
    W_xy[1:-1, 1:-1] = (W[2:, 2:] - W[2:, :-2] - W[:-2, 2:] + W[:-2, :-2]) / (4.0 * dx * dy)

    # 应变能密度
    energy_density = (W_xx + W_yy) ** 2 - 2.0 * (1.0 - 0.3) * (W_xx * W_yy - W_xy ** 2)

    # 数值积分（梯形法则）
    U = 0.5 * D * np.sum(energy_density[1:-1, 1:-1]) * dx * dy
    return U
