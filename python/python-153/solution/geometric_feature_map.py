"""
geometric_feature_map.py
基于项目 768_minimal_surface_exact (极小曲面方程) 与
1265_toms112 (点在多边形内判定) 的量子态空间几何分析模块。

核心数学模型:
1. 极小曲面方程 (平均曲率为零):
   (1 + U_x^2) U_{yy} - 2 U_x U_y U_{xy} + (1 + U_y^2) U_{xx} = 0

2. 悬链面 (Catenoid):
   U(X,Y) = acosh(a * sqrt(X^2 + Y^2)) / a
   要求: a * sqrt(X^2 + Y^2) > 1

3. Scherk 第一曲面:
   U(X,Y) = log(cos(a*Y) / cos(a*X)) / a
   要求: |a*X|, |a*Y| < pi/2

4. 螺旋面 (Helicoid):
   U(X,Y) = atan(X/Y)

5. 射线法 (Ray Casting) 判定点在多边形内:
   从点引水平射线向右，统计与多边形边界的交点数。
   奇数 -> 内部, 偶数 -> 外部。

6. 量子态空间的几何解释:
   将量子核方法中的特征映射视为从数据空间到量子态空间的嵌入。
   核函数 k(x, x') 与嵌入曲面的内蕴几何有关。
"""

import numpy as np
from typing import Tuple, List


def minimal_surface_catenoid(
    X: np.ndarray,
    Y: np.ndarray,
    a: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    悬链面 (Catenoid) 的精确解析解及各阶偏导数。
    U(X,Y) = acosh(a * sqrt(X^2 + Y^2)) / a

    返回: (U, Ux, Uy, Uxx, Uxy, Uyy)
    """
    if a <= 0:
        raise ValueError("Parameter a must be positive")

    R = np.sqrt(X ** 2 + Y ** 2)
    # 边界处理: 确保 a*R > 1
    R = np.maximum(R, 1.01 / a)

    U = np.arccosh(a * R) / a

    denom = R * np.sqrt((a * R) ** 2 - 1.0)
    denom = np.maximum(denom, 1e-15)

    Ux = X / denom
    Uy = Y / denom

    # 二阶导数
    a2R2 = (a * R) ** 2
    factor = (a2R2 - 1.0) ** 1.5
    factor = np.maximum(factor, 1e-15)

    Uxx = (Y ** 2 * (a2R2 - 1.0) + X ** 2) / (R ** 3 * factor)
    Uxy = -X * Y / (R ** 3 * factor)
    Uyy = (X ** 2 * (a2R2 - 1.0) + Y ** 2) / (R ** 3 * factor)

    return U, Ux, Uy, Uxx, Uxy, Uyy


def minimal_surface_scherk(
    X: np.ndarray,
    Y: np.ndarray,
    a: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scherk 第一曲面的精确解析解及各阶偏导数。
    U(X,Y) = log(cos(a*Y) / cos(a*X)) / a

    要求: |a*X|, |a*Y| < pi/2
    """
    if a <= 0:
        raise ValueError("Parameter a must be positive")

    # 边界处理: 限制在定义域内
    X = np.clip(X, -0.99 * np.pi / (2.0 * a), 0.99 * np.pi / (2.0 * a))
    Y = np.clip(Y, -0.99 * np.pi / (2.0 * a), 0.99 * np.pi / (2.0 * a))

    cos_aX = np.cos(a * X)
    cos_aY = np.cos(a * Y)
    cos_aX = np.maximum(np.abs(cos_aX), 1e-15) * np.sign(cos_aX + 1e-15)

    U = np.log(cos_aY / cos_aX) / a
    Ux = np.tan(a * X)
    Uy = -np.tan(a * Y)
    Uxx = a * (np.tan(a * X) ** 2 + 1.0)
    Uxy = np.zeros_like(X)
    Uyy = -a * (np.tan(a * Y) ** 2 + 1.0)

    return U, Ux, Uy, Uxx, Uxy, Uyy


def minimal_surface_residual(
    Uxx: np.ndarray,
    Uxy: np.ndarray,
    Uyy: np.ndarray,
    Ux: np.ndarray,
    Uy: np.ndarray
) -> np.ndarray:
    """
    计算极小曲面方程的残差。
    R = (1 + Ux^2) Uyy - 2 Ux Uy Uxy + (1 + Uy^2) Uxx
    理论上对精确解 R = 0。
    """
    R = (1.0 + Ux ** 2) * Uyy - 2.0 * Ux * Uy * Uxy + (1.0 + Uy ** 2) * Uxx
    return R


def point_in_polygon(
    x0: float,
    y0: float,
    poly_x: np.ndarray,
    poly_y: np.ndarray
) -> bool:
    """
    射线法 (Ray Casting) 判断点 (x0, y0) 是否在多边形内部。

    算法:
    1. 对每条边 (i) -> (i+1):
       a) 检查边是否跨越水平线 y = y0
       b) 计算交点 X 坐标
       c) 若交点在点右侧，inside 取反
    2. 奇数次交点 -> 内部; 偶数次 -> 外部
    """
    n = len(poly_x)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    if len(poly_y) != n:
        raise ValueError("poly_x and poly_y must have same length")

    inside = False
    for i in range(n):
        ip1 = (i + 1) % n

        yi = poly_y[i]
        yip1 = poly_y[ip1]

        # 边是否跨越水平线 y = y0 (排除水平边和恰好在顶点的情况)
        if ((yi > y0) != (yip1 > y0)) or (y0 == yi and yip1 > y0):
            # 计算交点 X 坐标
            xi = poly_x[i]
            xip1 = poly_x[ip1]

            if abs(yip1 - yi) < 1e-15:
                continue

            x_intersect = xi + (y0 - yi) * (xip1 - xi) / (yip1 - yi)

            if x0 < x_intersect:
                inside = not inside

    return inside


def quantum_state_bloch_region(
    state: np.ndarray,
    region_polygon_x: np.ndarray,
    region_polygon_y: np.ndarray
) -> bool:
    """
    判断量子态在布洛赫球面上的投影是否落在指定多边形区域内。
    对单量子比特，布洛赫坐标为:
    x = 2*Re(alpha*conj(beta))
    y = 2*Im(alpha*conj(beta))
    z = |alpha|^2 - |beta|^2
    """
    dim = len(state)
    if dim != 2:
        raise ValueError("This function only works for single-qubit states (dim=2)")

    alpha = state[0]
    beta = state[1]

    x_bloch = 2.0 * (alpha * np.conj(beta)).real
    y_bloch = 2.0 * (alpha * np.conj(beta)).imag

    return point_in_polygon(x_bloch, y_bloch, region_polygon_x, region_polygon_y)


def geometric_quantum_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    surface_type: str = "catenoid",
    a: float = 1.0
) -> float:
    """
    基于极小曲面几何的量子核函数。
    将数据点映射到极小曲面上的坐标，计算曲面上的测地线距离。
    """
    if len(x) < 2 or len(x_prime) < 2:
        raise ValueError("Input vectors must have at least 2 dimensions")

    # 将数据点映射到曲面坐标
    X = np.array([[x[0], x_prime[0]]])
    Y = np.array([[x[1], x_prime[1]]])

    if surface_type == "catenoid":
        U, _, _, _, _, _ = minimal_surface_catenoid(X, Y, a)
    elif surface_type == "scherk":
        U, _, _, _, _, _ = minimal_surface_scherk(X, Y, a)
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")

    # 核函数: exp(- (U(x) - U(x'))^2 )
    diff = U[0, 0] - U[0, 1]
    kernel = np.exp(-diff ** 2)
    return kernel


def quantum_feature_space_volume(
    n_qubits: int,
    n_samples: int = 1000
) -> float:
    """
    使用蒙特卡洛方法估计 n-qubit 量子特征空间的"有效体积"。
    在 2^n 维希尔伯特空间中随机采样量子态，统计满足几何约束的比例。
    """
    dim = 2 ** n_qubits
    count = 0

    for _ in range(n_samples):
        # 随机量子态 (Haar 随机)
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)

        # 检查是否满足"局部性"约束: 各基态振幅差异不太大
        probs = np.abs(psi) ** 2
        entropy = -np.sum(probs * np.log(probs + 1e-15))
        max_entropy = np.log(dim)

        # 若熵大于最大熵的一半，认为该态在特征空间"有效区域"内
        if entropy > 0.5 * max_entropy:
            count += 1

    return count / n_samples
