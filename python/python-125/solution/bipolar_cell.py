"""
bipolar_cell.py
双极细胞感受野建模与信号处理

基于以下种子项目合成：
- 661_legendre_polynomial: Legendre多项式计算与Gauss-Legendre求积
- 527_hexagon_integrals: 多边形矩计算

科学背景：
双极细胞（Bipolar Cell）是视网膜中连接光感受器与神经节细胞的重要中间神经元。
其感受野具有典型的中心-周边拮抗结构（center-surround antagonism）。

本模块实现：
1. 差异高斯（Difference of Gaussians, DoG）感受野模型
2. Legendre正交基展开感受野调谐曲线
3. Gauss-Legendre求积计算感受野卷积响应

关键公式：
- 感受野响应：R(x,y) = A_c * exp(-(x²+y²)/(2σ_c²)) - A_s * exp(-(x²+y²)/(2σ_s²))
- 空间频率调谐：通过Legendre基展开实现
"""

import numpy as np
from typing import Tuple, Callable


# =============================================================================
# Legendre多项式计算（基于661_legendre_polynomial）
# =============================================================================

def legendre_polynomial_value(n: int, x: np.ndarray) -> np.ndarray:
    """
    计算Legendre多项式 P_0(x) 到 P_n(x) 的值。
    
    使用三项递推关系：
        P_0(x) = 1
        P_1(x) = x
        P_k(x) = [(2k-1) * x * P_{k-1}(x) - (k-1) * P_{k-2}(x)] / k,  k ≥ 2
    
    参数:
        n: 最高阶数
        x: (M,) 评估点，需在[-1,1]区间内
    
    返回:
        V: (M, n+1) 多项式值矩阵，V[i,j] = P_j(x[i])
    """
    m = x.shape[0]
    V = np.zeros((m, n + 1), dtype=np.float64)
    
    V[:, 0] = 1.0  # P_0(x) = 1
    if n >= 1:
        V[:, 1] = x  # P_1(x) = x
    
    for k in range(2, n + 1):
        V[:, k] = ((2.0 * k - 1.0) * x * V[:, k - 1] - (k - 1.0) * V[:, k - 2]) / k
    
    return V


def legendre_polynomial_zeros(n: int) -> np.ndarray:
    """
    计算n阶Legendre多项式 P_n(x) 的零点。
    
    Legendre多项式的零点可通过求解对称三对角Jacobi矩阵的特征值获得。
    Jacobi矩阵J的构造：
        J_{i,i} = 0
        J_{i,i+1} = J_{i+1,i} = sqrt(i² / (4i² - 1)),  i = 1, ..., n-1
    
    P_n(x)的零点即为J的特征值，全部位于(-1, 1)区间内。
    
    参数:
        n: 多项式阶数
    
    返回:
        t: (n,) 零点数组
    """
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])
    
    # 构造Jacobi矩阵
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        b = np.sqrt(i * i / (4.0 * i * i - 1.0))
        J[i - 1, i] = b
        J[i, i - 1] = b
    
    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(J)
    return np.sort(eigenvalues)


def gauss_legendre_quadrature(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成Gauss-Legendre求积的节点和权重。
    
    在区间 [-1, 1] 上：
        ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i * f(x_i)
    
    节点 x_i 为n阶Legendre多项式的零点。
    权重 w_i = 2 / [(1 - x_i²) * (P_n'(x_i))²]
    
    利用导数递推：
        P_n'(x) = n / (1 - x²) * [P_{n-1}(x) - x * P_n(x)]
    
    参数:
        n: 求积阶数
    
    返回:
        x: (n,) 求积节点
        w: (n,) 求积权重
    """
    x = legendre_polynomial_zeros(n)
    
    # 计算P_{n-1}(x)和P_n(x)
    V = legendre_polynomial_value(n, x)
    P_n = V[:, n]
    P_n_minus_1 = V[:, n - 1]
    
    # 计算导数 P_n'(x)
    # P_n'(x) = n * (P_{n-1}(x) - x * P_n(x)) / (1 - x^2)
    denom = 1.0 - x ** 2
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    Pn_prime = n * (P_n_minus_1 - x * P_n) / denom
    
    # 权重
    w = 2.0 / ((1.0 - x ** 2) * Pn_prime ** 2 + 1e-14)
    
    return x, w


# =============================================================================
# 差异高斯感受野模型
# =============================================================================

def dog_receptive_field(
    x: np.ndarray, y: np.ndarray,
    A_c: float, sigma_c: float,
    A_s: float, sigma_s: float
) -> np.ndarray:
    """
    计算差异高斯（DoG）感受野的空间响应分布。
    
    双极细胞的感受野由中心（center）和周边（surround）两个拮抗的高斯函数组成：
    
        RF(x,y) = A_c * exp(-(x²+y²) / (2σ_c²)) - A_s * exp(-(x²+y²) / (2σ_s²))
    
    其中：
    - A_c, A_s: 中心和周边的振幅
    - σ_c, σ_s: 中心和周边的高斯标准差（通常 σ_s > σ_c）
    - 中心-周边比（center-surround ratio）= A_c / A_s
    
    参数:
        x, y: 空间坐标网格（由np.meshgrid生成）
        A_c, sigma_c: 中心振幅和标准差
        A_s, sigma_s: 周边振幅和标准差
    
    返回:
        rf: 与x,y同形的感受野响应
    """
    r2 = x ** 2 + y ** 2
    center = A_c * np.exp(-r2 / (2.0 * sigma_c ** 2))
    surround = A_s * np.exp(-r2 / (2.0 * sigma_s ** 2))
    return center - surround


def compute_bipolar_response_convolution(
    stimulus: np.ndarray,
    rf_params: dict,
    grid_spacing: float
) -> float:
    """
    通过数值积分计算双极细胞对给定刺激的感受野卷积响应。
    
    卷积积分：
        R = ∫∫ RF(x,y) * I(x,y) dx dy
    
    使用Gauss-Legendre求积在二维上进行数值积分：
        R ≈ Σ_i Σ_j w_i * w_j * RF(x_i, y_j) * I(x_i, y_j)
    
    参数:
        stimulus: (N, M) 刺激强度分布矩阵
        rf_params: 感受野参数字典，包含A_c, sigma_c, A_s, sigma_s
        grid_spacing: 网格间距（微米）
    
    返回:
        response: 双极细胞响应
    """
    ny, nx = stimulus.shape
    
    # 创建坐标网格（以感受野中心为原点）
    x_range = (nx - 1) * grid_spacing / 2.0
    y_range = (ny - 1) * grid_spacing / 2.0
    
    # 使用Gauss-Legendre求积
    n_quad = min(16, nx, ny)
    x_nodes, x_weights = gauss_legendre_quadrature(n_quad)
    
    # 将节点从[-1,1]映射到实际空间
    x_phys = x_range * x_nodes
    y_phys = y_range * x_nodes
    
    # 计算感受野在求积节点处的值
    A_c = rf_params.get('A_c', 1.0)
    sigma_c = rf_params.get('sigma_c', 10.0)
    A_s = rf_params.get('A_s', 0.5)
    sigma_s = rf_params.get('sigma_s', 30.0)
    
    response = 0.0
    scale_x = x_range
    scale_y = y_range
    
    for i in range(n_quad):
        for j in range(n_quad):
            # 将物理坐标映射到像素索引（最近邻插值）
            xi = (x_phys[i] + x_range) / (2.0 * x_range) * (nx - 1)
            yj = (y_phys[j] + y_range) / (2.0 * y_range) * (ny - 1)
            
            ix = int(np.clip(round(xi), 0, nx - 1))
            iy = int(np.clip(round(yj), 0, ny - 1))
            
            # 感受野值
            r2 = x_phys[i] ** 2 + y_phys[j] ** 2
            rf_val = A_c * np.exp(-r2 / (2.0 * sigma_c ** 2)) - A_s * np.exp(-r2 / (2.0 * sigma_s ** 2))
            
            response += scale_x * scale_y * x_weights[i] * x_weights[j] * rf_val * stimulus[iy, ix]
    
    return float(response)


# =============================================================================
# Legendre正交基展开感受野调谐曲线
# =============================================================================

def decompose_rf_with_legendre_basis(
    spatial_profile: Callable[[float], float],
    max_degree: int,
    n_quad: int = 64
) -> np.ndarray:
    """
    将一维空间感受野轮廓用Legendre正交基展开。
    
    展开式：
        f(x) = Σ_{k=0}^{N} c_k * P_k(x),  x ∈ [-1, 1]
    
    系数通过正交投影计算：
        c_k = (2k+1)/2 * ∫_{-1}^{1} f(x) * P_k(x) dx
    
    使用Gauss-Legendre求积计算积分：
        c_k ≈ (2k+1)/2 * Σ_i w_i * f(x_i) * P_k(x_i)
    
    参数:
        spatial_profile: 感受野空间轮廓函数 f(x), x∈[-1,1]
        max_degree: 最高展开阶数
        n_quad: 求积阶数
    
    返回:
        coeffs: (max_degree+1,) Legendre展开系数
    """
    x_nodes, weights = gauss_legendre_quadrature(n_quad)
    
    # 计算Legendre多项式值
    V = legendre_polynomial_value(max_degree, x_nodes)
    
    # 计算函数值
    f_vals = np.array([spatial_profile(xi) for xi in x_nodes], dtype=np.float64)
    
    coeffs = np.zeros(max_degree + 1, dtype=np.float64)
    for k in range(max_degree + 1):
        integrand = f_vals * V[:, k]
        integral = np.sum(weights * integrand)
        coeffs[k] = (2.0 * k + 1.0) / 2.0 * integral
    
    return coeffs


def reconstruct_rf_from_legendre_coeffs(
    coeffs: np.ndarray,
    x: np.ndarray
) -> np.ndarray:
    """
    从Legendre系数重构感受野轮廓。
    
    f(x) = Σ_{k=0}^{N} c_k * P_k(x)
    
    参数:
        coeffs: (N+1,) Legendre系数
        x: (M,) 评估点
    
    返回:
        f: (M,) 重构的函数值
    """
    n = len(coeffs) - 1
    V = legendre_polynomial_value(n, x)
    return V @ coeffs
