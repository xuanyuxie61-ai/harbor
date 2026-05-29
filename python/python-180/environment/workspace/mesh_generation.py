"""
mesh_generation.py
自适应网格生成与节点优化

融合种子项目:
  - 676_line_cvt_lloyd: Centroidal Voronoi Tessellation (CVT) Lloyd 迭代
  - 1247_tetrahedron_grid: 四面体重心坐标网格生成
  - 427_fibonacci_spiral: 黄金比例角增量采样

科学背景:
  在 SPDE 数值积分中，空间网格的质量直接影响 stiffness matrix 的条件数
  与离散误差。CVT 最小化能量泛函:
      E(z_1,...,z_n) = sum_{i=1}^n integral_{V_i} rho(x) ||x - z_i||^2 dx,
  其中 V_i 为 Voronoi 区域，rho(x) 为密度函数。
  Lloyd 算法通过迭代映射:
      z_i^{(k+1)} = centroid(V_i^{(k)})
  收敛到 CVT 的临界点。

核心公式:
  1. 密度加权质心:
       z_i = (int_{V_i} x rho(x) dx) / (int_{V_i} rho(x) dx)
  2. 1D 区间 [a,b] 的 Voronoi 区域即为中点分割的子区间。
  3. Fibonacci 球面采样（2D 推广）:
       theta_k = 2 pi k / phi^2,   r_k = R * sqrt(k / N)
       其中 phi = (1+sqrt(5))/2 为黄金比例。
  4. 四面体重心坐标:
       x = (i*v1 + j*v2 + k*v3 + l*v4) / n,  i+j+k+l=n
"""

import numpy as np
from typing import Callable, Optional, Tuple


def cvt_lloyd_1d(n: int,
                 a: float,
                 b: float,
                 density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 it_num: int = 100,
                 tol: float = 1e-10) -> np.ndarray:
    """
    一维 CVT Lloyd 迭代。

    输入:
        n: 生成器数量
        a, b: 区间端点
        density: 密度函数 rho(x)，默认均匀密度 rho(x)=1
        it_num: 最大迭代次数
        tol: 收敛容差

    输出:
        x: 优化后的生成器位置，shape (n,)
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if b <= a:
        raise ValueError("b must be > a")
    if it_num < 1:
        raise ValueError("it_num must be >= 1")

    if density is None:
        density = lambda x: np.ones_like(x, dtype=np.float64)

    # 初始均匀分布
    x = np.linspace(a, b, n, dtype=np.float64)

    for it in range(it_num):
        # 计算 Voronoi 边界: 相邻生成器的中点
        boundaries = np.zeros(n + 1, dtype=np.float64)
        boundaries[0] = a
        boundaries[-1] = b
        boundaries[1:-1] = 0.5 * (x[:-1] + x[1:])

        x_new = np.zeros(n, dtype=np.float64)
        for i in range(n):
            left = boundaries[i]
            right = boundaries[i + 1]
            # 数值积分：Simpson 法则，至少 5 个采样点
            nquad = max(5, int(100 * (right - left) / (b - a)) + 1)
            t = np.linspace(left, right, nquad)
            w = np.ones(nquad, dtype=np.float64)
            w[0] = 0.5
            w[-1] = 0.5
            w[1:-1:2] = 2.0
            w[2:-1:2] = 4.0
            w *= (right - left) / (3.0 * (nquad - 1) // 2 * 2) if (nquad % 2 == 1) else (right - left) / (nquad - 1)
            if nquad % 2 == 0:
                # 偶数点 fallback 到梯形法则保证稳定
                w = np.ones(nquad, dtype=np.float64)
                w[0] = 0.5
                w[-1] = 0.5
                w *= (right - left) / (nquad - 1)

            rho_vals = density(t)
            # 边界鲁棒性：防止密度为负或零
            rho_vals = np.clip(rho_vals, 1e-12, None)
            mass = np.sum(w * rho_vals)
            moment = np.sum(w * t * rho_vals)
            x_new[i] = moment / mass if mass > 0 else 0.5 * (left + right)

        # 排序并保持边界
        x_new = np.clip(np.sort(x_new), a, b)

        diff = np.max(np.abs(x_new - x))
        x = x_new
        if diff < tol:
            break

    return x


def tetrahedron_grid_count(n: int) -> int:
    """
    四面体网格点数量: C(n+3, 3) = (n+3)(n+2)(n+1)/6
    """
    if n < 0:
        return 0
    return (n + 3) * (n + 2) * (n + 1) // 6


def tetrahedron_grid(n: int, vertices: np.ndarray) -> np.ndarray:
    """
    生成标准四面体 [0,1]^3 上的重心坐标网格点，再映射到一般四面体。

    输入:
        n: 每条边分割数
        vertices: shape (3, 4)，四面体四个顶点坐标

    输出:
        tg: shape (3, ng)，网格点
    """
    if vertices.shape != (3, 4):
        raise ValueError("vertices must have shape (3, 4)")
    ng = tetrahedron_grid_count(n)
    tg = np.zeros((3, ng), dtype=np.float64)
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                coeff = np.array([i, j, k, l], dtype=np.float64) / n
                tg[:, p] = vertices @ coeff
                p += 1
    return tg


def fibonacci_spiral_disk(n: int, R: float = 1.0) -> np.ndarray:
    """
    Fibonacci 螺旋在圆盘上的均匀采样。

    黄金比例:
        phi = (1 + sqrt(5)) / 2
    采样公式:
        r_k = R * sqrt(k / (N - 0.5))
        theta_k = 2 * pi * k / phi^2
        x_k = r_k * cos(theta_k)
        y_k = r_k * sin(theta_k)

    该分布最小化离散 Riesz s-能量，具有拟最优的均匀性。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi / (phi ** 2)

    k = np.arange(n, dtype=np.float64)
    r = R * np.sqrt(k / (n - 0.5))
    theta = k * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def adaptive_density_function(x: np.ndarray,
                              steepness: float = 10.0,
                              center: float = 0.5) -> np.ndarray:
    """
    自适应密度函数：在中心区域加密。
    基于 Fisher-KPP 波前解的梯度集中特性:
        rho(x) ~ exp(- steepness * |x - center|) + 0.1
    确保最小密度不为零，避免数值奇异性。
    """
    dx = np.abs(x - center)
    rho = np.exp(-steepness * dx) + 0.1
    return rho


def generate_composite_mesh_1d(n_base: int = 65,
                               a: float = 0.0,
                               b: float = 1.0,
                               steepness: float = 12.0,
                               center: float = 0.5) -> np.ndarray:
    """
    生成一维复合自适应网格：先 CVT 优化，再在每个 Voronoi 区间内均匀加密。
    """
    density = lambda x: adaptive_density_function(x, steepness=steepness, center=center)
    cvt_pts = cvt_lloyd_1d(n_base, a, b, density=density, it_num=80, tol=1e-12)
    return cvt_pts
