"""
adaptive_meshing.py
===================
基于质心 Voronoi 剖分的自适应节点布置与参数空间映射

本模块将以下种子项目的核心算法融入结构力学：
  - 260_cvt_square_pdf_discrete : Lloyd CVT 迭代、离散 PDF 采样 → 柔性构件自适应节点加密
  - 1052_sammon_data : 单纯形坐标、螺旋/圆参数曲线 → 弯曲梁中心线参数空间映射

核心物理模型：
  - 对柔性梁的曲率分布 ρ(x) 作为采样密度，使用 CVT 在梁轴向上布置节点，
    使得节点密度与曲率成正比（高曲率区更密）。
  - CVT 最优性条件：每个生成子 g_i 恰好为其 Voronoi 单元 V_i 的质心：
        g_i = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx
  - Lloyd 算法迭代：
        1. 从离散 PDF 采样大量点
        2. 将每个点分配到最近的生成子
        3. 更新生成子为单元样本均值
        4. 重复直至收敛
  - 对空间弯曲梁，用三维螺旋/圆参数化中心线：
        r(s) = [R cos(s/R), R sin(s/R), h s / (2πR)]
    其中 s 为弧长参数，R 为曲率半径，h 为螺距。
"""

import numpy as np
from typing import Tuple, List, Callable


def discrete_pdf_sample_1d(pdf_values: np.ndarray, x_grid: np.ndarray,
                           n_samples: int) -> np.ndarray:
    """
    基于 260_cvt_square_pdf_discrete 思想，在一维离散网格上按概率密度采样。
    
    步骤：
        1. 归一化 pdf → 概率质量函数
        2. 累加得 CDF
        3. 生成均匀随机数 u ~ U(0,1)
        4. 逆变换采样：x = CDF^{-1}(u)
    """
    pdf = np.asarray(pdf_values, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    if len(pdf) != len(x):
        raise ValueError("pdf_values 与 x_grid 长度必须一致")
    if np.any(pdf < 0):
        raise ValueError("PDF 值必须非负")
    total = np.trapezoid(pdf, x)
    if total <= 1e-18:
        raise ValueError("PDF 积分为零")
    pdf_norm = pdf / total
    # 数值积分得 CDF
    cdf = np.zeros_like(x)
    for i in range(1, len(x)):
        cdf[i] = cdf[i - 1] + 0.5 * (pdf_norm[i - 1] + pdf_norm[i]) * (x[i] - x[i - 1])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    u = np.random.rand(n_samples)
    # 逆变换：在 CDF 表中查找
    samples = np.interp(u, cdf, x)
    return samples


def cvt_1d_lloyd(n_generators: int, pdf_func: Callable[[np.ndarray], np.ndarray],
                 x_range: Tuple[float, float], n_samples: int = 20000,
                 it_max: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    一维 Lloyd CVT 迭代，在区间 [a,b] 上按密度 pdf_func 布置生成子。
    
    算法：
        1. 初始化生成子为均匀分布
        2. 每步从 pdf 采样 n_samples 个点
        3. 将每个点分配到最近的生成子
        4. 更新生成子为单元内样本的加权平均
        5. 当生成子最大移动量 < tol 时收敛
    
    返回
    ----
    generators : 按升序排列的生成子坐标
    """
    a, b = x_range
    if a >= b:
        raise ValueError("区间无效")
    gens = np.linspace(a, b, n_generators)
    x_fine = np.linspace(a, b, 2000)
    pdf_fine = pdf_func(x_fine)
    # 保证 pdf 非负
    pdf_fine = np.maximum(pdf_fine, 0.0)
    for it in range(it_max):
        samples = discrete_pdf_sample_1d(pdf_fine, x_fine, n_samples)
        # 分配
        dist = np.abs(samples[:, None] - gens[None, :])
        labels = np.argmin(dist, axis=1)
        new_gens = np.zeros_like(gens)
        moved = False
        for k in range(n_generators):
            mask = labels == k
            if np.sum(mask) == 0:
                # 空单元：重新随机放置
                new_gens[k] = np.random.uniform(a, b)
                moved = True
            else:
                new_gens[k] = np.mean(samples[mask])
        new_gens = np.sort(new_gens)
        shift = np.max(np.abs(new_gens - gens))
        gens = new_gens
        if shift < tol:
            break
    return gens


def curvature_density(x: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    由梁挠度曲线 w(x) 计算曲率近似密度：
        κ(x) ≈ |w''(x)| / (1 + (w'(x))²)^{3/2}
    使用中心差分近似二阶导。
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(displacement, dtype=np.float64)
    if len(x) != len(w):
        raise ValueError("x 与 displacement 长度不一致")
    n = len(x)
    if n < 3:
        return np.zeros_like(x)
    dx = np.gradient(x)
    dw = np.gradient(w, x)
    d2w = np.gradient(dw, x)
    curvature = np.abs(d2w) / (1.0 + dw ** 2) ** 1.5
    # 平滑处理避免零点导致 PDF 退化
    curvature = np.maximum(curvature, 1e-6 * np.max(curvature))
    return curvature


def simplex_coordinates_nd(n: int) -> np.ndarray:
    """
    基于 1052_sammon_data 中 simplex_coordinates2 思想，构造 n 维正则单纯形顶点。
    
    顶点位于原点中心，顶点到质心距离为 1。构造方法：
        v_0 = (1, 0, ..., 0)
        v_1 = (a, b, 0, ..., 0)
        ...
        v_n = (a, a, ..., a, b)
    其中 a = (1 - √(n+1)) / n, b = √(1 - n a²)。
    然后整体平移至质心在原点。
    """
    if n < 1:
        raise ValueError("维度 n 必须 ≥ 1")
    a = (1.0 - np.sqrt(n + 1.0)) / n
    b_sq = 1.0 - n * a * a
    if b_sq < 0:
        raise ValueError("数值不稳定，b² < 0")
    b = np.sqrt(b_sq)
    verts = np.zeros((n + 1, n), dtype=np.float64)
    for i in range(n + 1):
        if i < n:
            verts[i, :i] = a
            verts[i, i] = (1.0 if i == 0 else b)
        else:
            verts[i, :] = a
    # 平移至质心在原点
    centroid = verts.mean(axis=0)
    verts -= centroid
    return verts


def helix_parametrization(s: np.ndarray, R: float, pitch: float) -> np.ndarray:
    """
    三维螺旋线（圆柱螺旋）参数方程，以弧长 s 为参数：
        x(s) = R cos(s / R)
        y(s) = R sin(s / R)
        z(s) = pitch · s / (2π R)
    
    返回
    ----
    coords : (len(s), 3) 空间坐标
    """
    s = np.asarray(s, dtype=np.float64)
    if R <= 0:
        raise ValueError("曲率半径 R 必须为正")
    theta = s / R
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = pitch * theta / (2.0 * np.pi)
    return np.column_stack((x, y, z))


def circle_parametrization(theta: np.ndarray, R: float) -> np.ndarray:
    """
    二维圆参数方程，用于环形柔性构件中心线。
    """
    theta = np.asarray(theta, dtype=np.float64)
    return np.column_stack((R * np.cos(theta), R * np.sin(theta)))


def map_nodes_to_space_curve(s_nodes: np.ndarray, curve_func: Callable[[np.ndarray], np.ndarray],
                             tangent_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    将一维节点坐标 s_nodes 映射到空间参数曲线，并计算局部切向量。
    
    返回
    ----
    coords : (n, 3) 空间坐标
    tangents : (n, 3) 单位切向量
    """
    coords = curve_func(s_nodes)
    # 数值切向量
    tangents = np.zeros_like(coords)
    if len(s_nodes) >= 2:
        tangents[0] = coords[1] - coords[0]
        tangents[-1] = coords[-1] - coords[-2]
        tangents[1:-1] = coords[2:] - coords[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-14, 1.0, norms)
    tangents = tangents / norms
    return coords, tangents
