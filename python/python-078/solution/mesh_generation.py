"""
mesh_generation.py
血管截面CVT自适应网格生成

融合来源:
- 260_cvt_square_pdf_discrete: 基于离散PDF的CVT Lloyd迭代、逆变换采样
- 239_cvt_1_movie: CVT单步迭代、最近邻搜索、能量泛函计算

科学背景:
在动脉脉动流的有限元/有限体积计算中，网格质量直接影响数值精度。
血管壁厚度在分叉处和弯曲处呈现非均匀分布（外弧厚、内弧薄）。
Centroidal Voronoi Tessellation (CVT) 提供了一种基于密度函数的自适应网格生成方法，
使生成器自动聚集在密度高的区域（如壁面剪切应力高的区域）。

CVT能量泛函:
    E(r_1, ..., r_N) = ∫_Ω ρ(x) min_j ||x - r_j||² dx

其中ρ(x)为密度函数（此处取血管壁厚度的倒数，厚处稀疏、薄处密集）。
Lloyd算法通过迭代将生成器移动到对应Voronoi单元的质心来最小化E。
"""

import numpy as np
from typing import Tuple


# ======================================================================
# 来自 239_cvt_1_movie / 260_cvt_square_pdf_discrete 的核心算法
# ======================================================================

def find_closest(generators: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    暴力最近邻搜索：为每个采样点找到最近的生成器索引。

    参数:
        generators: (N, 2) 生成器坐标
        samples: (M, 2) 采样点坐标

    返回:
        indices: (M,) 最近生成器索引
    """
    n_gen = generators.shape[0]
    n_samp = samples.shape[0]
    nearest = np.zeros(n_samp, dtype=int)

    for i in range(n_samp):
        dists = np.sum((generators - samples[i]) ** 2, axis=1)
        nearest[i] = int(np.argmin(dists))
    return nearest


def cvt_iterate(generators: np.ndarray, n_samples: int,
                density_func=None, seed: int = None) -> Tuple[np.ndarray, float, float]:
    """
    执行一次CVT Lloyd迭代。

    算法步骤:
    1. 在区域内生成 n_samples 个随机采样点（可按密度加权）
    2. 对每个采样点，找到最近的生成器
    3. 将属于同一单元的采样点取平均，得到新的质心
    4. 用质心替换原生成器

    参数:
        generators: (N, 2) 当前生成器坐标
        n_samples: 采样点数量（通常为生成器数量的50-100倍）
        density_func: 可选的密度函数，接受(x,y)向量返回密度值
        seed: 随机种子

    返回:
        new_generators: (N, 2) 新生成器坐标
        diff: 新旧生成器的L2位移之和
        energy: 离散化CVT能量
    """
    if seed is not None:
        np.random.seed(seed)

    n_gen = generators.shape[0]
    if n_gen < 1:
        return generators.copy(), 0.0, 0.0

    # 采样
    if density_func is None:
        samples = np.random.rand(n_samples, 2)
    else:
        # 接受-拒绝采样或逆变换采样（简化：均匀采样后按密度加权重采样）
        # 先生成较多候选点
        candidates = np.random.rand(n_samples * 5, 2)
        weights = density_func(candidates[:, 0], candidates[:, 1])
        weights = np.maximum(weights, 1e-15)
        weights /= weights.sum()
        idx = np.random.choice(len(candidates), size=n_samples, p=weights, replace=True)
        samples = candidates[idx]

    # 确保采样点在[0,1]²内
    samples = np.clip(samples, 0.0, 1.0)

    nearest = find_closest(generators, samples)
    new_generators = np.zeros_like(generators)
    counts = np.zeros(n_gen)

    for j in range(n_gen):
        mask = (nearest == j)
        if np.any(mask):
            new_generators[j] = np.mean(samples[mask], axis=0)
            counts[j] = np.sum(mask)
        else:
            # 空单元：随机重置
            new_generators[j] = np.random.rand(2)
            counts[j] = 1

    # 计算位移
    diff = float(np.sum(np.linalg.norm(new_generators - generators, axis=1)))

    # 计算离散能量 E ≈ (1/N_s) Σ ρ(x_i) min_j ||x_i - r_j||²
    energy = 0.0
    for i in range(n_samples):
        j = nearest[i]
        d2 = np.sum((samples[i] - generators[j]) ** 2)
        if density_func is not None:
            rho = density_func(np.array([samples[i, 0]]), np.array([samples[i, 1]]))[0]
            energy += rho * d2
        else:
            energy += d2
    energy /= n_samples

    return new_generators, diff, energy


def generate_cvt_mesh(n_generators: int, n_samples_per_gen: int = 100,
                      max_iter: int = 100, tol: float = 1e-5,
                      density_func=None, seed: int = None) -> np.ndarray:
    """
    生成CVT网格：迭代直到收敛。

    参数:
        n_generators: 生成器数量
        n_samples_per_gen: 每个生成器对应的采样点数
        max_iter: 最大迭代次数
        tol: 收敛容差（生成器总位移 < tol）
        density_func: 密度函数
        seed: 随机种子

    返回:
        generators: (N, 2) 最终生成器坐标
    """
    if seed is not None:
        np.random.seed(seed)

    generators = np.random.rand(n_generators, 2)
    n_samples = n_generators * n_samples_per_gen

    for it in range(max_iter):
        generators, diff, energy = cvt_iterate(
            generators, n_samples, density_func=density_func, seed=None
        )
        if diff < tol:
            break

    return generators


# ======================================================================
# 血管特异性密度函数
# ======================================================================

def vessel_wall_density(x: np.ndarray, y: np.ndarray,
                        thickness_center: float = 0.5,
                        thickness_amplitude: float = 0.3,
                        n_modes: int = 3) -> np.ndarray:
    """
    模拟血管壁厚度的非均匀空间分布，作为CVT密度函数的倒数。

    模型假设:
    - 血管壁厚度在分叉外弧处增加，在内弧处减小
    - 密度 ρ ∝ 1 / thickness，即厚处网格稀疏、薄处网格密集

    参数:
        x, y: 归一化坐标（0到1）
        thickness_center: 平均厚度
        thickness_amplitude: 厚度变化幅度
        n_modes: 空间变化模态数

    返回:
        density: 与输入同形的密度值
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    thickness = thickness_center * np.ones_like(x)

    for k in range(1, n_modes + 1):
        thickness += thickness_amplitude * (
            np.sin(2.0 * np.pi * k * x) * np.cos(2.0 * np.pi * k * y)
        ) / k

    thickness = np.clip(thickness, 0.1 * thickness_center, 3.0 * thickness_center)
    density = 1.0 / thickness
    return density


def map_cvt_to_annulus(generators: np.ndarray,
                       inner_radius: float, outer_radius: float) -> np.ndarray:
    """
    将单位正方形[0,1]²上的CVT生成器映射到环形血管截面区域。

    映射关系（极坐标）:
        r = r_inner + y * (r_outer - r_inner)
        θ = 2π x

    参数:
        generators: (N, 2) 单位正方形内坐标 [x,y]
        inner_radius: 血管内半径 [m]（血液流动区域）
        outer_radius: 血管外半径 [m]（含壁厚）

    返回:
        points: (N, 2) 笛卡尔坐标 [x_cart, y_cart]
    """
    x = generators[:, 0]
    y = generators[:, 1]

    theta = 2.0 * np.pi * x
    r = inner_radius + y * (outer_radius - inner_radius)

    x_cart = r * np.cos(theta)
    y_cart = r * np.sin(theta)
    return np.column_stack([x_cart, y_cart])


# ======================================================================
# 辅助数据结构
# ======================================================================

class VascularCVTMesh:
    """
    血管截面CVT网格封装。
    包含生成器、Voronoi单元质心、以及到几何参数的映射。
    """
    def __init__(self, generators: np.ndarray, inner_r: float, outer_r: float):
        self.generators = generators.copy()
        self.inner_radius = inner_r
        self.outer_radius = outer_r
        self.cartesian = map_cvt_to_annulus(generators, inner_r, outer_r)

    def radial_coordinates(self) -> np.ndarray:
        """返回各生成器的径向坐标。"""
        pts = self.cartesian
        return np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

    def angular_coordinates(self) -> np.ndarray:
        """返回各生成器的角坐标。"""
        pts = self.cartesian
        return np.arctan2(pts[:, 1], pts[:, 0])

    def wall_thickness_distribution(self,
                                     thickness_center: float = 1.0e-3,
                                     amplitude: float = 0.3e-3) -> np.ndarray:
        """
        基于生成器位置估算局部壁厚。
        外半径 - 内半径 = 壁厚，此处模拟非均匀壁厚。
        """
        theta = self.angular_coordinates()
        # 模拟分叉处外弧壁厚增加
        thickness = thickness_center + amplitude * np.cos(theta)
        return np.clip(thickness, 0.2e-3, 2.0e-3)
