"""
monte_carlo_contact.py
接触问题的蒙特卡洛不确定性分析模块
融合种子项目：
  - 888_polygon_monte_carlo（多边形区域采样与积分）
  - 935_pyramid_monte_carlo（三维金字塔区域采样）
  - 1092_snakes_and_ladders_simulation（蒙特卡洛统计框架）
"""
import numpy as np
from typing import Tuple, Callable, List
from mesh_generator import TriMesh2D


def polygon_triangulate(nv: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    简单多边形扇形三角剖分（假设多边形为凸或近似凸）。
    返回三角形索引 (nv-2, 3)。
    """
    tri = np.zeros((nv - 2, 3), dtype=int)
    for i in range(nv - 2):
        tri[i] = [0, i + 1, i + 2]
    return tri


def triangle_area_2d(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    r"""
    三角形有向面积：
    A = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    """
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def polygon_sample_uniform(vertices: np.ndarray, n_samples: int) -> np.ndarray:
    r"""
    在二维多边形内均匀采样（融合 888_polygon_monte_carlo）。

    算法：
    1. 将多边形扇形三角剖分
    2. 按面积加权随机选择三角形
    3. 在三角形内均匀采样：
       x = (1 - \sqrt{r1}) * A + \sqrt{r1} * (1 - r2) * B + \sqrt{r1} * r2 * C
    """
    nv = vertices.shape[0]
    if nv < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    tri = polygon_triangulate(nv, vertices[:, 0], vertices[:, 1])
    areas = np.array([triangle_area_2d(
        vertices[t[0], 0], vertices[t[0], 1],
        vertices[t[1], 0], vertices[t[1], 1],
        vertices[t[2], 0], vertices[t[2], 1]) for t in tri])
    total_area = np.sum(areas)
    if total_area < 1e-20:
        raise ValueError("Degenerate polygon")
    probs = areas / total_area
    cumsum = np.cumsum(probs)

    samples = np.zeros((2, n_samples))
    for j in range(n_samples):
        r_area = np.random.rand()
        tri_idx = np.searchsorted(cumsum, r_area)
        t = tri[tri_idx]
        r1, r2 = np.random.rand(2)
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        A = vertices[t[0]]
        B = vertices[t[1]]
        C = vertices[t[2]]
        samples[:, j] = (1.0 - r1 - r2) * A + r1 * B + r2 * C
    return samples


def pyramid01_sample(n: int) -> np.ndarray:
    r"""
    在单位金字塔内均匀采样（融合 935_pyramid_monte_carlo）。
    金字塔体积：V = 4/3。

    映射公式：
    z = 1 - r_3^{1/3}
    y = (1 - z) * (2r_2 - 1)
    x = (1 - z) * (2r_1 - 1)
    """
    one_third = 1.0 / 3.0
    x = np.random.rand(3, n)
    x[2, :] = 1.0 - x[2, :] ** one_third
    x[1, :] = (1.0 - x[2, :]) * (2.0 * x[1, :] - 1.0)
    x[0, :] = (1.0 - x[2, :]) * (2.0 * x[0, :] - 1.0)
    return x


def monte_carlo_contact_force_variance(mesh: TriMesh2D,
                                        pressure_sampler: Callable[[np.ndarray], np.ndarray],
                                        n_samples: int = 5000) -> Tuple[float, float, float]:
    r"""
    蒙特卡洛估计接触力的统计量（均值、方差、最大值）。
    融合 1092_snakes_and_ladders_simulation 的统计框架。

    在接触区域（网格底部）进行随机采样，统计压力分布。
    """
    # 提取底部边界作为多边形
    y_min = np.min(mesh.nodes[:, 1])
    boundary_nodes = np.where(np.abs(mesh.nodes[:, 1] - y_min) < 1e-6)[0]
    if len(boundary_nodes) < 3:
        # 退化为线段，使用均匀分布
        x_coords = mesh.nodes[boundary_nodes, 0]
        samples = np.zeros((2, n_samples))
        samples[0, :] = np.random.uniform(np.min(x_coords), np.max(x_coords), n_samples)
        samples[1, :] = y_min
    else:
        # 按 x 排序
        order = np.argsort(mesh.nodes[boundary_nodes, 0])
        poly = mesh.nodes[boundary_nodes[order]]
        # 检查是否共线（面积接近零）
        poly_area = 0.0
        for i in range(1, len(poly) - 1):
            poly_area += triangle_area_2d(poly[0,0], poly[0,1], poly[i,0], poly[i,1], poly[i+1,0], poly[i+1,1])
        if poly_area < 1e-20:
            # 共线退化为线段
            samples = np.zeros((2, n_samples))
            samples[0, :] = np.random.uniform(np.min(poly[:,0]), np.max(poly[:,0]), n_samples)
            samples[1, :] = y_min
        else:
            samples = polygon_sample_uniform(poly, n_samples)

    pressures = pressure_sampler(samples.T)
    mean_p = float(np.mean(pressures))
    var_p = float(np.var(pressures))
    max_p = float(np.max(pressures))
    return mean_p, var_p, max_p


def monte_carlo_friction_coefficient_sensitivity(
    base_solver_func: Callable[[float], float],
    mu_mean: float = 0.3,
    mu_std: float = 0.05,
    n_batches: int = 10,
    n_per_batch: int = 100
) -> dict:
    r"""
    摩擦系数随机扰动的蒙特卡洛敏感性分析。
    融合 snakes_and_ladders_simulation 的批量统计思想。

    模型：\mu \sim \mathcal{N}(\mu_{mean}, \mu_{std}^2)
    计算目标量 Q(\mu) 的统计分布。
    """
    all_results = []
    batch_stats = []
    for batch in range(n_batches):
        batch_vals = []
        for _ in range(n_per_batch):
            mu_sample = max(0.01, np.random.normal(mu_mean, mu_std))
            try:
                q = base_solver_func(mu_sample)
            except Exception:
                q = np.nan
            batch_vals.append(q)
        batch_vals = np.array(batch_vals)
        valid = batch_vals[~np.isnan(batch_vals)]
        if len(valid) > 0:
            batch_stats.append({
                "batch": batch + 1,
                "min": float(np.min(valid)),
                "mean": float(np.mean(valid)),
                "max": float(np.max(valid)),
                "std": float(np.std(valid))
            })
            all_results.extend(valid.tolist())
    all_results = np.array(all_results)
    summary = {
        "overall_mean": float(np.mean(all_results)),
        "overall_std": float(np.std(all_results)),
        "overall_min": float(np.min(all_results)),
        "overall_max": float(np.max(all_results)),
        "batch_stats": batch_stats
    }
    return summary
