"""
adaptive_grid.py
================
基于质心 Voronoi 剖分 (CVT) 的自适应采样与三角插值

融合原始项目：
  - 253_cvt_circle_nonuniform：非均匀密度下的 CVT 计算
  - 596_interp_trig：三角函数插值（周期信号重构）

数学物理模型：
  1. CVT 能量泛函（非均匀密度 ρ(x)）：
       F(P) = Σ_{i=1}^k ∫_{V_i} ρ(x) ||x - p_i||² dx
     其中 V_i 为第 i 个生成元 p_i 的 Voronoi 区域。
     最优配置满足 p_i 为 V_i 关于密度 ρ 的质心：
       p_i = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx

  2. Lloyd 迭代算法：
       a) 在当前生成元处构造 Voronoi 剖分
       b) 将每个生成元移到其 Voronoi 区域的质心
       c) 重复直至收敛

  3. 三角函数插值（用于周期 chemoattractant 信号）：
       对 n 个等距节点 {x_j} 上的数据 {y_j}，周期为 nh 的三角插值：
         I(x) = Σ_{j=1}^n y_j · τ_j(x)
       其中 τ_j 为三角基函数（cardinal function）：
         n 为奇数：τ_j(x) = sin(π(x-x_j)/h) / [ n sin(π(x-x_j)/(nh)) ]
         n 为偶数：τ_j(x) = sin(π(x-x_j)/h) / [ n tan(π(x-x_j)/(nh)) ]
"""

import numpy as np


# ---------------------------------------------------------------------------
# Trigonometric Interpolation (from 596_interp_trig)
# ---------------------------------------------------------------------------
def trigcardinal(xi, xdj, nd, h):
    """
    三角基函数（cardinal function）。

    公式：
        nd 为奇数：τ(x) = sin(π(x-x_j)/h) / [ nd · sin(π(x-x_j)/(nd·h)) ]
        nd 为偶数：τ(x) = sin(π(x-x_j)/h) / [ nd · tan(π(x-x_j)/(nd·h)) ]

    在 x = x_j 处 τ = 1，在其他数据节点处 τ = 0。
    """
    xi = np.asarray(xi, dtype=float)
    diff = np.pi * (xi - xdj) / h
    # 避免除零
    if nd % 2 == 1:
        denom = nd * np.sin(diff / nd)
    else:
        denom = nd * np.tan(diff / nd)
    tau = np.sin(diff) / (denom + 1e-300)
    # 精确修复插值节点处的值
    tau[np.isclose(xi, xdj, atol=1e-12)] = 1.0
    return tau


def trig_interpolant(xd, yd, xi):
    """
    对等距数据节点 xd 上的值 yd 进行三角插值，在点 xi 处求值。

    参数
    ----
    xd : np.ndarray, shape (nd,)
        等距数据节点（假定间隔 h = xd[1]-xd[0]）
    yd : np.ndarray, shape (nd,)
        数据值
    xi : np.ndarray 或 float
        插值目标点

    返回
    ----
    yi : np.ndarray 或 float
        插值结果
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    nd = xd.size
    if nd < 2:
        raise ValueError("trig_interpolant: 至少需要 2 个数据点")
    h = xd[1] - xd[0]
    if abs(h) < 1e-15:
        raise ValueError("trig_interpolant: 节点间隔为零")

    scalar_input = np.isscalar(xi)
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=float))
    yi = np.zeros_like(xi_arr)
    for j in range(nd):
        yi += yd[j] * trigcardinal(xi_arr, xd[j], nd, h)
    return float(yi[0]) if scalar_input else yi


# ---------------------------------------------------------------------------
# CVT with nonuniform density (from 253_cvt_circle_nonuniform)
# ---------------------------------------------------------------------------
def cvt_circle_nonuniform(n_generators: int,
                          density_func,
                          n_samples: int = None,
                          n_iterations: int = 20,
                          domain_radius: float = 1.0):
    """
    在圆形区域内计算非均匀密度下的 CVT。

    参数
    ----
    n_generators : int
        生成元数量
    density_func : callable
        ρ(x,y) -> float，非负密度函数
    n_samples : int
        采样点数，默认 5000 * n_generators
    n_iterations : int
        Lloyd 迭代次数
    domain_radius : float
        圆形区域半径

    返回
    ----
    generators : np.ndarray, shape (n_generators, 2)
        CVT 生成元坐标
    """
    n_generators = max(1, int(n_generators))
    if n_samples is None:
        n_samples = 5000 * n_generators
    n_samples = max(n_generators * 10, int(n_samples))

    # 初始生成元：在圆内均匀随机分布
    rng = np.random.default_rng(seed=42)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_generators)
    r = domain_radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_generators))
    generators = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    for it in range(n_iterations):
        # 生成非均匀采样点（拒绝采样）
        samples = np.zeros((n_samples, 2), dtype=float)
        accepted = 0
        max_density = 0.0
        # 粗略估计最大密度（在生成元附近采样）
        for g in generators:
            max_density = max(max_density, density_func(g))
        max_density = max(max_density, 1.0) * 1.5

        while accepted < n_samples:
            batch = min(n_samples - accepted, 2000)
            cand_theta = rng.uniform(0.0, 2.0 * np.pi, size=batch)
            cand_r = domain_radius * np.sqrt(rng.uniform(0.0, 1.0, size=batch))
            cand = np.column_stack([cand_r * np.cos(cand_theta),
                                    cand_r * np.sin(cand_theta)])
            rho_vals = np.array([density_func(pt) for pt in cand])
            mask = rng.uniform(0.0, max_density, size=batch) <= rho_vals
            n_acc = int(np.sum(mask))
            end = min(accepted + n_acc, n_samples)
            samples[accepted:end] = cand[mask][:end - accepted]
            accepted = end

        # 将每个样本分配到最近的生成元
        # 使用向量化距离计算
        diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]  # (n_samples, n_gen, 2)
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)

        # 计算每个 Voronoi 区域的加权质心
        new_generators = np.zeros_like(generators)
        for i in range(n_generators):
            mask = nearest == i
            if np.sum(mask) == 0:
                new_generators[i] = generators[i]
            else:
                pts = samples[mask]
                rhos = np.array([density_func(pt) for pt in pts])
                weights = rhos + 1e-15
                new_generators[i] = np.average(pts, axis=0, weights=weights)

        # 限制在圆内
        norms = np.linalg.norm(new_generators, axis=1)
        scale = np.where(norms > domain_radius, domain_radius / (norms + 1e-15), 1.0)
        new_generators *= scale[:, np.newaxis]
        generators = new_generators

    return generators


class AdaptiveChemotaxisSampler:
    """
    基于 CVT 的自适应采样器，用于在 chemoattractant 高浓度区域加密采样。
    """

    def __init__(self, concentration_field_func, domain=((-1, 1), (-1, 1))):
        self.c_func = concentration_field_func
        self.domain = domain

    def density_at(self, pt):
        """
        采样密度与浓度成正比（高浓度区域更多采样点）。
        """
        c = self.c_func(pt)
        return max(0.0, c)

    def sample_adaptive(self, n_points: int = 30, n_iter: int = 15):
        """
        返回自适应分布的采样点坐标。
        """
        # 将密度函数包装到单位圆（通过坐标缩放）
        xmid = 0.5 * (self.domain[0][0] + self.domain[0][1])
        ymid = 0.5 * (self.domain[1][0] + self.domain[1][1])
        rx = 0.5 * (self.domain[0][1] - self.domain[0][0])
        ry = 0.5 * (self.domain[1][1] - self.domain[1][0])

        def circle_density(q):
            x = xmid + rx * q[0]
            y = ymid + ry * q[1]
            return self.density_at(np.array([x, y]))

        gens = cvt_circle_nonuniform(n_points, circle_density,
                                      n_iterations=n_iter, domain_radius=1.0)
        # 映射回原始区域
        points = np.zeros_like(gens)
        points[:, 0] = xmid + rx * gens[:, 0]
        points[:, 1] = ymid + ry * gens[:, 1]
        return points

    def interpolate_periodic_signal(self, t_values, signal_values, t_query):
        """
        对周期化学信号使用三角插值进行重构。

        参数
        ----
        t_values : np.ndarray
            等距时间采样点（已排序）
        signal_values : np.ndarray
            对应信号值
        t_query : np.ndarray 或 float
            查询时间点

        返回
        ----
        s_query : np.ndarray 或 float
            插值信号值
        """
        return trig_interpolant(t_values, signal_values, t_query)
