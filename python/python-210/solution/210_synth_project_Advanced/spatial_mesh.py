"""
spatial_mesh.py - 随机空间网格引擎：区域识别与积分点生成

本模块实现随机空间中的网格生成、区域判定和数值积分，
为可靠性分析中的高维积分提供离散化基础。

1. 网格生成:
   在标准超立方体 [-1,1]^n 中生成编号网格。
   支持多种经典区域: 正方形(S)、L形(L)、圆盘(D)、环形(A)、心形(H)。

2. 区域判定:
   给定点集, 判定哪些点位于指定区域内。
   用于拒绝采样和条件概率估计。

3. 数值积分:
   基于网格的 Monte Carlo 积分:
   I ≈ (Vol/n) sum_{x_i in domain} f(x_i)

种子项目映射:
  820_numgrid      → 区域编号与网格生成
  468_geometry     → 几何判定函数
  182_circle_positive_distance → 正圆域采样
"""

import numpy as np


def numgrid(region, n):
    """在 [-1,1]^2 中生成编号网格

    种子项目 820_numgrid 的 Python 实现。

    Parameters
    ----------
    region : str
      'S' - 正方形 (全区域)
      'L' - L 形域
      'C' - 1/4 圆切割
      'D' - 圆盘
      'A' - 环形
      'H' - 心形
      'B' - 蝴蝶形外部
    n : int, 每维网格点数

    Returns
    -------
    G : np.ndarray (n, n), 网格编号 (0=域外, >0=域内编号)
    x : np.ndarray (n,), x 坐标
    y : np.ndarray (n,), y 坐标
    active_points : np.ndarray (K, 2), 域内点的坐标
    """
    if n < 2:
        raise ValueError("n 必须 >= 2")

    x1d = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(x1d, x1d[::-1])

    if region == 'S':
        mask = (x > -1) & (x < 1) & (y > -1) & (y < 1)
    elif region == 'L':
        mask = (x > -1) & (x < 1) & (y > -1) & (y < 1) & ((x > 0) | (y > 0))
    elif region == 'C':
        mask = (x > -1) & (x < 1) & (y > -1) & (y < 1) & ((x + 1) ** 2 + (y + 1) ** 2 > 1)
    elif region == 'D':
        mask = x ** 2 + y ** 2 < 1
    elif region == 'A':
        mask = (x ** 2 + y ** 2 < 1) & (x ** 2 + y ** 2 > 1.0 / 3.0)
    elif region == 'H':
        rho = 0.75
        sigma = 0.75
        mask = (x ** 2 + y ** 2) * (x ** 2 + y ** 2 - sigma * y) < rho * x ** 2
    elif region == 'B':
        t = np.arctan2(y, x)
        r = np.sqrt(x ** 2 + y ** 2)
        mask = (r >= np.sin(2.0 * t) + 0.2 * np.sin(8.0 * t)) & \
               (x > -1) & (x < 1) & (y > -1) & (y < 1)
    else:
        raise ValueError(f"未知区域类型: {region}")

    # 编号
    G = np.zeros((n, n), dtype=int)
    k = np.where(mask)
    G[k] = np.arange(1, len(k[0]) + 1)

    # 域内点坐标
    active_points = np.column_stack([x[mask], y[mask]])

    return G, x1d, x1d[::-1], active_points


def sample_in_region(region, n_points, rng=None):
    """在指定区域内均匀采样 (拒绝采样)

    Parameters
    ----------
    region : str, 区域标识
    n_points : int, 采样数

    Returns
    -------
    points : np.ndarray (n_points, 2)
    """
    rng = rng or np.random.default_rng()

    def inside(pts):
        x, y = pts[:, 0], pts[:, 1]
        if region == 'S':
            return (np.abs(x) < 1) & (np.abs(y) < 1)
        elif region == 'D':
            return x ** 2 + y ** 2 < 1
        elif region == 'A':
            r2 = x ** 2 + y ** 2
            return (r2 < 1) & (r2 > 1.0 / 3.0)
        elif region == 'L':
            return (np.abs(x) < 1) & (np.abs(y) < 1) & ((x > 0) | (y > 0))
        elif region == 'H':
            rho = 0.75
            sigma = 0.75
            r2 = x ** 2 + y ** 2
            return r2 * (r2 - sigma * y) < rho * x ** 2
        else:
            return (np.abs(x) < 1) & (np.abs(y) < 1)

    points = []
    n_rejected = 0
    batch = max(1000, n_points * 2)

    while len(points) < n_points:
        s = rng.uniform(-1, 1, size=(batch, 2))
        ok = inside(s)
        for j in range(batch):
            if ok[j]:
                points.append(s[j])
                if len(points) >= n_points:
                    break
            else:
                n_rejected += 1
        if n_rejected > 20 * n_points:
            break

    return np.array(points[:n_points])


def monte_carlo_integral(func, region, n_samples, rng=None):
    """Monte Carlo 积分: I ≈ Vol * mean(f(x_i))

    Parameters
    ----------
    func : callable (points) -> values
    region : str, 积分区域
    n_samples : int

    Returns
    -------
    integral : float
    std_error : float
    """
    rng = rng or np.random.default_rng()
    points = sample_in_region(region, n_samples, rng)
    values = func(points)
    values = np.nan_to_num(values, nan=0.0)

    # 计算区域体积 (通过 MC 估计)
    n_total = n_samples * 4  # 在 [-1,1]^2 中的总采样
    vol = 4.0 * len(points) / max(n_total, 1)

    integral = vol * np.mean(values)
    std_error = vol * np.std(values) / np.sqrt(max(len(values), 1))
    return float(integral), float(std_error)


class FailureDomainAnalyzer:
    """失效域几何分析

    利用网格和采样分析失效域 g(u)<=0 的几何特征:
    - 失效域面积/体积
    - 失效边界到原点的距离 (= 可靠度指标 beta)
    - 失效域的重心 (设计点估计)
    """

    def __init__(self, lsf, n_dim=2):
        self.lsf = lsf
        self.n_dim = n_dim

    def estimate_failure_volume(self, n_samples=5000, bound=4.0, rng=None):
        """Monte Carlo 估计失效域体积"""
        rng = rng or np.random.default_rng()
        samples = rng.uniform(-bound, bound, size=(n_samples, self.n_dim))
        g_vals = self.lsf.evaluate(samples)
        fail_ratio = np.mean(g_vals <= 0)
        vol_box = (2 * bound) ** self.n_dim
        return float(fail_ratio * vol_box)

    def estimate_boundary_distance(self, n_directions=100, rng=None):
        """通过射线法估计 beta (原点到失效面的最小距离)

        从原点向多个方向发射射线, 找到每条射线与 g(u)=0 的交点,
        取最小距离。
        """
        rng = rng or np.random.default_rng()
        min_dist = float('inf')

        for _ in range(n_directions):
            # 随机方向
            direction = rng.standard_normal(self.n_dim)
            direction /= np.linalg.norm(direction) + 1e-30

            # 二分法找交点
            t_lo, t_hi = 0.0, 10.0
            g_lo = float(np.atleast_1d(self.lsf.evaluate(
                (t_lo * direction).reshape(1, -1)))[0])
            g_hi = float(np.atleast_1d(self.lsf.evaluate(
                (t_hi * direction).reshape(1, -1)))[0])

            if g_lo <= 0:
                min_dist = 0.0
                break
            if g_hi > 0:
                continue

            for _ in range(50):
                t_mid = 0.5 * (t_lo + t_hi)
                g_mid = float(np.atleast_1d(self.lsf.evaluate(
                    (t_mid * direction).reshape(1, -1)))[0])
                if g_mid > 0:
                    t_lo = t_mid
                else:
                    t_hi = t_mid

            min_dist = min(min_dist, t_hi)

        return float(min_dist)

    def estimate_failure_centroid(self, n_samples=5000, bound=4.0, rng=None):
        """失效域重心估计"""
        rng = rng or np.random.default_rng()
        samples = rng.uniform(-bound, bound, size=(n_samples, self.n_dim))
        g_vals = self.lsf.evaluate(samples)
        fail_mask = g_vals <= 0
        if np.sum(fail_mask) == 0:
            return np.zeros(self.n_dim)
        return np.mean(samples[fail_mask], axis=0)
