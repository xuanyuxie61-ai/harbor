"""
集合预报参数采样与插值模块 (Ensemble Generator & Interpolation)

集成种子项目:
- 558_hypercube_grid: 多维超立方体网格生成
- 792_nearest_interp_1d: 一维最近邻插值

科学背景:
  中尺度对流系统集合预报需要在大规模参数空间中生成代表性样本.
  使用结构化张量积网格 (hypercube grid) 对关键参数进行采样:
    - 初始水汽扰动幅度
    - 微物理凝结时间尺度
    - 边界层交换系数
    - 地表热通量

  对探空廓线数据, 使用最近邻插值快速生成垂直剖面.

核心公式:
  网格生成 (含 5 种居中方式):
    1. Uniform:    包含端点
    2. Interior:   不含端点
    3. Left-closed:  含左端点
    4. Right-closed: 含右端点
    5. Midpoint:   中点偏移

  最近邻插值:
    f̂(x) = f(x_{k*}),  k* = argmin_k |x - x_k|
"""

import numpy as np
from typing import List, Tuple


def hypercube_grid(dim: int, ns: List[int], bounds: List[Tuple[float, float]],
                   centering: List[int] = None) -> np.ndarray:
    """
    生成 M 维超立方体上的结构化张量积网格 (基于 558_hypercube_grid).

    参数:
      dim: 维度
      ns:  每维的格点数
      bounds: 每维的 (min, max)
      centering: 每维的居中方式 (默认全部 Uniform)
        0=Uniform, 1=Interior, 2=Left-closed, 3=Right-closed, 4=Midpoint

    返回:
      grid: (N, dim) 数组, N = prod(ns)
    """
    if centering is None:
        centering = [0] * dim

    # 生成每维的 1D 网格
    grids_1d = []
    for d in range(dim):
        a, b = bounds[d]
        n = max(2, ns[d])
        ctype = centering[d]

        if ctype == 0:  # Uniform
            x = np.linspace(a, b, n)
        elif ctype == 1:  # Interior
            if n == 1:
                x = np.array([(a+b)/2.0])
            else:
                dx = (b - a) / (n + 1.0)
                x = a + dx * np.arange(1, n + 1)
        elif ctype == 2:  # Left-closed
            dx = (b - a) / n
            x = a + dx * np.arange(n)
        elif ctype == 3:  # Right-closed
            dx = (b - a) / n
            x = a + dx * np.arange(1, n + 1)
        elif ctype == 4:  # Midpoint
            dx = (b - a) / n
            x = a + dx * (np.arange(n) + 0.5)
        else:
            x = np.linspace(a, b, n)
        grids_1d.append(x)

    # 张量积构造
    N = int(np.prod(ns))
    grid = np.zeros((N, dim))

    # 使用 stride-based 直接积构造
    idx = 0
    def recurse(d, current):
        nonlocal idx
        if d == dim:
            grid[idx, :] = current
            idx += 1
            return
        for val in grids_1d[d]:
            recurse(d + 1, current + [val])

    recurse(0, [])
    return grid


def nearest_interp_1d(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    一维最近邻插值 (基于 792_nearest_interp_1d).

    对每个查询点 xi, 找到最近的 xd 并返回对应 yd.
    """
    xd = np.asarray(xd)
    yd = np.asarray(yd)
    xi = np.asarray(xi)
    result = np.zeros_like(xi)

    for i, x in enumerate(xi.flat):
        # 寻找最近邻
        dist = np.abs(xd - x)
        idx = int(np.argmin(dist))
        result.flat[i] = yd[idx]
    return result


class EnsembleParameterSampler:
    """
    集合预报参数采样器.
    """

    def __init__(self, param_names: List[str], param_bounds: List[Tuple[float, float]],
                 samples_per_dim: int = 3):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.dim = len(param_names)
        self.samples_per_dim = samples_per_dim
        self.grid = hypercube_grid(self.dim, [samples_per_dim] * self.dim,
                                   param_bounds, centering=[1] * self.dim)
        self.n_ensemble = len(self.grid)

    def get_member_params(self, member_idx: int) -> dict:
        """获取第 member_idx 个集合成员的参数."""
        if member_idx < 0 or member_idx >= self.n_ensemble:
            raise IndexError("Member index out of range")
        return {name: float(self.grid[member_idx, d])
                for d, name in enumerate(self.param_names)}


class SoundingProfileInterpolator:
    """
    探空廓线最近邻插值器.
    """

    def __init__(self, pressure_levels: np.ndarray, temperature: np.ndarray,
                 dewpoint: np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray):
        self.p_src = np.asarray(pressure_levels)
        self.T_src = np.asarray(temperature)
        self.Td_src = np.asarray(dewpoint)
        self.u_src = np.asarray(wind_u)
        self.v_src = np.asarray(wind_v)
        # 保护排序 (气压递减)
        if len(self.p_src) > 1 and self.p_src[0] < self.p_src[1]:
            self.p_src = self.p_src[::-1]
            self.T_src = self.T_src[::-1]
            self.Td_src = self.Td_src[::-1]
            self.u_src = self.u_src[::-1]
            self.v_src = self.v_src[::-1]

    def interpolate(self, p_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray, np.ndarray]:
        """
        对目标气压层进行最近邻插值.
        """
        T = nearest_interp_1d(self.p_src, self.T_src, p_target)
        Td = nearest_interp_1d(self.p_src, self.Td_src, p_target)
        u = nearest_interp_1d(self.p_src, self.u_src, p_target)
        v = nearest_interp_1d(self.p_src, self.v_src, p_target)
        return p_target, T, Td, u, v
