"""
travel_time_tree.py
================================================================================
地下水质点运移的 travel-time 树结构与反向追踪模块

基于种子项目：
  - 196_collatz：离散动力系统的逆映射、树层级遍历与周期分析

科学背景：
  在地下水污染修复中，确定污染物从源区到抽取井的 travel time（运移时间）
  分布至关重要。正向粒子追踪（forward particle tracking）模拟质点随流场的
  运动；反向粒子追踪（backward particle tracking）则从观测井出发，逆向追踪
  可能的污染源区。

  将 collatz 的逆映射树结构概念迁移到地下水领域：
    - 正向映射 F：给定位置 x(t)，计算 x(t+Δt) = x(t) + v(x) Δt
    - 反向映射 F^{-1}：给定位置 x(t+Δt)，寻找所有可能的 x(t)
      由于弥散作用，单个下游点可能对应多个上游前像（preimage），
      形成分支树结构。

  对于纯对流问题（忽略弥散），反向映射是确定性的；
  考虑弥散时，反向映射在高斯随机游走的意义下形成概率树。

  Travel-time 概率密度函数（TTD）定义为：
      g(t) = |∂x/∂t|^{-1}  在流管截面上归一化
  或等价地通过粒子到达时间直方图估计：
      g(t) ≈ (1/N_p) Σ_{p=1}^{N_p} δ(t - τ_p)
================================================================================
"""

import numpy as np
from typing import List, Callable, Optional
from collections import deque


class TravelTimeTree:
    """
    Travel-time 树：用于反向追踪污染物潜在源区的树形数据结构。
    """

    def __init__(self, x0: float, t0: float, v_func: Callable[[float], float],
                 D: float = 0.0, dt: float = 1.0):
        """
        参数
        ----------
        x0 : float
            初始位置（通常为观测井位置）
        t0 : float
            初始时间
        v_func : callable
            流速场函数 v(x)
        D : float
            弥散系数（用于构造反向随机分支）
        dt : float
            反向时间步长
        """
        self.x0 = float(x0)
        self.t0 = float(t0)
        self.v_func = v_func
        self.D = float(D)
        self.dt = float(dt)

    def forward_step(self, x: float) -> float:
        """正向一步：x(t+dt) = x(t) + v(x) dt。"""
        return x + self.v_func(x) * self.dt

    def backward_step(self, x: float, n_branches: int = 1,
                      rng: Optional[np.random.Generator] = None) -> list[float]:
        """
        反向一步：从 x(t+dt) 回溯到可能的 x(t)。

        纯对流（D=0）：确定性单值逆映射
            x(t) ≈ x(t+dt) - v(x) dt

        含弥散（D>0）：引入随机分支，模拟弥散导致的源区不确定性
            x(t) = x(t+dt) - v(x) dt + ξ,  ξ ~ N(0, 2D dt)
        """
        if rng is None:
            rng = np.random.default_rng()
        v = self.v_func(x)
        x_base = x - v * self.dt
        if self.D <= 0.0 or n_branches <= 1:
            return [x_base]

        branches = []
        std = np.sqrt(2.0 * self.D * self.dt)
        for _ in range(n_branches):
            xi = rng.normal(0.0, std)
            branches.append(x_base + xi)
        return branches

    def build_backward_tree(self, max_levels: int = 10,
                            n_branches: int = 2) -> dict:
        """
        构建反向追踪树，返回树结构字典。

        树的层级 k 对应反向时间 k·dt，节点为可能的上游位置。
        类似于 collatz_level 函数：从目标点出发，逐层应用反向映射。

        返回
        -------
        dict
            {"levels": [[x1, x2, ...], ...], "times": [t0, t0+dt, ...]}
        """
        rng = np.random.default_rng(42)
        levels = [[self.x0]]
        times = [self.t0]

        for level in range(1, max_levels + 1):
            current_level = []
            for x_parent in levels[-1]:
                children = self.backward_step(x_parent, n_branches=n_branches, rng=rng)
                current_level.extend(children)
            levels.append(current_level)
            times.append(self.t0 + level * self.dt)

        return {"levels": levels, "times": times}

    def compute_travel_time_distribution(self, x_source: float,
                                         n_particles: int = 10000,
                                         max_steps: int = 500,
                                         x_bounds: tuple = (-1e9, 1e9)) -> np.ndarray:
        """
        使用正向随机行走粒子追踪估计 travel-time 分布。

        算法：
          1. 在 x_source 处释放 N_p 个粒子
          2. 每个粒子执行随机行走：x_{k+1} = x_k + v(x_k)Δt + √(2DΔt) Z_k
          3. 记录每个粒子到达 x0 附近的时间
          4. 构造到达时间直方图作为 TTD 估计

        参数
        ----------
        x_source : float
            污染源位置
        n_particles : int
            粒子数
        max_steps : int
            最大时间步数
        x_bounds : tuple
            若粒子超出此范围则标记为"未到达"

        返回
        -------
        np.ndarray
            到达时间数组（未到达粒子标记为 -1）
        """
        if n_particles < 1:
            raise ValueError("粒子数必须 ≥ 1")
        rng = np.random.default_rng(123)
        arrival_times = np.full(n_particles, -1.0)

        for p in range(n_particles):
            x = float(x_source)
            for step in range(1, max_steps + 1):
                v = self.v_func(x)
                std = np.sqrt(2.0 * self.D * self.dt) if self.D > 0 else 0.0
                x = x + v * self.dt + rng.normal(0.0, std)
                t = step * self.dt

                # 检查是否到达观测点附近（容差 = 2 * v * dt）
                tol = max(abs(v) * self.dt * 2.0, 1e-3)
                if abs(x - self.x0) < tol:
                    arrival_times[p] = t
                    break
                if x < x_bounds[0] or x > x_bounds[1]:
                    break

        return arrival_times


def discrete_dynamical_stability_map(v_func: Callable[[float], float],
                                     x_grid: np.ndarray,
                                     dt: float = 1.0,
                                     n_iter: int = 100) -> np.ndarray:
    """
    计算离散动力系统 x_{k+1} = x_k + v(x_k) dt 的稳定性图。

    类似于 Collatz 序列中研究收敛性和周期性，这里分析流速场中
    不同起始位置的质点是否收敛到固定点或出现周期性行为。

    返回每个网格点的 Lyapunov 指数估计：
        λ(x) = (1/n) Σ_{k=0}^{n-1} ln |1 + v'(x_k) dt|
    """
    if len(x_grid) == 0:
        raise ValueError("网格不能为空")
    lyap = np.zeros_like(x_grid)
    h_deriv = 1e-6

    for i, x0 in enumerate(x_grid):
        x = x0
        lam_sum = 0.0
        for _ in range(n_iter):
            # 数值微分估计 v'(x)
            vp = v_func(x + h_deriv)
            vm = v_func(x - h_deriv)
            dv_dx = (vp - vm) / (2.0 * h_deriv)
            jac = abs(1.0 + dv_dx * dt)
            if jac > 1e-15:
                lam_sum += np.log(jac)
            x = x + v_func(x) * dt
            if not np.isfinite(x):
                lam_sum = np.nan
                break
        lyap[i] = lam_sum / n_iter if np.isfinite(lam_sum) else np.nan

    return lyap


if __name__ == "__main__":
    # 测试：均匀流速 v = 0.5
    tree = TravelTimeTree(x0=50.0, t0=0.0, v_func=lambda x: 0.5, D=0.1, dt=2.0)
    btree = tree.build_backward_tree(max_levels=5, n_branches=2)
    assert len(btree["levels"]) == 6

    ttd = tree.compute_travel_time_distribution(x_source=0.0, n_particles=500, max_steps=200)
    reached = ttd[ttd > 0]
    assert len(reached) > 0

    x_grid = np.linspace(0, 100, 50)
    lyap = discrete_dynamical_stability_map(lambda x: 0.5, x_grid)
    assert np.allclose(lyap, np.log(1.0), atol=0.1)
    print("travel_time_tree: 自测试通过")
