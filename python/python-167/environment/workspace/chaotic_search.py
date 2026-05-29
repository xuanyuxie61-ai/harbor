"""
chaotic_search.py
基于混沌动力学的全局步态参数优化模块。
融入种子项目：
  - 655_leaf_chaos（迭代函数系统 IFS → 步态参数空间的全局探索）

科学背景：
多足机器人步态优化是一个高度非凸、多模态的优化问题。
传统梯度法易陷入局部最优。混沌系统（如 Logistic 映射、Henon 映射、
Barnsley 蕨类 IFS）具有遍历性、初值敏感性与伪随机性，
可用于构造高效的混沌搜索算子，在参数空间进行全局探索。

本模块实现：
1. 基于 Logistic 映射的混沌序列生成
2. 基于 Barnsley 蕨类 IFS 的二维参数空间采样
3. 混沌退火优化算法（Chaotic Simulated Annealing）
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from utils import clip_to_bounds


class LogisticMap:
    """
    Logistic 映射：
        x_{n+1} = r · x_n · (1 - x_n)
    当 r ∈ [3.57, 4] 时系统进入混沌状态。
    具有遍历性：轨道在 [0,1] 上均匀分布（长期平均）。
    """

    def __init__(self, r: float = 4.0, x0: float = 0.3):
        if not (3.57 <= r <= 4.0):
            r = clip_to_bounds(np.array([r]), np.array([3.57]), np.array([4.0]))[0]
        self.r = r
        self.x = x0

    def next(self) -> float:
        self.x = self.r * self.x * (1.0 - self.x)
        return self.x

    def generate(self, n: int) -> np.ndarray:
        seq = np.zeros(n)
        for i in range(n):
            seq[i] = self.next()
        return seq


class BarnsleyFernIFS:
    """
    源自 leaf_chaos.m 的 Barnsley 蕨类迭代函数系统。

    IFS 定义：
    由 N 个仿射变换 { w_k(x) = A_k·x + b_k }_{k=1}^N 组成，
    每个变换以概率 p_k 被选取。迭代轨道 {x_n} 的吸引子为分形集。

    在步态优化中，将参数空间中的"优质解区域"视为吸引子，
    用 IFS 在参数空间中生成候选点。
    """

    def __init__(self):
        # 4 个仿射变换（标准 Barnsley 蕨参数）
        self.A = [
            np.array([[0.0, 0.0], [0.0, 0.16]]),
            np.array([[0.85, 0.04], [-0.04, 0.85]]),
            np.array([[0.2, -0.26], [0.23, 0.22]]),
            np.array([[-0.15, 0.28], [0.26, 0.24]]),
        ]
        self.b = [
            np.array([0.0, 0.0]),
            np.array([0.0, 1.6]),
            np.array([0.0, 1.6]),
            np.array([0.0, 0.44]),
        ]
        self.probs = np.array([0.01, 0.85, 0.07, 0.07])
        self.state = np.array([0.0, 0.0])

    def step(self) -> np.ndarray:
        """
        执行一次 IFS 迭代，返回新状态。
        """
        k = np.random.choice(len(self.A), p=self.probs)
        self.state = self.A[k] @ self.state + self.b[k]
        return self.state.copy()

    def sample(self, n: int, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        生成 n 个 2D 样本，并线性映射到 bounds 指定的前 2 维参数区间。
        bounds: (lower, upper) 可为任意维度，仅取前 2 维映射。
        """
        lower, upper = bounds
        lower2 = np.asarray(lower).flatten()[:2]
        upper2 = np.asarray(upper).flatten()[:2]
        samples = np.zeros((n, 2))
        for i in range(n):
            s = self.step()
            # 映射到目标区间（Barnsley 蕨的吸引子大致在 x∈[-2.5,2.5], y∈[0,10]）
            s_mapped = lower2 + (s - np.array([-2.5, 0.0])) / np.array([5.0, 10.0]) * (upper2 - lower2)
            samples[i] = clip_to_bounds(s_mapped, lower2, upper2)
        return samples


class ChaoticSimulatedAnnealing:
    """
    混沌模拟退火（CSA）优化器。

    算法思想：
    将混沌变量引入模拟退火的扰动机制，利用混沌遍历性避免局部最优。
    扰动公式：
        x_new = x_current + β·T·(2·c - 1)
    其中 c 为混沌变量（Logistic 映射），T 为温度，β 为步长缩放因子。
    接受准则（Metropolis 准则）：
        P(accept) = exp( -ΔE / T )   if ΔE > 0
                  = 1                if ΔE ≤ 0
    """

    def __init__(self, dim: int, bounds: np.ndarray,
                 T0: float = 10.0, T_min: float = 1e-4,
                 cooling_rate: float = 0.95, max_iter: int = 500,
                 logistic_r: float = 4.0):
        self.dim = dim
        self.bounds = np.asarray(bounds, dtype=float)  # (dim, 2)
        self.T0 = T0
        self.T_min = T_min
        self.alpha = cooling_rate
        self.max_iter = max_iter
        self.chaos_maps = [LogisticMap(r=logistic_r, x0=0.1 + 0.1 * i) for i in range(dim)]

    def optimize(self, objective: Callable[[np.ndarray], float],
                 x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        最小化目标函数 objective(x)。
        返回 (x_best, f_best)。
        """
        if x0 is None:
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        x_current = clip_to_bounds(x0, self.bounds[:, 0], self.bounds[:, 1])
        f_current = objective(x_current)
        x_best = x_current.copy()
        f_best = f_current
        T = self.T0
        beta = 0.5

        for iteration in range(self.max_iter):
            if T < self.T_min:
                break
            # 混沌扰动
            chaos_vals = np.array([m.next() for m in self.chaos_maps])
            delta = beta * T * (2.0 * chaos_vals - 1.0)
            x_new = x_current + delta
            x_new = clip_to_bounds(x_new, self.bounds[:, 0], self.bounds[:, 1])
            f_new = objective(x_new)
            delta_E = f_new - f_current

            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (T + 1e-12)):
                x_current = x_new.copy()
                f_current = f_new
                if f_new < f_best:
                    x_best = x_new.copy()
                    f_best = f_new

            T *= self.alpha

        return x_best, f_best


class GaitParameterOptimizer:
    """
    综合步态参数优化器，结合混沌搜索与局部精细搜索。
    优化变量：步态周期 T_gait、步幅 stride_length、抬腿高度 swing_height、
              相位耦合强度 coupling_k、阻尼系数 damping。
    """

    def __init__(self):
        self.bounds = np.array([
            [0.4, 1.5],    # T_gait (s)
            [0.05, 0.30],  # stride_length (m)
            [0.02, 0.10],  # swing_height (m)
            [1.0, 15.0],   # coupling_k
            [0.5, 5.0],    # damping
        ])
        self.csa = ChaoticSimulatedAnnealing(
            dim=5, bounds=self.bounds, T0=5.0, max_iter=300
        )
        self.ifs = BarnsleyFernIFS()

    def evaluate_gait_fitness(self, params: np.ndarray,
                              stability_margin_func: Callable[[np.ndarray], float],
                              energy_cost_func: Callable[[np.ndarray], float]) -> float:
        """
        综合适应度函数：
            f = -w1·margin + w2·energy + w3·penalty
         margin 为稳定性裕度（越大越好，取负）
         energy 为能量消耗估计（越小越好）
         penalty 为约束违反惩罚。
        """
        T_gait, stride, swing_h, coupling, damping = params
        # 简化评估：调用外部函数
        margin = stability_margin_func(params)
        energy = energy_cost_func(params)
        # 约束惩罚
        penalty = 0.0
        if swing_h > 0.5 * stride:
            penalty += 100.0 * (swing_h - 0.5 * stride)
        if T_gait < 0.3:
            penalty += 50.0 * (0.3 - T_gait)
        fitness = -5.0 * margin + 2.0 * energy + penalty
        return fitness

    def optimize(self, stability_margin_func: Callable[[np.ndarray], float],
                 energy_cost_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """
        主优化接口。
        """
        def obj(x):
            return self.evaluate_gait_fitness(x, stability_margin_func, energy_cost_func)

        # 先用 IFS 生成若干初始猜测
        best_f = float('inf')
        best_x = None
        ifs_samples = self.ifs.sample(20, (self.bounds[:, 0], self.bounds[:, 1]))
        for s in ifs_samples:
            # 补全到 5 维（IFS 只生成 2 维）
            s5 = np.concatenate([s[:2], np.random.uniform(self.bounds[2:, 0], self.bounds[2:, 1])])
            f = obj(s5)
            if f < best_f:
                best_f = f
                best_x = s5.copy()

        # 混沌模拟退火精细搜索
        x_opt, f_opt = self.csa.optimize(obj, x0=best_x)
        return x_opt, f_opt
