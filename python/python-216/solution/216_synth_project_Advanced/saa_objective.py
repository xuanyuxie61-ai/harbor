# -*- coding: utf-8 -*-
"""
saa_objective.py
================

样本平均近似 (Sample Average Approximation, SAA) 核心引擎.

数学形式:
  原始随机优化问题:
      min_{x in X}  F(x) := E_xi[ f(x, xi) ]

  SAA 近似:
      min_{x in X}  F_N(x) := (1/N) sum_{i=1}^N f(x, xi_i)

  其中 xi_i 是 iid 样本, f(x, xi) = J(u(x, xi)) 是亥姆霍兹 PDE 的
  目标函数值, 而 u(x, xi) 是在随机场实现 xi 下的 PDE 解.

融合的种子项目:
  - 711_mandelbrot_area    -> Monte Carlo 积分思想 (样本计数比率估计期望)
  - 993_r8row              -> 行数据的运行平均 (running average)
  - 773_mnist_neural       -> 小批量 SGD 训练循环结构
  - 022_asa_graphs         -> 样本的随机排列 (小批量洗牌)

SAA 统计保证 (Shapiro et al. 2021):
  - F_N -> F 一致收敛 (大数定律)
  - sqrt(N) (x_N - x*) -> N(0, Sigma)  (CLT)
  - 统计误差 O(1/sqrt(N))
"""

import math
from typing import List, Callable, Tuple, Optional
from middle_square_rng import MiddleSquareRNG
from random_field import RandomFieldKL, EllipsoidConfidenceSampler
from helmholtz_solver import HelmholtzFD
from quadrature import integrate_on_rectangle
from scientific_formulas import (
    saa_statistical_error_bound, kluncer_bounds,
    variance_reduction_factor
)


class RunningStatistics:
    """运行统计量 (复刻 seed project 993_r8row running_average/running_sum).

    在 SAA 中用于:
      1. 跟踪 F_N(x_t) 的运行平均, 作为收敛监控;
      2. 在线更新样本方差, 用于构造置信区间;
      3. Polyak-Ruppert 平均化: x_bar_t = (1/t) sum x_k.
    """

    def __init__(self):
        self.n = 0
        self.sum_vals = 0.0
        self.sum_sq = 0.0
        self.history: List[float] = []

    def update(self, val: float) -> None:
        """添加新观测."""
        self.n += 1
        self.sum_vals += val
        self.sum_sq += val * val
        self.history.append(val)

    def mean(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sum_vals / self.n

    def variance(self) -> float:
        """无偏样本方差."""
        if self.n < 2:
            return 0.0
        m = self.mean()
        return (self.sum_sq / self.n - m * m) * self.n / (self.n - 1)

    def std(self) -> float:
        return math.sqrt(max(self.variance(), 0.0))

    def running_averages(self) -> List[float]:
        """返回前缀运行平均序列 (复刻 r8row_running_average.m)."""
        out = []
        s = 0.0
        for i, v in enumerate(self.history):
            s += v
            out.append(s / (i + 1))
        return out

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """F_N 的 (1-alpha) 置信区间."""
        if self.n < 2:
            m = self.mean()
            return (m, m)
        z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.96)
        m = self.mean()
        se = self.std() / math.sqrt(self.n)
        return (m - z * se, m + z * se)


class SAAObjective:
    """SAA 目标函数封装.

    给定:
      - 设计变量 x (此处为 KL 系数的平移, 即随机场的均值参数)
      - 亥姆霍兹求解器
      - 随机场模型
    计算:
      F_N(x) = (1/N) sum_{i=1}^N J(u(x, xi_i))

    同时提供:
      - 单样本评估 f(x, xi_i)
      - 小批量评估
      - 有限差分梯度
    """

    def __init__(self, nx_fd: int = 15, ny_fd: int = 15,
                 Lx: float = 1.0, Ly: float = 1.0,
                 k_wave: float = 3.0, damping: float = 1e-2,
                 K_kl: int = 4, sigma_field: float = 0.3,
                 ell_field: float = 0.2,
                 source_amplitude: float = 1.0):
        self.nx_fd = nx_fd
        self.ny_fd = ny_fd
        self.Lx = Lx
        self.Ly = Ly
        self.k_wave = k_wave
        self.damping = damping
        self.K_kl = K_kl
        self.sigma_field = sigma_field
        self.ell_field = ell_field
        self.source_amplitude = source_amplitude

        # 亥姆霍兹求解器 (共用, 但每个样本 a_field 不同)
        self.solver = HelmholtzFD(nx_fd, ny_fd, Lx, Ly, k_wave, damping)

        # 随机场模型
        self.rf = RandomFieldKL(L=Lx, sigma=sigma_field,
                                ell=ell_field, K=K_kl, m_gram=32)

        # 运行统计
        self.stats = RunningStatistics()

    def _make_source(self) -> Callable:
        """源项 f(x, y) = A sin(pi x / Lx) sin(pi y / Ly).
        选择此形式使得源在边界为零, 与 Dirichlet BC 相容。
        """
        A = self.source_amplitude
        Lx, Ly = self.Lx, self.Ly

        def f(x, y):
            return A * math.sin(math.pi * x / Lx) * math.sin(math.pi * y / Ly)
        return f

    def _make_a_field(self, xi: List[float],
                      x0_offset: List[float]) -> Callable:
        """给定随机实现 xi 和设计偏移 x0_offset, 构造 a_field(x, y).

        a(x, y) = a_KL(x; xi + x0_offset)
        这里把 "设计变量" 解释为 KL 系数的均值平移.
        """
        K = self.K_kl
        xi_shifted = [xi[k] + (x0_offset[k] if k < len(x0_offset) else 0.0)
                      for k in range(K)]
        rf = self.rf
        Lx = self.Lx
        Ly = self.Ly

        def a_field(x, y):
            # 一维 KL 沿 x, 在 y 方向均匀 (简化)
            val = rf.evaluate(x, xi_shifted)
            return val
        return a_field

    def evaluate_single(self, x_design: List[float],
                        xi: List[float]) -> float:
        """单次样本评估: 给定设计 x_design 和随机实现 xi,
        求解亥姆霍兹并返回 J(u).
        """
        f_func = self._make_source()
        a_field = self._make_a_field(xi, x_design)
        u, _, _ = self.solver.solve(f_func, a_field)
        J = self.solver.compute_objective(u)
        # 数值保护: J 必须非负
        return max(J, 0.0)

    def evaluate_batch(self, x_design: List[float],
                       xi_batch: List[List[float]]) -> float:
        """小批量 SAA 评估: F_N(x) = (1/N) sum f(x, xi_i)."""
        if not xi_batch:
            return 0.0
        total = 0.0
        for xi in xi_batch:
            total += self.evaluate_single(x_design, xi)
        return total / len(xi_batch)

    def finite_difference_gradient(self, x_design: List[float],
                                   xi_batch: List[List[float]],
                                   eps: float = 1e-3
                                   ) -> List[float]:
        """中心差分梯度 of F_N w.r.t. x_design.

        d F_N / d x_k ≈ (F_N(x + e_k eps) - F_N(x - e_k eps)) / (2 eps)
        """
        K = len(x_design)
        grad = []
        f0 = self.evaluate_batch(x_design, xi_batch)
        for k in range(K):
            x_plus = list(x_design)
            x_minus = list(x_design)
            x_plus[k] += eps
            x_minus[k] -= eps
            fp = self.evaluate_batch(x_plus, xi_batch)
            fm = self.evaluate_batch(x_minus, xi_batch)
            grad.append((fp - fm) / (2.0 * eps))
        return grad


class SAASampler:
    """SAA 样本生成器: 融合多种采样策略.

    1. 纯 Monte Carlo (MC): xi_i iid N(0, I_K)  (复刻 711_mandelbrot_area)
    2. 排列洗牌 (permutation): 固定 N_max 个样本, 每 epoch 洗牌
       (复刻 022_asa_graphs TSP 中的随机排列思想)
    3. 椭球置信域采样 (low-discrepancy): 见 random_field.EllipsoidConfidenceSampler
    """

    def __init__(self, rng: MiddleSquareRNG, K: int,
                 strategy: str = "mc"):
        """
        Args:
            rng: 伪随机数发生器
            K: KL 维度
            strategy: "mc" | "permutation" | "ellipsoid"
        """
        self.rng = rng
        self.K = K
        self.strategy = strategy

        # 预生成一批样本 (用于 permutation 策略)
        self.pool: List[List[float]] = []
        self.perm_index = 0
        self.perm_order: List[int] = []

    def pregenerate(self, N: int) -> None:
        """预生成 N 个样本, 用于 permutation 策略."""
        self.pool = []
        for _ in range(N):
            xi = self.rng.next_gaussian_vector(self.K)
            self.pool.append(xi)
        self.perm_order = list(range(N))
        self.perm_index = 0

    def next_batch(self, batch_size: int) -> List[List[float]]:
        """获取一个 batch 的样本."""
        if self.strategy == "mc":
            return [self.rng.next_gaussian_vector(self.K)
                    for _ in range(batch_size)]
        elif self.strategy == "permutation":
            if not self.pool:
                self.pregenerate(max(batch_size * 10, 100))
            batch = []
            for _ in range(batch_size):
                if self.perm_index >= len(self.perm_order):
                    # 重新洗牌 (复刻 022 的 next_perm 思想)
                    self.perm_order = self.rng.next_permutation(
                        len(self.perm_order))
                    self.perm_index = 0
                idx = self.perm_order[self.perm_index]
                self.perm_index += 1
                batch.append(self.pool[idx])
            return batch
        elif self.strategy == "ellipsoid":
            # 使用椭球采样器
            radii = [3.0] * self.K  # 3-sigma 置信椭球
            sampler = EllipsoidConfidenceSampler(radii, n_per_axis=4)
            pts = sampler.generate()
            # 截取 batch_size
            if len(pts) >= batch_size:
                return pts[:batch_size]
            # 不够就重复 + 加噪
            out = list(pts)
            while len(out) < batch_size:
                noise = [self.rng.next_gaussian() * 0.1
                         for _ in range(self.K)]
                base = pts[len(out) % len(pts)]
                out.append([base[k] + noise[k] for k in range(self.K)])
            return out
        else:
            raise ValueError("未知采样策略: %s" % self.strategy)


class MonteCarloAreaEstimator:
    """Monte Carlo 面积估计器 (复刻 seed project 711_mandelbrot_area).

    在 SAA 中类比: 用样本计数比率估计概率 / 期望.
    例如: 估计 P{ J(u(xi)) > threshold } 的概率.

    源自 mandelbrot_area.m:
        在复平面上随机采样 N 点, 计数落在 Mandelbrot 集中的比例,
        乘以总面积得到面积估计. 此处我们用相同思想估计 SAA 的风险度量.
    """

    def __init__(self, rng: MiddleSquareRNG):
        self.rng = rng

    def estimate_probability(self, indicator_func: Callable,
                             n_samples: int,
                             volume: float = 1.0) -> dict:
        """估计 indicator_func(xi) = 1 的概率 * volume.

        Returns:
            dict with keys: estimate, std_error, n_samples, hit_count
        """
        hits = 0
        for _ in range(n_samples):
            xi = self.rng.next_gaussian_vector(4)  # 4D 测试
            if indicator_func(xi):
                hits += 1
        p_hat = hits / max(n_samples, 1)
        # 二项分布标准误差
        se = math.sqrt(p_hat * (1 - p_hat) / max(n_samples, 1))
        return {
            "estimate": p_hat * volume,
            "std_error": se * volume,
            "n_samples": n_samples,
            "hit_count": hits,
        }
