"""
cvt_optimizer.py
================
非均匀密度 Centroidal Voronoi Tessellation (CVT) 采样优化。

源自 cvt_1d_nonuniform 项目的核心思想，扩展到多维空间，用于
优化分子动力学积分和蒙特卡洛采样中的节点分布。

核心数学原理
------------
CVT 定义：给定区域 Ω 和密度函数 ρ(x)，一组生成点 {z_i} 称为
CVT 生成点，如果每个 z_i 同时是其 Voronoi 单元 V_i 的质心：

    z_i = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx

其中 Voronoi 单元定义为：

    V_i = { x ∈ Ω : ‖x - z_i‖ ≤ ‖x - z_j‖, ∀ j ≠ i }

CVT 最小化能量泛函（量化误差）：

    E(z₁,...,z_N) = Σ_i ∫_{V_i} ρ(x) ‖x - z_i‖² dx

对于分子动力学应用，取密度 ρ(x) ∝ |∇V(x)|²，使得势能变化剧烈
的区域获得更多采样点。

Lloyd 算法（固定点迭代）：
    1. 给定生成点 {z_i}
    2. 构造 Voronoi 剖分 {V_i}
    3. 更新 z_i ← Centroid(V_i)
    4. 重复直到收敛

收敛判据（能量变化率）：
    |E^{(k+1)} - E^{(k)}| / E^{(k)} < tol
"""

import numpy as np
from typing import Callable, Tuple


def density_transform(s: float, density_type: int = 0) -> float:
    """
    非均匀密度变换函数（1D 参考实现）。
    
    将均匀随机变量 s ∈ [0,1] 映射到服从特定密度的随机变量。
    
    类型:
        0: ρ(s) = s        (恒等)
        1: ρ(s) = √s
        2: ρ(s) = s^(1/3)
        3: ρ(s) = s^(1/4)
        4: ρ(s) = log( e / (e - s(e-1)) )
        5: ρ(s) = 0.5 + atan(50(s-0.5))/π
        6: ρ(s) = sin(π(s-0.5))   (Chebyshev-like)
    """
    s = float(np.clip(s, 0.0, 1.0))
    if density_type == 0:
        return s
    elif density_type == 1:
        return np.sqrt(s)
    elif density_type == 2:
        return s ** (1.0 / 3.0)
    elif density_type == 3:
        return s ** (1.0 / 4.0)
    elif density_type == 4:
        euler = np.e
        return np.log(euler / (euler - s * (euler - 1.0)))
    elif density_type == 5:
        return 0.5 + np.arctan(50.0 * (s - 0.5)) / np.pi
    elif density_type == 6:
        return np.sin(np.pi * (s - 0.5))
    else:
        return s


class CVTOptimizer:
    """
    多维 CVT 优化器。
    """

    def __init__(self, dim: int = 2, n_generators: int = 16,
                 domain: Tuple[np.ndarray, np.ndarray] = None,
                 density_func: Callable = None,
                 max_iter: int = 100,
                 tol: float = 1e-5):
        """
        参数:
            dim: 空间维度
            n_generators: 生成点数量
            domain: (low, high)，每个维度的边界
            density_func: ρ(x) 密度函数，默认为均匀密度
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.dim = dim
        self.n = n_generators
        if domain is None:
            self.low = np.zeros(dim)
            self.high = np.ones(dim)
        else:
            self.low = np.asarray(domain[0])
            self.high = np.asarray(domain[1])
        self.density_func = density_func if density_func is not None else lambda x: 1.0
        self.max_iter = max_iter
        self.tol = tol
        self.generators = None
        self.energy_history = []

    def initialize_generators(self, method: str = "latin_hypercube"):
        """
        初始化生成点。
        
        方法:
            - random: 均匀随机
            - grid: 均匀网格
            - latin_hypercube: Latin 超立方采样
            - zeros: 全零
        """
        if method == "random":
            self.generators = np.random.rand(self.n, self.dim)
            self.generators = self.low + self.generators * (self.high - self.low)
        elif method == "grid":
            # 尽量均匀划分
            n_side = int(np.ceil(self.n ** (1.0 / self.dim)))
            coords = [np.linspace(0, 1, n_side) for _ in range(self.dim)]
            pts = []
            import itertools
            for combo in itertools.product(*coords):
                pts.append(combo)
                if len(pts) >= self.n:
                    break
            self.generators = np.array(pts[:self.n])
            self.generators = self.low + self.generators * (self.high - self.low)
        elif method == "latin_hypercube":
            # Latin hypercube
            self.generators = np.zeros((self.n, self.dim))
            for d in range(self.dim):
                perm = np.random.permutation(self.n)
                self.generators[:, d] = (perm + 0.5) / self.n
            self.generators = self.low + self.generators * (self.high - self.low)
        elif method == "zeros":
            self.generators = np.zeros((self.n, self.dim))
        else:
            self.generators = np.random.rand(self.n, self.dim)
            self.generators = self.low + self.generators * (self.high - self.low)

    def _find_nearest(self, sample: np.ndarray) -> int:
        """找到距离 sample 最近的生成点索引。"""
        diffs = self.generators - sample
        dists_sq = np.sum(diffs ** 2, axis=1)
        return int(np.argmin(dists_sq))

    def _quantize_error(self) -> float:
        """通过蒙特卡洛采样估计量化误差 E。"""
        n_samples = max(self.n * 50, 500)
        error = 0.0
        for _ in range(n_samples):
            s = np.random.rand(self.dim)
            sample = self.low + s * (self.high - self.low)
            nearest = self._find_nearest(sample)
            rho = self.density_func(sample)
            dist_sq = np.sum((sample - self.generators[nearest]) ** 2)
            error += rho * dist_sq
        return error / n_samples

    def optimize(self, sample_multiplier: int = 50) -> np.ndarray:
        """
        执行 Lloyd 算法优化 CVT。
        
        参数:
            sample_multiplier: 每步采样数 = n_generators × multiplier
        
        返回:
            优化后的生成点 N×dim
        """
        if self.generators is None:
            self.initialize_generators()

        n_samples = self.n * sample_multiplier
        self.energy_history = []

        for iteration in range(self.max_iter):
            generator_new = np.zeros_like(self.generators)
            tally = np.zeros(self.n)

            for _ in range(n_samples):
                sample = self.low + np.random.rand(self.dim) * (self.high - self.low)
                nearest = self._find_nearest(sample)
                generator_new[nearest] += sample
                tally[nearest] += 1.0

            # 更新生成点为 Voronoi 单元的质心
            for j in range(self.n):
                if tally[j] > 0:
                    self.generators[j] = generator_new[j] / tally[j]

            # 能量监控
            if iteration % 5 == 0:
                energy = self._quantize_error()
                self.energy_history.append(energy)
                if len(self.energy_history) > 1:
                    rel_change = abs(self.energy_history[-1] - self.energy_history[-2])
                    if rel_change < self.tol * abs(self.energy_history[-2] + 1e-12):
                        break

        return self.generators.copy()

    def get_sampling_weights(self) -> np.ndarray:
        """
        基于 Voronoi 单元体积估计各生成点的采样权重。
        
        使用蒙特卡洛估计：w_i ≈ N_i / N_total
        """
        n_test = self.n * 100
        counts = np.zeros(self.n)
        for _ in range(n_test):
            sample = self.low + np.random.rand(self.dim) * (self.high - self.low)
            nearest = self._find_nearest(sample)
            counts[nearest] += 1.0
        return counts / n_test


def cvt_1d_nonuniform_python(n_generators: int = 10,
                              density_type: int = 0,
                              n_steps: int = 100,
                              n_samples_per_step: int = 1000) -> np.ndarray:
    """
    一维非均匀密度 CVT 的直接实现（源自原项目算法）。
    
    参数:
        n_generators: 生成点数量
        density_type: 密度函数类型（见 density_transform）
        n_steps: 迭代步数
        n_samples_per_step: 每步采样数
    
    返回:
        最终生成点位置
    """
    # 初始化
    generators = np.sort(np.random.rand(n_generators))

    for step in range(n_steps):
        gen_new = np.zeros(n_generators)
        tally = np.zeros(n_generators)

        for _ in range(n_samples_per_step):
            s = np.random.rand()
            s_transformed = density_transform(s, density_type)
            s_transformed = np.clip(s_transformed, 0.0, 1.0)

            # 找到最近生成点
            dists = np.abs(generators - s_transformed)
            nearest = int(np.argmin(dists))
            gen_new[nearest] += s_transformed
            tally[nearest] += 1.0

        for j in range(n_generators):
            if tally[j] > 0:
                generators[j] = gen_new[j] / tally[j]

    return generators
