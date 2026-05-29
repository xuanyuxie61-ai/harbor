"""
基于 Centroidal Voronoi Tessellation (CVT) 的微反应器催化剂最优分布算法
===========================================================================
将 CVT 算法应用于微反应器催化剂颗粒的空间排布优化，使得催化剂在反应器
横截面上的覆盖率最均匀，从而强化混合与传质。

核心公式：
    给定区域 Ω ⊂ ℝ²，CVT 将 Ω 划分为 N 个 Voronoi 单元 {V_i}，使得
    能量泛函最小：

        E = Σᵢ ∫_{V_i} ρ(x) ||x - z_i||² dx

    其中 z_i 为单元 V_i 的质心（即生成元），ρ(x) 为催化剂密度权重函数。
    最优条件：每个生成元恰好是其 Voronoi 单元的质心：

        z_i = ∫_{V_i} ρ(x) x dx / ∫_{V_i} ρ(x) dx

    本模块使用 Lloyd 迭代逼近 CVT，并通过 Monte Carlo 采样估计积分。
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


class CatalystCVTPlacer:
    """
    基于 CVT 的二维/三维催化剂分布优化器。
    """

    def __init__(
        self,
        dim: int = 2,
        n_generators: int = 64,
        bounds: Optional[np.ndarray] = None,
        density_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sample_num: int = 20000,
        max_iter: int = 50,
        tol: float = 1.0e-5,
    ):
        if dim not in (2, 3):
            raise ValueError("仅支持 2D 或 3D 催化剂分布")
        if n_generators < 1:
            raise ValueError("生成元数量必须至少为 1")
        if sample_num < 100:
            raise ValueError("采样点数量过少")

        self.dim = dim
        self.n = n_generators
        self.sample_num = sample_num
        self.max_iter = max_iter
        self.tol = tol

        if bounds is None:
            self.bounds = np.array([[0.0, 1.0]] * dim, dtype=float)
        else:
            self.bounds = np.array(bounds, dtype=float)
            if self.bounds.shape != (dim, 2):
                raise ValueError("bounds 形状应为 (dim, 2)")

        self.density_func = density_func
        if density_func is None:
            # 默认均匀密度
            self.density_func = lambda pts: np.ones(pts.shape[0])

        # 初始化生成元（均匀随机）
        self.generators = self._init_generators()

    def _init_generators(self) -> np.ndarray:
        """在边界框内均匀初始化生成元。"""
        gens = np.zeros((self.n, self.dim))
        for d in range(self.dim):
            lo, hi = self.bounds[d]
            gens[:, d] = np.random.uniform(lo, hi, self.n)
        return gens

    def _sample_points(self) -> np.ndarray:
        """在边界框内按照密度函数进行重要性采样。"""
        # 拒绝采样：先生成均匀样本，再以 density/max_density 接受
        samples = []
        batch = min(self.sample_num * 2, 50000)
        while len(samples) < self.sample_num:
            pts = np.zeros((batch, self.dim))
            for d in range(self.dim):
                lo, hi = self.bounds[d]
                pts[:, d] = np.random.uniform(lo, hi, batch)
            dens = self.density_func(pts)
            if np.max(dens) <= 0.0:
                dens = np.ones_like(dens)
            prob = dens / np.max(dens)
            mask = np.random.rand(batch) < prob
            accepted = pts[mask]
            samples.extend(accepted.tolist())
        samples = np.array(samples[: self.sample_num])
        return samples

    def _find_closest(self, points: np.ndarray) -> np.ndarray:
        """
        对每个采样点，找到最近的生成元索引。
        时间复杂度 O(sample_num * n)。
        """
        # 向量化计算欧氏距离矩阵
        # distances[i,j] = ||points[i] - generators[j]||
        diff = points[:, np.newaxis, :] - self.generators[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        closest = np.argmin(dists, axis=1)
        return closest

    def _compute_energy(self, samples: np.ndarray, closest: np.ndarray) -> float:
        """
        计算 CVT 能量泛函：
            E = (1/M) Σᵢ Σ_{x∈V_i} ρ(x) ||x - z_i||²
        """
        dens = self.density_func(samples)
        energy = 0.0
        for i in range(self.n):
            mask = closest == i
            if np.any(mask):
                diff = samples[mask] - self.generators[i]
                dist2 = np.sum(diff ** 2, axis=1)
                energy += np.sum(dens[mask] * dist2)
        return energy / len(samples)

    def iterate(self) -> Tuple[np.ndarray, float, float]:
        """
        执行 Lloyd 迭代求解 CVT。

        返回:
            generators: 优化后的生成元坐标 (n, dim)
            energy: 最终能量
            max_shift: 最后一次迭代中生成元的最大移动距离
        """
        energy_history = []
        for it in range(self.max_iter):
            samples = self._sample_points()
            closest = self._find_closest(samples)
            dens = self.density_func(samples)

            new_gens = np.zeros_like(self.generators)
            counts = np.zeros(self.n)
            for i in range(self.n):
                mask = closest == i
                if np.any(mask):
                    weights = dens[mask]
                    total_weight = np.sum(weights)
                    if total_weight > 0.0:
                        new_gens[i] = np.sum(samples[mask] * weights[:, np.newaxis], axis=0) / total_weight
                    else:
                        new_gens[i] = self.generators[i]
                    counts[i] = total_weight
                else:
                    # 空单元：随机重置
                    new_gens[i] = self._init_generators()[0]

            # 边界裁剪
            for d in range(self.dim):
                lo, hi = self.bounds[d]
                new_gens[:, d] = np.clip(new_gens[:, d], lo, hi)

            shifts = np.sqrt(np.sum((new_gens - self.generators) ** 2, axis=1))
            max_shift = np.max(shifts)
            self.generators = new_gens

            energy = self._compute_energy(samples, closest)
            energy_history.append(energy)

            if max_shift < self.tol:
                break

        return self.generators, energy, max_shift

    def compute_uniformity_index(self) -> float:
        """
        计算催化剂分布均匀度指数：
            η = 1 - σ_V / μ_V
        其中 σ_V 为各 Voronoi 单元体积（权重）的标准差，μ_V 为均值。
        η → 1 表示完全均匀。
        """
        samples = self._sample_points()
        closest = self._find_closest(samples)
        dens = self.density_func(samples)
        volumes = np.zeros(self.n)
        for i in range(self.n):
            mask = closest == i
            if np.any(mask):
                volumes[i] = np.sum(dens[mask])
        mu = np.mean(volumes)
        if mu < 1.0e-12:
            return 0.0
        sigma = np.std(volumes)
        eta = max(0.0, 1.0 - sigma / mu)
        return eta

    def get_catalyst_loading_map(self, grid_res: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        在规则网格上生成催化剂负载密度图。
        返回 (grid_density, grid_coords)。
        """
        if self.dim != 2:
            raise NotImplementedError("仅 2D 支持网格密度图")
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], grid_res)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], grid_res)
        xv, yv = np.meshgrid(x, y)
        pts = np.column_stack([xv.ravel(), yv.ravel()])
        closest = self._find_closest(pts)
        # 密度图以生成元为中心高斯衰减
        density = np.zeros(len(pts))
        for i in range(len(pts)):
            gen = self.generators[closest[i]]
            dist2 = np.sum((pts[i] - gen) ** 2)
            density[i] = np.exp(-dist2 / (2.0 * 0.01 ** 2))
        density_map = density.reshape((grid_res, grid_res))
        return density_map, np.stack([xv, yv], axis=-1)
