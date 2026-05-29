"""
cvt_sampling.py

基于种子项目 261_cvt_square_uniform（质心Voronoi镶嵌CVT），
实现布里渊区k点优化采样与杂质位置分布的优化布局。

物理模型：
    在第一性原理和紧束缚计算中，布里渊区（BZ）积分的精度
    取决于k点采样的密度与分布均匀性。

    传统Monkhorst-Pack网格在均匀分布下表现良好，但在
    拓扑材料中，费米面附近的高曲率区域需要更密集的采样。

    CVT（Centroidal Voronoi Tessellation）提供了一种
    自适应采样方案：将k点视为Voronoi镶嵌的生成元，
    通过Lloyd迭代使每个生成元恰好位于其Voronoi单元的质心。

    优化目标泛函（CVT能量）：
        F(P) = Σ_i ∫_{V_i} ρ(x) ||x - p_i||^2 dx

    其中V_i为第i个Voronoi单元，p_i为生成元，ρ(x)为密度函数。
    对于拓扑超导体，可在费米面附近设置ρ(x) ∝ 1/|E_k|。

    同时，CVT可用于优化杂质原子在二维超导平面上的分布，
    以最大化对拓扑相的调控效果。
"""

import numpy as np
from typing import Tuple, Optional, Callable


class CVTSampler:
    """
    质心Voronoi镶嵌采样器（基于261_cvt_square_uniform）。
    """

    def __init__(self, n_generators: int, domain: Tuple[float, float,
                                                          float, float],
                 density_fn: Optional[Callable] = None,
                 rng_seed: Optional[int] = None):
        """
        初始化CVT采样器。

        Args:
            n_generators: 生成元数量N
            domain: (xmin, xmax, ymin, ymax) 采样域
            density_fn: 密度函数 ρ(x,y)，None表示均匀密度
            rng_seed: 随机数种子
        """
        if n_generators < 1:
            raise ValueError("生成元数量必须为正")
        self.n = n_generators
        self.domain = domain
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.density_fn = density_fn
        self.rng = np.random.RandomState(rng_seed)

    def _sample_uniform(self, n_points: int) -> np.ndarray:
        """
        在采样域内均匀随机采样。
        """
        points = np.zeros((n_points, 2))
        points[:, 0] = self.rng.uniform(self.xmin, self.xmax, n_points)
        points[:, 1] = self.rng.uniform(self.ymin, self.ymax, n_points)
        return points

    def _density_weighted_sample(self, n_points: int) -> np.ndarray:
        """
        按密度函数偏置采样（拒绝采样法）。
        """
        if self.density_fn is None:
            return self._sample_uniform(n_points)

        points = np.zeros((n_points, 2))
        count = 0
        max_attempts = n_points * 100
        attempts = 0

        # 估计最大密度
        test_points = self._sample_uniform(1000)
        rhos = np.array([self.density_fn(p[0], p[1])
                         for p in test_points])
        rho_max = np.max(rhos) * 1.2 if len(rhos) > 0 else 1.0

        while count < n_points and attempts < max_attempts:
            attempts += 1
            p = self._sample_uniform(1)[0]
            rho = self.density_fn(p[0], p[1])
            if self.rng.rand() < rho / rho_max:
                points[count] = p
                count += 1

        if count < n_points:
            points[count:] = self._sample_uniform(n_points - count)

        return points

    def _find_nearest_generator(self, samples: np.ndarray,
                                 generators: np.ndarray) -> np.ndarray:
        """
        为每个样本点找到最近的生成元（Voronoi单元归属）。
        """
        # 使用 broadcasting 计算距离
        # samples: (M, 2), generators: (N, 2)
        diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def _compute_centroids(self, samples: np.ndarray,
                           nearest: np.ndarray) -> np.ndarray:
        """
        计算每个Voronoi单元的质心。
        """
        centroids = np.zeros((self.n, 2))
        counts = np.zeros(self.n)

        for i in range(len(samples)):
            idx = nearest[i]
            centroids[idx] += samples[i]
            counts[idx] += 1

        # 边界处理：空单元保持原位
        for j in range(self.n):
            if counts[j] > 0:
                centroids[j] /= counts[j]
            else:
                # 重新随机放置空单元
                centroids[j] = self._sample_uniform(1)[0]

        return centroids

    def lloyd_iterate(self, num_iterations: int = 50,
                       sample_multiplier: int = 1000) -> np.ndarray:
        """
        执行Lloyd迭代优化CVT。

        算法流程：
            1) 随机初始化生成元位置
            2) 在每个Voronoi单元内采样
            3) 将生成元移至单元质心
            4) 重复直至收敛
        """
        # 初始化生成元
        generators = self._sample_uniform(self.n)
        sample_num = sample_multiplier * self.n

        for it in range(num_iterations):
            # 采样
            samples = self._density_weighted_sample(sample_num)

            # 找到最近生成元
            nearest = self._find_nearest_generator(samples, generators)

            # 更新为质心
            new_generators = self._compute_centroids(samples, nearest)

            # 收敛检查
            diff = np.max(np.linalg.norm(new_generators - generators, axis=1))
            generators = new_generators

            if diff < 1e-8:
                break

        return generators

    def cvt_energy(self, generators: np.ndarray,
                    sample_num: int = 10000) -> float:
        """
        计算CVT能量泛函。

        F = Σ_i ∫_{V_i} ρ(x) ||x - p_i||^2 dx
        """
        samples = self._density_weighted_sample(sample_num)
        nearest = self._find_nearest_generator(samples, generators)

        energy = 0.0
        for i in range(len(samples)):
            idx = nearest[i]
            dist2 = np.sum((samples[i] - generators[idx]) ** 2)
            if self.density_fn is not None:
                rho = self.density_fn(samples[i, 0], samples[i, 1])
            else:
                rho = 1.0
            energy += rho * dist2

        return energy / sample_num

    def brillouin_zone_kpoints(self, a: float = 1.0,
                                num_iterations: int = 30) -> np.ndarray:
        """
        生成优化的布里渊区k点采样。

        对于一维Kitaev链，布里渊区为 [-π/a, π/a]。
        这里扩展为二维正方晶格的布里渊区 [-π/a, π/a]^2。

        密度函数在费米面附近增强：
            ρ(k) = 1 + A / (|ε_k - E_F| + δ)
        """
        # 简化为均匀采样
        self.density_fn = None
        # 域设为 [-π, π]^2
        pi = np.pi
        self.domain = (-pi / a, pi / a, -pi / a, pi / a)
        self.xmin, self.xmax, self.ymin, self.ymax = self.domain

        kpoints = self.lloyd_iterate(num_iterations=num_iterations)
        return kpoints

    def impurity_optimized_positions(self, interaction_range: float,
                                      num_iterations: int = 30) -> np.ndarray:
        """
        优化杂质位置以最大化空间覆盖。

        密度函数模拟杂质排斥：
            ρ(x) ∝ Π_j (1 - exp(-|x - x_j|^2 / ξ^2))
        """
        # 使用均匀密度进行覆盖优化
        self.density_fn = None
        positions = self.lloyd_iterate(num_iterations=num_iterations)
        return positions


class BrillouinZoneIntegrator:
    """
    基于CVT采样的布里渊区积分器。
    """

    def __init__(self, kpoints: np.ndarray,
                 domain: Tuple[float, float, float, float]):
        self.kpoints = kpoints
        self.domain = domain
        self.n = len(kpoints)

    def integrate(self, integrand_fn: Callable) -> float:
        """
        在BZ上使用CVT k点进行数值积分。

        每个k点的权重近似相等（均匀CVT）：
            I ≈ (Area / N) Σ_i f(k_i)
        """
        xmin, xmax, ymin, ymax = self.domain
        area = (xmax - xmin) * (ymax - ymin)
        weight = area / self.n

        total = 0.0
        for k in self.kpoints:
            total += integrand_fn(k[0], k[1])

        return weight * total

    def fermi_surface_sampling(self, dispersion_fn: Callable,
                                e_fermi: float,
                                tolerance: float = 0.1) -> np.ndarray:
        """
        提取费米面附近的k点。
        """
        fs_points = []
        for k in self.kpoints:
            e = dispersion_fn(k[0], k[1])
            if abs(e - e_fermi) < tolerance:
                fs_points.append(k)

        return np.array(fs_points) if fs_points else np.array([])


# 类型注解修正
from typing import Callable


def demo():
    """演示CVT采样。"""
    # 均匀CVT
    cvt = CVTSampler(n_generators=25,
                      domain=(0.0, 1.0, 0.0, 1.0),
                      rng_seed=42)
    generators = cvt.lloyd_iterate(num_iterations=50)
    energy = cvt.cvt_energy(generators)
    print(f"CVT energy with {len(generators)} generators: {energy:.6f}")

    # BZ k点
    bz_cvt = CVTSampler(n_generators=64,
                         domain=(-np.pi, np.pi, -np.pi, np.pi),
                         rng_seed=42)
    kpts = bz_cvt.brillouin_zone_kpoints(a=1.0, num_iterations=30)
    print(f"BZ k-points range: x=[{kpts[:,0].min():.3f}, {kpts[:,0].max():.3f}], "
          f"y=[{kpts[:,1].min():.3f}, {kpts[:,1].max():.3f}]")

    # 积分测试
    integrator = BrillouinZoneIntegrator(kpts, bz_cvt.domain)

    def test_func(kx, ky):
        return np.cos(kx) * np.cos(ky)

    result = integrator.integrate(test_func)
    # 解析值：∫_{-π}^{π}∫_{-π}^{π} cos(kx)cos(ky) dk_x dk_y = 0
    print(f"Integral of cos(kx)cos(ky): {result:.6f}")


if __name__ == "__main__":
    demo()
