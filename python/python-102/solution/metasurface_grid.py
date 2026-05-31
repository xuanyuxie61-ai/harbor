"""
metasurface_grid.py
===================
基于重心 Voronoi 镶嵌（Centroidal Voronoi Tessellation, CVT）的超构表面
单元最优排布算法。

本模块融合项目 242_cvt_4_movie（CVT 迭代计算）与 725_matlab_map
（地理区域采样与 Voronoi 划分）的核心思想，将 CVT 应用于超构表面
纳米柱阵列的空间排布优化。

科学背景：
在超构表面设计中，纳米柱的排布密度需要适应局部相位梯度。
高相位梯度区域需要更密集的采样（更小的单元），而低梯度区域可以
使用较大的单元。CVT 提供了一种最优的、与密度函数相适应的
空间划分方法。

数学模型：
    给定密度函数 ρ(x,y) > 0，CVT 最小化能量泛函：
        F({z_i}) = Σ_i ∫_{V_i} ρ(x,y) ||(x,y) - z_i||² dA
    其中 V_i 为 Voronoi 单元，z_i 为其重心：
        z_i = ∫_{V_i} ρ(x,y) (x,y) dA / ∫_{V_i} ρ(x,y) dA
"""

import numpy as np


class MetasurfaceCVT:
    """
    超构表面 CVT 网格生成器。
    """

    def __init__(self, region=(-5.0e-6, 5.0e-6, -5.0e-6, 5.0e-6)):
        """
        region : tuple (xmin, xmax, ymin, ymax)
            超构表面物理区域 [m]
        """
        self.xmin, self.xmax, self.ymin, self.ymax = region
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

    def density_function(self, x, y, target_phase_func=None):
        """
        超构表面局部采样密度函数 ρ(x,y)。

        密度与目标相位梯度的模成正比：
            ρ(x,y) ∝ 1 + α |∇Φ_target(x,y)|
        其中 Φ_target 为期望的波前相位分布。

        若未提供 target_phase_func，则使用默认的聚焦透镜相位：
            Φ(x,y) = -k0 * sqrt(x² + y² + f²)
        """
        if target_phase_func is None:
            # 默认：焦距 f = 20 μm 的理想球面波相位
            f = 20.0e-6
            k0 = 2.0 * np.pi / 1.55e-6
            r2 = x ** 2 + y ** 2
            # 相位梯度模的近似
            grad_mag = k0 * np.sqrt(r2) / np.sqrt(r2 + f ** 2)
        else:
            # 数值差分计算梯度
            dx = self.Lx * 1e-4
            dy = self.Ly * 1e-4
            dxph = (target_phase_func(x + dx, y) - target_phase_func(x - dx, y)) / (2 * dx)
            dyph = (target_phase_func(x, y + dy) - target_phase_func(x, y - dy)) / (2 * dy)
            grad_mag = np.sqrt(dxph ** 2 + dyph ** 2)

        alpha = 0.5e-6  # 梯度耦合系数
        rho = 1.0 + alpha * grad_mag
        # 边界增强：在边缘处增加密度以抑制衍射损耗
        edge_factor = 1.0 + 0.3 * (
            np.exp(-((x - self.xmin) / (0.3e-6)) ** 2) +
            np.exp(-((x - self.xmax) / (0.3e-6)) ** 2) +
            np.exp(-((y - self.ymin) / (0.3e-6)) ** 2) +
            np.exp(-((y - self.ymax) / (0.3e-6)) ** 2)
        )
        return rho * edge_factor

    def sample_density(self, n, target_phase_func=None):
        """
        使用拒绝采样在 region 内按照密度函数 ρ(x,y) 生成 n 个样本点。
        """
        samples = np.zeros((n, 2), dtype=np.float64)
        count = 0
        max_rho = 5.0  # 密度上界估计
        while count < n:
            # 均匀候选
            x_cand = np.random.uniform(self.xmin, self.xmax, size=n * 2)
            y_cand = np.random.uniform(self.ymin, self.ymax, size=n * 2)
            rho_cand = self.density_function(x_cand, y_cand, target_phase_func)
            u = np.random.uniform(0, max_rho, size=n * 2)
            mask = u <= rho_cand
            valid_x = x_cand[mask]
            valid_y = y_cand[mask]
            n_valid = len(valid_x)
            take = min(n_valid, n - count)
            samples[count:count + take, 0] = valid_x[:take]
            samples[count:count + take, 1] = valid_y[:take]
            count += take
        return samples

    def voronoi_centroid(self, generator, samples):
        """
        使用 Monte-Carlo 样本近似计算 Voronoi 单元的重心。

        对于生成器 generator（shape (n,2)），将样本分配到最近的生成器，
        然后计算每个生成器对应样本的加权平均重心。
        """
        n_gen = generator.shape[0]
        # 最近邻分配
        # dist[i,j] = ||sample_i - generator_j||²
        # 使用广播避免显式循环
        dx = samples[:, 0][:, None] - generator[:, 0][None, :]
        dy = samples[:, 1][:, None] - generator[:, 1][None, :]
        dist2 = dx ** 2 + dy ** 2
        nearest = np.argmin(dist2, axis=1)

        new_generator = np.zeros_like(generator)
        counts = np.zeros(n_gen, dtype=np.int32)
        for j in range(n_gen):
            mask = nearest == j
            counts[j] = np.sum(mask)
            if counts[j] > 0:
                new_generator[j] = np.mean(samples[mask], axis=0)
            else:
                # 空单元：保持原位（边界退化情况）
                new_generator[j] = generator[j]
        return new_generator, counts

    def compute_cvt(self, n_generators, n_samples_per_iter=5000,
                    max_iter=50, tol=1.0e-8,
                    target_phase_func=None):
        """
        Lloyd 算法迭代求解 CVT。

        Parameters
        ----------
        n_generators : int
            生成器（纳米柱中心）数量
        n_samples_per_iter : int
            每轮迭代使用的 Monte-Carlo 样本数
        max_iter : int
            最大迭代次数
        tol : float
            生成器位移容差

        Returns
        -------
        generators : ndarray, shape (n_generators, 2)
            优化后的纳米柱中心坐标
        energy_history : list
            CVT 能量泛函历史
        """
        # 初始生成器：按密度函数采样
        generators = self.sample_density(n_generators, target_phase_func)
        energy_history = []

        for it in range(max_iter):
            # 生成 Monte-Carlo 样本（带密度）
            samples = self.sample_density(n_samples_per_iter, target_phase_func)

            new_generators, counts = self.voronoi_centroid(generators, samples)

            # 计算能量泛函（Monte-Carlo 估计）
            dx = samples[:, 0][:, None] - generators[:, 0][None, :]
            dy = samples[:, 1][:, None] - generators[:, 1][None, :]
            dist2 = dx ** 2 + dy ** 2
            nearest = np.argmin(dist2, axis=1)
            min_dist2 = np.min(dist2, axis=1)
            rho_s = self.density_function(samples[:, 0], samples[:, 1], target_phase_func)
            energy = np.mean(rho_s * min_dist2)
            energy_history.append(energy)

            shift = np.max(np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)))
            generators = new_generators

            if shift < tol * max(self.Lx, self.Ly):
                print(f"[metasurface_grid] CVT 收敛于迭代 {it}, shift={shift:.3e}")
                break

        return generators, energy_history

    def assign_pillar_parameters(self, generators, phase_func,
                                  height_range=(0.3e-6, 1.2e-6),
                                  width_range=(0.15e-6, 0.5e-6)):
        """
        根据目标相位函数为每个 CVT 生成器分配纳米柱几何参数，
        使得局部透射相位近似等于目标相位。

        使用简单的查表映射：
            Φ_target(x_i, y_i) → (h_i, w_i)
        其中 (h, w) 与相位的映射关系由预先计算的数据库给出。
        这里使用简化的物理模型：
            相位延迟 ≈ (n_eff - n_air) * k0 * h
        其中 n_eff 为波导有效折射率，与宽度 w 相关。
        """
        n = generators.shape[0]
        x = generators[:, 0]
        y = generators[:, 1]
        target_phases = phase_func(x, y)

        # 归一化到 [0, 2π]
        target_phases = np.mod(target_phases, 2.0 * np.pi)

        # 简化的参数空间映射
        # 假设相位与高度近似线性，宽度调制提供精细调节
        h_min, h_max = height_range
        w_min, w_max = width_range

        # 等效折射率随宽度的经验公式（简化模型）
        # n_eff(w) ≈ n_air + (n_si - n_air) * (w / w_max)^β
        beta = 0.7

        # 逆问题：给定目标相位，求解 (h, w)
        # 先固定 w = w_max，计算所需 h；若超出范围，则调节 w
        k0 = 2.0 * np.pi / 1.55e-6
        n_si = 3.48
        n_air = 1.0

        heights = np.zeros(n, dtype=np.float64)
        widths = np.zeros(n, dtype=np.float64)

        for i in range(n):
            phi = target_phases[i]
            # 尝试最大宽度
            w_try = w_max
            n_eff = n_air + (n_si - n_air) * (w_try / w_max) ** beta
            h_needed = phi / (k0 * (n_eff - n_air))
            if h_min <= h_needed <= h_max:
                heights[i] = h_needed
                widths[i] = w_try
            elif h_needed < h_min:
                # 需要更小 n_eff → 更小宽度
                heights[i] = h_min
                n_eff_needed = phi / (k0 * h_min) + n_air
                if n_eff_needed > n_si:
                    n_eff_needed = n_si
                if n_eff_needed < n_air:
                    n_eff_needed = n_air
                w_needed = w_max * ((n_eff_needed - n_air) / (n_si - n_air)) ** (1.0 / beta)
                widths[i] = np.clip(w_needed, w_min, w_max)
            else:
                # h_needed > h_max，需要更大 n_eff
                heights[i] = h_max
                n_eff_needed = phi / (k0 * h_max) + n_air
                if n_eff_needed > n_si:
                    n_eff_needed = n_si
                w_needed = w_max * ((n_eff_needed - n_air) / (n_si - n_air)) ** (1.0 / beta)
                widths[i] = np.clip(w_needed, w_min, w_max)

        return heights, widths

    def compute_voronoi_areas(self, generators, n_samples=200000):
        """
        使用 Monte-Carlo 采样估计每个 Voronoi 单元的面积。
        """
        x = np.random.uniform(self.xmin, self.xmax, size=n_samples)
        y = np.random.uniform(self.ymin, self.ymax, size=n_samples)
        dx = x[:, None] - generators[:, 0][None, :]
        dy = y[:, None] - generators[:, 1][None, :]
        dist2 = dx ** 2 + dy ** 2
        nearest = np.argmin(dist2, axis=1)
        n_gen = generators.shape[0]
        areas = np.zeros(n_gen, dtype=np.float64)
        total_area = self.Lx * self.Ly
        for j in range(n_gen):
            areas[j] = total_area * np.sum(nearest == j) / n_samples
        return areas


def demo():
    """演示：为聚焦超构表面生成 CVT 排布。"""
    grid = MetasurfaceCVT(region=(-5.0e-6, 5.0e-6, -5.0e-6, 5.0e-6))

    # 目标相位：球面波聚焦
    k0 = 2.0 * np.pi / 1.55e-6
    f = 20.0e-6

    def phase_func(x, y):
        return -k0 * (np.sqrt(x ** 2 + y ** 2 + f ** 2) - f)

    generators, energy = grid.compute_cvt(
        n_generators=200,
        n_samples_per_iter=8000,
        max_iter=30,
        target_phase_func=phase_func
    )
    print(f"[metasurface_grid] CVT 能量最终值: {energy[-1]:.6e}")

    heights, widths = grid.assign_pillar_parameters(generators, phase_func)
    areas = grid.compute_voronoi_areas(generators)
    print(f"[metasurface_grid] 平均纳米柱高度: {np.mean(heights):.3e} m, "
          f"平均宽度: {np.mean(widths):.3e} m")
    print(f"[metasurface_grid] 平均单元面积: {np.mean(areas):.3e} m², "
          f"填充因子≈{np.mean(widths**2 / areas)*100:.1f}%")
    return generators, heights, widths, areas


if __name__ == "__main__":
    demo()
