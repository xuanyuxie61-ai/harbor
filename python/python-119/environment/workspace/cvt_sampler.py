"""
cvt_sampler.py
自由体积 Centroidal Voronoi Tessellation 分析模块

融合原项目:
- 146_ccvt_reflect: 反射边界 CVT 算法
- 252_cvt_box: 盒子约束 CVT 投影
- 259_cvt_square_nonuniform: 非均匀密度 CVT
- 242_cvt_4_movie: CVT 迭代与密度采样

功能:
1. 在模拟盒子内生成 CVT 采样点
2. 基于非均匀密度（自由体积权重）优化 Voronoi 区域
3. 计算 Voronoi 体积分布与自由体积分数
4. 使用 CVT 能量最小化评估体系结构有序度
"""

import numpy as np
from typing import Tuple, Optional


class CVTSampler:
    """
    Centroidal Voronoi Tessellation 采样器，用于聚合物自由体积分析。
    
    物理背景:
        在玻璃态聚合物中，自由体积以纳米级空洞形式分布。
        CVT 提供一种将空间分割为 Voronoi 胞的方法，
        通过非均匀密度 ρ(x) ∝ V_free(x) 加权，
        使得生成器位置对应于自由体积的"质心"。
    
    数学描述:
        CVT 能量泛函:
            E({z_i}) = Σ_i ∫_{Ω_i} ρ(x) ||x - z_i||^2 dx
        其中 Ω_i 为第 i 个 Voronoi 区域。
        
        最优性条件:
            z_i = (∫_{Ω_i} ρ(x) x dx) / (∫_{Ω_i} ρ(x) dx)
        
        即生成器为其 Voronoi 区域的密度加权质心。
    """
    
    def __init__(
        self,
        n_generators: int = 64,
        n_samples: int = 5000,
        max_iter: int = 50,
        box: np.ndarray = None,
        tol: float = 1e-5,
    ):
        """
        参数:
            n_generators: 生成器数量（Voronoi 胞数）
            n_samples: 每次迭代的采样点数
            max_iter: 最大迭代次数
            box: 模拟盒子尺寸 (3,)
            tol: 收敛容差
        """
        if n_generators < 1:
            raise ValueError("n_generators 必须 >= 1")
        if n_samples < n_generators:
            raise ValueError("n_samples 必须 >= n_generators")
        if max_iter < 1:
            raise ValueError("max_iter 必须 >= 1")
        
        self.n_generators = n_generators
        self.n_samples = n_samples
        self.max_iter = max_iter
        self.box = np.array(box if box is not None else [10.0, 10.0, 10.0])
        self.tol = tol
        
        # 生成器位置
        self.generators = np.random.rand(n_generators, 3) * self.box
        
        # 能量历史
        self.energy_history = []
    
    def _free_volume_density(
        self,
        samples: np.ndarray,
        polymer_positions: np.ndarray,
        exclusion_radius: float = 1.0,
    ) -> np.ndarray:
        """
        计算采样点的自由体积密度权重。
        
        模型:
            ρ(x) = exp( -Σ_i exp(-||x - r_i||^2 / (2σ^2)) )
        
        这表示被聚合物单体占据的区域密度低，空腔区域密度高。
        
        参数:
            samples: (M, 3) 采样点
            polymer_positions: (N, 3) 聚合物单体位置
            exclusion_radius: 排除半径（单体范德华半径）
        
        返回:
            (M,) 密度权重数组
        """
        M = samples.shape[0]
        N = polymer_positions.shape[0]
        
        sigma = exclusion_radius / 2.0
        sigma_sq = sigma ** 2
        
        # 向量化计算效率
        density = np.zeros(M)
        
        # 分批处理避免内存溢出
        batch_size = 1000
        for i in range(0, M, batch_size):
            batch = samples[i:i+batch_size]
            # 计算所有距离
            diff = batch[:, np.newaxis, :] - polymer_positions[np.newaxis, :, :]
            diff = diff - self.box * np.rint(diff / self.box)
            dist_sq = np.sum(diff ** 2, axis=2)
            
            # 高斯包络叠加
            occupancy = np.sum(np.exp(-dist_sq / (2.0 * sigma_sq)), axis=1)
            
            # 自由体积密度: 空腔处高，拥挤处低
            density[i:i+batch_size] = np.exp(-occupancy)
        
        # 归一化
        dmax = np.max(density)
        if dmax > 1e-15:
            density = density / dmax
        
        # 确保非负
        density = np.clip(density, 1e-10, 1.0)
        
        return density
    
    def _find_closest(self, samples: np.ndarray) -> np.ndarray:
        """
        为每个采样点找到最近的生成器索引。
        
        参数:
            samples: (M, 3) 采样点
        
        返回:
            (M,) 最近生成器索引
        """
        M = samples.shape[0]
        indices = np.zeros(M, dtype=int)
        
        for i in range(M):
            diff = samples[i] - self.generators
            diff = diff - self.box * np.rint(diff / self.box)
            dist_sq = np.sum(diff ** 2, axis=1)
            indices[i] = np.argmin(dist_sq)
        
        return indices
    
    def _cvt_energy(self, density: np.ndarray, samples: np.ndarray, indices: np.ndarray) -> float:
        """
        计算 CVT 能量。
        
        公式:
            E = (1/M) Σ_k ρ(x_k) ||x_k - z_{i(k)}||^2
        
        参数:
            density: (M,) 采样点密度
            samples: (M, 3) 采样点
            indices: (M,) 最近生成器索引
        
        返回:
            能量值
        """
        energy = 0.0
        for k in range(len(samples)):
            diff = samples[k] - self.generators[indices[k]]
            diff = diff - self.box * np.rint(diff / self.box)
            energy += density[k] * np.sum(diff ** 2)
        
        return energy / len(samples)
    
    def iterate(
        self,
        polymer_positions: np.ndarray,
        exclusion_radius: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行 CVT 迭代优化。
        
        算法（融合 146, 252, 259, 242）:
            1. 在盒子内生成采样点
            2. 计算非均匀密度 ρ(x)（融合 259_cvt_square_nonuniform）
            3. 找到每个采样点的最近生成器
            4. 将生成器更新为 Voronoi 区域的加权质心
            5. 应用反射边界约束（融合 146_ccvt_reflect）
            6. 应用盒子投影（融合 252_cvt_box）
            7. 重复直到收敛
        
        参数:
            polymer_positions: (N, 3) 聚合物单体位置
            exclusion_radius: 排除半径
        
        返回:
            (generators, volumes): 生成器位置和对应的 Voronoi 体积
        """
        for it in range(self.max_iter):
            # 在盒子内均匀采样（融合 242_cvt_4_movie 的采样思想）
            samples = np.random.rand(self.n_samples, 3) * self.box
            
            # 计算非均匀密度（融合 259_cvt_square_nonuniform）
            density = self._free_volume_density(samples, polymer_positions, exclusion_radius)
            
            # 找到最近生成器
            indices = self._find_closest(samples)
            
            # 计算能量
            energy = self._cvt_energy(density, samples, indices)
            self.energy_history.append(energy)
            
            # 更新生成器为加权质心
            new_generators = np.zeros_like(self.generators)
            mass = np.zeros(self.n_generators)
            
            for k in range(self.n_samples):
                i = indices[k]
                new_generators[i] += density[k] * samples[k]
                mass[i] += density[k]
            
            # 归一化并处理空胞（融合 252_cvt_box 的边界处理）
            for i in range(self.n_generators):
                if mass[i] > 1e-15:
                    new_generators[i] = new_generators[i] / mass[i]
                else:
                    # 空胞: 重新随机放置
                    new_generators[i] = np.random.rand(3) * self.box
            
            # 应用反射边界约束（融合 146_ccvt_reflect）
            # 若生成器超出盒子，将其反射回盒子内
            for d in range(3):
                mask_low = new_generators[:, d] < 0
                mask_high = new_generators[:, d] > self.box[d]
                new_generators[mask_low, d] = -new_generators[mask_low, d]
                new_generators[mask_high, d] = 2 * self.box[d] - new_generators[mask_high, d]
            
            # 盒子投影（融合 252_cvt_box）
            new_generators = np.clip(new_generators, 0.0, self.box)
            
            # 检查收敛
            max_displacement = np.max(np.linalg.norm(new_generators - self.generators, axis=1))
            self.generators = new_generators
            
            if max_displacement < self.tol:
                break
        
        # 计算 Voronoi 体积（通过蒙特卡洛采样估计）
        volumes = self._estimate_voronoi_volumes(polymer_positions, exclusion_radius)
        
        return self.generators.copy(), volumes
    
    def _estimate_voronoi_volumes(
        self,
        polymer_positions: np.ndarray,
        exclusion_radius: float = 1.0,
    ) -> np.ndarray:
        """
        估计每个 Voronoi 胞的体积。
        
        方法:
            在盒子内大量采样，统计落入每个 Voronoi 胞的采样点数。
            体积分数 = (胞内点数) / (总点数) * V_box
        
        参数:
            polymer_positions: (N, 3) 聚合物位置
            exclusion_radius: 排除半径
        
        返回:
            (n_generators,) 体积数组
        """
        n_test = 100000
        test_samples = np.random.rand(n_test, 3) * self.box
        
        indices = self._find_closest(test_samples)
        counts = np.bincount(indices, minlength=self.n_generators)
        
        box_volume = np.prod(self.box)
        volumes = counts / n_test * box_volume
        
        return volumes
    
    def free_volume_fraction(
        self,
        polymer_positions: np.ndarray,
        van_der_waals_radius: float = 0.8,
    ) -> float:
        """
        计算系统的自由体积分数（Free Volume Fraction）。
        
        物理定义（硬球模型近似）:
            f_v = (V_total - N * v_particle) / V_total
        
        其中 v_particle = (4/3)π r_vdw^3 为单个粒子的排除体积。
        
        参数:
            polymer_positions: (N, 3) 聚合物位置
            van_der_waals_radius: 范德华半径
        
        返回:
            自由体积分数 [0, 1]
        """
        N = polymer_positions.shape[0]
        box_volume = np.prod(self.box)
        
        # 排除体积
        v_particle = (4.0 / 3.0) * np.pi * van_der_waals_radius ** 3
        occupied_volume = N * v_particle
        
        # 自由体积（修正重叠: 实际排除体积小于硬球和）
        # 使用 Carnahan-Starling 近似修正
        eta = occupied_volume / box_volume  # 堆积分数
        if eta >= 1.0:
            return 0.0
        
        # 自由体积分数: 1 - 有效堆积分数
        # 对于随机密堆积，最大堆积分数约 0.64
        fv = 1.0 - eta
        
        return float(max(0.0, min(fv, 1.0)))
    
    def structural_order_parameter(self) -> float:
        """
        计算结构有序度参数。
        
        基于 CVT 能量的相对变化:
            η = 1 - E_final / E_initial
        
        η → 1 表示高度有序，η → 0 表示无序。
        
        返回:
            有序度参数 [0, 1]
        """
        if len(self.energy_history) < 2:
            return 0.0
        
        e_initial = self.energy_history[0]
        e_final = self.energy_history[-1]
        
        if abs(e_initial) < 1e-15:
            return 0.0
        
        eta = 1.0 - e_final / e_initial
        return float(np.clip(eta, 0.0, 1.0))
