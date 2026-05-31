"""
spray_dynamics.py - 喷雾液滴动力学与分布优化
===============================================
基于CVT理论的火箭发动机喷雾液滴最优空间分布计算。

原项目映射:
- 238_cvt      -> Centroidal Voronoi Tessellation优化液滴空间分布
- 255_cvt_corn -> CVT径向生长模型用于液滴蒸发前沿演化

科学背景:
=========
液体火箭发动机中，推进剂通过喷注器雾化后形成喷雾。
液滴的空间分布对燃烧效率和稳定性至关重要:

1. 液滴尺寸分布: 通常用Rosin-Rammler分布描述
       Q(d) = 1 - exp[-(d/d_0)^n]
   其中 d_0 为特征直径, n 为均匀度指数

2. 液滴蒸发率 (d^2-law):
       d(t)^2 = d_0^2 - K·t
   其中 K = 8·k_g·ln(1+B) / (ρ_l·C_p,g)
   B = C_p,g·(T_g - T_b) / L_v  为Spalding传质数

3. 最优分布准则:
   液滴应尽可能均匀分布以最大化蒸发面积，
   同时避免局部过浓导致的不稳定燃烧。
   这等价于最小化CVT能量泛函:
       E(Ω; {x_i}) = Σ_i ∫_{V_i} ||x - x_i||^2 ρ(x) dx
"""

import numpy as np
from utils import check_finite_array, safe_divide, robust_sqrt


class SprayDistributionCVT:
    """
    基于CVT的喷雾液滴空间分布优化。
    
    将液滴视为Voronoi单元的生成器，
    通过Lloyd迭代使生成器趋近于各自Voronoi区域的质心，
    从而获得最优空间分布。
    """
    
    def __init__(self,
                 chamber_radius: float = 0.15,
                 chamber_length: float = 0.60,
                 n_droplets: int = 500,
                 droplet_diameter_mean: float = 80.0e-6,  # 80 μm
                 droplet_diameter_std: float = 20.0e-6,
                 gas_temperature: float = 3000.0,
                 gas_pressure: float = 7.0e6):
        
        self.R_c = chamber_radius
        self.L_c = chamber_length
        self.n_droplets = n_droplets
        self.d_0 = droplet_diameter_mean
        self.d_std = droplet_diameter_std
        self.T_g = gas_temperature
        self.P_g = gas_pressure
        
        # 液滴位置 ( cylindrical coordinates: z, r, θ )
        self.positions = None
        self.velocities = None
        self.diameters = None
        self.masses = None
        
        # CVT迭代参数
        self.energy_history = []
    
    def generate_initial_distribution(self, seed: int = 42) -> np.ndarray:
        """
        生成初始液滴分布 (准随机采样)。
        
        分布策略:
            - 轴向: 在喷注面板附近密集，下游稀疏
            - 径向: 避开中心轴线 (防止回流区过浓)
            - 周向: 均匀分布
        """
        rng = np.random.RandomState(seed)
        
        # 轴向分布: 指数衰减 (喷注面板附近密集)
        lambda_z = 1.0 / (0.15 * self.L_c)  # 特征衰减长度
        z = rng.exponential(1.0 / lambda_z, self.n_droplets)
        z = np.clip(z, 0.0, self.L_c)
        
        # 径向分布: 避开中心, 壁面附近稀疏
        # 使用Beta分布
        r_norm = rng.beta(2.0, 2.0, self.n_droplets)
        r = 0.1 * self.R_c + 0.8 * self.R_c * r_norm
        r = np.clip(r, 0.0, self.R_c)
        
        # 周向均匀
        theta = 2.0 * np.pi * rng.rand(self.n_droplets)
        
        self.positions = np.column_stack([z, r, theta])
        
        # 初始化速度 (轴向为主)
        u_z = 15.0 + 10.0 * rng.randn(self.n_droplets)  # m/s
        u_r = 2.0 * rng.randn(self.n_droplets)
        u_theta = 5.0 * rng.randn(self.n_droplets)
        self.velocities = np.column_stack([u_z, u_r, u_theta])
        
        # 液滴直径: Rosin-Rammler分布采样
        n_rr = 3.5  # 均匀度指数
        d_0_rr = self.d_0 * (np.log(2.0)) ** (1.0 / n_rr)
        u = rng.rand(self.n_droplets)
        d = d_0_rr * (-np.log(1.0 - u)) ** (1.0 / n_rr)
        self.diameters = np.clip(d, 10e-6, 200e-6)
        
        # 质量
        rho_l = 807.0  # RP-1密度 kg/m^3
        self.masses = (np.pi / 6.0) * rho_l * self.diameters ** 3
        
        return self.positions
    
    def _find_closest_generator(self, samples: np.ndarray, generators: np.ndarray) -> np.ndarray:
        """
        对每个采样点找到最近的生成器。
        
        距离度量 (柱坐标):
            d^2 = Δz^2 + r_1^2 + r_2^2 - 2·r_1·r_2·cos(Δθ)
        
        这是欧氏距离在柱坐标下的表达。
        """
        n_samples = samples.shape[0]
        n_gen = generators.shape[0]
        
        # 转换为笛卡尔坐标进行距离计算
        # 样本点
        xs = samples[:, 1:2] * np.cos(samples[:, 2:3])
        ys = samples[:, 1:2] * np.sin(samples[:, 2:3])
        zs = samples[:, 0:1]
        
        # 生成器
        xg = generators[:, 1:2] * np.cos(generators[:, 2:3])
        yg = generators[:, 1:2] * np.sin(generators[:, 2:3])
        zg = generators[:, 0:1]
        
        # 广播计算距离
        dx = xs.T - xg  # (n_gen, n_samples)
        dy = ys.T - yg
        dz = zs.T - zg
        
        dist_sq = dx ** 2 + dy ** 2 + dz ** 2
        nearest = np.argmin(dist_sq, axis=0)
        min_dist_sq = np.min(dist_sq, axis=0)
        
        return nearest, min_dist_sq
    
    def cvt_iterate(self, n_samples: int = 10000) -> tuple:
        """
        执行一次CVT迭代 (Lloyd算法)。
        
        算法步骤:
            1. 在燃烧室内采样大量点
            2. 对每个采样点找到最近的液滴(生成器)
            3. 更新每个生成器为其Voronoi区域的质心
            4. 计算能量泛函
        
        能量泛函:
            E = (1/N_s) Σ_s min_i ||x_s - g_i||^2
        
        原项目映射: 238_cvt / cvt_iterate.m
        """
        if self.positions is None:
            raise RuntimeError("Call generate_initial_distribution first.")
        
        generators = self.positions.copy()
        n_gen = generators.shape[0]
        
        # 在燃烧室内均匀采样
        rng = np.random.RandomState(123)
        z_samp = rng.rand(n_samples) * self.L_c
        r_samp = self.R_c * np.sqrt(rng.rand(n_samples))  # 面积均匀
        theta_samp = 2.0 * np.pi * rng.rand(n_samples)
        samples = np.column_stack([z_samp, r_samp, theta_samp])
        
        # 找到最近生成器
        nearest, min_dist_sq = self._find_closest_generator(samples, generators)
        
        # 计算新的质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_gen)
        energy = 0.0
        
        for j in range(n_samples):
            idx = nearest[j]
            new_generators[idx] += samples[j]
            counts[idx] += 1
            energy += min_dist_sq[j]
        
        # 避免除零
        for j in range(n_gen):
            if counts[j] > 0:
                new_generators[j] /= counts[j]
            else:
                # 未被采到的生成器保持原位
                new_generators[j] = generators[j]
        
        # 周向角归一化到 [0, 2π)
        new_generators[:, 2] = new_generators[:, 2] % (2.0 * np.pi)
        
        # 边界处理
        new_generators[:, 0] = np.clip(new_generators[:, 0], 0.0, self.L_c)
        new_generators[:, 1] = np.clip(new_generators[:, 1], 0.0, self.R_c)
        
        # 计算变化量
        it_diff = np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)).sum()
        energy = energy / n_samples
        
        self.positions = new_generators
        self.energy_history.append(energy)
        
        return it_diff, energy
    
    def optimize_distribution(self, n_iterations: int = 50, n_samples: int = 10000,
                              tolerance: float = 1e-6) -> dict:
        """
        执行CVT优化直到收敛。
        
        参数:
            n_iterations: 最大迭代次数
            n_samples: 每次迭代的采样点数
            tolerance: 收敛容差
        
        返回:
            优化结果统计
        """
        if self.positions is None:
            self.generate_initial_distribution()
        
        for it in range(n_iterations):
            it_diff, energy = self.cvt_iterate(n_samples=n_samples)
            
            if it_diff < tolerance:
                break
        
        return {
            "iterations": it + 1,
            "final_energy": energy,
            "energy_history": np.array(self.energy_history),
            "final_positions": self.positions,
            "mean_diameter": float(np.mean(self.diameters)),
            "std_diameter": float(np.std(self.diameters))
        }
    
    def compute_evaporation_rate(self) -> np.ndarray:
        """
        计算每个液滴的蒸发速率常数 K。
        
        d^2-law:
            d(d^2)/dt = -K
        
        K = 8·k_g·ln(1+B) / (ρ_l·C_p,g)
        
        其中:
            k_g: 气体导热系数, W/(m·K)
            B = C_p,g·(T_g - T_b) / L_v: Spalding传质数
            T_b: 液滴沸点 (RP-1 ~ 500K at 7MPa)
        """
        if self.diameters is None:
            raise RuntimeError("Droplets not initialized.")
        
        T_b = 500.0  # RP-1沸点 at 7MPa, K (近似)
        L_v = 2.13e6  # 汽化潜热, J/kg
        k_g = 0.08  # 燃烧产物导热系数, W/(m·K)
        C_p_g = 1800.0  # J/(kg·K)
        rho_l = 807.0  # kg/m^3
        
        # Spalding数
        B = C_p_g * (self.T_g - T_b) / L_v
        B = np.clip(B, 0.01, 10.0)
        
        K = 8.0 * k_g * np.log(1.0 + B) / (rho_l * C_p_g)
        
        return K
    
    def simulate_droplet_lifetime(self, dt: float = 1.0e-5, n_steps: int = 1000) -> dict:
        """
        模拟液滴蒸发寿命。
        
        原项目映射: 255_cvt_corn 的径向生长演化思想
        
        时间推进:
            d_i(t+Δt)^2 = d_i(t)^2 - K·Δt
        
        同时更新位置 (考虑阻力):
            dv/dt = (3/4)·(ρ_g/ρ_l)·(C_d/d)·|u_g - v|·(u_g - v)
        """
        if self.positions is None or self.diameters is None:
            raise RuntimeError("Droplets not initialized.")
        
        d_current = self.diameters.copy()
        pos_current = self.positions.copy()
        vel_current = self.velocities.copy()
        
        K = self.compute_evaporation_rate()
        
        # 气体参数
        rho_g = 15.0  # 燃烧产物密度 at 7MPa, kg/m^3
        mu_g = 8.5e-5  # Pa·s
        
        # 记录历史
        d_history = [d_current.copy()]
        pos_history = [pos_current.copy()]
        
        for step in range(n_steps):
            # 蒸发
            d_sq = d_current ** 2 - K * dt
            d_current = np.sqrt(np.maximum(d_sq, 0.0))
            
            # 完全蒸发的液滴标记
            evaporated = d_current < 1e-9
            
            # 阻力 (Stokes阻力, Re < 1)
            # C_d = 24/Re for Stokes flow
            u_rel = 50.0 - vel_current[:, 0]  # 假设气体轴向速度50 m/s
            u_rel = np.clip(u_rel, -500.0, 500.0)
            Re = rho_g * np.abs(u_rel) * d_current / mu_g
            Re = np.clip(Re, 1e-6, 1000.0)
            
            # Schiller-Naumann阻力系数
            C_d = np.where(Re < 1.0, 24.0 / Re,
                           24.0 / Re * (1.0 + 0.15 * Re ** 0.687))
            C_d = np.clip(C_d, 0.0, 500.0)
            
            # 加速度
            d_safe = np.maximum(d_current, 1e-9)
            a_z = (3.0 / 4.0) * (rho_g / 807.0) * (C_d / d_safe) * \
                  np.abs(u_rel) * u_rel
            a_z = np.clip(a_z, -1e6, 1e6)
            
            vel_current[:, 0] += a_z * dt
            pos_current[:, 0] += vel_current[:, 0] * dt
            
            # 边界条件
            pos_current[:, 0] = np.clip(pos_current[:, 0], 0.0, self.L_c)
            pos_current[:, 1] = np.clip(pos_current[:, 1], 0.0, self.R_c)
            
            # 蒸发完的液滴停留在原位
            pos_current[evaporated] = pos_history[0][evaporated]
            
            if step % 100 == 0:
                d_history.append(d_current.copy())
                pos_history.append(pos_current.copy())
        
        return {
            "final_diameters": d_current,
            "final_positions": pos_current,
            "evaporation_fraction": float(np.mean(d_current < 1e-9)),
            "diameter_history": d_history,
            "position_history": pos_history
        }
    
    def compute_spray_statistics(self) -> dict:
        """计算喷雾统计特性。"""
        if self.positions is None:
            raise RuntimeError("Droplets not initialized.")
        
        # Sauter平均直径 (SMD)
        if self.diameters is not None:
            d32 = np.sum(self.diameters ** 3) / np.sum(self.diameters ** 2)
        else:
            d32 = self.d_0
        
        # 体积浓度分布 (轴向)
        n_bins = 20
        z_bins = np.linspace(0, self.L_c, n_bins + 1)
        bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
        concentrations = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (self.positions[:, 0] >= z_bins[i]) & (self.positions[:, 0] < z_bins[i+1])
            if np.sum(mask) > 0:
                bin_volume = np.pi * self.R_c ** 2 * (z_bins[i+1] - z_bins[i])
                concentrations[i] = np.sum(self.masses[mask]) / bin_volume if self.masses is not None else np.sum(mask) / bin_volume
        
        return {
            "sauter_mean_diameter": float(d32),
            "n_droplets": self.n_droplets,
            "mean_axial_position": float(np.mean(self.positions[:, 0])),
            "std_axial_position": float(np.std(self.positions[:, 0])),
            "mean_radial_position": float(np.mean(self.positions[:, 1])),
            "concentration_profile": concentrations,
            "bin_centers": bin_centers
        }


if __name__ == "__main__":
    spray = SprayDistributionCVT(n_droplets=200)
    spray.generate_initial_distribution()
    
    result = spray.optimize_distribution(n_iterations=30, n_samples=5000)
    print(f"CVT converged in {result['iterations']} iterations")
    print(f"Final energy: {result['final_energy']:.6e}")
    
    lifetime = spray.simulate_droplet_lifetime(n_steps=500)
    print(f"Evaporation fraction: {lifetime['evaporation_fraction']:.3f}")
    
    stats = spray.compute_spray_statistics()
    print(f"Sauter mean diameter: {stats['sauter_mean_diameter']*1e6:.2f} μm")
