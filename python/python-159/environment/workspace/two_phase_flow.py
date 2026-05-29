"""
two_phase_flow.py - 燃烧室内气液两相流动分析
=============================================
基于Stokes方程的液滴周围低速流动场计算与阻力分析。

原项目映射:
- 1172_stokes_2d_exact -> 2D Stokes方程精确解用于液滴周围流动场

科学背景:
=========
在液体火箭发动机燃烧室中，液滴相对气相的速度较低
(典型滑移速度 < 50 m/s, 液滴直径 < 100 μm)，
液滴周围的局部流动可用低雷诺数Stokes近似描述。

Stokes方程 (不可压缩, 低雷诺数):
    ∇p = μ·∇²u + f
    ∇·u = 0

对于绕球形液滴的轴对称流动，Stokes流解析解给出阻力:
    F_d = 6·π·μ·R·U_∞ · C_s

其中滑移系数 C_s 考虑液滴内部循环:
    C_s = (2/3 + μ_int/μ_ext) / (1 + μ_int/μ_ext)

对于蒸发液滴，还需考虑Basset历史和虚拟质量效应。
"""

import numpy as np
from utils import check_finite_array, safe_divide, robust_sqrt


class StokesDropletFlow:
    """
    基于Stokes方程的液滴周围流动场分析。
    
    使用Wang等人(2009)的精确解框架计算
    液滴周围的速度场、压力场和应力分布。
    """
    
    def __init__(self,
                 droplet_radius: float = 40.0e-6,
                 free_stream_velocity: float = 30.0,  # m/s
                 gas_viscosity: float = 8.5e-5,  # Pa·s
                 gas_density: float = 15.0,  # kg/m^3
                 liquid_viscosity: float = 1.0e-3,  # Pa·s (RP-1)
                 surface_tension: float = 0.02):  # N/m
        
        self.R_d = droplet_radius
        self.U_inf = free_stream_velocity
        self.mu_g = gas_viscosity
        self.rho_g = gas_density
        self.mu_l = liquid_viscosity
        self.sigma = surface_tension
        
        # 特征无量纲数
        self.Re = self.rho_g * self.U_inf * (2 * self.R_d) / self.mu_g
        self.Ca = self.mu_g * self.U_inf / self.sigma  # Capillary数
        self.We = self.rho_g * self.U_inf ** 2 * (2 * self.R_d) / self.sigma  # Weber数
    
    def stokes_drag_coefficient(self) -> float:
        """
        计算考虑内部循环的Stokes阻力系数。
        
        Hadamard-Rybczynski解:
            对于球形液滴/气泡，阻力为固体球的:
                C_s = (2/3 + λ) / (1 + λ)
            其中 λ = μ_int / μ_ext
        
        当 λ → ∞ (固体球), C_s → 1
        当 λ → 0 (气泡), C_s → 2/3
        
        对于RP-1液滴在燃气中: λ ≈ 10-20, C_s ≈ 0.9-0.95
        """
        lam = safe_divide(self.mu_l, self.mu_g, default=100.0)
        lam = np.clip(lam, 1e-6, 1e6)
        C_s = (2.0 / 3.0 + lam) / (1.0 + lam)
        return float(C_s)
    
    def stokes_drag_force(self) -> float:
        """
        计算液滴受到的Stokes阻力。
        
        F_d = 6·π·μ·R·U_∞·C_s
        """
        C_s = self.stokes_drag_coefficient()
        F_d = 6.0 * np.pi * self.mu_g * self.R_d * self.U_inf * C_s
        return float(F_d)
    
    def velocity_field_stokes(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        计算Stokes流速度场 (2D)。
        
        使用Wang-Stokes精确解#1的框架，
        对于绕球流动在局部2D截面上的近似。
        
        速度场 (Stokes流绕圆柱, Lamb解):
            u_r = U_∞·cos(θ)·(1 - (3/2)(R/r) + (1/2)(R/r)^3)·C_s
            u_θ = -U_∞·sin(θ)·(1 - (3/4)(R/r) - (1/4)(R/r)^3)·C_s
        
        参数:
            x, y: 空间坐标数组 (相对于液滴中心)
        
        返回:
            u, v: x和y方向速度分量
        """
        C_s = self.stokes_drag_coefficient()
        
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        
        # 避免在液滴内部计算
        r = np.maximum(r, self.R_d * 1.001)
        
        # 径向和切向速度 (球坐标, 轴对称)
        eta = self.R_d / r
        u_r = self.U_inf * np.cos(theta) * (1.0 - 1.5 * eta + 0.5 * eta ** 3) * C_s
        u_theta = -self.U_inf * np.sin(theta) * (1.0 - 0.75 * eta - 0.25 * eta ** 3) * C_s
        
        # 转换为笛卡尔分量
        u = u_r * np.cos(theta) - u_theta * np.sin(theta)
        v = u_r * np.sin(theta) + u_theta * np.cos(theta)
        
        return u, v
    
    def pressure_field_stokes(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算Stokes流压力场。
        
        p = p_∞ - (3/2)·μ·U_∞·R·cos(θ)/r^2 · C_s
        
        返回:
            压力相对于远场的偏差, Pa
        """
        C_s = self.stokes_drag_coefficient()
        r = np.sqrt(x ** 2 + y ** 2)
        r = np.maximum(r, self.R_d * 1.001)
        theta = np.arctan2(y, x)
        
        dp = -1.5 * self.mu_g * self.U_inf * self.R_d * np.cos(theta) / (r ** 2) * C_s
        return dp
    
    def shear_stress_distribution(self, n_points: int = 100) -> tuple:
        """
        计算液滴表面的剪切应力分布。
        
        表面剪切应力:
            τ_rθ|_{r=R} = (3/2)·μ·U_∞·sin(θ)/R · (1/(1+λ))
        
        返回:
            theta: 角度数组
            tau: 剪切应力数组
        """
        theta = np.linspace(0, np.pi, n_points)
        lam = safe_divide(self.mu_l, self.mu_g, default=100.0)
        lam = np.clip(lam, 1e-6, 1e6)
        
        tau = 1.5 * self.mu_g * self.U_inf * np.sin(theta) / self.R_d * (1.0 / (1.0 + lam))
        return theta, tau
    
    def compute_nusselt_number(self, Prandtl: float = 0.7) -> float:
        """
        计算液滴的Nusselt数 (传热)。
        
        对于蒸发液滴的Ranz-Marshall关联式:
            Nu = 2 + 0.6·Re^{1/2}·Pr^{1/3}
        
        对于低雷诺数Stokes流 (Re << 1):
            Nu ≈ 2 (纯导热极限)
        """
        Re_d = max(self.Re, 1e-10)
        Nu = 2.0 + 0.6 * (Re_d ** 0.5) * (Prandtl ** (1.0 / 3.0))
        return float(Nu)
    
    def compute_sherwood_number(self, Schmidt: float = 1.0) -> float:
        """
        计算Sherwood数 (传质)。
        
        Ranz-Marshall类关联式:
            Sh = 2 + 0.6·Re^{1/2}·Sc^{1/3}
        """
        Re_d = max(self.Re, 1e-10)
        Sh = 2.0 + 0.6 * (Re_d ** 0.5) * (Schmidt ** (1.0 / 3.0))
        return float(Sh)
    
    def basset_history_force(self, velocity_history: np.ndarray, dt: float) -> float:
        """
        计算Basset历史力 (记忆效应)。
        
        非定常Stokes流中，液滴受到的附加阻力包含历史积分项:
            F_B = 6·π·μ·R^2 · ∫_{-∞}^{t} (dU/dτ) / √(π·ν·(t-τ)) dτ
        
        其中 ν = μ/ρ 为运动粘度。
        
        参数:
            velocity_history: 速度历史数组
            dt: 时间步长
        
        返回:
            Basset力, N
        """
        nu = safe_divide(self.mu_g, self.rho_g, default=1e-5)
        
        n = len(velocity_history)
        if n < 2:
            return 0.0
        
        F_basset = 0.0
        for k in range(n - 1):
            dU_dt = (velocity_history[-1] - velocity_history[k]) / ((n - 1 - k) * dt)
            tau = (n - 1 - k) * dt
            if tau > 0:
                F_basset += dU_dt / np.sqrt(np.pi * nu * tau) * dt
        
        F_basset *= 6.0 * np.pi * self.mu_g * self.R_d ** 2
        return float(F_basset)


class TwoPhaseFlowSolver:
    """
    简化的燃烧室内两相流动求解器。
    
    耦合液滴运动与气相流动的1D近似模型。
    """
    
    def __init__(self,
                 chamber_geometry,
                 n_z: int = 100,
                 gas_velocity_inlet: float = 50.0,
                 gas_temperature_inlet: float = 500.0):
        
        self.geo = chamber_geometry
        self.n_z = n_z
        self.z = np.linspace(0, self.geo.L_c, n_z)
        self.dz = self.z[1] - self.z[0]
        
        self.u_g = np.ones(n_z) * gas_velocity_inlet
        self.T_g = np.ones(n_z) * gas_temperature_inlet
        self.rho_g = np.ones(n_z) * 15.0
        self.P_g = np.ones(n_z) * 7.0e6
        
        # 液滴相参数
        self.d_droplet = np.ones(n_z) * 80e-6  # 初始直径
        self.u_d = np.ones(n_z) * 30.0  # 液滴速度
        self.n_droplet = np.ones(n_z) * 1e8  # 数密度
    
    def solve_steady_1d(self, droplet_source: np.ndarray = None) -> dict:
        """
        求解稳态1D两相流动。
        
        气相控制方程:
            d(ρ_g·u_g·A)/dz = Σ m_dot_d  (质量源)
            d(ρ_g·u_g^2·A)/dz + A·dP/dz = F_d  (动量)
            d(ρ_g·u_g·A·h_g)/dz = Q_dot  (能量)
        
        液滴相:
            d(d^2)/dz = -K / u_d  (d^2-law沿流线)
            d(u_d)/dz = F_d / m_d  (动量)
        """
        if droplet_source is None:
            droplet_source = np.zeros(self.n_z)
            droplet_source[:10] = 1e8  # 在入口附近注入
        
        # 简化解: 逐步推进
        for i in range(1, self.n_z):
            A = self.geo.area_at_z(self.z[i])
            A_prev = self.geo.area_at_z(self.z[i-1])
            
            # 液滴蒸发
            stokes = StokesDropletFlow(
                droplet_radius=self.d_droplet[i-1] / 2.0,
                free_stream_velocity=self.u_g[i-1] - self.u_d[i-1]
            )
            K = stokes.compute_evaporation_rate() if hasattr(stokes, 'compute_evaporation_rate') else 1e-8
            # 使用简化蒸发模型
            k_g = 0.08
            B = 1.5
            K = 8.0 * k_g * np.log(1.0 + B) / (807.0 * 1800.0)
            
            # d^2-law
            d_sq_new = self.d_droplet[i-1] ** 2 - K * self.dz / max(self.u_d[i-1], 1.0)
            self.d_droplet[i] = np.sqrt(max(d_sq_new, 0.0))
            
            # 液滴动量
            F_d = stokes.stokes_drag_force()
            m_d = (np.pi / 6.0) * 807.0 * self.d_droplet[i-1] ** 3
            du_d = F_d / max(m_d, 1e-15) * self.dz / max(self.u_d[i-1], 1.0)
            self.u_d[i] = self.u_d[i-1] + du_d
            
            # 气相质量守恒 (简化的连续方程)
            mass_source = droplet_source[i] * (np.pi / 6.0) * 807.0 * \
                          (self.d_droplet[i-1] ** 3 - self.d_droplet[i] ** 3)
            self.rho_g[i] = self.rho_g[i-1] * self.u_g[i-1] * A_prev / \
                            (max(self.u_g[i-1], 1.0) * A) + mass_source / (A * self.dz)
            self.rho_g[i] = np.clip(self.rho_g[i], 1.0, 50.0)
            
            # 气相速度 (简化)
            self.u_g[i] = self.u_g[i-1] * (A_prev / A) * (self.rho_g[i-1] / self.rho_g[i])
            self.u_g[i] = np.clip(self.u_g[i], 1.0, 500.0)
        
        return {
            "z": self.z,
            "gas_velocity": self.u_g,
            "gas_density": self.rho_g,
            "droplet_diameter": self.d_droplet,
            "droplet_velocity": self.u_d,
            "evaporation_length": self._find_evaporation_length()
        }
    
    def _find_evaporation_length(self) -> float:
        """找到液滴完全蒸发的特征长度。"""
        evaporated = np.where(self.d_droplet < 1e-9)[0]
        if len(evaporated) > 0:
            return float(self.z[evaporated[0]])
        return float(self.z[-1])


if __name__ == "__main__":
    from geometry_model import CombustionChamberGeometry
    
    geo = CombustionChamberGeometry()
    flow = TwoPhaseFlowSolver(geo, n_z=200)
    result = flow.solve_steady_1d()
    
    print(f"Evaporation length: {result['evaporation_length']:.4f} m")
    print(f"Max gas velocity: {np.max(result['gas_velocity']):.2f} m/s")
    
    stokes = StokesDropletFlow()
    print(f"Stokes drag coefficient: {stokes.stokes_drag_coefficient():.4f}")
    print(f"Drag force: {stokes.stokes_drag_force():.6e} N")
    print(f"Nusselt number: {stokes.compute_nusselt_number():.3f}")
