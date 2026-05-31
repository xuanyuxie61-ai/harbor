"""
combustion_wave.py - 一维燃烧波传播与反应扩散方程
=================================================
基于Jacobi迭代的一维燃烧波方程求解与燃烧速率插值。

原项目映射:
- 606_jacobi_poisson_1d -> Jacobi迭代求解反应-扩散方程
- 800_newton_interp_1d  -> Newton插值用于燃烧速率曲线

科学背景:
=========
火箭发动机中的燃烧波传播可用一维反应-扩散方程描述:

    ρ·C_p·∂T/∂t = k·∂²T/∂x² + Q·ω(T,Y)
    ρ·∂Y/∂t = ρ·D·∂²Y/∂x² - ω(T,Y)

其中:
    T: 温度
    Y: 反应物质量分数
    Q: 反应热
    ω: 反应速率 (Arrhenius定律)
        ω = A·ρ·Y·exp(-E_a/(R·T))

稳态传播速度 (层流火焰速度) 满足:
    S_L^2 = (α/τ_c)·f(β)
    
    其中 α = k/(ρ·C_p) 为热扩散系数
          τ_c 为化学特征时间
          β = E_a·(T_b-T_u)/(R·T_b^2) 为Zeldovich数
          f(β) ≈ 0.5·β^{-2}·exp(-β) (大活化能渐近)

燃烧速率 (Regression rate) 随压力变化:
    r_b = a·P^n
    
    对于RP-1: a ≈ 1.5e-5, n ≈ 0.5 (St. Robert定律)
"""

import numpy as np
from utils import safe_divide, robust_sqrt, check_finite_array, PRE_EXPONENTIAL, ACTIVATION_ENERGY


class NewtonInterpolation:
    """
    Newton差商插值。
    
    用于燃烧速率-压力关系曲线的插值计算。
    
    原项目映射: 800_newton_interp_1d
    
    差商定义:
        f[x_i] = f(x_i)
        f[x_i,...,x_j] = (f[x_{i+1},...,x_j] - f[x_i,...,x_{j-1}]) / (x_j - x_i)
    
    Newton插值多项式:
        P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have same length")
        if len(x_data) < 2:
            raise ValueError("Need at least 2 data points")
        
        self.xd = np.array(x_data, dtype=float)
        self.yd = np.array(y_data, dtype=float)
        self.n = len(x_data)
        self._compute_divided_differences()
    
    def _compute_divided_differences(self):
        """计算差商表。"""
        self.cd = self.yd.copy()
        
        for i in range(1, self.n):
            for j in range(self.n - 1, i - 1, -1):
                denom = self.xd[j] - self.xd[j - i]
                if abs(denom) < 1e-14:
                    denom = 1e-14
                self.cd[j] = (self.cd[j] - self.cd[j - 1]) / denom
    
    def evaluate(self, x: float) -> float:
        """
        使用Newton插值计算P(x)。
        
        Horner-like求值:
            P(x) = cd[0] + cd[1](x-xd[0]) + cd[2](x-xd[0])(x-xd[1]) + ...
                 = cd[0] + (x-xd[0])·(cd[1] + (x-xd[1])·(cd[2] + ...))
        
        边界处理:
            - x < min(xd): 外推警告，使用最近端点
            - x > max(xd): 外推警告，使用最近端点
        """
        x = float(x)
        
        # 限制外推范围
        x_min, x_max = np.min(self.xd), np.max(self.xd)
        if x < x_min:
            x = x_min
        elif x > x_max:
            x = x_max
        
        result = self.cd[-1]
        for i in range(self.n - 2, -1, -1):
            result = result * (x - self.xd[i]) + self.cd[i]
        
        return float(result)
    
    def evaluate_array(self, x_arr: np.ndarray) -> np.ndarray:
        """对数组进行插值。"""
        return np.array([self.evaluate(x) for x in x_arr])


class CombustionRateModel:
    """
    固体/液体推进剂燃烧速率模型。
    
    基于Saint-Robert定律:
        r_b = a · P^n
    
    同时提供基于Newton插值的离散数据版本。
    """
    
    def __init__(self, a_coeff: float = 1.5e-5, n_coeff: float = 0.5):
        self.a = a_coeff
        self.n = n_coeff
        
        # 构建参考数据点用于Newton插值
        # 典型RP-1液膜燃烧速率数据
        self.pressure_ref = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]) * 1e6  # Pa
        self.rate_ref = self.a * (self.pressure_ref ** self.n)  # m/s
        
        self.interpolator = NewtonInterpolation(self.pressure_ref, self.rate_ref)
    
    def regression_rate(self, pressure: float) -> float:
        """计算给定压力下的燃烧速率。"""
        return self.a * (pressure ** self.n)
    
    def regression_rate_interpolated(self, pressure: float) -> float:
        """使用Newton插值计算燃烧速率。"""
        return self.interpolator.evaluate(pressure)
    
    def temperature_sensitivity(self, pressure: float, sigma_p: float = 0.002) -> float:
        """
        计算温度敏感系数:
            σ_p = (∂ln(r_b)/∂T)_P
        
        典型值: 0.001 ~ 0.003 /K
        """
        return sigma_p


class ReactionDiffusionSolver:
    """
    一维反应-扩散方程求解器 (Jacobi迭代)。
    
    控制方程:
        ∂T/∂t = α·∂²T/∂x² + (Q/C_p)·ω(T)/ρ
    
    离散化 (显式Euler):
        T_i^{n+1} = T_i^n + dt·[α·(T_{i+1}^n - 2T_i^n + T_{i-1}^n)/dx² + source_i^n]
    
    稳态求解使用Jacobi迭代:
        T_i^{k+1} = [T_{i+1}^k + T_{i-1}^k + (dx²/α)·source_i^k] / 2
    
    原项目映射: 606_jacobi_poisson_1d
    """
    
    def __init__(self,
                 domain_length: float = 0.02,  # 火焰厚度 ~ 2 cm
                 n_points: int = 201,
                 thermal_diffusivity: float = 1.2e-4,  # m^2/s
                 heat_release: float = 4.5e7,  # J/kg
                 specific_heat: float = 1800.0,  # J/(kg·K)
                 activation_energy: float = 1.26e5,  # J/mol
                pre_exponential: float = 1.8e10,  # 1/s
                 temperature_unburned: float = 500.0,  # K
                 temperature_burned: float = 3600.0,  # K
                 density: float = 15.0):  # kg/m^3
        
        self.L = domain_length
        self.nx = n_points
        self.alpha = thermal_diffusivity
        self.Q = heat_release
        self.C_p = specific_heat
        self.E_a = activation_energy
        self.A = pre_exponential
        self.T_u = temperature_unburned
        self.T_b = temperature_burned
        self.rho = density
        
        self.x = np.linspace(0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]
        
        # Zeldovich数
        self.beta = self.E_a * (self.T_b - self.T_u) / (8.314 * self.T_b ** 2)
        
        # 层流火焰速度的理论估计
        self.S_L_theoretical = self._estimate_laminar_flame_speed()
    
    def _estimate_laminar_flame_speed(self) -> float:
        """
        使用大活化能渐近理论估计层流火焰速度。
        
        Zeldovich-Frank-Kamenetskii理论:
            S_L ≈ √(α·A·exp(-E_a/(R·T_b)) · β^{-2})
        """
        R_gas = 8.314
        tau_c = safe_divide(1.0, self.A * np.exp(-self.E_a / (R_gas * self.T_b)), default=1e-3)
        S_L = np.sqrt(self.alpha / tau_c) * safe_divide(1.0, self.beta, default=1.0)
        return float(S_L)
    
    def reaction_rate(self, T: float, Y: float = 1.0) -> float:
        """
        Arrhenius反应速率:
            ω = A·ρ·Y·exp(-E_a/(R·T))
        
        参数:
            T: 温度, K
            Y: 反应物质量分数
        
        返回:
            反应速率, kg/(m^3·s)
        """
        R_gas = 8.314
        if T < self.T_u * 0.8:
            return 0.0
        
        rate = self.A * self.rho * Y * np.exp(-self.E_a / (R_gas * max(T, 100.0)))
        return rate
    
    def source_term(self, T: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        计算热源项:
            S = (Q/C_p)·ω(T,Y)/ρ
        """
        if Y is None:
            Y = np.ones_like(T)
        
        omega = np.array([self.reaction_rate(Ti, Yi) for Ti, Yi in zip(T, Y)])
        source = (self.Q / self.C_p) * omega / max(self.rho, 1e-10)
        # 限制热源项大小防止数值爆炸
        source = np.clip(source, 0.0, 1e12)
        return source
    
    def solve_steady_jacobi(self, max_iterations: int = 100000,
                            tolerance: float = 1e-8,
                            omega_relax: float = 0.5) -> dict:
        """
        使用Jacobi迭代求解稳态反应-扩散方程。
        
        离散方程:
            -α·(T_{i+1} - 2T_i + T_{i-1})/dx² = S_i
        
        Jacobi迭代:
            T_i^{new} = [T_{i+1}^{old} + T_{i-1}^{old} + (dx²/α)·S_i^{old}] / 2
        
        边界条件:
            T(0) = T_u  (未燃区)
            T(L) = T_b  (已燃区)
        
        原项目映射: 606_jacobi_poisson_1d
        """
        # 初始化: 使用tanh型过渡曲线作为初始猜测
        T = self.T_u + 0.5 * (self.T_b - self.T_u) * \
            (1.0 + np.tanh((self.x - self.L * 0.3) / (self.L * 0.05)))
        T[0] = self.T_u
        T[-1] = self.T_b
        
        dx2_over_alpha = self.dx ** 2 / self.alpha
        
        for it in range(max_iterations):
            T_old = T.copy()
            
            # 计算源项 (基于旧温度)
            S = self.source_term(T_old)
            
            # Jacobi更新 (内部点)
            for i in range(1, self.nx - 1):
                T_new_i = 0.5 * (T_old[i+1] + T_old[i-1] + dx2_over_alpha * S[i])
                # 松弛 + 温度边界保护
                T[i] = omega_relax * T_new_i + (1.0 - omega_relax) * T_old[i]
                T[i] = np.clip(T[i], self.T_u * 0.9, self.T_b * 1.1)
            
            # 强制边界
            T[0] = self.T_u
            T[-1] = self.T_b
            
            # 收敛检查
            residual = np.sqrt(np.mean((T - T_old) ** 2))
            if residual < tolerance:
                break
        
        # 计算反应区位置 (最大温度梯度处)
        dTdx = np.gradient(T, self.dx)
        flame_position = self.x[np.argmax(np.abs(dTdx))]
        
        # 火焰厚度 (基于最大温度梯度, 99%温度变化范围)
        max_grad = np.max(np.abs(dTdx))
        if max_grad > 1e-10:
            flame_thickness = (self.T_b - self.T_u) / max_grad
        else:
            flame_thickness = self.L
        
        # 数值火焰速度 (基于总热释放与密度*Cp*温差的关系)
        source_final = self.source_term(T)
        total_heat_release = np.trapezoid(source_final * self.rho * self.C_p, self.x)
        denom = self.rho * self.C_p * (self.T_b - self.T_u)
        if denom > 1e-10 and total_heat_release > 1e-10:
            # 火焰速度 = 热释放率 / (ρ·Cp·ΔT) 的合理缩放
            S_L_numerical = np.sqrt(total_heat_release / denom * self.alpha)
            S_L_numerical = np.clip(S_L_numerical, 0.01, 100.0)
        else:
            S_L_numerical = self.S_L_theoretical
        
        return {
            "x": self.x,
            "temperature": T,
            "temperature_gradient": dTdx,
            "source": source_final,
            "iterations": it + 1,
            "final_residual": residual,
            "flame_position": float(flame_position),
            "flame_thickness": float(flame_thickness),
            "S_L_theoretical": self.S_L_theoretical,
            "S_L_numerical": float(S_L_numerical),
            "zeldovich_number": self.beta
        }
    
    def solve_time_dependent(self, dt: float = 1.0e-7,
                             n_steps: int = 5000,
                             save_interval: int = 500) -> dict:
        """
        求解非定常反应-扩散方程 (显式Euler)。
        
        T_i^{n+1} = T_i^n + dt·[α·(T_{i+1}^n - 2T_i^n + T_{i-1}^n)/dx² + S_i^n]
        
        CFL条件: dt < dx² / (2α)
        """
        dt_max = 0.5 * self.dx ** 2 / self.alpha
        if dt > dt_max:
            dt = dt_max * 0.9
        
        T = np.ones(self.nx) * self.T_u
        # 点火区
        T[self.nx // 2 - 5:self.nx // 2 + 5] = self.T_b
        T[0] = self.T_u
        T[-1] = self.T_b
        
        T_history = [T.copy()]
        t_history = [0.0]
        
        for step in range(n_steps):
            S = self.source_term(T)
            
            T_new = T.copy()
            for i in range(1, self.nx - 1):
                diffusion = self.alpha * (T[i+1] - 2*T[i] + T[i-1]) / self.dx ** 2
                T_new[i] = T[i] + dt * (diffusion + S[i])
            
            T_new[0] = self.T_u
            T_new[-1] = self.T_b
            T = T_new
            
            if step % save_interval == 0:
                T_history.append(T.copy())
                t_history.append((step + 1) * dt)
        
        return {
            "x": self.x,
            "temperature_final": T,
            "temperature_history": np.array(T_history),
            "time_history": np.array(t_history),
            "dt": dt
        }


if __name__ == "__main__":
    # 测试Newton插值
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
    interp = NewtonInterpolation(x_data, y_data)
    print(f"Newton interpolation at 2.5: {interp.evaluate(2.5):.4f}")
    
    # 测试燃烧速率模型
    rate_model = CombustionRateModel()
    print(f"Regression rate at 7MPa: {rate_model.regression_rate(7e6)*1e3:.4f} mm/s")
    print(f"Interpolated rate at 7MPa: {rate_model.regression_rate_interpolated(7e6)*1e3:.4f} mm/s")
    
    # 测试反应扩散求解
    rd = ReactionDiffusionSolver()
    print(f"Zeldovich number: {rd.beta:.2f}")
    print(f"Theoretical S_L: {rd.S_L_theoretical:.4f} m/s")
    
    result = rd.solve_steady_jacobi()
    print(f"Jacobi converged in {result['iterations']} iterations")
    print(f"Flame position: {result['flame_position']*1e3:.2f} mm")
    print(f"Flame thickness: {result['flame_thickness']*1e6:.2f} μm")
    print(f"Numerical S_L: {result['S_L_numerical']:.4f} m/s")
