"""
reaction_diffusion.py
表面催化反应-扩散方程求解器

整合原项目:
  - 283_diffusion_pde: 扩散 PDE 时间演化
  - 358_fd1d_bvp: 一维有限差分边值问题

科学背景:
  在表面催化反应的宏观尺度上，吸附物种的浓度分布
  服从反应-扩散方程 (Reaction-Diffusion Equation):
  
    ∂c/∂t = D * ∇²c + R(c)
  
  其中:
    c(x,t): 表面吸附物种的覆盖度 (或浓度)
    D: 表面扩散系数 (m²/s)
    R(c): 反应源项 (包含吸附、脱附和表面反应)
  
  对于 Langmuir-Hinshelwood 机理 (CO + O → CO2):
    R_CO = k_ads^CO * P_CO * (1 - θ_CO - θ_O) - k_des^CO * θ_CO
           - k_rxn * θ_CO * θ_O
    R_O  = 2 * k_ads^O2 * P_O2 * (1 - θ_CO - θ_O)² - k_des^O * θ_O
           - k_rxn * θ_CO * θ_O
  
  在稳态下 (∂c/∂t = 0)，方程退化为边值问题:
    D * d²θ/dx² + R(θ) = 0
"""

import numpy as np
from typing import Callable, Tuple, Optional


class ReactionDiffusion1D:
    """
    一维反应-扩散方程求解器
    
    方程形式:
      -d/dx (D(x) dc/dx) + k_r(x) * c(x) = f(x)
    
    边界条件:
      c(x_0) = c_L, c(x_N) = c_R  (Dirichlet)
      或
      dc/dx|_{x_0} = 0, dc/dx|_{x_N} = 0  (Neumann)
    """

    def __init__(self, x_grid: np.ndarray, diffusivity: float,
                 bc_type: str = "dirichlet"):
        self.x = np.asarray(x_grid, dtype=float)
        self.n = len(self.x)
        self.D = diffusivity
        self.bc_type = bc_type

    def solve_steady_state(self, reaction_func: Callable[[np.ndarray], np.ndarray],
                           bc_values: Tuple[float, float]) -> np.ndarray:
        """
        求解稳态反应-扩散方程
        
        离散化 (非均匀网格有限差分):
          -D * (c_{i-1} - 2c_i + c_{i+1}) / (dx_L * dx_R)
          + R(c_i) = 0
        
        使用 Newton-Raphson 迭代处理非线性反应项
        """
        if self.bc_type == "dirichlet":
            return self._solve_steady_dirichlet(reaction_func, bc_values)
        elif self.bc_type == "neumann":
            return self._solve_steady_neumann(reaction_func, bc_values)
        else:
            raise ValueError(f"不支持的边界条件类型: {self.bc_type}")

    def _solve_steady_dirichlet(self, reaction_func, bc_values):
        c = np.linspace(bc_values[0], bc_values[1], self.n)
        tol = 1e-10
        max_iter = 100

        for _ in range(max_iter):
            # 构建 Jacobian
            A = np.zeros((self.n, self.n))
            rhs = np.zeros(self.n)

            A[0, 0] = 1.0
            rhs[0] = bc_values[0]
            A[self.n - 1, self.n - 1] = 1.0
            rhs[self.n - 1] = bc_values[1]

            for i in range(1, self.n - 1):
                dx_l = self.x[i] - self.x[i - 1]
                dx_r = self.x[i + 1] - self.x[i]
                dx = self.x[i + 1] - self.x[i - 1]

                # 扩散算子离散化
                A[i, i - 1] = -2.0 * self.D / (dx_l * dx)
                A[i, i] = 2.0 * self.D / (dx_l * dx_r)
                A[i, i + 1] = -2.0 * self.D / (dx_r * dx)

                # 反应项 (线性化)
                r_val = reaction_func(c[i])
                # 简化为: R(c) ≈ -k * c + s
                # 这里使用当前值作为源项
                rhs[i] = -r_val
                # 对反应项的导数近似添加到 Jacobian 对角线
                dc = 1e-8
                dr_dc = (reaction_func(c[i] + dc) - r_val) / dc
                A[i, i] += dr_dc

            delta_c = np.linalg.solve(A, rhs)
            c = c + delta_c
            if np.linalg.norm(delta_c) < tol:
                break

        return c

    def _solve_steady_neumann(self, reaction_func, bc_values):
        c = np.ones(self.n) * 0.5
        tol = 1e-10
        max_iter = 100

        for _ in range(max_iter):
            A = np.zeros((self.n, self.n))
            rhs = np.zeros(self.n)

            # Neumann 边界: zero flux
            A[0, 0] = 1.0
            A[0, 1] = -1.0
            rhs[0] = 0.0
            A[self.n - 1, self.n - 1] = 1.0
            A[self.n - 1, self.n - 2] = -1.0
            rhs[self.n - 1] = 0.0

            for i in range(1, self.n - 1):
                dx_l = self.x[i] - self.x[i - 1]
                dx_r = self.x[i + 1] - self.x[i]
                dx = self.x[i + 1] - self.x[i - 1]

                A[i, i - 1] = -2.0 * self.D / (dx_l * dx)
                A[i, i] = 2.0 * self.D / (dx_l * dx_r)
                A[i, i + 1] = -2.0 * self.D / (dx_r * dx)

                r_val = reaction_func(c[i])
                rhs[i] = -r_val
                dc = 1e-8
                dr_dc = (reaction_func(c[i] + dc) - r_val) / dc
                A[i, i] += dr_dc

            delta_c = np.linalg.solve(A, rhs)
            c = c + delta_c
            if np.linalg.norm(delta_c) < tol:
                break

        return c

    def solve_time_dependent(self, c0: np.ndarray, t_end: float,
                             reaction_func: Callable[[np.ndarray], np.ndarray],
                             n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        时间依赖的反应-扩散方程求解
        
        采用隐式 Euler 时间积分:
          (c^{n+1} - c^n) / dt = D * L * c^{n+1} + R(c^{n+1})
        
        简化为显式 Euler (稳定性要求 dt < dx² / (2D)):
          c^{n+1} = c^n + dt * (D * L * c^n + R(c^n))
        """
        c = np.asarray(c0, dtype=float).copy()
        dt = t_end / n_steps
        dx_min = np.min(np.diff(self.x))
        dt_max = dx_min ** 2 / (2.0 * self.D)
        if dt > dt_max:
            # 自动调整步数以满足稳定性
            n_steps = int(np.ceil(t_end / (0.9 * dt_max)))
            dt = t_end / n_steps

        trajectory = [c.copy()]
        for _ in range(n_steps):
            # 扩散项 (中心差分)
            laplacian = np.zeros(self.n)
            for i in range(1, self.n - 1):
                dx_l = self.x[i] - self.x[i - 1]
                dx_r = self.x[i + 1] - self.x[i]
                laplacian[i] = 2.0 * (
                    (c[i + 1] - c[i]) / dx_r - (c[i] - c[i - 1]) / dx_l
                ) / (dx_l + dx_r)

            # 边界条件处理
            if self.bc_type == "neumann":
                laplacian[0] = laplacian[1]
                laplacian[self.n - 1] = laplacian[self.n - 2]
            else:
                laplacian[0] = 0.0
                laplacian[self.n - 1] = 0.0

            reaction = reaction_func(c)
            c = c + dt * (self.D * laplacian + reaction)

            # 浓度物理约束
            c = np.clip(c, 0.0, 1.0)
            trajectory.append(c.copy())

        return np.array(trajectory), np.linspace(0, t_end, n_steps + 1)


class LangmuirHinshelwoodKinetics:
    """
    Langmuir-Hinshelwood 表面反应动力学模型
    
    反应机理:
      CO(g) + *  ⇌ CO*
      O2(g) + 2* ⇌ 2O*
      CO* + O*  → CO2(g) + 2*
    
    覆盖度演化方程 (mean-field):
      dθ_CO/dt = k_ads^CO * P_CO * (1 - θ_CO - θ_O) - k_des^CO * θ_CO
                 - k_rxn * θ_CO * θ_O
      dθ_O/dt  = 2 * k_ads^O2 * P_O2 * (1 - θ_CO - θ_O)² - k_des^O * θ_O
                 - k_rxn * θ_CO * θ_O
    
    速率常数 (Arrhenius):
      k = A * exp(-E_a / (k_B T))
    """

    def __init__(self, temperature_k: float = 500.0,
                 p_co_pa: float = 1.0e3,
                 p_o2_pa: float = 5.0e2):
        self.T = temperature_k
        self.p_co = p_co_pa
        self.p_o2 = p_o2_pa
        from utils import kb_t_ev
        self.kb_t = kb_t_ev(temperature_k)

        # 速率参数 (典型值)
        self.a_ads_co = 1.0e6   # Pa^-1 s^-1
        self.ea_ads_co = 0.0    # eV (非活化吸附)
        self.a_des_co = 1.0e13  # s^-1
        self.ea_des_co = 1.3    # eV

        self.a_ads_o2 = 5.0e5   # Pa^-1 s^-1
        self.ea_ads_o2 = 0.0
        self.a_des_o = 1.0e13   # s^-1
        self.ea_des_o = 2.0     # eV

        self.a_rxn = 1.0e13     # s^-1
        self.ea_rxn = 0.8       # eV (Langmuir-Hinshelwood 势垒)

    def _rate_constants(self) -> dict:
        """计算所有速率常数"""
        return {
            'k_ads_co': self.a_ads_co * self.p_co * np.exp(-self.ea_ads_co / self.kb_t),
            'k_des_co': self.a_des_co * np.exp(-self.ea_des_co / self.kb_t),
            'k_ads_o2': self.a_ads_o2 * self.p_o2 * np.exp(-self.ea_ads_o2 / self.kb_t),
            'k_des_o': self.a_des_o * np.exp(-self.ea_des_o / self.kb_t),
            'k_rxn': self.a_rxn * np.exp(-self.ea_rxn / self.kb_t),
        }

    def rhs(self, theta: np.ndarray) -> np.ndarray:
        """
        计算覆盖度时间导数
        
        theta[0] = θ_CO, theta[1] = θ_O
        """
        theta = np.clip(theta, 0.0, 1.0)
        th_co = theta[0]
        th_o = theta[1]
        th_free = max(0.0, 1.0 - th_co - th_o)

        k = self._rate_constants()

        dth_co_dt = (k['k_ads_co'] * th_free
                     - k['k_des_co'] * th_co
                     - k['k_rxn'] * th_co * th_o)

        dth_o_dt = (2.0 * k['k_ads_o2'] * th_free ** 2
                    - k['k_des_o'] * th_o
                    - k['k_rxn'] * th_co * th_o)

        return np.array([dth_co_dt, dth_o_dt])

    def integrate_ode(self, theta0: np.ndarray, t_end: float,
                      n_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Runge-Kutta 4 阶方法积分覆盖度演化
        """
        theta = np.asarray(theta0, dtype=float).copy()
        dt = t_end / n_steps
        trajectory = [theta.copy()]
        times = [0.0]

        for _ in range(n_steps):
            k1 = self.rhs(theta)
            k2 = self.rhs(theta + 0.5 * dt * k1)
            k3 = self.rhs(theta + 0.5 * dt * k2)
            k4 = self.rhs(theta + dt * k3)
            theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            theta = np.clip(theta, 0.0, 1.0)
            trajectory.append(theta.copy())
            times.append(times[-1] + dt)

        return np.array(trajectory), np.array(times)

    def steady_state_coverage(self) -> np.ndarray:
        """
        计算稳态覆盖率
        
        求解非线性方程组:
          dθ_CO/dt = 0
          dθ_O/dt = 0
        
        使用伪时间演化配合自适应小步长显式 Euler，
        确保大速率常数下的数值稳定性。
        """
        theta = np.array([0.3, 0.3])
        dt = 1e-12  # 初始小步长 (s)
        for step in range(5000000):
            dth = self.rhs(theta)
            # 自适应步长: 确保单步变化不超过 0.01
            max_dth = np.max(np.abs(dth))
            if max_dth > 1e-300:
                dt_safe = min(dt, 0.01 / max_dth)
            else:
                dt_safe = dt
            theta_new = theta + dt_safe * dth
            theta_new = np.clip(theta_new, 0.0, 1.0)
            if np.linalg.norm(dth) < 1e-14:
                break
            theta = theta_new
            # 逐渐增大步长以加速收敛
            if step % 1000 == 0 and dt < 1e-6:
                dt *= 2.0
        return theta
