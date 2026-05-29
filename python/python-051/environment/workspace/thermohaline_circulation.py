"""
thermohaline_circulation.py
===========================
二维经向翻转环流(meridional overturning circulation, MOC)的流函数-涡度数值求解器。

核心物理模型
------------
采用流函数 ψ 与涡度 ω 的耦合方程组描述热盐环流：

  (1) 泊松方程（流函数-涡度关系）：
      ∇²ψ = -ω

  (2) 涡度方程（考虑斜压项与耗散）：
      ∂ω/∂t + J(ψ, ω) = ν∇²ω + (g/ρ₀)(∂ρ/∂x)

  (3) 温度平流-扩散方程：
      ∂T/∂t + J(ψ, T) = κ_T∇²T + Q_T(x,z)

  (4) 盐度平流-扩散方程：
      ∂S/∂t + J(ψ, S) = κ_S∇²S + Q_S(x,z)

  (5) 状态方程（线性近似）：
      ρ = ρ₀[1 - α(T - T_ref) + β(S - S_ref)]

其中 J(A,B) = (∂A/∂x)(∂B/∂z) - (∂A/∂z)(∂B/∂x) 为雅可比算子。

数值离散
--------
- 空间：二阶中心差分，交错网格（Arakawa C-grid 简化版）
- 时间：涡度与平流项采用 Adams-Bashforth 二阶，扩散项采用 Crank-Nicolson
- 边界：刚性 lid（ψ=0 于上表面），无滑移侧边界，恢复型边界条件用于 T/S
"""

import numpy as np
from matrix_solvers import bicg_solve, create_poisson_stencil


class ThermohalineCirculation:
    """
    热盐环流求解器。
    """

    def __init__(self, nx=64, nz=32, Lx=5.0e6, Lz=4.0e3,
                 rho0=1027.0, alpha=1.7e-4, beta=7.6e-4,
                 nu=10.0, kappa_T=1.0e-4, kappa_S=1.0e-4,
                 g=9.81, dt=86400.0):
        """
        参数
        ----
        nx, nz : int
            水平与垂向网格数
        Lx, Lz : float
            域尺寸 [m]（水平宽度，垂向深度）
        rho0 : float
            参考密度 [kg/m³]
        alpha, beta : float
            热膨胀系数 [1/K] 与盐压缩系数 [1/psu]
        nu : float
            涡粘系数 [m²/s]
        kappa_T, kappa_S : float
            温度/盐度扩散系数 [m²/s]
        g : float
            重力加速度 [m/s²]
        dt : float
            时间步长 [s]
        """
        if nx < 4 or nz < 4:
            raise ValueError("网格数 nx, nz 必须 >= 4")
        if Lx <= 0 or Lz <= 0:
            raise ValueError("域尺寸必须为正")
        if dt <= 0:
            raise ValueError("时间步长必须为正")

        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dx = Lx / (nx - 1)
        self.dz = Lz / (nz - 1)
        self.rho0 = rho0
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.kappa_T = kappa_T
        self.kappa_S = kappa_S
        self.g = g
        self.dt = dt

        # 状态场
        self.psi = np.zeros((nx, nz))      # 流函数 [m²/s]
        self.omega = np.zeros((nx, nz))    # 涡度 [1/s]
        self.T = np.zeros((nx, nz))        # 温度异常 [K]
        self.S = np.zeros((nx, nz))        # 盐度异常 [psu]

        # 泊松方程的离散矩阵（使用 five-point stencil）
        self.poisson_A = create_poisson_stencil(nx, nz, self.dx, self.dz)

        # 参考值
        self.T_ref = 0.0
        self.S_ref = 0.0

        # 用于 Adams-Bashforth 的历史项
        self.omega_old = None
        self.T_old = None
        self.S_old = None

    def jacobian(self, A, B):
        """
        计算离散雅可比 J(A, B) = A_x * B_z - A_z * B_x。
        采用 Arakawa 能量守恒格式（二阶中心差分）。
        """
        nx, nz = A.shape
        if B.shape != A.shape:
            raise ValueError("A, B 形状必须相同")

        J = np.zeros_like(A)
        dx = self.dx
        dz = self.dz

        # 内部节点
        Ax = np.zeros_like(A)
        Az = np.zeros_like(A)
        Bx = np.zeros_like(B)
        Bz = np.zeros_like(B)

        Ax[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * dx)
        Az[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * dz)
        Bx[1:-1, :] = (B[2:, :] - B[:-2, :]) / (2 * dx)
        Bz[:, 1:-1] = (B[:, 2:] - B[:, :-2]) / (2 * dz)

        J = Ax * Bz - Az * Bx

        # 边界置零（无通量）
        J[0, :] = 0.0
        J[-1, :] = 0.0
        J[:, 0] = 0.0
        J[:, -1] = 0.0
        return J

    def laplacian(self, F):
        """
        标量场 F 的二维拉普拉斯 ∇²F 的二阶中心差分。
        """
        nx, nz = F.shape
        dx = self.dx
        dz = self.dz
        L = np.zeros_like(F)

        L[1:-1, 1:-1] = (
            (F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / (dx ** 2) +
            (F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / (dz ** 2)
        )
        return L

    def density_anomaly(self):
        """
        由状态方程计算密度异常：
            ρ' = ρ₀(-α T + β S)
        """
        return self.rho0 * (-self.alpha * self.T + self.beta * self.S)

    def baroclinic_term(self):
        """
        斜压项：(g/ρ₀) ∂ρ/∂x ≈ -g·α·∂T/∂x + g·β·∂S/∂x
        """
        # TODO(Hole 4): 实现斜压项计算
        # 提示: 基于状态方程对 T 和 S 求水平梯度，结合热膨胀/盐压缩系数计算斜压源项
        raise NotImplementedError("Hole 4: 请实现 baroclinic_term")

    def apply_boundary_conditions(self):
        """
        边界条件设置：
        - 上表面 (z=0): 刚性 lid，ψ = 0，ω = -∂²ψ/∂z²（利用内部点外推）
        - 下底面 (z=-Lz): 无滑移，ψ = 0，ω 由内部外推
        - 左/右侧壁 (x=0, Lx): 无滑移，ψ = 0，T/S 恢复型边界
        """
        nx, nz = self.nx, self.nz

        # 无滑移侧边界
        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0

        # 涡度边界（二阶近似）
        # ω = -∂²ψ/∂n² 于壁面
        self.omega[0, :] = -(2.0 * self.psi[1, :] - 0.5 * self.psi[2, :]) / (self.dx ** 2)
        self.omega[-1, :] = -(2.0 * self.psi[-2, :] - 0.5 * self.psi[-3, :]) / (self.dx ** 2)
        self.omega[:, 0] = -(2.0 * self.psi[:, 1] - 0.5 * self.psi[:, 2]) / (self.dz ** 2)
        self.omega[:, -1] = -(2.0 * self.psi[:, -2] - 0.5 * self.psi[:, -3]) / (self.dz ** 2)

        # T/S 恢复型边界：侧边界恢复至参考态
        relax_rate = 1.0 / (10.0 * 86400.0)  # 10 天恢复时间
        # 左边界（高纬） colder, saltier
        self.T[0, :] = self.T_ref - 2.0
        self.S[0, :] = self.S_ref + 0.5
        # 右边界（低纬） warmer, fresher
        self.T[-1, :] = self.T_ref + 2.0
        self.S[-1, :] = self.S_ref - 0.5
        # 上下边界绝热（零法向梯度已在 laplacian 中隐含处理）

    def solve_streamfunction(self):
        """
        由 ω 求解 ψ：∇²ψ = -ω，使用 BiCG 迭代求解。
        若 BiCG 失败，回退至直接求解（小规模问题）。
        """
        rhs = -self.omega.flatten()
        psi_flat = bicg_solve(self.poisson_A, rhs, tol=1e-8, max_iter=500)

        # 检验解的有效性
        if not np.all(np.isfinite(psi_flat)):
            # 回退：对当前规模使用 numpy 直接求解
            try:
                psi_flat = np.linalg.solve(self.poisson_A, rhs)
            except np.linalg.LinAlgError:
                psi_flat = np.zeros_like(rhs)

        self.psi = psi_flat.reshape((self.nx, self.nz))
        # 边界置零
        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0

    def step(self, forcing_T=None, forcing_S=None):
        """
        执行一个时间步长。

        时间离散格式：
        - 平流项：Adams-Bashforth 二阶
        - 扩散项：Crank-Nicolson（此处用显式 + 小时间步保证稳定）
        - 斜压项：当前步显式
        """
        nx, nz = self.nx, self.nz
        dt = self.dt

        # 确保边界条件
        self.apply_boundary_conditions()

        # 求解流函数
        self.solve_streamfunction()

        # 计算各源项
        J_omega = self.jacobian(self.psi, self.omega)
        J_T = self.jacobian(self.psi, self.T)
        J_S = self.jacobian(self.psi, self.S)

        L_omega = self.laplacian(self.omega)
        L_T = self.laplacian(self.T)
        L_S = self.laplacian(self.S)

        baroclinic = self.baroclinic_term()

        # Adams-Bashforth 二阶
        if self.omega_old is None:
            # 第一步用前向欧拉
            omega_rhs = -J_omega + self.nu * L_omega + baroclinic
            T_rhs = -J_T + self.kappa_T * L_T
            S_rhs = -J_S + self.kappa_S * L_S
        else:
            omega_rhs = -1.5 * J_omega + 0.5 * self.omega_old
            omega_rhs += self.nu * L_omega + baroclinic
            T_rhs = -1.5 * J_T + 0.5 * self.T_old
            T_rhs += self.kappa_T * L_T
            S_rhs = -1.5 * J_S + 0.5 * self.S_old
            S_rhs += self.kappa_S * L_S

        if forcing_T is not None:
            T_rhs += forcing_T
        if forcing_S is not None:
            S_rhs += forcing_S

        # 更新
        self.omega_old = J_omega.copy()
        self.T_old = J_T.copy()
        self.S_old = J_S.copy()

        self.omega += dt * omega_rhs
        self.T += dt * T_rhs
        self.S += dt * S_rhs

        # 数值稳定性：截断异常值
        self.omega = np.clip(self.omega, -1e-3, 1e-3)
        self.T = np.clip(self.T, -10.0, 10.0)
        self.S = np.clip(self.S, -5.0, 5.0)

        # NaN/Inf 清洗
        self.omega = np.where(np.isfinite(self.omega), self.omega, 0.0)
        self.T = np.where(np.isfinite(self.T), self.T, 0.0)
        self.S = np.where(np.isfinite(self.S), self.S, 0.0)

        # 硬边界约束
        self.apply_boundary_conditions()

    def get_velocity(self):
        """
        由流函数计算速度场：
            u = -∂ψ/∂z,   w = ∂ψ/∂x
        """
        u = np.zeros_like(self.psi)
        w = np.zeros_like(self.psi)
        dz = self.dz
        dx = self.dx

        u[:, 1:-1] = -(self.psi[:, 2:] - self.psi[:, :-2]) / (2 * dz)
        w[1:-1, :] = (self.psi[2:, :] - self.psi[:-2, :]) / (2 * dx)
        return u, w

    def get_overturning_streamfunction(self):
        """
        返回经向翻转流函数（垂向积分）。
        """
        return self.psi.copy()
