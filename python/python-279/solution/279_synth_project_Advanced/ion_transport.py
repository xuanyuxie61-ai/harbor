"""
ion_transport.py
================
修正 Poisson-Nernst-Planck 离子输运求解器。

科学背景:
  固态电解质中的离子传输由耦合的 PNP 方程组描述:

  (1) Poisson 方程 (电势):
      -eps * d^2(phi)/dx^2 = e * (c_Li - c_fixed + c_anion)
      其中 eps = eps_0 * eps_r 为介电常数, c_fixed 为固定电荷密度

  (2) Nernst-Planck 方程 (Li+ 浓度):
      dc/dt = D * d^2c/dx^2 + (D*e/(kB*T)) * d/dx(c * dphi/dx)
      第一项: Fick 扩散; 第二项: 迁移 (drift)

  (3) 修正项 (steric effect, Bazant 2011):
      考虑离子有限体积, 引入:
        mu_ex = kB*T * log(1 - sum_i c_i/c_max)
      额外漂移项:
        -D * c * (1 - c/c_max)^{-1} * dc/dx

  边界条件:
    - 电极界面: Butler-Volmer 动力学 或 Dirichlet
    - 电解质/电解质界面: 通量连续

  Butler-Volmer 动力学:
    j = j_0 * [exp(alpha_a*e*eta/(kB*T)) - exp(-alpha_c*e*eta/(kB*T))]
    其中 eta = phi_s - phi - U_eq 为过电位, j_0 为交换电流密度

无量纲化:
  x* = x/L, t* = t*D/L^2, phi* = e*phi/(kB*T), c* = c/c_ref

  无量纲 PNP:
    -lambda^2 * d^2(phi*)/dx*^2 = c_Li* - c_fixed*
    dc_Li*/dt* = d^2c_Li*/dx*^2 + d/dx(c_Li* * dphi*/dx*)

  其中 lambda = lambda_D / L 为无量纲 Debye 长度
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from material_constants import (
    KB, ELEMENTARY_CHARGE, EPSILON_0, FARADAY,
    LLZO_RELATIVE_PERMITTIVITY, LLZO_DIFFUSION_COEFF, SMALL_NUMBER, CONVERGENCE_TOL,
)
from high_order_fd import central_fd_1d_uniform, nonuniform_fd_matrix


class ModifiedPNPSolver:
    """
    一维修正 PNP 方程求解器。

    离散化: 高阶有限差分 + 隐式时间积分 (Backward Euler)
    非线性: Picard 迭代 (固定点迭代) 或 Newton-Raphson
    """
    def __init__(self, N, L, T_kelvin=300.0, c_ref=1000.0,
                 fd_order=4, time_scheme='backward_euler'):
        """
        参数:
            N: 内部网格点数
            L: 电解质厚度 [m]
            T_kelvin: 温度 [K]
            c_ref: 参考浓度 [mol/m^3]
            fd_order: 有限差分精度阶数
            time_scheme: 'backward_euler' 或 'crank_nicolson'
        """
        self.N = N
        self.L = L
        self.T = T_kelvin
        self.c_ref = c_ref
        self.h = L / (N + 1)
        self.fd_order = fd_order
        self.time_scheme = time_scheme

        # 物理参数
        self.eps = EPSILON_0 * LLZO_RELATIVE_PERMITTIVITY
        self.D_Li = LLZO_DIFFUSION_COEFF
        self.V_T = KB * T_kelvin / ELEMENTARY_CHARGE  # 热电压 [V]
        # Debye 长度: lambda_D = sqrt(eps * kB * T / (2 * e^2 * NA * c))
        # 其中 c 为摩尔浓度 [mol/m^3], NA * c 为离子数密度
        from material_constants import AVOGADRO
        n_0 = c_ref * AVOGADRO  # 离子数密度 [1/m^3]
        self.lambda_D = np.sqrt(
            self.eps * KB * T_kelvin / (2.0 * ELEMENTARY_CHARGE**2 * n_0)
        )

        # 无量纲 Debye 长度
        self.lambda_sq = (self.lambda_D / L) ** 2

        # 网格
        self.x = np.linspace(0, L, N + 2)

        # 差分算子
        self.D1 = central_fd_1d_uniform(N, self.h, order=min(fd_order, 4), deriv=1)
        self.D2 = central_fd_1d_uniform(N, self.h, order=min(fd_order, 4), deriv=2)

        # 存储
        self.phi = np.zeros(N)       # 电势 [V]
        self.c_Li = np.ones(N) * c_ref  # Li 浓度 [mol/m^3]
        self.c_fixed = np.zeros(N)   # 固定电荷

    def set_initial_conditions(self, phi_init=None, c_Li_init=None, c_fixed_profile=None):
        """设置初始条件"""
        if phi_init is not None:
            self.phi = np.asarray(phi_init, dtype=np.float64).copy()
        if c_Li_init is not None:
            self.c_Li = np.asarray(c_Li_init, dtype=np.float64).copy()
        if c_fixed_profile is not None:
            self.c_fixed = np.asarray(c_fixed_profile, dtype=np.float64).copy()

    def _poisson_rhs(self, c_Li_star, c_fixed_star):
        """
        Poisson 方程右端 (无量纲):
          -lambda^2 * D2 * phi = c_Li* - c_fixed*
        """
        rhs = c_Li_star - c_fixed_star
        # 求解 phi
        A = -self.lambda_sq * self.D2
        # 添加 Dirichlet BC (phi=0 在两端)
        A_dense = A.toarray()
        phi = np.linalg.solve(A_dense, rhs)
        return phi

    def _nernst_planck_flux(self, c_Li, phi):
        """
        NP 通量 (无量纲):
          J = -D * dc/dx - D * c * dphi/dx

        包括 steric 修正:
          J_steric = J + D * c^2/c_max * (1 - c/c_max)^{-1} * dphi/dx
        """
        dc_dx = self.D1.dot(c_Li)
        dphi_dx = self.D1.dot(phi)
        # 标准 NP 通量
        J = -self.D_Li * dc_dx - (self.D_Li / self.V_T) * c_Li * dphi_dx
        return J

    def _steric_correction(self, c_Li, c_max):
        """
        Steric 修正项 (Bazant 2011):

        化学势修正:
          mu_ex = kB*T * log(1 - c/c_max)
        修正通量:
          J_ex = -D * c * d(mu_ex/kBT)/dx
               = D * c / (c_max - c) * dc/dx
        """
        dc_dx = self.D1.dot(c_Li)
        c_ratio = c_Li / np.maximum(c_max - c_Li, SMALL_NUMBER)
        c_ratio = np.clip(c_ratio, -1e6, 1e6)
        J_ex = self.D_Li * c_ratio * dc_dx
        return J_ex

    def butler_volmer_flux(self, phi_s, phi_m, c_Li_surf, j_0=1.0, alpha=0.5, U_eq=0.0):
        """
        Butler-Volmer 电极动力学:

        j = j_0 * [exp(alpha*e*eta/(kB*T)) - exp(-(1-alpha)*e*eta/(kB*T))]
        eta = phi_s - phi_m - U_eq

        返回: 电流密度 j [A/m^2]
        """
        eta = phi_s - phi_m - U_eq
        eta_star = eta / self.V_T
        eta_star = np.clip(eta_star, -50, 50)  # 防溢出
        j = j_0 * (np.exp(alpha * eta_star) - np.exp(-(1 - alpha) * eta_star))
        return j

    def step_backward_euler(self, dt, c_max=None, picard_tol=1e-8, picard_max=20):
        """
        Backward Euler 时间步进 + Picard 迭代。

        离散:
          (c^{n+1} - c^n) / dt = -div(J^{n+1})
          c^{n+1} + dt * div(J(c^{n+1}, phi^{n+1})) = c^n

        Picard 线性化:
          固定 phi^{(k)}, 求解 c^{(k+1)}
          固定 c^{(k+1)}, 求解 phi^{(k+1)}
        """
        c_old = self.c_Li.copy()
        c_star = self.c_Li / self.c_ref
        c_fixed_star = self.c_fixed / self.c_ref

        for p_iter in range(picard_max):
            c_old_iter = c_star.copy()

            # (1) 求解 Poisson
            phi = self._poisson_rhs(c_star, c_fixed_star)

            # (2) 求解 NP (线性化)
            # c* + dt* * (-D2*c* - d/dx(c* * dphi/dx)) = c_old*
            dphi_dx = self.D1.dot(phi)
            # 对流项: d/dx(c * dphi/dx) ≈ dphi_dx * D1 * c + c * D2 * phi
            # 简化为矩阵形式: A * c* = c_old*
            A_np = sparse.eye(self.N) - dt * self.D2
            rhs_np = c_old / self.c_ref

            # 添加 Dirichlet BC
            A_dense = A_np.toarray()
            c_new_star = np.linalg.solve(A_dense, rhs_np)

            # Steric 约束
            if c_max is not None:
                c_new_star = np.minimum(c_new_star, c_max / self.c_ref * 0.99)
            c_new_star = np.maximum(c_new_star, SMALL_NUMBER)

            # 收敛检查
            res = np.linalg.norm(c_new_star - c_old_iter) / max(np.linalg.norm(c_old_iter), SMALL_NUMBER)
            c_star = c_new_star

            if res < picard_tol:
                break

        self.phi = phi
        self.c_Li = c_star * self.c_ref
        return res

    def compute_charge_density(self):
        """
        空间电荷密度 [C/m^3]:
          rho = e * (c_Li - c_fixed)
        """
        return ELEMENTARY_CHARGE * (self.c_Li - self.c_fixed)

    def compute_current_density(self):
        """
        电流密度 [A/m^2]:
          i = e * J_Li = e * (-D*dc/dx - D*c/(kB*T) * dphi/dx)
        """
        J = self._nernst_planck_flux(self.c_Li, self.phi)
        return ELEMENTARY_CHARGE * J

    def ionic_conductivity(self):
        """
        由 Nernst-Einstein 关系计算离子电导率:

          sigma = e^2 * c * D / (kB * T)

        返回: sigma [S/m]
        """
        c_avg = np.mean(self.c_Li)
        sigma = (ELEMENTARY_CHARGE**2) * c_avg * self.D_Li / (KB * self.T)
        return sigma

    def mass_conservation_check(self):
        """
        质量守恒检查:
          integral(c dx) 应保持不变

        返回相对偏差:
          delta = |int(c) - int(c_0)| / int(c_0)
        """
        return np.sum(self.c_Li) * self.h
