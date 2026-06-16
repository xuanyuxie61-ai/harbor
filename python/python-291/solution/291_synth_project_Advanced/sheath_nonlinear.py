"""
sheath_nonlinear.py
===================
非线性 Poisson-Boltzmann 方程求解器。

本模块实现等离子体鞘层的自洽电势求解。核心方程：
    Poisson 方程 (无量纲)：
        d²φ/dx² = -ρ/ε₀ → d²φ/dx² = (n_e - n_i)  (归一化后)

    修正 Boltzmann 电子密度 (含二次电子发射)：
        n_e(φ) = n₀ * [δ exp(eφ/kT_e) + (1-δ)] * exp(eφ/(χ kT_e))
    其中：
        δ: SEE 系数
        χ: 有效电子温度降低因子

    冷离子流体密度 (能量守恒)：
        n_i(φ) = n₀ / sqrt(1 - 2eφ/(m_i u₀²))
    其中 u₀ ≥ c_s (Bohm 条件)

    数值方法：
        1. Newton-Raphson 迭代 (含阻尼)
        2. 不动点迭代 (Picard)
        3. 伪时间推进
        4. 延拓法 (continuation)
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, Optional
import math

from fd_highorder import fd_second_deriv_matrix


class PoissonBoltzmannSolver:
    """
    非线性 Poisson-Boltzmann 方程求解器

    方程：
        L φ = n_e(φ) - n_i(φ) =: g(φ)

    其中 L = d²/dx² 为二阶微分算子

    Jacobian:
        J = L - ∂g/∂φ
        ∂g/∂φ = ∂n_e/∂φ - ∂n_i/∂φ
    """

    def __init__(
        self,
        x: np.ndarray,
        phi_wall: float,
        chi_see: float = 1.0,
        gamma_see: float = 0.05,
        mass_ratio: float = 1e-5,
        u_bohm: float = 1.0,
        fd_order: int = 4,
    ):
        """
        参数：
            x: 网格坐标数组
            phi_wall: 无量纲壁面电势
            chi_see: SEE 有效因子
            gamma_see: SEE 系数
            mass_ratio: m_e / m_i
            u_bohm: 无量纲 Bohm 速度
            fd_order: 有限差分阶数
        """
        self.x = x
        self.N = len(x)
        self.phi_wall = phi_wall
        self.chi = chi_see
        self.gamma = gamma_see
        self.mu = mass_ratio
        self.u_bohm = u_bohm
        self.fd_order = fd_order

        # 构造二阶微分矩阵
        self.D2 = fd_second_deriv_matrix(x, order=fd_order)

        # 存储结果
        self.phi = None
        self.n_e = None
        self.n_i = None
        self.E_field = None
        self.converged = False
        self.n_iterations = 0
        self.residual_history = []

    def electron_density(self, phi: np.ndarray) -> np.ndarray:
        """
        修正 Boltzmann 电子密度

            n_e(φ) = [(1-γ) exp(φ/χ) + γ exp(φ)]
        其中 φ = eΦ/(kT_e) 为无量纲电势（φ ≤ 0 在鞘层内）

        第一项：来自等离子体的电子
        第二项：二次电子发射贡献

        为数值稳定性，对指数进行截断：
            exp(φ/χ) → min(exp(φ/χ), exp_max)
        """
        phi_safe = np.clip(phi, -50.0, 0.0)
        chi_safe = max(self.chi, 1e-6)
        gamma_safe = min(max(self.gamma, 0.0), 0.99)

        n_e_plasma = (1.0 - gamma_safe) * np.exp(phi_safe / chi_safe)
        n_e_see = gamma_safe * np.exp(phi_safe)
        return n_e_plasma + n_e_see

    def ion_density(self, phi: np.ndarray) -> np.ndarray:
        """
        冷离子流体密度

            n_i(φ) = 1 / sqrt(1 - 2φ / u₀²)
        其中 u₀ ≥ 1 (Bohm 速度，无量纲)

        物理约束：1 - 2φ/u₀² > 0 → φ < u₀²/2
        由于鞘层内 φ ≤ 0，此条件自动满足
        """
        u0_sq = self.u_bohm**2
        arg = 1.0 - 2.0 * phi / u0_sq

        # 数值保护
        arg = np.maximum(arg, 1e-10)
        return 1.0 / np.sqrt(arg)

    def charge_density(self, phi: np.ndarray) -> np.ndarray:
        """
        净电荷密度
            ρ(φ) = n_i(φ) - n_e(φ)
        Poisson 方程：d²φ/dx² = -ρ
        """
        return self.ion_density(phi) - self.electron_density(phi)

    def residual(self, phi: np.ndarray) -> np.ndarray:
        """
        非线性残差
            R(φ) = D² φ + ρ(φ) = 0

        边界条件：
            φ(0) = 0 (鞘层边缘)
            φ(L) = φ_wall
        """
        R = self.D2 @ phi + self.charge_density(phi)

        # 施加边界条件
        R[0] = phi[0] - 0.0
        R[-1] = phi[-1] - self.phi_wall

        return R

    def jacobian(self, phi: np.ndarray) -> np.ndarray:
        """
        Jacobian 矩阵
            J = D² + dρ/dφ

        dρ/dφ = dn_i/dφ - dn_e/dφ

        dn_e/dφ = [(1-γ)/χ exp(φ/χ) + γ exp(φ)]
        dn_i/dφ = 1 / (u₀² (1 - 2φ/u₀²)^{3/2})
        """
        u0_sq = self.u_bohm**2
        chi_safe = max(self.chi, 1e-6)
        gamma_safe = min(max(self.gamma, 0.0), 0.99)

        phi_safe = np.clip(phi, -50.0, 0.0)

        # d n_e / d φ
        dne_dphi = (
            (1.0 - gamma_safe) / chi_safe * np.exp(phi_safe / chi_safe)
            + gamma_safe * np.exp(phi_safe)
        )

        # d n_i / d φ
        arg = np.maximum(1.0 - 2.0 * phi / u0_sq, 1e-10)
        dni_dphi = 1.0 / (u0_sq * arg**1.5)

        # d ρ / d φ
        drho_dphi = dni_dphi - dne_dphi

        # Jacobian = D² + diag(dρ/dφ)
        J = self.D2.copy()
        for i in range(self.N):
            J[i, i] += drho_dphi[i]

        # 边界行
        J[0, :] = 0.0
        J[0, 0] = 1.0
        J[-1, :] = 0.0
        J[-1, -1] = 1.0

        return J

    def solve_newton(
        self,
        phi_init: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-10,
        damping: float = 1.0,
    ) -> Tuple[np.ndarray, bool]:
        """
        Newton-Raphson 求解（含阻尼）

        迭代格式：
            J(φ_k) δφ = -R(φ_k)
            φ_{k+1} = φ_k + α δφ

        参数：
            phi_init: 初始猜测，默认为线性插值
            max_iter: 最大迭代次数
            tol: 收敛容差 (L2 范数)
            damping: 阻尼系数 α ∈ (0, 1]

        返回：
            phi: 收敛解
            converged: 是否收敛
        """
        if phi_init is None:
            phi = np.linspace(0.0, self.phi_wall, self.N)
        else:
            phi = phi_init.copy()

        # 确保边界条件
        phi[0] = 0.0
        phi[-1] = self.phi_wall

        self.residual_history = []

        for k in range(max_iter):
            R = self.residual(phi)
            res_norm = np.linalg.norm(R) / self.N
            self.residual_history.append(res_norm)

            if res_norm < tol:
                self.converged = True
                self.n_iterations = k
                break

            J = self.jacobian(phi)

            try:
                delta_phi = la.solve(J, -R)
            except la.LinAlgError:
                # Jacobian 奇异，使用伪逆
                delta_phi = la.lstsq(J, -R)[0]

            # 阻尼更新 + 回溯线搜索
            alpha = damping
            for _ in range(10):
                phi_new = phi + alpha * delta_phi
                R_new = self.residual(phi_new)
                res_new = np.linalg.norm(R_new) / self.N
                if res_new < res_norm:
                    break
                alpha *= 0.5

            phi = phi + alpha * delta_phi

            # 数值保护：限制电势范围
            phi = np.clip(phi, -100.0, 1.0)
            phi[0] = 0.0
            phi[-1] = self.phi_wall

        self.phi = phi
        self.n_e = self.electron_density(phi)
        self.n_i = self.ion_density(phi)

        return phi, self.converged

    def solve_picard(
        self,
        phi_init: Optional[np.ndarray] = None,
        max_iter: int = 500,
        tol: float = 1e-8,
        omega: float = 0.3,
    ) -> Tuple[np.ndarray, bool]:
        """
        不动点迭代 (Picard 迭代) 含 SOR 加速

        迭代格式：
            φ_{k+1} = (1-ω) φ_k + ω L^{-1} [-ρ(φ_k)]

        参数：
            omega: SOR 松弛因子
        """
        if phi_init is None:
            phi = np.linspace(0.0, self.phi_wall, self.N)
        else:
            phi = phi_init.copy()

        phi[0] = 0.0
        phi[-1] = self.phi_wall

        # 预分解 D2（去掉边界行）
        D2_interior = self.D2[1:-1, 1:-1]
        try:
            D2_inv = la.inv(D2_interior)
        except la.LinAlgError:
            D2_inv = la.pinv(D2_interior)

        self.residual_history = []
        converged = False

        for k in range(max_iter):
            rho = self.charge_density(phi)

            # 求解 D² φ_new = -ρ
            rhs = -rho[1:-1]
            phi_new_interior = D2_inv @ rhs

            phi_new = phi.copy()
            phi_new[1:-1] = phi_new_interior
            phi_new[0] = 0.0
            phi_new[-1] = self.phi_wall

            # SOR 更新
            phi_updated = (1.0 - omega) * phi + omega * phi_new
            phi_updated[0] = 0.0
            phi_updated[-1] = self.phi_wall

            # 检查收敛
            diff = np.linalg.norm(phi_updated - phi) / self.N
            self.residual_history.append(diff)

            if diff < tol:
                converged = True
                self.n_iterations = k
                break

            phi = phi_updated
            phi = np.clip(phi, -100.0, 1.0)
            phi[0] = 0.0
            phi[-1] = self.phi_wall

        self.phi = phi
        self.n_e = self.electron_density(phi)
        self.n_i = self.ion_density(phi)
        self.converged = converged

        return phi, converged

    def pseudo_time_step(
        self,
        phi_init: Optional[np.ndarray] = None,
        dt: float = 0.01,
        n_steps: int = 1000,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, bool]:
        """
        伪时间推进法

        ∂φ/∂τ = D²φ + ρ(φ)

        使用隐式 Euler：
            (I - dt D²) φ^{n+1} = φ^n + dt ρ(φ^n)
        """
        if phi_init is None:
            phi = np.linspace(0.0, self.phi_wall, self.N)
        else:
            phi = phi_init.copy()

        phi[0] = 0.0
        phi[-1] = self.phi_wall

        # 构造左端矩阵
        A = np.eye(self.N) - dt * self.D2
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0

        self.residual_history = []
        converged = False

        for step in range(n_steps):
            rho = self.charge_density(phi)
            rhs = phi + dt * rho
            rhs[0] = 0.0
            rhs[-1] = self.phi_wall

            try:
                phi_new = la.solve(A, rhs)
            except la.LinAlgError:
                phi_new = la.lstsq(A, rhs)[0]

            phi_new = np.clip(phi_new, -100.0, 1.0)
            phi_new[0] = 0.0
            phi_new[-1] = self.phi_wall

            diff = np.linalg.norm(phi_new - phi) / self.N
            self.residual_history.append(diff)

            if diff < tol:
                converged = True
                self.n_iterations = step
                break

            phi = phi_new

        self.phi = phi
        self.n_e = self.electron_density(phi)
        self.n_i = self.ion_density(phi)
        self.converged = converged

        return phi, converged

    def continuation_solve(
        self,
        phi_wall_target: float,
        n_steps: int = 20,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, bool]:
        """
        延拓法 (continuation method)

        逐渐改变壁面电势从 0 到目标值，
        每一步使用前一步的解作为初始猜测。

        参数：
            phi_wall_target: 目标壁面电势
            n_steps: 延拓步数
            tol: 收敛容差
        """
        phi = np.zeros(self.N)  # 零电势初始

        phi_values = np.linspace(0.0, phi_wall_target, n_steps + 1)

        for step in range(1, n_steps + 1):
            self.phi_wall = phi_values[step]
            phi, converged = self.solve_newton(
                phi_init=phi,
                max_iter=50,
                tol=tol,
                damping=0.8,
            )
            if not converged and step > 1:
                # 使用 Picard 作为后备
                phi, converged = self.solve_picard(
                    phi_init=phi,
                    max_iter=200,
                    tol=tol,
                )
            if not converged:
                break

        self.phi = phi
        self.n_e = self.electron_density(phi)
        self.n_i = self.ion_density(phi)
        return phi, converged

    def compute_sheath_properties(self) -> dict:
        """
        计算鞘层物理量

        返回字典包含：
            sheath_thickness: 鞘层厚度
            max_field: 最大电场强度
            wall_charge: 壁面电荷密度
            bohman_ratio: Bohm 判据比值
        """
        if self.phi is None:
            raise RuntimeError("需要先求解电势")

        # 电场 E = -dφ/dx (使用中心差分)
        dx = np.gradient(self.x)
        E = -np.gradient(self.phi) / dx
        self.E_field = E

        # 鞘层厚度：φ 从 0.01*phi_wall 到 0.99*phi_wall 的区间
        threshold_low = 0.01 * abs(self.phi_wall)
        threshold_high = 0.99 * abs(self.phi_wall)
        inside = np.where(np.abs(self.phi) >= threshold_low)[0]
        if len(inside) > 0:
            x_sheath_start = self.x[inside[0]]
        else:
            x_sheath_start = self.x[0]

        # Bohm 比
        if self.u_bohm > 0:
            # 在鞘层边缘的离子速度（估计）
            u_edge = self.u_bohm
        else:
            u_edge = 0.0

        return {
            'sheath_start': x_sheath_start,
            'max_field': np.max(np.abs(E)),
            'wall_field': E[-1],
            'max_electron_density': np.max(self.n_e),
            'max_ion_density': np.max(self.n_i),
            'charge_imbalance_max': np.max(np.abs(self.n_i - self.n_e)),
            'phi_min': np.min(self.phi),
            'bohm_ratio': u_edge,
        }
