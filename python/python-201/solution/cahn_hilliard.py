"""
cahn_hilliard.py — 随机 Cahn-Hilliard 方程求解器
====================================================
实现带有随机参数的 Cahn-Hilliard 相场方程的数值求解,
结合多项式混沌展开进行不确定性传播。

核心方程:
  Cahn-Hilliard 方程:
    ∂c/∂t = ∇·(M(ξ) ∇μ)
    μ = δF/δc = γ(ξ)(c^3 - c) - κ(ξ)∇²c

  自由能泛函:
    F[c] = ∫_Ω [γ(ξ)/4 (c^2 - 1)^2 + κ(ξ)/2 |∇c|^2] dx

  能量耗散律:
    dF/dt = -∫_Ω M(ξ) |∇μ|^2 dx ≤ 0

  PCE 展开:
    c(x,t,ξ) ≈ Σ_{k=0}^{P} ĉ_k(x,t) Ψ_k(ξ)

  数值离散:
    空间: FFT 谱方法 (周期性边界)
    时间: 半隐式 Crank-Nicolson / BDF2

映射种子项目:
  - 787_navier_stokes_2d_exact: Navier-Stokes 精确解 → 验证方法
  - 211_continuity_exact: 无散度场 → 速度场约束
  - 1298_Eric6669: Port-Hamiltonian 结构 → 能量保持结构
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from config import GlobalConfig, CahnHilliardConfig
from grid import create_physical_mesh, generate_initial_condition
from utils import spectral_derivative_2d
from polynomial_basis import OrthogonalPolynomialBasis
from galerkin import StochasticGalerkinProjector


class CahnHilliardSolver:
    """
    确定性 Cahn-Hilliard 求解器 (用于单个随机样本)。

    使用 Fourier 谱方法离散空间, 半隐式时间积分。

    映射 787_navier_stokes_2d_exact:
      借鉴 Taylor-Green 涡旋的精确解验证方法,
      使用类似的正交函数验证 CH 求解器。

    映射 211_continuity_exact:
      质量守恒: ∫c dx = const, 类似连续性方程的无散度约束。
    """

    def __init__(self, config: CahnHilliardConfig):
        self.config = config
        self.X, self.Y, self.dx, self.dy = create_physical_mesh(config)
        self.nx = config.nx
        self.ny = config.ny
        self.Lx = config.Lx
        self.Ly = config.Ly

        # 波数 (用于谱方法)
        self.kx = 2 * np.pi * np.fft.fftfreq(
            self.nx, d=self.Lx / self.nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(
            self.ny, d=self.Ly / self.ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K4 = self.K2 ** 2

    def solve(self, c_init: np.ndarray,
              kappa: float, gamma: float, mobility: float,
              n_steps: Optional[int] = None,
              dt: Optional[float] = None
              ) -> Tuple[np.ndarray, List[float]]:
        """
        求解 Cahn-Hilliard 方程。

        半隐式方案:
          (c^{n+1} - c^n) / dt = M ∇²[γ((c^n)^3 - c^n) - κ∇²c^{n+1}]

        线性化 (凸分裂):
          (c^{n+1} - c^n) / dt = M ∇²[γ(c_S^3 - c_A^n) - κ∇²c^{n+1}]

        其中 c_S = c^n (显式部分), c_A = c^{n+1} (隐式部分),
        使用凸分裂: c^3 ≈ 3(c^n)^2 c^{n+1} - 2(c^n)^3

        参数:
            c_init:   初始浓度场, shape (ny, nx)
            kappa:    梯度能量系数 κ
            gamma:    双阱势参数 γ
            mobility: 迁移率 M
            n_steps:  时间步数
            dt:       时间步长

        返回:
            c_final:  最终浓度场
            energies: 各时间步的自由能列表
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        if dt is None:
            dt = self.config.dt

        M = mobility
        c = c_init.copy()
        energies = []

        # 初始能量
        E0 = self.compute_free_energy(c, kappa, gamma)
        energies.append(E0)

        for step in range(n_steps):
            c = self._step_semi_implicit(c, kappa, gamma, M, dt)

            # 每 50 步记录能量
            if step % 50 == 0 or step == n_steps - 1:
                E = self.compute_free_energy(c, kappa, gamma)
                energies.append(E)

        return c, energies

    def _step_semi_implicit(self, c: np.ndarray,
                            kappa: float, gamma: float,
                            M: float, dt: float) -> np.ndarray:
        """
        单个半隐式时间步。

        使用凸分裂 + Fourier 谱方法:
          显式: γ(c^3 - c) 的凸部分
          隐式: -κ∇²c + γ*c 的凹部分

        Fourier 空间求解:
          ĉ^{n+1}(k) = [ĉ^n(k) + dt*M*K2 * F{γ*凸分裂项}]
                        / [1 + dt*M*K2*(κ*K2 + γ)]

        添加 stabilization: S*dt*(c^{n+1} - c^n) 确保能量稳定。
        """
        nx, ny = self.nx, self.ny
        K2 = self.K2

        # 凸分裂: c^3 - c ≈ (2c^3 - 3c^n * (c^n)^2 + ...)
        # 简化: 使用 stabilization 方法
        S = gamma * 2.0 + kappa / (self.dx * self.dy)  # stabilization

        # 显式部分
        f_exp = gamma * (c ** 3 - c)

        # Fourier 变换
        c_hat = np.fft.fft2(c)
        f_hat = np.fft.fft2(f_exp)

        # 半隐式分母
        # 1 + dt*M*K2*(κ*K2 + γ + S)
        denom = 1.0 + dt * M * K2 * (kappa * K2 + S)
        # 防止除以零 (K=0 模式)
        denom[0, 0] = max(denom[0, 0], 1.0)

        # 分子
        numer = c_hat - dt * M * K2 * f_hat + dt * M * K2 * S * c_hat

        # 求解
        c_new_hat = numer / denom

        # 守恒修正: 确保零波数 (均值) 不变
        c_new_hat[0, 0] = c_hat[0, 0]

        c_new = np.real(np.fft.ifft2(c_new_hat))

        # 数值界限: c ∈ [-1, 1]
        c_new = np.clip(c_new, -1.0 - 0.1, 1.0 + 0.1)

        return c_new

    def compute_free_energy(self, c: np.ndarray,
                            kappa: float, gamma: float) -> float:
        """
        计算 Cahn-Hilliard 自由能。

        F[c] = ∫_Ω [γ/4 (c^2 - 1)^2 + κ/2 |∇c|^2] dx

        使用梯形法则近似空间积分。

        参数:
            c:     浓度场
            kappa: κ
            gamma: γ

        返回:
            F: 自由能值
        """
        # 双阱势密度: γ/4 * (c^2 - 1)^2
        f_bulk = gamma / 4.0 * (c ** 2 - 1.0) ** 2

        # 梯度能量: κ/2 * |∇c|^2
        _, _, lap, _ = spectral_derivative_2d(c, self.Lx, self.Ly)
        # |∇c|^2 近似: -c * ∇²c (分部积分)
        grad_sq = np.abs(lap) * np.abs(c)  # 粗略估计

        # 实际上应使用谱导数计算 |∇c|^2
        _, dfdy, _, _ = spectral_derivative_2d(c, self.Lx, self.Ly)
        # 重新计算梯度
        c_hat = np.fft.fft2(c)
        dc_dx = np.real(np.fft.ifft2(1j * self.KX * c_hat))
        dc_dy = np.real(np.fft.ifft2(1j * self.KY * c_hat))
        grad_sq = dc_dx ** 2 + dc_dy ** 2

        f_grad = kappa / 2.0 * grad_sq

        # 积分
        dA = self.dx * self.dy
        F = np.sum(f_bulk + f_grad) * dA

        return float(F)

    def verify_conservation(self, c: np.ndarray,
                            c_init: np.ndarray) -> float:
        """
        验证质量守恒。

        ∫c dx - ∫c_0 dx 应 ≈ 0

        返回相对误差。
        """
        dA = self.dx * self.dy
        mass_init = np.sum(c_init) * dA
        mass_curr = np.sum(c) * dA
        if abs(mass_init) > 1e-15:
            return abs(mass_curr - mass_init) / abs(mass_init)
        return abs(mass_curr - mass_init)

    def compute_energy_dissipation(self, energies: List[float]
                                   ) -> float:
        """
        验证能量耗散律。

        dF/dt ≤ 0: 自由能应单调递减。

        返回: 能量增加的比例 (应 ≈ 0 或负)。
        """
        if len(energies) < 2:
            return 0.0

        violations = 0
        for i in range(1, len(energies)):
            if energies[i] > energies[i - 1] + 1e-10:
                violations += 1

        return violations / max(len(energies) - 1, 1)


class StochasticCahnHilliard:
    """
    随机 Cahn-Hilliard 求解器。

    结合 PCE 和确定性求解器, 通过非侵入式配点法
    或 Galerkin 投影法求解随机问题。

    映射 1298_Eric6669: 利用 Port-Hamiltonian 能量结构
    确保随机解的物理一致性。
    """

    def __init__(self, config: GlobalConfig,
                 basis: OrthogonalPolynomialBasis,
                 projector: StochasticGalerkinProjector):
        self.config = config
        self.basis = basis
        self.projector = projector
        self.solver = CahnHilliardSolver(config.cahn_hilliard)

    def solve_collocation(self, quadrature_nodes: np.ndarray,
                          quadrature_weights: np.ndarray
                          ) -> Tuple[np.ndarray, List[float]]:
        """
        非侵入式配点法求解。

        对每个求积节点 ξ_q:
          1. 求解确定性 CH 方程 (参数为 ξ_q)
          2. 记录最终场 c(x, ξ_q)
        然后:
          3. 通过 PCE 回归得到系数 ĉ_k(x)

        参数:
            quadrature_nodes:   shape (n_q, d)
            quadrature_weights: shape (n_q,)

        返回:
            pce_coeffs:  shape (ny, nx, P), PCE 系数场
            mean_energy: 平均自由能历史
        """
        n_q = quadrature_nodes.shape[0]
        P = self.basis.n_basis
        nx = self.config.cahn_hilliard.nx
        ny = self.config.cahn_hilliard.ny

        # 初始条件
        X, Y, _, _ = create_physical_mesh(self.config.cahn_hilliard)
        c_init = generate_initial_condition(self.config.cahn_hilliard,
                                            X, Y)

        # 对每个求积点求解
        solutions = np.zeros((n_q, ny, nx))
        all_energies = []

        for q in range(n_q):
            xi_q = quadrature_nodes[q]

            # 提取物理参数
            kappa = float(xi_q[0]) if len(xi_q) > 0 else 0.01
            kappa = max(kappa, 1e-6)  # 确保正值

            gamma = float(xi_q[1]) if len(xi_q) > 1 else 1.0
            gamma = max(gamma, 0.1)  # 确保正值

            mobility = float(xi_q[2]) if len(xi_q) > 2 else 1.0
            mobility = max(mobility, 0.01)  # 确保正值

            # 求解 (使用较少的步数以节省计算)
            n_steps_solve = min(self.config.cahn_hilliard.n_steps, 100)
            c_final, energies = self.solver.solve(
                c_init, kappa, gamma, mobility,
                n_steps=n_steps_solve)

            solutions[q] = c_final
            all_energies.append(energies)

        # PCE 回归: 求解 ĉ_k = Σ_q w_q c(x, ξ_q) Ψ_k(ξ_q) / <Ψ_k, Ψ_k>
        Psi = self.basis.evaluate(quadrature_nodes)  # (n_q, P)
        pce_coeffs = np.zeros((ny, nx, P))

        for p_idx in range(P):
            psi_q = Psi[:, p_idx]  # (n_q,)
            for iy in range(ny):
                for ix in range(nx):
                    c_at_point = solutions[:, iy, ix]  # (n_q,)
                    pce_coeffs[iy, ix, p_idx] = np.sum(
                        quadrature_weights * c_at_point * psi_q)

        # 平均能量历史
        mean_energies = []
        n_energy_pts = min(len(e) for e in all_energies)
        for t_idx in range(n_energy_pts):
            e_avg = np.mean([e[t_idx] for e in all_energies])
            mean_energies.append(e_avg)

        return pce_coeffs, mean_energies

    def reconstruct_solution(self, pce_coeffs: np.ndarray,
                             xi_sample: np.ndarray) -> np.ndarray:
        """
        从 PCE 系数重建解在特定样本点的值。

        c(x, ξ*) = Σ_k ĉ_k(x) Ψ_k(ξ*)

        参数:
            pce_coeffs: shape (ny, nx, P)
            xi_sample:  shape (d,), 样本点

        返回:
            c_reconstructed: shape (ny, nx)
        """
        xi_2d = xi_sample.reshape(1, -1)
        Psi = self.basis.evaluate(xi_2d)  # (1, P)

        c_recon = np.zeros(pce_coeffs.shape[:2])
        for p_idx in range(self.basis.n_basis):
            c_recon += pce_coeffs[:, :, p_idx] * Psi[0, p_idx]

        return c_recon
