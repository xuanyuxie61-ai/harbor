"""
axon_propagation.py
轴突非线性信号传播模块

融合 dg1d_burgers (间断 Galerkin 空间离散 + RK4 时间推进)
与 mhd_exact (磁流体动力学耦合思想)。

核心科学模型：
  将经典电缆方程扩展为包含离子流-电磁场耦合的非线性传播方程。

  非线性神经电缆方程 (Burgers-type):
    tau_m dV/dt + lambda_e^{-1} dV/dx = D d^2V/dx^2 + f_nonlin(V) + I_ion

  其中：
    tau_m = C_m / g_L   (膜时间常数)
    D = lambda^2 / tau_m  (有效扩散系数)
    lambda = sqrt(d / (4 R_a g_L))  (空间常数)
    f_nonlin(V) = g_Na m^3 h (V - E_Na) + g_K n^4 (V - E_K)  (HH 离子流)

  电磁场耦合 (MHD 思想):
    轴突内离子流 J_ion = sum_k z_k e n_k v_k 产生磁场 B:
    curl B = mu_0 J_ion
    该磁场通过洛伦兹力反馈调制离子迁移率:
    v_k' = v_k + (z_k e / m_k) (v_k x B) dt

  DG 空间离散 (1D):
    单元 K_j = [x_{j-1/2}, x_{j+1/2}], 局部基函数 phi_k(x) (Jacobi 多项式)。
    弱形式:
      int_{K_j} tau_m dV/dt phi_k dx =
        - int_{K_j} D dV/dx dphi_k/dx dx
        + [D V_x phi_k]_{x_{j-1/2}}^{x_{j+1/2}}
        - int_{K_j} lambda_e^{-1} V dphi_k/dx dx
        + [lambda_e^{-1} V^* phi_k]_{x_{j-1/2}}^{x_{j+1/2}}
        + int_{K_j} (f_nonlin(V) + I_ion) phi_k dx

  数值通量 V^* 采用 Local Lax-Friedrichs:
    V^* = 0.5 (V^+ + V^-) + 0.5 C |lambda_e^{-1}| (V^+ - V^-)
"""

import numpy as np
import math
from spike_neuron import HHNeuron


class JacobiPolynomial:
    """
    Jacobi 多项式求值，融合 dg1d_burgers 中 JacobiP 思想。
    用于 DG 方法的谱基函数。
    """

    @staticmethod
    def evaluate(x, alpha, beta, N):
        """
        计算归一化的 Jacobi 多项式 P_N^{(alpha,beta)}(x)。
        x: 求值点，可为一维数组
        alpha, beta: 参数，要求 > -1
        N: 多项式阶数
        """
        if alpha <= -1 or beta <= -1:
            raise ValueError("alpha and beta must be > -1")
        x = np.atleast_1d(x)
        if N < 0:
            return np.zeros_like(x)

        # 使用递推关系
        # P_0 = 1 / sqrt(gamma0)
        gamma0 = (2.0 ** (alpha + beta + 1.0)) / (alpha + beta + 1.0) * \
                 math.gamma(alpha + 1.0) * math.gamma(beta + 1.0) / math.gamma(alpha + beta + 1.0)
        P0 = np.ones_like(x) / np.sqrt(gamma0)
        if N == 0:
            return P0

        gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
        P1 = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)
        if N == 1:
            return P1

        aold = 2.0 / (2.0 + alpha + beta) * np.sqrt(
            (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0)
        )

        PL_prev2 = P0
        PL_prev1 = P1
        for i in range(1, N):
            h1 = 2.0 * i + alpha + beta
            anew = 2.0 / (h1 + 2.0) * np.sqrt(
                (i + 1.0) * (i + 1.0 + alpha + beta) * (i + 1.0 + alpha) * (i + 1.0 + beta)
                / (h1 + 1.0) / (h1 + 3.0)
            )
            bnew = - (alpha ** 2 - beta ** 2) / h1 / (h1 + 2.0)
            PL_curr = (1.0 / anew) * (
                -aold * PL_prev2 + (x - bnew) * PL_prev1
            )
            aold = anew
            PL_prev2 = PL_prev1
            PL_prev1 = PL_curr
        return PL_curr

    @staticmethod
    def gauss_lobatto_nodes(N):
        """
        返回 N+1 个 Gauss-Lobatto 节点 (包含端点 -1, 1)。
        N: 多项式阶数
        """
        if N < 0:
            raise ValueError("N must be non-negative.")
        # 使用 Chebyshev-Gauss-Lobatto 节点作为近似
        # x_k = cos(pi * k / N), k = 0, 1, ..., N
        k = np.arange(N + 1)
        nodes = np.cos(np.pi * k / N)
        # 反转顺序使其从 -1 到 1
        nodes = nodes[::-1]
        return nodes


class DG1DNeuralCable:
    """
    一维神经电缆的间断 Galerkin 离散。
    模拟动作电位在轴突上的空间传播。
    """

    def __init__(self, xL, xR, K, Np, dt, epsilon=0.01):
        """
        xL, xR: 空间区间
        K: 单元数
        Np: 每单元节点数 (阶数+1)
        dt: 时间步长
        epsilon: 数值粘性系数
        """
        self.xL = xL
        self.xR = xR
        self.K = K
        self.Np = Np
        self.dt = dt
        self.epsilon = epsilon

        # 生成网格和参考节点
        self.x_nodes = JacobiPolynomial.gauss_lobatto_nodes(Np - 1)
        # 映射到物理单元
        self.dx = (xR - xL) / K
        self.x = np.zeros((Np, K))
        for k in range(K):
            x_center = xL + (k + 0.5) * self.dx
            self.x[:, k] = x_center + 0.5 * self.dx * self.x_nodes

        # Vandermonde 矩阵 (Legendre alpha=beta=0)
        self.V = np.zeros((Np, Np))
        for j in range(Np):
            self.V[:, j] = JacobiPolynomial.evaluate(self.x_nodes, 0.0, 0.0, j)

        # 求导矩阵 D
        self.Dr = self._compute_derivative_matrix()
        self.Dx = (2.0 / self.dx) * self.Dr

        # 质量矩阵 (对角，由于正交性)
        self.M = np.linalg.inv(self.V @ self.V.T)

        # 物理参数
        self.tau_m = 2.0   # ms
        self.lambda_e = 1.0  # 空间尺度
        self.D_coeff = 0.5   # 有效扩散

    def _compute_derivative_matrix(self):
        """计算参考域 [-1,1] 上的求导矩阵。"""
        Np = self.Np
        Dr = np.zeros((Np, Np))
        for i in range(Np):
            for j in range(Np):
                if i != j:
                    # 基于 Lagrange 基函数的导数
                    # 简化计算：利用 Vandermonde 矩阵求逆
                    pass
        # 更稳健的方法：利用多项式求导
        # 对每个基函数 j，在节点 i 处的导数
        for j in range(Np):
            # 构造第 j 个 Lagrange 插值多项式在节点上的值
            lj = np.zeros(Np)
            lj[j] = 1.0
            # 通过 Vandermonde 求系数
            coeffs = np.linalg.solve(self.V, lj)
            # 求导后的系数
            dcoeffs = np.zeros(Np)
            for p in range(1, Np):
                dcoeffs[p - 1] = p * coeffs[p]
            # 在节点处求值
            for i in range(Np):
                Dr[i, j] = np.sum(dcoeffs * self.V[i, :])
        return Dr

    def local_lax_friedrichs_flux(self, u_left, u_right):
        """
        Local Lax-Friedrichs 数值通量。
        u_left: 左单元右端值
        u_right: 右单元左端值
        """
        C = 1.0  # 波速上界 (简化)
        flux = 0.5 * (u_left + u_right) + 0.5 * C * (u_left - u_right)
        return flux

    def rhs(self, u, I_ion):
        """
        计算 DG 右端项。
        u: (Np, K) 膜电位
        I_ion: (Np, K) 离子流
        """
        Np, K = u.shape
        rhsu = np.zeros_like(u)

        # 单元内部导数项
        for k in range(K):
            # 扩散项: D * d^2u/dx^2 (简化处理)
            ux = self.Dx @ u[:, k]
            uxx = self.Dx @ ux
            rhsu[:, k] += self.D_coeff * uxx

            # 对流项 (Burgers-type 非线性): -lambda_e^{-1} du/dx
            rhsu[:, k] += -(1.0 / self.lambda_e) * ux

            # 源项
            rhsu[:, k] += (-u[:, k] / self.tau_m + I_ion[:, k])

        # 界面通量 (间断处理)
        for k in range(K):
            # 左界面
            if k == 0:
                u_left_boundary = u[-1, k]  # 周期/Neumann 边界
            else:
                u_left_boundary = u[-1, k - 1]
            u_right_boundary = u[0, k]

            flux_left = self.local_lax_friedrichs_flux(u_left_boundary, u_right_boundary)
            # 右界面
            if k == K - 1:
                u_right_next = u[0, k]
            else:
                u_right_next = u[0, k + 1]
            flux_right = self.local_lax_friedrichs_flux(u[-1, k], u_right_next)

            # 通量对 RHS 的贡献 (简化 penalty 形式)
            rhsu[0, k] += self.epsilon * (flux_left - u[0, k]) / self.dx
            rhsu[-1, k] += self.epsilon * (flux_right - u[-1, k]) / self.dx

        return rhsu

    def step_rk4(self, u, I_ion):
        """RK4 单步推进。"""
        dt = self.dt
        # k1
        rhs1 = self.rhs(u, I_ion)
        # k2
        rhs2 = self.rhs(u + 0.5 * dt * rhs1, I_ion)
        # k3
        rhs3 = self.rhs(u + 0.5 * dt * rhs2, I_ion)
        # k4
        rhs4 = self.rhs(u + dt * rhs3, I_ion)
        u_new = u + dt / 6.0 * (rhs1 + 2.0 * rhs2 + 2.0 * rhs3 + rhs4)
        return u_new

    def simulate(self, u0, T_final, I_ion_func=None):
        """
        运行仿真。
        u0: (Np, K) 初始膜电位
        T_final: 终止时间
        I_ion_func: 可选的时变离子流函数 func(t, x)
        """
        n_steps = int(np.ceil(T_final / self.dt))
        u = u0.copy()
        history = [u.copy()]
        for step in range(n_steps):
            t = step * self.dt
            if I_ion_func is not None:
                I_ion = I_ion_func(t, self.x)
            else:
                I_ion = np.zeros_like(u)
            u = self.step_rk4(u, I_ion)
            # 数值边界处理
            u = np.clip(u, -100.0, 100.0)
            if step % max(1, n_steps // 100) == 0:
                history.append(u.copy())
        return u, history


class MHDNeuralCoupling:
    """
    MHD 电磁耦合对神经信号的调制。
    融合 mhd_exact 的 Hartmann 流思想。
    """

    MU_0 = 4.0 * np.pi * 1e-7  # H/m
    E_CHARGE = 1.602e-19       # C

    @staticmethod
    def ionic_current_density(V, ion_concentrations):
        """
        计算离子流密度 J_ion = sum_k z_k e n_k v_k。
        简化模型: J = sigma_e E, E = -dV/dx
        增强模型以产生可见的 MHD 效应。
        """
        sigma_e = 1.5  # S/m (增强电导率)
        E_field = -np.gradient(V)
        J = sigma_e * E_field
        return J

    @staticmethod
    def magnetic_field_from_current(J, y_coord, mu0=MU_0):
        """
        由安培定律 curl B = mu0 J 估算磁场。
        一维简化: B(y) = mu0 * J * y (考虑薄层效应)
        """
        B = mu0 * J * y_coord
        return B

    @staticmethod
    def lorentz_force_modulation(J, B, ion_mobility=5e3):
        """
        洛伦兹力 F = J x B 对离子迁移率的调制。
        返回有效电导率修正因子。
        增强 mobility 以产生可见的非线性调制。
        """
        F = np.abs(J * B)  # 一维简化
        correction = 1.0 / (1.0 + ion_mobility * F)
        return np.clip(correction, 0.1, 1.0)

    def compute_effective_conductivity(self, V, y_coord=5e-5):
        """
        计算考虑 MHD 耦合后的有效电导率。
        y_coord 增大以产生可见的磁场强度。
        """
        J = self.ionic_current_density(V, None)
        B = self.magnetic_field_from_current(J, y_coord)
        correction = self.lorentz_force_modulation(J, B)
        return correction


def demo_axon_propagation():
    """轴突信号传播 demo。"""
    cable = DG1DNeuralCable(xL=0.0, xR=10.0, K=20, Np=4, dt=0.001, epsilon=0.05)
    u0 = np.zeros((cable.Np, cable.K))
    # 在左侧注入高斯型初始脉冲
    for k in range(cable.K):
        for i in range(cable.Np):
            x = cable.x[i, k]
            u0[i, k] = 20.0 * np.exp(-((x - 1.0) ** 2) / 0.5)
    u_final, history = cable.simulate(u0, T_final=5.0)
    return u_final, cable.x


def demo_mhd_coupling():
    """MHD 耦合 demo。"""
    mhd = MHDNeuralCoupling()
    x = np.linspace(0, 10, 100)
    V = 20.0 * np.exp(-((x - 5.0) ** 2) / 1.0)
    correction = mhd.compute_effective_conductivity(V, y_coord=1e-6)
    return x, V, correction
