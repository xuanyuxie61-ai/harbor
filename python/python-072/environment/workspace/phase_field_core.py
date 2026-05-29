"""
phase_field_core.py
===================
相场方程核心模块 (Phase-Field Core Equation)

基于 Allen-Cahn / Ginzburg-Landau 相场模型，描述二元合金凝固过程中的
液固界面演化。序参量 φ 取值为 +1（固相）和 -1（液相），界面过渡层
宽度由参数 ε 控制。

核心物理方程：
    τ ∂φ/∂t = ε²∇²φ - W'(φ) - λ(T - T_M)(1 - φ²)² - μ(C - C_e)(1 - φ²)²

其中：
    W(φ) = (1/4)(φ² - 1)²   为双阱势函数
    W'(φ) = φ(φ² - 1)       为双阱势的导数
    λ 为热耦合系数
    μ 为溶质耦合系数
    T_M 为熔点温度
    C_e 为平衡浓度
"""

import numpy as np


class PhaseFieldModel:
    """
    相场模型核心类，实现 Allen-Cahn 型相场方程的离散与求解。
    """

    def __init__(self, nx, ny, dx, dy, epsilon=0.01, tau=1.0,
                 lambda_thermal=1.0, lambda_solute=1.0,
                 T_m=1.0, C_e=0.5, mobility=1.0):
        """
        初始化相场模型参数。

        Parameters
        ----------
        nx, ny : int
            网格点数。
        dx, dy : float
            空间步长。
        epsilon : float
            界面宽度参数。
        tau : float
            弛豫时间。
        lambda_thermal : float
            热耦合系数 λ_T。
        lambda_solute : float
            溶质耦合系数 λ_C。
        T_m : float
            熔点温度。
        C_e : float
            平衡浓度。
        mobility : float
            界面迁移率 M_φ。
        """
        if nx < 3 or ny < 3:
            raise ValueError("网格维度 nx, ny 必须至少为 3")
        if dx <= 0 or dy <= 0:
            raise ValueError("空间步长 dx, dy 必须为正")
        if epsilon <= 0:
            raise ValueError("界面宽度参数 epsilon 必须为正")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
        self.tau = tau
        self.lambda_thermal = lambda_thermal
        self.lambda_solute = lambda_solute
        self.T_m = T_m
        self.C_e = C_e
        self.mobility = mobility

        # 预计算 Laplacian 系数（五点差分）
        self.cx = 1.0 / (dx * dx)
        self.cy = 1.0 / (dy * dy)

    def double_well_potential(self, phi):
        """
        双阱势函数 W(φ) = (1/4)(φ² - 1)²。

        在相场理论中，双阱势保证了系统的两个稳定相态（固相 φ=+1 和
        液相 φ=-1），并在界面处产生能量势垒。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            双阱势值。
        """
        return 0.25 * (phi ** 2 - 1.0) ** 2

    def double_well_derivative(self, phi):
        """
        双阱势的导数 W'(φ) = φ(φ² - 1) = φ³ - φ。

        这是相场方程中的主要非线性驱动力，驱使系统向两相态演化。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            双阱势导数值。
        """
        return phi ** 3 - phi

    def interpolation_function(self, phi):
        """
        插值函数 h(φ) = (1/2)(1 + φ)，用于在固液相间插值物理性质。

        物理量 P 的相依赖值可表示为：
            P(φ) = P_solid * h(φ) + P_liquid * (1 - h(φ))

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            插值函数值，范围 [0, 1]。
        """
        return 0.5 * (1.0 + np.clip(phi, -1.0, 1.0))

    def laplacian_5point(self, field):
        """
        五点差分格式计算二维 Laplacian：
            ∇²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²

        采用齐次 Neumann 边界条件（零法向梯度），通过镜像延拓实现。

        Parameters
        ----------
        field : ndarray, shape (nx, ny)
            输入场。

        Returns
        -------
        ndarray, shape (nx, ny)
            Laplacian 值。
        """
        lap = np.zeros_like(field)

        # 内部点
        lap[1:-1, 1:-1] = (
            self.cx * (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) +
            self.cy * (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2])
        )

        # Neumann 边界：镜像延拓
        # x = 0 边界
        lap[0, 1:-1] = (
            self.cx * (field[1, 1:-1] - field[0, 1:-1]) +
            self.cy * (field[0, 2:] - 2.0 * field[0, 1:-1] + field[0, :-2])
        )
        # x = nx-1 边界
        lap[-1, 1:-1] = (
            self.cx * (field[-2, 1:-1] - field[-1, 1:-1]) +
            self.cy * (field[-1, 2:] - 2.0 * field[-1, 1:-1] + field[-1, :-2])
        )
        # y = 0 边界
        lap[1:-1, 0] = (
            self.cx * (field[2:, 0] - 2.0 * field[1:-1, 0] + field[:-2, 0]) +
            self.cy * (field[1:-1, 1] - field[1:-1, 0])
        )
        # y = ny-1 边界
        lap[1:-1, -1] = (
            self.cx * (field[2:, -1] - 2.0 * field[1:-1, -1] + field[:-2, -1]) +
            self.cy * (field[1:-1, -2] - field[1:-1, -1])
        )

        # 角点处理
        lap[0, 0] = self.cx * (field[1, 0] - field[0, 0]) + self.cy * (field[0, 1] - field[0, 0])
        lap[-1, 0] = self.cx * (field[-2, 0] - field[-1, 0]) + self.cy * (field[-1, 1] - field[-1, 0])
        lap[0, -1] = self.cx * (field[1, -1] - field[0, -1]) + self.cy * (field[0, -2] - field[0, -1])
        lap[-1, -1] = self.cx * (field[-2, -1] - field[-1, -1]) + self.cy * (field[-1, -2] - field[-1, -1])

        return lap

    def phase_field_rhs(self, phi, T, C, velocity_x=None, velocity_y=None):
        """
        计算相场方程的右端项（时间导数）。

        完整相场方程：
            τ ∂φ/∂t = ε²∇²φ - W'(φ) - λ_T(1 - φ²)²(T - T_M) - λ_C(1 - φ²)²(C - C_e)

        若给定速度场，则包含对流项：
            τ (∂φ/∂t + v·∇φ) = ...

        Parameters
        ----------
        phi : ndarray, shape (nx, ny)
            当前序参量场。
        T : ndarray, shape (nx, ny)
            温度场。
        C : ndarray, shape (nx, ny)
            浓度场。
        velocity_x, velocity_y : ndarray, optional
            速度分量场。

        Returns
        -------
        ndarray
            ∂φ/∂t。
        """
        # ============================================================
        # HOLE 1: 实现 Allen-Cahn 相场方程的右端项
        #
        # 需要完成以下物理量的计算：
        #   1. 扩散项: ε² ∇²φ （调用 self.laplacian_5point）
        #   2. 化学驱动力: -W'(φ) （调用 self.double_well_derivative）
        #   3. 热耦合项: -λ_T (1 - φ²)² (T - T_M)
        #   4. 溶质耦合项: -λ_C (1 - φ²)² (C - C_e)
        #   5. 总右端项除以弛豫时间 τ
        #   6. 若给定速度场，用一阶迎风格式减去对流项 v·∇φ
        #
        # 提示：注意 phi 的数值稳定性，必要时进行 clip
        # ============================================================
        raise NotImplementedError("HOLE 1: 请实现 phase_field_rhs 方法")

    def interface_energy_density(self, phi):
        """
        计算界面能量密度泛函：
            f_int = (ε²/2)|∇φ|² + W(φ)

        总界面能：E_int = ∫ f_int dΩ

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            界面能量密度。
        """
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * self.dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dy)

        grad_sq = grad_x ** 2 + grad_y ** 2
        potential = self.double_well_potential(phi)

        return 0.5 * self.epsilon ** 2 * grad_sq + potential

    def initialize_circular_nucleus(self, center_x, center_y, radius,
                                    solid_value=1.0, liquid_value=-1.0):
        """
        初始化圆形晶核：
            φ(r) = solid_value,  当 r ≤ radius
            φ(r) = liquid_value, 当 r > radius

        并在界面处采用 tanh 剖面光滑过渡：
            φ(r) = tanh((radius - r) / (√2 ε))

        Parameters
        ----------
        center_x, center_y : float
            晶核中心坐标（物理坐标）。
        radius : float
            晶核半径。
        solid_value, liquid_value : float
            固相和液相的序参量值。

        Returns
        -------
        ndarray
            初始序参量场。
        """
        x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y = np.linspace(0, (self.ny - 1) * self.dy, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # 采用平衡界面剖面：tanh((radius - r)/(sqrt(2)*ε))
        # 这满足一维稳态 Allen-Cahn 方程
        interface_width = np.sqrt(2.0) * self.epsilon
        phi = np.tanh((radius - r) / interface_width)

        return phi

    def compute_interface_normal(self, phi):
        """
        计算界面单位法向量：
            n = ∇φ / |∇φ|

        在界面附近 |∇φ| > 0 处定义良好。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        tuple of ndarray
            (n_x, n_y) 法向量分量。
        """
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * self.dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dy)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        # 避免除零
        grad_mag = np.maximum(grad_mag, 1e-12)

        n_x = grad_x / grad_mag
        n_y = grad_y / grad_mag

        return n_x, n_y

    def compute_curvature(self, phi):
        """
        计算界面平均曲率 κ = ∇·(∇φ/|∇φ|)。

        曲率在 Gibbs-Thomson 效应中起关键作用：
            T_interface = T_M - Γκ
        其中 Γ = σ/(ΔS) 为 Gibbs-Thomson 系数。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            界面曲率场。
        """
        n_x, n_y = self.compute_interface_normal(phi)

        # 计算 div(n)
        dn_x_dx = np.zeros_like(n_x)
        dn_y_dy = np.zeros_like(n_y)

        dn_x_dx[1:-1, :] = (n_x[2:, :] - n_x[:-2, :]) / (2.0 * self.dx)
        dn_y_dy[:, 1:-1] = (n_y[:, 2:] - n_y[:, :-2]) / (2.0 * self.dy)

        curvature = dn_x_dx + dn_y_dy

        # 仅在界面附近保留曲率值
        interface_mask = np.abs(phi) < 0.9
        curvature = curvature * interface_mask

        return curvature
