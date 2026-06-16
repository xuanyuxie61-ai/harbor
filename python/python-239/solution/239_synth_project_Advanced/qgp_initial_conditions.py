"""
qgp_initial_conditions.py — QGP 初始条件: Woods-Saxon 核叠加与流形混合
=========================================================================

融合种子项目:
  - 1104_vikasverma1077_manifold_mixup (流形混合正则化)
  - 1167_NikeHop_UniversalPolicies (条件扩散模型)

本模块生成 QGP 流体动力学模拟的初始条件.
物理模型: 两个相对论性重核 (Au 或 Pb) 在超相对论碰撞中
          产生高温高密物质 (QGP), 初始能量密度分布由核几何决定.

Woods-Saxon 核密度分布:
    rho_A(r) = rho_0 / (1 + exp((r - R_A) / a_A))

其中 R_A 为核半径, a_A 为表面弥散参数.

碰撞核叠加 (Glauber 模型):
    T_A(b, s) = integral rho_A(sqrt(s^2 + z^2)) dz  (厚度函数)
    N_coll(b, x, y) = T_A(x - b/2, y) * T_B(x + b/2, y) * sigma_NN
    N_part(b, x, y) = T_A * (1 - exp(-sigma * T_B)) + T_B * (1 - exp(-sigma * T_A))

初始能量密度 (Glauber-Glasma 混合):
    e(x, y, tau_0) = (1-f) * N_part/N_part_max * e_max
                    + f * N_coll/N_coll_max * e_max
    其中 f ~ 0.15 为硬散射分量比例

流形混合 (源自 1104_manifold_mixup):
    将两种不同的初始条件模型 (Woods-Saxon 和 MC-KLN) 在流形上
    进行线性混合, 生成连续的初始条件参数空间:
        IC_mixed = lambda * IC_WS + (1 - lambda) * IC_MCKLN
    混合参数 lambda 在 MCMC 参数估计中被约束.

条件扩散生成 (源自 1167_UniversalPolicies):
    使用 EDM (Elucidating Diffusion Models) 框架生成
    涨落初始条件, 条件于碰撞几何参数 (b, sqrt(s)):
        dx/dt = -sigma'(t)/sigma(t) * x + sigma(t) * score(x, t)
    其中 score = nabla_x log p(x|condition)
"""

import numpy as np
from qgp_config import (
    NumericalParams, InitialConditionParams, SIGMA_SB
)
from qgp_grid import QGPGrid
from qgp_eos import QGPEquationOfState


def woods_saxon_density(r: np.ndarray, R: float, a: float) -> np.ndarray:
    """
    Woods-Saxon 核密度分布

    rho(r) = rho_0 / (1 + exp((r - R) / a))

    物理: 描述原子核中核子的空间分布
        R: 核半径 (~ 1.2 * A^{1/3} fm)
        a: 表面弥散参数 (~ 0.5 fm)

    对于 Au 核 (A=197):
        R_Au ~ 6.38 fm
        a_Au ~ 0.535 fm

    归一化: integral rho(r) d^3r = A (核子数)

    Args:
        r: 径向距离 (fm)
        R: 核半径参数 (fm)
        a: 弥散参数 (fm)

    Returns:
        归一化核密度 rho(r) / rho_0
    """
    return 1.0 / (1.0 + np.exp((r - R) / np.maximum(a, 1.0e-10)))


def nuclear_thickness(x: np.ndarray, y: np.ndarray,
                       R: float, a: float, nz: int = 50) -> np.ndarray:
    """
    核厚度函数 T(x, y) = integral_{-inf}^{inf} rho(sqrt(x^2+y^2+z^2)) dz

    数值积分采用 Gauss-Legendre 求积:
        T(x, y) ≈ sum_{k=1}^{nz} w_k * rho(sqrt(x^2 + y^2 + z_k^2))

    其中 z_k, w_k 为 Gauss-Legendre 节点和权重,
    积分范围 [-z_max, z_max], z_max ~ 2*R + 10*a.

    物理: T(x,y) 正比于穿过核的核子面密度,
    是 Glauber 模型中的基本量.

    Args:
        x, y: 横向坐标 (fm)
        R: 核半径 (fm)
        a: 弥散参数 (fm)
        nz: 纵向积分点数

    Returns:
        核厚度函数 T(x, y)
    """
    z_max = 2.0 * R + 10.0 * a
    # Gauss-Legendre 节点和权重
    z_nodes, weights = np.polynomial.legendre.leggauss(nz)
    z_nodes = z_nodes * z_max  # 映射到 [-z_max, z_max]
    weights = weights * z_max

    T = np.zeros_like(x)
    for k in range(nz):
        r = np.sqrt(x**2 + y**2 + z_nodes[k]**2)
        T += weights[k] * woods_saxon_density(r, R, a)
    return T


class QGPInitialConditions:
    """
    QGP 初始条件生成器

    方法:
    1. Glauber 模型: 核叠加产生初始能量密度
    2. 流形混合: 两种模型的凸组合 (源自 manifold_mixup)
    3. 扩散生成: 条件扩散模型添加事件-by-事件涨落 (源自 EDM)
    """

    def __init__(self, grid: QGPGrid, eos: QGPEquationOfState):
        """
        Args:
            grid: 计算网格
            eos: 状态方程
        """
        self.grid = grid
        self.eos = eos
        self.ic_params = InitialConditionParams

        # Woods-Saxon 参数
        self.R_A = self.ic_params.RADIUS_AU
        self.a_A = self.ic_params.DIFFUSENESS_AU

    def glauber_model(self, impact_parameter: float,
                       mixing_fraction: float = 0.15) -> np.ndarray:
        """
        Glauber 模型初始能量密度

        e(x, y) = (1-f) * N_part_norm + f * N_coll_norm

        其中:
            N_part = T_A(x-b/2) * [1-exp(-sigma*T_B(x+b/2))]
                   + T_B(x+b/2) * [1-exp(-sigma*T_A(x-b/2))]
            N_coll = T_A(x-b/2) * T_B(x+b/2) * sigma_NN

        核-核非弹性截面: sigma_NN ~ 42 mb (sqrt(s)=200 GeV)
        转换为 fm^2: sigma_NN ~ 4.2 fm^2

        Args:
            impact_parameter: 碰撞参数 b (fm)
            mixing_fraction: 硬散射混合比例 f

        Returns:
            初始能量密度 e(x, y) (GeV/fm^3)
        """
        ng = self.grid.ng
        nx, ny = self.grid.nx, self.grid.ny
        X = self.grid.X[ng:ng+ny, ng:ng+nx]
        Y = self.grid.Y[ng:ng+ny, ng:ng+nx]

        # 核 A 中心在 (-b/2, 0), 核 B 在 (+b/2, 0)
        sigma_NN = 4.2  # fm^2 (200 GeV Au+Au)

        T_A = nuclear_thickness(X + impact_parameter/2.0, Y,
                                 self.R_A, self.a_A)
        T_B = nuclear_thickness(X - impact_parameter/2.0, Y,
                                 self.R_A, self.a_A)

        # 参与者核子数密度
        prob_A = 1.0 - np.exp(-sigma_NN * T_B)
        prob_B = 1.0 - np.exp(-sigma_NN * T_A)
        N_part = T_A * prob_A + T_B * prob_B

        # 碰撞核子数密度
        N_coll = T_A * T_B * sigma_NN

        # 归一化
        N_part_max = np.max(N_part) + 1.0e-15
        N_coll_max = np.max(N_coll) + 1.0e-15

        N_part_norm = N_part / N_part_max
        N_coll_norm = N_coll / N_coll_max

        # 混合初始能量密度
        e_profile = (1.0 - mixing_fraction) * N_part_norm + \
                    mixing_fraction * N_coll_norm

        # 标度到物理能量密度
        e_max = SIGMA_SB * self.ic_params.T_INITIAL_MAX**4
        e_init = e_max * e_profile

        return np.maximum(e_init, NumericalParams.ENERGY_FLOOR)

    def mckln_model(self, impact_parameter: float) -> np.ndarray:
        """
        MC-KLN 模型初始条件 (简化版本)

        基于参与者和碰撞者的偏心度, 添加方位角调制:
            e(r, phi) = e_0 * (1 + 2*v2*cos(2*phi) + 2*v3*cos(3*phi))
            * exp(-r^2 / (2*R_G^2))

        其中:
            v2 = 空间偏心度 epsilon_2
            epsilon_2 = <y^2 - x^2> / <y^2 + x^2>
            R_G = 高斯宽度参数

        此模型产生与 Glauber 不同的初始条件形状,
        用于流形混合.

        Args:
            impact_parameter: 碰撞参数 b (fm)

        Returns:
            初始能量密度
        """
        ng = self.grid.ng
        nx, ny = self.grid.nx, self.ny
        X = self.grid.X_int
        Y = self.grid.Y_int
        R = self.grid.R_int
        PHI = np.arctan2(Y, X)

        # 空间偏心度 (与碰撞参数相关)
        b_max = 2.0 * self.R_A  # 最大碰撞参数
        epsilon_2 = 0.5 * (impact_parameter / b_max)**2
        epsilon_2 = min(epsilon_2, 0.3)

        # 三角形偏心度 (涨落驱动)
        epsilon_3 = 0.05

        # 高斯包络
        R_G = self.ic_params.SMOOTHING_SIGMA * 3.0
        envelope = np.exp(-R**2 / (2.0 * R_G**2))

        # 方位角调制
        angular_mod = 1.0 + 2.0 * epsilon_2 * np.cos(2.0 * PHI) + \
                      2.0 * epsilon_3 * np.cos(3.0 * PHI)

        e_max = SIGMA_SB * self.ic_params.T_INITIAL_MAX**4
        e_init = e_max * envelope * angular_mod

        return np.maximum(e_init, NumericalParams.ENERGY_FLOOR)

    def manifold_mix(self, impact_parameter: float,
                      mix_lambda: float = 0.5) -> np.ndarray:
        """
        流形混合初始条件 (源自 1104_manifold_mixup)

        IC_mixed = lambda * IC_Glauber + (1-lambda) * IC_MCKLN

        流形混合的物理动机:
        - Glauber 模型: 强调核子的几何叠加
        - MC-KLN 模型: 强调参与者的偏心度
        - 真实初始条件介于两者之间
        - lambda 是待确定的物理参数

        数学性质:
        - lambda=1: 纯 Glauber
        - lambda=0: 纯 MC-KLN
        - 0<lambda<1: 凸组合, 保证正定性

        Args:
            impact_parameter: 碰撞参数 b (fm)
            mix_lambda: 混合参数 (0 到 1)

        Returns:
            混合后的初始能量密度
        """
        lambda_clipped = np.clip(mix_lambda, 0.0, 1.0)

        e_glauber = self.glauber_model(impact_parameter)
        e_mckln = self.mckln_model(impact_parameter)

        e_mixed = lambda_clipped * e_glauber + (1.0 - lambda_clipped) * e_mckln

        return np.maximum(e_mixed, NumericalParams.ENERGY_FLOOR)

    def add_diffusion_fluctuations(self, e_base: np.ndarray,
                                     seed: int = 42) -> np.ndarray:
        """
        使用扩散模型添加事件-by-事件涨落 (源自 1167_UniversalPolicies)

        EDM (Elucidating Diffusion Models) 简化版:
        对基底初始条件添加相关涨落:
            e_fluct = e_base + sigma * xi_correlated

        其中 xi_correlated 为空间相关的高斯噪声:
            xi(x, y) = sum_k a_k * exp(-|x-x_k|^2 / (2*l_c^2))

        相关长度 l_c ~ 0.5 fm (部分子尺度)

        扩散过程的 Heun 采样 (二阶 ODE 求解器):
            从纯噪声出发, 逐步去噪:
            x_{t+1} = x_t + (x'_t + x'_{t+1}) * dt / 2

        Args:
            e_base: 基底初始能量密度 (ny, nx)
            seed: 随机种子 (用于可重复性)

        Returns:
            含涨落的初始能量密度
        """
        rng = np.random.RandomState(seed)
        ny, nx = e_base.shape
        l_c = 0.5  # 相关长度 (fm)

        # 生成空间相关的涨落场
        # 方法: 白噪声通过高斯滤波器
        xi_white = rng.randn(ny, nx)

        # 高斯滤波 (卷积实现相关噪声)
        sigma_filter = l_c / min(self.grid.dx, self.grid.dy)
        sigma_filter = max(sigma_filter, 1.0)

        # 使用傅里叶空间滤波
        kx = np.fft.fftfreq(nx, d=self.grid.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.grid.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2

        # 高斯滤波核: exp(-l_c^2 * k^2 / 2)
        filter_kernel = np.exp(-l_c**2 * K2 / 2.0)
        xi_filtered = np.real(np.fft.ifft2(
            np.fft.fft2(xi_white) * filter_kernel
        ))

        # 归一化
        xi_std = np.std(xi_filtered)
        if xi_std > 1.0e-10:
            xi_filtered /= xi_std

        # 涨落幅度: 正比于 e_base 的平方根 (Poisson 型涨落)
        fluctuation_amplitude = 0.1 * np.sqrt(e_base + 1.0e-10)
        e_fluct = e_base + fluctuation_amplitude * xi_filtered

        return np.maximum(e_fluct, NumericalParams.ENERGY_FLOOR)

    def generate_full_initial_state(self, impact_parameter: float,
                                     mix_lambda: float = 0.5,
                                     add_fluctuations: bool = True,
                                     seed: int = 42) -> tuple:
        """
        生成完整的初始状态

        流程:
        1. 流形混合生成基底能量密度
        2. (可选) 扩散模型添加涨落
        3. 高斯平滑
        4. 计算初始温度、流速、重子密度

        初始条件:
            vx = vy = 0  (Bjorken 初始无横向流)
            nB = small_uniform  (RHIC 能量下重子密度很低)
            T = T(e) 通过 EoS 反解

        Args:
            impact_parameter: 碰撞参数 (fm)
            mix_lambda: 混合参数
            add_fluctuations: 是否添加涨落
            seed: 随机种子

        Returns:
            (e, vx, vy, nB) 初始原始变量 (含鬼单元)
        """
        # 1. 生成内部区域的基底能量密度
        e_int = self.manifold_mix(impact_parameter, mix_lambda)

        # 2. 添加涨落
        if add_fluctuations:
            e_int = self.add_diffusion_fluctuations(e_int, seed)

        # 3. 高斯平滑 (移除亚网格涨落)
        e_int = self._gaussian_smooth(e_int,
                                       self.ic_params.SMOOTHING_SIGMA)

        # 4. 构建完整网格 (含鬼单元)
        ng = self.grid.ng
        e_full = np.zeros((self.grid.ny + 2*ng, self.grid.nx + 2*ng))
        e_full[ng:ng+self.grid.ny, ng:ng+self.grid.nx] = e_int
        e_full = self.grid.apply_boundary_conditions(e_full)

        # 5. 初始流速 (Bjorken 近似: 初始无横向流)
        vx = np.zeros_like(e_full) + self.ic_params.INIT_FLOW_VELOCITY
        vy = np.zeros_like(e_full) + self.ic_params.INIT_FLOW_VELOCITY

        # 6. 重子密度 (RHIC 下很小)
        nB = np.ones_like(e_full) * self.ic_params.BARYON_CHEM_POT

        return e_full, vx, vy, nB

    def _gaussian_smooth(self, field: np.ndarray,
                          sigma: float) -> np.ndarray:
        """
        高斯平滑 (傅里叶空间实现)

        G_sigma(x) = (2*pi*sigma^2)^{-1/2} * exp(-x^2 / (2*sigma^2))
        f_smooth = f * G_sigma (卷积)
        => f_smooth(k) = f(k) * exp(-sigma^2 * k^2 / 2) (傅里叶空间)

        Args:
            field: 二维场 (ny, nx)
            sigma: 平滑宽度 (fm)

        Returns:
            平滑后的场
        """
        ny, nx = field.shape
        kx = np.fft.fftfreq(nx, d=self.grid.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.grid.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        filter_kernel = np.exp(-sigma**2 * K2 / 2.0)
        smoothed = np.real(np.fft.ifft2(np.fft.fft2(field) * filter_kernel))
        return smoothed

    def compute_spatial_eccentricity(self, e: np.ndarray) -> dict:
        """
        计算初始空间偏心度

        epsilon_n = sqrt(<r^n * cos(n*phi)>^2 + <r^n * sin(n*phi)>^2)
                    / <r^n>

        其中 <...> 表示以能量密度为权的平均:
            <O> = integral O * e(x,y) dx dy / integral e(x,y) dx dy

        epsilon_2 (椭圆率): 驱动椭圆流 v_2
        epsilon_3 (三角形率): 由涨落驱动三角流 v_3

        Args:
            e: 能量密度 (内部区域)

        Returns:
            偏心度字典 {'eps2': float, 'eps3': float, 'psi2': float, 'psi3': float}
        """
        X = self.grid.X_int
        Y = self.grid.Y_int
        R = self.grid.R_int
        PHI = np.arctan2(Y, X)

        weight = e * R**2
        total_weight = np.sum(weight) + 1.0e-30

        # 二阶偏心度
        cos2 = np.sum(weight * np.cos(2.0 * PHI)) / total_weight
        sin2 = np.sum(weight * np.sin(2.0 * PHI)) / total_weight
        eps2 = np.sqrt(cos2**2 + sin2**2)
        psi2 = 0.5 * np.arctan2(sin2, cos2)

        # 三阶偏心度
        cos3 = np.sum(weight * np.cos(3.0 * PHI)) / total_weight
        sin3 = np.sum(weight * np.sin(3.0 * PHI)) / total_weight
        eps3 = np.sqrt(cos3**2 + sin3**2)
        psi3 = np.arctan2(sin3, cos3) / 3.0

        return {'eps2': eps2, 'eps3': eps3, 'psi2': psi2, 'psi3': psi3}
