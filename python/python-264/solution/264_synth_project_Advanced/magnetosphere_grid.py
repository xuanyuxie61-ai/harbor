# -*- coding: utf-8 -*-
"""
magnetosphere_grid.py
=====================

磁层相空间网格构造模块.

构建 (L, E) 二维相空间网格, 其中:
  - L: McIlwain L 参数, 表征磁壳径向位置 (无量纲, 以 R_E 为单位)
  - E: 电子动能 (MeV), 或等价地用洛伦兹因子 gamma

网格类型:
  - 均匀网格: 用于高阶有限差分格式
  - 自适应对数网格: 在近磁顶区域加密
  - 高斯-洛巴托网格: 用于谱方法兼容性

核心物理背景:
  地球辐射带 (Van Allen 带) 由内带 (L ~ 1.2-2) 和外带 (L ~ 3-7) 组成,
  其间存在槽区 (slot region, L ~ 2-3). 粒子分布函数 f(L, E, t) 的
  演化由扩散方程控制:

    df/dt = (1/G) * d/dL (G * D_LL * df/dL) + S - L_loss

  其中 G 为度量因子, D_LL 为径向扩散系数, S 为源项, L_loss 为损失项.

本模块还包含损失锥投掷角的计算:
  sin^2(alpha_LC) = B_eq / B_atm ~ 1/L^3 (偶极近似)

参考文献:
  [1] Roederer, J.G. & Tian, Q., "Dynamics of Magnetically Trapped Particles" (2016)
  [2] Bourdarie, S. & Maget, V., "Numerical modeling of radiation belts", AGU (2020)
"""

import numpy as np
import physical_constants as pc


class MagnetosphereGrid:
    """
    磁层 (L, E) 相空间网格.

    属性
    ----
    L : ndarray, shape (n_L,)
        McIlwain L 参数数组
    E_MeV : ndarray, shape (n_E,)
        电子动能数组 (MeV)
    gamma : ndarray, shape (n_E,)
        对应的洛伦兹因子
    p : ndarray, shape (n_E,)
        对应的相对论动量 [kg*m/s]
    B_eq : ndarray, shape (n_L,)
        赤道面磁场强度 [T]
    alpha_lc : ndarray, shape (n_L,)
        损失锥投掷角 [rad]
    dL : float
        L 方向网格间距 (均匀网格)
    dE : ndarray, shape (n_E,)
        E 方向网格间距 (可非均匀)
    metric_G : ndarray, shape (n_L,)
        扩散方程度量因子 G(L)

    参数
    ----
    L_min, L_max : float
        L 参数范围
    n_L : int
        L 方向网格点数
    E_min_MeV, E_max_MeV : float
        动能范围 (MeV)
    n_E : int
        能量方向网格点数
    grid_type : str
        网格类型: 'uniform', 'log', 'gauss_lobatto'
    """

    def __init__(self,
                 L_min=None, L_max=None, n_L=None,
                 E_min_MeV=None, E_max_MeV=None, n_E=None,
                 grid_type='uniform'):
        # 使用默认参数
        L_min = pc.L_MIN if L_min is None else L_min
        L_max = pc.L_MAX if L_max is None else L_max
        n_L = pc.L_SHELL_NUM if n_L is None else n_L
        E_min_MeV = pc.ENERGY_MIN_MeV if E_min_MeV is None else E_min_MeV
        E_max_MeV = pc.ENERGY_MAX_MeV if E_max_MeV is None else E_max_MeV
        n_E = pc.ENERGY_NUM if n_E is None else n_E

        self.L_min = L_min
        self.L_max = L_max
        self.n_L = n_L
        self.E_min_MeV = E_min_MeV
        self.E_max_MeV = E_max_MeV
        self.n_E = n_E
        self.grid_type = grid_type

        # 构造 L 网格
        self.L = self._build_L_grid()
        self.dL = self.L[1] - self.L[0] if self.n_L > 1 else 1.0

        # 构造 E 网格
        self.E_MeV = self._build_E_grid()
        self.gamma = pc.kinetic_to_lorentz(self.E_MeV)
        self.p = pc.momentum_from_gamma(self.gamma)

        # 能量网格间距
        self.dE = np.diff(self.E_MeV)
        self.dE_avg = np.mean(self.dE) if self.n_E > 1 else 1.0

        # 物理量: 赤道磁场、损失锥
        self.B_eq = pc.dipole_field_magnitude(self.L)
        self.alpha_lc = self._compute_loss_cone()

        # 度量因子 G(L)
        self.metric_G = self._compute_metric()

        # 预计算差分算子所需的网格数据
        self._precompute_stencils()

    # -----------------------------------------------------------------
    #  网格构造
    # -----------------------------------------------------------------
    def _build_L_grid(self):
        """构造 L 方向网格."""
        if self.grid_type == 'uniform':
            return np.linspace(self.L_min, self.L_max, self.n_L)
        elif self.grid_type == 'log':
            # 对数网格: 在近磁顶区域加密
            log_L = np.linspace(np.log(self.L_min), np.log(self.L_max), self.n_L)
            return np.exp(log_L)
        elif self.grid_type == 'gauss_lobatto':
            # 高斯-洛巴托节点 (边界加密)
            # x_i = -cos(pi*i/(N-1)), 映射到 [L_min, L_max]
            i = np.arange(self.n_L)
            x = -np.cos(np.pi * i / (self.n_L - 1))
            return 0.5 * (self.L_max - self.L_min) * (x + 1.0) + self.L_min
        else:
            raise ValueError(f"未知网格类型: {self.grid_type}")

    def _build_E_grid(self):
        """构造 E 方向网格."""
        # 默认使用对数-线性混合网格
        # 低能区 (< 0.5 MeV): 线性, 高能区: 对数
        if self.E_min_MeV < 0.5 and self.E_max_MeV > 0.5:
            n_low = self.n_E // 3
            n_high = self.n_E - n_low
            E_low = np.linspace(self.E_min_MeV, 0.5, n_low, endpoint=False)
            E_high = np.logspace(np.log10(0.5), np.log10(self.E_max_MeV), n_high)
            return np.concatenate([E_low, E_high])
        else:
            return np.linspace(self.E_min_MeV, self.E_max_MeV, self.n_E)

    # -----------------------------------------------------------------
    #  物理量计算
    # -----------------------------------------------------------------
    def _compute_loss_cone(self):
        """
        计算每个 L 壳层的损失锥投掷角.

        物理公式 (偶极场近似):
          sin^2(alpha_LC) = B_eq(L) / B_atm
          B_eq(L) = B_0 / L^3

        其中 B_atm 为大气层顶 (~100 km) 的磁场强度, 取 5e-5 T.

        当 L 很大时, alpha_LC 接近 90 度 (捕获区宽);
        当 L 接近 1 时, alpha_LC 很小 (大部分粒子进入大气损失).
        """
        B_atm = pc.B_ATMOSPHERE
        sin2_alpha = self.B_eq / B_atm
        # 截断到 [0, 1] 防止数值异常
        sin2_alpha = np.clip(sin2_alpha, 0.0, 1.0)
        return np.arcsin(np.sqrt(sin2_alpha))

    def _compute_metric(self):
        """
        计算扩散方程度量因子 G(L).

        在偶极场近似下, 径向扩散方程的度量因子为:
          G(L) = L^4 * sqrt(L)  (来自通量管体积积分)

        更精确地 (Hamlin et al. 1961):
          G(L) = L^4 * I(L)

        其中 I(L) 为沿磁力线的积分:
          I(L) = integral_{-B_m}^{B_m} (1 - B/B_m)^{-1/2} / B * dl

        这里使用简化的偶极场近似 G(L) = L^4.
        """
        return self.L**4

    def _precompute_stencils(self):
        """预计算差分算子所需的网格坐标."""
        # L 方向的节点坐标 (内部节点)
        self.L_interior = self.L[1:-1]
        # L 方向的半节点 (用于通量计算)
        if self.n_L > 1:
            self.L_half = 0.5 * (self.L[:-1] + self.L[1:])
        else:
            self.L_half = np.array([self.L[0]])

        # 度量因子在半节点处的值
        self.metric_G_half = self.L_half**4

        # 赤道磁场在半节点处的值
        self.B_eq_half = pc.dipole_field_magnitude(self.L_half)

    # -----------------------------------------------------------------
    #  扩散系数计算
    # -----------------------------------------------------------------
    def radial_diffusion_coefficient(self, L=None, Kp=4):
        """
        计算径向扩散系数 D_LL (经验公式).

        采用 Brautigam & Albert (2000) 经验公式:
          D_LL = 10^{a*Kp + b} * L^{c}

        其中 a, b, c 为拟合系数, Kp 为行星际磁场活动指数.

        简化形式 (O'Brien et al. 2014):
          log10(D_LL) = 0.506*Kp - 9.325 + 10*L_factor

        参数
        ----
        L : ndarray, optional
            L 参数数组, 默认使用网格 L
        Kp : float
            Kp 指数 (0-9), 默认 4 (中等活动)

        返回
        -------
        D_LL : ndarray
            径向扩散系数 [1/s] (以 L 参数为自变量)
        """
        if L is None:
            L = self.L
        L = np.asarray(L, dtype=np.float64)

        # Brautigam & Albert (2000) 参数化
        # log10(D_LL) = a0 + a1*Kp + a2*L
        # D_LL 单位: 1/s, L 无量纲
        a0 = -9.325
        a1 = 0.506
        a2 = 0.0  # L 依赖由 L^10 处理

        # 简化: D_LL ~ 10^(a0 + a1*Kp) * L^10
        # 这是强 L 依赖的扩散, 符合观测
        base_coeff = 10.0**(a0 + a1 * Kp)
        L_factor = L**10

        # 防止数值溢出: 截断大 L 处的扩散系数
        D_LL = base_coeff * L_factor
        D_LL = np.clip(D_LL, 0.0, 1.0e3)

        return D_LL

    def energy_diffusion_coefficient(self, E_MeV=None, L=None, wave_mode='chorus'):
        """
        计算能量扩散系数 D_EE (准线性理论).

        对于 chorus 波, 准线性扩散系数为:
          D_EE = (pi/4) * Omega_ce^2 * (B_w/B_0)^2 * E * g(alpha_eq)

        其中 B_w 为波振幅, g(alpha_eq) 为投掷角散射函数.

        参数
        ----
        E_MeV : ndarray, optional
            动能数组 (MeV)
        L : ndarray, optional
            L 参数数组
        wave_mode : str
            波模式: 'chorus', 'hiss', 'emice'

        返回
        -------
        D_EE : ndarray, shape (n_L, n_E)
            能量扩散系数 [MeV^2/s]
        """
        if E_MeV is None:
            E_MeV = self.E_MeV
        if L is None:
            L = self.L

        E_MeV = np.asarray(E_MeV)
        L = np.asarray(L)

        # 网格外积
        L2d, E2d = np.meshgrid(L, E_MeV, indexing='ij')
        gamma_2d = pc.kinetic_to_lorentz(E2d)

        if wave_mode == 'chorus':
            B_wave = pc.B_CHORUS_AMPLITUDE
            f_res = 0.2 * pc.OMEGA_CE_REF / (2.0 * np.pi)
        elif wave_mode == 'hiss':
            B_wave = 20.0e-12  # 20 pT
            f_res = 0.5 * pc.OMEGA_CE_REF / (2.0 * np.pi)
        elif wave_mode == 'emice':
            B_wave = 50.0e-12  # 50 pT
            f_res = 0.3 * pc.OMEGA_CE_REF / (2.0 * np.pi)
        else:
            raise ValueError(f"未知波模式: {wave_mode}")

        # 局部磁场 (赤道)
        B_local = pc.dipole_field_magnitude(L2d)
        Omega_ce = pc.gyrofrequency(B_local)

        # 简化准线性扩散系数
        # D_EE ~ (B_w / B_0)^2 * Omega_ce * E_kin / gamma
        ratio = (B_wave / np.maximum(B_local, pc.EPSILON_NUM))**2
        D_EE = ratio * Omega_ce * E2d / np.maximum(gamma_2d, 1.0)

        return D_EE

    # -----------------------------------------------------------------
    #  网格质量检查
    # -----------------------------------------------------------------
    def check_grid_quality(self):
        """
        检查网格质量, 返回诊断信息.

        检查项目:
          - L 网格单调性
          - 能量网格单调性
          - 最小/最大网格间距
          - 网格长宽比
        """
        dL_arr = np.diff(self.L)
        dE_arr = np.diff(self.E_MeV)

        info = {
            'L_monotone': np.all(dL_arr > 0),
            'E_monotone': np.all(dE_arr > 0),
            'L_min_spacing': float(np.min(dL_arr)) if len(dL_arr) > 0 else 0.0,
            'L_max_spacing': float(np.max(dL_arr)) if len(dL_arr) > 0 else 0.0,
            'E_min_spacing': float(np.min(dE_arr)) if len(dE_arr) > 0 else 0.0,
            'E_max_spacing': float(np.max(dE_arr)) if len(dE_arr) > 0 else 0.0,
            'aspect_ratio_max': float(np.max(dL_arr) / max(np.min(dE_arr), pc.EPSILON_NUM)),
            'n_total': self.n_L * self.n_E,
        }
        return info

    def summary(self):
        """打印网格摘要信息."""
        info = self.check_grid_quality()
        print(f"磁层相空间网格:")
        print(f"  L 范围: [{self.L_min:.2f}, {self.L_max:.2f}], n_L = {self.n_L}")
        print(f"  E 范围: [{self.E_min_MeV:.2f}, {self.E_max_MeV:.2f}] MeV, n_E = {self.n_E}")
        print(f"  网格类型: {self.grid_type}")
        print(f"  总网格点数: {info['n_total']}")
        print(f"  L 间距: [{info['L_min_spacing']:.4e}, {info['L_max_spacing']:.4e}]")
        print(f"  E 间距: [{info['E_min_spacing']:.4e}, {info['E_max_spacing']:.4e}]")
        print(f"  网格长宽比 (最大): {info['aspect_ratio_max']:.2f}")
        print(f"  损失锥范围: [{np.degrees(self.alpha_lc[0]):.2f}, {np.degrees(self.alpha_lc[-1]):.2f}] deg")


if __name__ == "__main__":
    grid = MagnetosphereGrid()
    grid.summary()

    D_LL = grid.radial_diffusion_coefficient(Kp=4)
    print(f"\nD_LL (L=4, Kp=4): {D_LL[grid.n_L//2]:.4e}")

    D_EE = grid.energy_diffusion_coefficient(wave_mode='chorus')
    print(f"D_EE 形状: {D_EE.shape}, 最大值: {np.max(D_EE):.4e}")
