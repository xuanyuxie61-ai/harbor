"""
thermal_transport.py
====================
热质传输方程求解模块

求解温度场和浓度场的对流-扩散方程：

温度场（能量守恒）：
    ∂T/∂t + v·∇T = α_T ∇²T + (L_f / c_p) ∂h/∂t

其中：
    α_T = k / (ρ c_p)     热扩散系数
    L_f                   凝固潜热
    h(φ) = (1+φ)/2        固相分数
    (L_f/c_p) ∂h/∂t       相变潜热源项

浓度场（溶质守恒）：
    ∂C/∂t + v·∇C = ∇·(D(φ)∇C) + k_p C(1 - k_p) ∂φ/∂t / (k_p + (1-k_p)h)

其中：
    D(φ) = D_solid * h(φ) + D_liquid * (1 - h(φ))
    k_p                   溶质分配系数（partition coefficient）

溶质过冷度（Gibbs-Thomson 修正）：
    ΔT_solute = -m_L * (C_l - C_e)
    其中 m_L 为液相线斜率，C_l 为界面液相浓度。
"""

import numpy as np


class ThermalTransportSolver:
    """
    热质传输方程求解器，采用有限差分法离散。
    """

    def __init__(self, nx, ny, dx, dy, dt,
                 thermal_diffusivity=1.0,
                 latent_heat=1.0,
                 specific_heat=1.0,
                 solute_diffusivity_solid=0.01,
                 solute_diffusivity_liquid=1.0,
                 partition_coefficient=0.3,
                 liquidus_slope=-1.0):
        """
        初始化热质传输求解器。

        Parameters
        ----------
        nx, ny : int
            网格点数。
        dx, dy : float
            空间步长。
        dt : float
            时间步长。
        thermal_diffusivity : float
            热扩散系数 α_T。
        latent_heat : float
            单位质量凝固潜热 L_f。
        specific_heat : float
            比热容 c_p。
        solute_diffusivity_solid : float
            固相中溶质扩散系数 D_s。
        solute_diffusivity_liquid : float
            液相中溶质扩散系数 D_l。
        partition_coefficient : float
            平衡分配系数 k_p。
        liquidus_slope : float
            液相线斜率 m_L（通常 < 0）。
        """
        if nx < 3 or ny < 3:
            raise ValueError("网格维度必须至少为 3")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("步长参数必须为正")
        if not (0 < partition_coefficient <= 1.0):
            raise ValueError("分配系数 k_p 必须在 (0, 1] 范围内")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.alpha_T = thermal_diffusivity
        self.latent_heat = latent_heat
        self.specific_heat = specific_heat
        self.D_s = solute_diffusivity_solid
        self.D_l = solute_diffusivity_liquid
        self.k_p = partition_coefficient
        self.m_L = liquidus_slope

        # 稳定性条件检查
        max_diff = max(thermal_diffusivity, solute_diffusivity_liquid)
        dt_diff_limit = 0.25 * min(dx ** 2, dy ** 2) / max_diff
        if dt > dt_diff_limit:
            # 不强制报错，但提醒
            pass

    def solid_fraction(self, phi):
        """
        计算固相分数 h(φ) = (1 + φ) / 2。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            固相分数，范围 [0, 1]。
        """
        return np.clip(0.5 * (1.0 + phi), 0.0, 1.0)

    def effective_diffusivity(self, phi):
        """
        计算与相场相关的有效扩散系数：
            D(φ) = D_solid * h(φ) + D_liquid * (1 - h(φ))

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            有效扩散系数场。
        """
        h = self.solid_fraction(phi)
        return self.D_s * h + self.D_l * (1.0 - h)

    def laplacian_with_variable_coeff(self, field, coeff):
        """
        计算变系数 Laplacian：∇·(coeff ∇field)。

        采用守恒型差分格式：
            ∇·(D∇C) ≈ [D_{i+1/2}(C_{i+1}-C_i) - D_{i-1/2}(C_i-C_{i-1})] / h²

        Parameters
        ----------
        field : ndarray
            待求导场。
        coeff : ndarray
            变系数场。

        Returns
        -------
        ndarray
            ∇·(coeff ∇field)。
        """
        result = np.zeros_like(field)

        # x 方向
        coeff_x_plus = np.zeros_like(coeff)
        coeff_x_minus = np.zeros_like(coeff)
        coeff_x_plus[1:-1, :] = 0.5 * (coeff[2:, :] + coeff[1:-1, :])
        coeff_x_minus[1:-1, :] = 0.5 * (coeff[1:-1, :] + coeff[:-2, :])

        result[1:-1, :] += (
            coeff_x_plus[1:-1, :] * (field[2:, :] - field[1:-1, :]) -
            coeff_x_minus[1:-1, :] * (field[1:-1, :] - field[:-2, :])
        ) / (self.dx ** 2)

        # y 方向
        coeff_y_plus = np.zeros_like(coeff)
        coeff_y_minus = np.zeros_like(coeff)
        coeff_y_plus[:, 1:-1] = 0.5 * (coeff[:, 2:] + coeff[:, 1:-1])
        coeff_y_minus[:, 1:-1] = 0.5 * (coeff[:, 1:-1] + coeff[:, :-2])

        result[:, 1:-1] += (
            coeff_y_plus[:, 1:-1] * (field[:, 2:] - field[:, 1:-1]) -
            coeff_y_minus[:, 1:-1] * (field[:, 1:-1] - field[:, :-2])
        ) / (self.dy ** 2)

        # 边界处理（Neumann：零通量）
        result[0, :] = result[1, :]
        result[-1, :] = result[-2, :]
        result[:, 0] = result[:, 1]
        result[:, -1] = result[:, -2]

        return result

    def convection_term(self, field, vx, vy):
        """
        计算对流项 v·∇field，采用一阶迎风格式。

        Parameters
        ----------
        field : ndarray
            被对流的场。
        vx, vy : ndarray
            速度分量。

        Returns
        -------
        ndarray
            对流项 v·∇field。
        """
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)

        # x 方向迎风格式
        mask_pos = vx >= 0
        grad_x[1:-1, :][mask_pos[1:-1, :]] = (
            field[1:-1, :][mask_pos[1:-1, :]] - field[:-2, :][mask_pos[1:-1, :]]
        ) / self.dx
        mask_neg = vx < 0
        grad_x[1:-1, :][mask_neg[1:-1, :]] = (
            field[2:, :][mask_neg[1:-1, :]] - field[1:-1, :][mask_neg[1:-1, :]]
        ) / self.dx

        # y 方向迎风格式
        mask_pos = vy >= 0
        grad_y[:, 1:-1][mask_pos[:, 1:-1]] = (
            field[:, 1:-1][mask_pos[:, 1:-1]] - field[:, :-2][mask_pos[:, 1:-1]]
        ) / self.dy
        mask_neg = vy < 0
        grad_y[:, 1:-1][mask_neg[:, 1:-1]] = (
            field[:, 2:][mask_neg[:, 1:-1]] - field[:, 1:-1][mask_neg[:, 1:-1]]
        ) / self.dy

        return vx * grad_x + vy * grad_y

    def latent_heat_source(self, phi, phi_old):
        """
        计算相变潜热源项：
            Q_latent = (L_f / c_p) * (h(φ) - h(φ_old)) / Δt

        Parameters
        ----------
        phi : ndarray
            新时刻序参量场。
        phi_old : ndarray
            旧时刻序参量场。

        Returns
        -------
        ndarray
            潜热源项。
        """
        h_new = self.solid_fraction(phi)
        h_old = self.solid_fraction(phi_old)
        return (self.latent_heat / self.specific_heat) * (h_new - h_old) / self.dt

    def solute_rejection_source(self, phi, phi_old, C):
        """
        计算溶质排出源项（solute rejection）：

        在凝固过程中，溶质被排斥到液相：
            Q_C = C * (1 - k_p) * (∂h/∂t) / (k_p + (1-k_p)h)

        Parameters
        ----------
        phi : ndarray
            新时刻序参量场。
        phi_old : ndarray
            旧时刻序参量场。
        C : ndarray
            浓度场。

        Returns
        -------
        ndarray
            溶质源项。
        """
        h_new = self.solid_fraction(phi)
        h_old = self.solid_fraction(phi_old)
        dh_dt = (h_new - h_old) / self.dt

        denom = self.k_p + (1.0 - self.k_p) * h_new
        denom = np.maximum(denom, 1e-12)  # 避免除零

        return C * (1.0 - self.k_p) * dh_dt / denom

    def temperature_rhs(self, T, phi, phi_old, vx, vy):
        """
        计算温度场方程的右端项：
            ∂T/∂t = α_T ∇²T - v·∇T + Q_latent

        Parameters
        ----------
        T : ndarray
            当前温度场。
        phi, phi_old : ndarray
            当前和上一时刻的序参量场。
        vx, vy : ndarray
            速度场。

        Returns
        -------
        ndarray
            ∂T/∂t。
        """
        # 热扩散项
        lap_T = np.zeros_like(T)
        lap_T[1:-1, 1:-1] = (
            (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dx ** 2) +
            (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dy ** 2)
        )
        diffusion = self.alpha_T * lap_T

        # 对流项
        convection = self.convection_term(T, vx, vy)

        # 潜热源
        q_latent = self.latent_heat_source(phi, phi_old)

        return diffusion - convection + q_latent

    def concentration_rhs(self, C, phi, phi_old, vx, vy):
        """
        计算浓度场方程的右端项：
            ∂C/∂t = ∇·(D(φ)∇C) - v·∇C + Q_C

        Parameters
        ----------
        C : ndarray
            当前浓度场。
        phi, phi_old : ndarray
            当前和上一时刻序参量场。
        vx, vy : ndarray
            速度场。

        Returns
        -------
        ndarray
            ∂C/∂t。
        """
        # ============================================================
        # HOLE 2: 实现浓度场方程的右端项
        #
        # 需要完成以下物理量的计算：
        #   1. 变系数扩散项: ∇·(D(φ) ∇C)
        #      （调用 self.effective_diffusivity 获取 D_eff，
        #       再调用 self.laplacian_with_variable_coeff）
        #   2. 对流项: -v·∇C （调用 self.convection_term）
        #   3. 溶质排出源项 Q_C （调用 self.solute_rejection_source）
        #   4. 返回 diffusion - convection + q_solute
        #
        # 物理意义：凝固过程中溶质被排斥到液相，需正确计算
        # 相场依赖的有效扩散系数和溶质源项
        # ============================================================
        raise NotImplementedError("HOLE 2: 请实现 concentration_rhs 方法")

    def compute_thermal_undercooling(self, T, T_m):
        """
        计算热过冷度：
            ΔT_thermal = T_M - T

        Parameters
        ----------
        T : ndarray
            温度场。
        T_m : float
            熔点温度。

        Returns
        -------
        ndarray
            热过冷度场。
        """
        return T_m - T

    def compute_solutal_undercooling(self, C, C_e):
        """
        计算溶质过冷度：
            ΔT_solutal = -m_L * (C - C_e)

        Parameters
        ----------
        C : ndarray
            浓度场。
        C_e : float
            平衡浓度。

        Returns
        -------
        ndarray
            溶质过冷度场。
        """
        return -self.m_L * (C - C_e)

    def compute_total_undercooling(self, T, C, T_m, C_e, gamma, curvature):
        """
        计算总过冷度（Gibbs-Thomson 关系）：
            ΔT_total = ΔT_thermal + ΔT_solutal - Γ κ

        其中 Γ = γ / ΔS_f 为 Gibbs-Thomson 系数，
        κ 为界面曲率。

        Parameters
        ----------
        T : ndarray
            温度场。
        C : ndarray
            浓度场。
        T_m : float
            熔点温度。
        C_e : float
            平衡浓度。
        gamma : float
            界面能 γ。
        curvature : ndarray
            界面曲率场。

        Returns
        -------
        ndarray
            总过冷度场。
        """
        dT_thermal = self.compute_thermal_undercooling(T, T_m)
        dT_solutal = self.compute_solutal_undercooling(C, C_e)
        return dT_thermal + dT_solutal - gamma * curvature
