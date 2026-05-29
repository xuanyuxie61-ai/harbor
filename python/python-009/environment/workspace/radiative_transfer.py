"""
radiative_transfer.py
行星大气辐射传输求解模块。

融合原始项目：384_fem1d_adaptive（自适应有限元求解边值问题）
             577_image_diffuse（扩散方程数值平滑与稳定化）

本模块求解平面平行大气中的辐射传输方程：
    μ dI(τ, μ)/dτ = I(τ, μ) - S(τ, μ)

其中:
    I(τ, μ): 比辐射强度
    τ: 光学深度
    μ = cos(θ): 天顶角余弦
    S: 源函数

对于透射光谱计算，需要求解从行星边缘穿过大气的辐射衰减。
"""

import numpy as np
from typing import Tuple, Optional, Callable
from sparse_linear_algebra import CRSMatrix, crs_gmres


class RadiativeTransferSolver:
    """
    1D 平面平行大气辐射传输求解器。

    物理模型:
        在平面平行近似下，辐射传输方程为:
            μ dI/dτ = I - (1-ω) B - (ω/4π) ∫ P(μ, μ') I(μ') dΩ'

        对于纯吸收（无散射，ω=0）:
            μ dI/dτ = I - B
        其解为:
            I(τ, μ) = I(0, μ) exp(-τ/μ) + ∫_0^τ B(τ') exp(-(τ-τ')/μ) dτ'/μ

    对于透射几何（行星凌日）:
        观测路径与行星边缘相切，有效路径长度为:
            s(z) = 2 √[(R_p + z)² - (R_p + z_min)²]
        其中 z_min 是切线最低点高度。
    """

    def __init__(self, wavelength: np.ndarray, planet_radius_m: float):
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.R_p = planet_radius_m
        self.n_wl = len(wavelength)

    def compute_optical_depth(self, pressure: np.ndarray, temperature: np.ndarray,
                              abundance: dict, cross_sections: dict,
                              gravity: np.ndarray,
                              rayleigh_cross_section: Optional[np.ndarray] = None,
                              cloud_optical_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算每层的光学深度。

        公式:
            dτ_i = Σ_s σ_s(λ, P_i, T_i) * n_s(z_i) * Δz_i
                 + σ_R(λ) * n_tot(z_i) * Δz_i
                 + dτ_cloud

        其中:
            n_s = VMR_s * P / (k_B T)   (数密度)
            Δz_i ≈ H_i * ΔlnP_i         (层厚度，标高近似)

        参数:
            pressure: 压强数组 (Pa)，形状 (n_layers,)
            temperature: 温度数组 (K)，形状 (n_layers,)
            abundance: 物种丰度字典 {species: vmr_array}
            cross_sections: 吸收截面字典 {species: sigma_array(wl, layers)}
            gravity: 重力加速度数组 (m/s²)
            rayleigh_cross_section: 瑞利散射截面 (cm²)，形状 (n_wl,)
            cloud_optical_depth: 云光学厚度，形状 (n_layers, n_wl) 或 (n_layers,)

        返回:
            tau: 光学深度，形状 (n_layers, n_wl)，累积光学深度从高层到低层
        """
        n_layers = len(pressure)
        # TODO Hole 1: 实现光学深度计算
        # 需要计算：
        # 1. 数密度 n_total = P / (k_B * T)
        # 2. 大气标高 H = k_B * T / (mu_avg * amu * gravity)
        # 3. 层厚度 dz = H * |dlnP|
        # 4. 吸收光学深度 tau_abs = sum_s (n_s * sigma_s * dz) [注意单位转换 cm^2 -> m^2]
        # 5. 加上瑞利散射和云层贡献
        # 6. 累积光学深度 tau_cumulative = cumsum(tau_abs, axis=0)
        tau_cumulative = np.zeros((n_layers, self.n_wl), dtype=np.float64)
        return tau_cumulative

    def transit_depth_spectrum(self, pressure: np.ndarray, temperature: np.ndarray,
                               optical_depth: np.ndarray,
                               altitude: np.ndarray) -> np.ndarray:
        """
        计算透射光谱（Transit Depth Spectrum）。

        行星凌日时，恒星光线穿过行星大气边缘。
        对于切线路径，有效光学深度远大于垂直光学深度。

        公式（等效高度法）:
            R_p(λ)² = R_p0² + 2 ∫_{z_min}^{∞} (z - z_min) [1 - exp(-τ_eff(z, λ))] dz

        其中有效光学深度 τ_eff 沿切线路径:
            τ_eff(λ, z_tan) = 2 ∫_{z_tan}^{∞} α(λ, z) ds
            ds = (R_p + z) / √[(R_p + z)² - (R_p + z_tan)²] dz

        对于简化计算，使用等效高度近似:
            若某层 τ_vert > 1，则该层对波长 λ 不透明。
            有效半径 R_eff(λ) = R_p + z(λ)
            其中 z(λ) 满足 τ_vert(λ, z) ≈ 1。

        透射深度:
            δ(λ) = (R_eff(λ) / R_star)²

        参数:
            pressure: 压强数组
            temperature: 温度数组
            optical_depth: 累积光学深度 (n_layers, n_wl)
            altitude: 海拔高度数组 (m)

        返回:
            transit_depth: 透射深度 (ppm)
        """
        n_layers = len(pressure)
        transit_depth = np.zeros(self.n_wl, dtype=np.float64)

        for iwl in range(self.n_wl):
            tau_vert = optical_depth[:, iwl]

            # 找到 τ ≈ 1 的高度（有效吸收高度）
            # 使用对数插值
            if tau_vert[-1] < 1.0:
                # 即使最底层也不够不透明
                z_eff = altitude[-1]
            elif tau_vert[0] > 1.0:
                # 即使最高层也太不透明
                z_eff = altitude[0]
            else:
                # 插值找到 τ=1 的高度
                log_tau = np.log10(np.maximum(tau_vert, 1e-10))
                target_log_tau = 0.0  # log10(1)
                z_eff = np.interp(target_log_tau, log_tau, altitude)

            # 有效半径
            R_eff = self.R_p + z_eff
            # 透射深度（假设 R_star = 10 R_p 的简化）
            R_star = 10.0 * self.R_p
            depth = ((R_eff / R_star)**2 - (self.R_p / R_star)**2) * 1e6  # ppm
            transit_depth[iwl] = max(depth, 0.0)

        return transit_depth

    def finite_element_rte_solve(self, mu: float, tau_grid: np.ndarray,
                                  source_function: np.ndarray,
                                  boundary_top: float = 0.0,
                                  boundary_bot: Optional[float] = None) -> np.ndarray:
        """
        使用有限元方法求解单角度辐射传输方程。

        方程:
            μ dI/dτ = I - S

        这是关于 τ 的一阶线性 ODE。转化为标准形式：
            dI/dτ - (1/μ) I = -S/μ

        使用 Galerkin 有限元离散化：
            在单元 e = [τ_i, τ_{i+1}] 上，设 I ≈ N_1 I_i + N_2 I_{i+1}
            N_1 = (τ_{i+1} - τ) / h,  N_2 = (τ - τ_i) / h

        弱形式:
            ∫ N_j (dI/dτ - I/μ) dτ = -∫ N_j S/μ dτ

        融合 fem1d_adaptive 的有限元组装思想。

        参数:
            mu: 天顶角余弦
            tau_grid: 光学深度网格 (n,)
            source_function: 源函数 S(τ)，形状 (n,)
            boundary_top: 顶层边界条件 I(τ=0)
            boundary_bot: 底层边界条件 I(τ=τ_max)（若为 None 则用向外辐射条件）

        返回:
            intensity: 辐射强度 I(τ)
        """
        tau = np.asarray(tau_grid, dtype=np.float64)
        S = np.asarray(source_function, dtype=np.float64)
        n = len(tau)

        if n < 2:
            raise ValueError("光学深度网格至少需 2 个点")
        if len(S) != n:
            raise ValueError("源函数与网格维度不匹配")
        if abs(mu) < 1e-15:
            raise ValueError("μ 不能为零")

        # 组装三对角线性系统
        # 使用两点 Gauss-Legendre 积分
        adiag = np.zeros(n, dtype=np.float64)
        aleft = np.zeros(n, dtype=np.float64)
        arite = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)

        for i in range(n - 1):
            h = tau[i + 1] - tau[i]
            if h < 1e-30:
                continue

            # Gauss-Legendre 点
            gp = h / 2.0 * np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]) + (tau[i] + tau[i + 1]) / 2.0
            w = h / 2.0 * np.array([1.0, 1.0])

            for iq in range(2):
                t = gp[iq]
                ww = w[iq]

                # 形状函数
                N1 = (tau[i + 1] - t) / h
                N2 = (t - tau[i]) / h

                # 形状函数导数
                dN1 = -1.0 / h
                dN2 = 1.0 / h

                # 局部源函数插值
                S_local = N1 * S[i] + N2 * S[i + 1]

                # 单元刚度矩阵贡献
                # ∫ N_j (dN_k/dτ - N_k/μ) dτ
                coeff = 1.0 / mu

                # 节点 i 的方程
                adiag[i] += ww * (N1 * dN1 - coeff * N1 * N1)
                arite[i] += ww * (N1 * dN2 - coeff * N1 * N2)
                rhs[i] += -ww * coeff * N1 * S_local

                # 节点 i+1 的方程
                aleft[i + 1] += ww * (N2 * dN1 - coeff * N2 * N1)
                adiag[i + 1] += ww * (N2 * dN2 - coeff * N2 * N2)
                rhs[i + 1] += -ww * coeff * N2 * S_local

        # 边界条件
        I_sol = np.zeros(n, dtype=np.float64)

        if mu > 0:
            # 向下辐射：顶层入射已知
            adiag[0] = 1.0
            arite[0] = 0.0
            rhs[0] = boundary_top
            aleft[0] = 0.0

            if boundary_bot is not None:
                adiag[-1] = 1.0
                aleft[-1] = 0.0
                rhs[-1] = boundary_bot
        else:
            # 向上辐射：底层入射已知
            adiag[-1] = 1.0
            aleft[-1] = 0.0
            rhs[-1] = boundary_bot if boundary_bot is not None else 0.0
            arite[-1] = 0.0

            if boundary_top is not None:
                adiag[0] = 1.0
                arite[0] = 0.0
                rhs[0] = boundary_top

        # 求解三对角系统（Thomas 算法）
        I_sol = self._thomas_algorithm(aleft, adiag, arite, rhs)

        # 融合 image_diffuse 的扩散平滑思想：对解进行局部平滑以提高数值稳定性
        I_sol = self._diffuse_smooth(I_sol, iterations=1, c=0.1)

        return I_sol

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray,
                          c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Thomas 算法求解三对角系统。

        系统形式:
            a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i

        前向消元:
            b'_i = b_i - a_i * c'_{i-1}
            d'_i = d_i - a_i * d'_{i-1}

        回代:
            x_n = d'_n / b'_n
            x_i = (d'_i - c_i x_{i+1}) / b'_i
        """
        n = len(b)
        cp = np.zeros(n, dtype=np.float64)
        dp = np.zeros(n, dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]

        return x

    def _diffuse_smooth(self, arr: np.ndarray, iterations: int = 1,
                        c: float = 0.1) -> np.ndarray:
        """
        数值扩散平滑。

        融合 image_diffuse 的扩散平滑思想，用于抑制数值振荡。

        公式:
            u^{new}_i = c * (u_{i-1} + u_{i+1}) / 2 + (1-c) * u_i
        """
        arr = np.asarray(arr, dtype=np.float64).copy()
        n = len(arr)
        if n < 3:
            return arr

        for _ in range(iterations):
            arr_new = arr.copy()
            arr_new[1:-1] = c * 0.5 * (arr[:-2] + arr[2:]) + (1.0 - c) * arr[1:-1]
            # 边界处理：单边扩散
            arr_new[0] = c * arr[1] + (1.0 - c) * arr[0]
            arr_new[-1] = c * arr[-2] + (1.0 - c) * arr[-1]
            arr = arr_new

        return arr

    def adaptive_rte_refinement(self, tau_grid: np.ndarray,
                                 source_function: np.ndarray,
                                 mu: float,
                                 tol: float = 1e-4,
                                 max_levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应加密求解辐射传输方程。

        融合 fem1d_adaptive 的自适应加密思想：
        1. 在粗网格上求解
        2. 估计局部误差
        3. 在误差大的区域加密
        4. 重复直到满足精度

        误差估计:
            η_i ≈ |I_fine(τ_i) - I_coarse(τ_i)|
        """
        tau = np.asarray(tau_grid, dtype=np.float64)
        S = np.asarray(source_function, dtype=np.float64)

        for level in range(max_levels):
            I_coarse = self.finite_element_rte_solve(mu, tau, S)

            # 在中间点构造细网格解
            tau_fine = np.zeros(2 * len(tau) - 1, dtype=np.float64)
            S_fine = np.zeros(2 * len(tau) - 1, dtype=np.float64)
            for i in range(len(tau) - 1):
                tau_fine[2 * i] = tau[i]
                tau_fine[2 * i + 1] = 0.5 * (tau[i] + tau[i + 1])
                S_fine[2 * i] = S[i]
                S_fine[2 * i + 1] = 0.5 * (S[i] + S[i + 1])
            tau_fine[-1] = tau[-1]
            S_fine[-1] = S[-1]

            I_fine = self.finite_element_rte_solve(mu, tau_fine, S_fine)

            # 在粗网格点上估计误差
            error = np.abs(I_fine[::2] - I_coarse)
            max_err = np.max(error)

            if max_err < tol or level == max_levels - 1:
                return tau_fine, I_fine

            # 在误差大的区间加密
            refine_mask = error[:-1] > tol
            if not np.any(refine_mask):
                return tau_fine, I_fine

            new_tau = [tau[0]]
            new_S = [S[0]]
            for i in range(len(tau) - 1):
                if refine_mask[i]:
                    new_tau.append(0.5 * (tau[i] + tau[i + 1]))
                    new_S.append(0.5 * (S[i] + S[i + 1]))
                new_tau.append(tau[i + 1])
                new_S.append(S[i + 1])

            tau = np.array(new_tau)
            S = np.array(new_S)

        return tau, self.finite_element_rte_solve(mu, tau, S)
