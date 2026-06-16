"""
Landau-Ginzburg-Devonshire 自由能泛函模块
==========================================
对应种子项目: 486_gray_scott_movie (反应扩散系统 → 极化-磁化耦合动力学)
              1270_zruan_PAC_desensitization (MD 能量提取 → 自由能分量提取)

物理背景:
    多铁性材料的热力学由 LGD 自由能泛函描述:
        F[P, M] = ∫{f_Landau + f_elastic + f_gradient + f_electric
                     + f_magnetic + f_magnetoelectric} dV

    对于 BiFeO3, 极化 P = (P₁, P₂, P₃) 和反铁磁矢量 L = (L₁, L₂, L₃)
    是主要序参量. 为简化, 本代码用有效磁化 M 替代 L.

LGD 自由能密度分量:
    1. Landau 体能量:
       f_L = α₁(P₁² + P₂² + P₃²) + α₁₁(P₁⁴ + P₂⁴ + P₃⁴)
           + α₁₂(P₁²P₂² + P₂²P₃² + P₁²P₃²)
           + α₁₁₁(P₁⁶ + P₂⁶ + P₃⁶)
           + α₁₁₂(P₁⁴(P₂²+P₃²) + P₂⁴(P₁²+P₃²) + P₃⁴(P₁²+P₂²))
           + α₁₂₃P₁²P₂²P₃²

    2. 梯度能:
       f_G = ½G₁₁(P₁,₁² + P₂,₂² + P₃,₃²)
           + ½G₁₂((P₁,₂² + P₁,₃²) + (P₂,₁² + P₂,₃²) + (P₃,₁² + P₃,₂²))
           + G₄₄(P₁,₂P₂,₁ + P₁,₃P₃,₁ + P₂,₃P₃,₂)

    3. 磁能:
       f_M = -½β₁(M₁² + M₂² + M₃²) + ¼β₁₁(M₁⁴ + M₂⁴ + M₃⁴)
           + K₁(M₁⁴M₂⁴ + M₂⁴M₃⁴ + M₁⁴M₃⁴)/M_s⁸

    4. 磁电耦合:
       f_ME = α(P₁M₁² + P₂M₂² + P₃M₃²)  (线性)
            + γ((P₁M₁)² + (P₂M₂)² + (P₃M₃)²) (双线性)

有效场 (变分导数):
    H_eff^P = -δF/δP    → 驱动极化演化的有效电 field
    H_eff^M = -δF/δM    → 驱动磁化演化的有效磁 field
"""

import numpy as np
from multiferroic_constants import BiFeO3LGD


class LGDFreeEnergyFunctional:
    """
    Landau-Ginzburg-Devonshire 自由能泛函.

    在 2D 网格上计算自由能密度及其变分导数 (有效场).
    采用周期性边界条件或 Dirichlet/Neumann 边界.

    网格约定:
        P[i,j,k]: k=0,1,2 对应 P₁, P₂, P₃ 分量
        M[i,j,k]: k=0,1,2 对应 M₁, M₂, M₃ 分量
        空间索引 (i,j) 对应 (x,y) 方向
    """

    def __init__(self, lgd_params=None):
        """
        参数:
            lgd_params: BiFeO3LGD 实例. 若为 None 则使用默认参数.
        """
        self.params = lgd_params if lgd_params is not None else BiFeO3LGD()

    # ============================================================
    # Landau 体能量
    # ============================================================

    def f_landau(self, P, T):
        """
        计算 Landau 体自由能密度 (J/m³).

        f_L = α₁(T)·(P₁² + P₂² + P₃²)
            + α₁₁·(P₁⁴ + P₂⁴ + P₃⁴)
            + α₁₂·(P₁²·P₂² + P₂²·P₃² + P₁²·P₃²)
            + α₁₁₁·(P₁⁶ + P₂⁶ + P₃⁶)
            + α₁₁₂·(P₁⁴·(P₂²+P₃²) + P₂⁴·(P₁²+P₃²) + P₃⁴·(P₁²+P₂²))
            + α₁₂₃·P₁²·P₂²·P₃²

        参数:
            P: 极化场数组, shape (nx, ny, 3)
            T: 温度 (K)

        返回:
            f_L: 自由能密度, shape (nx, ny)
        """
        p = self.params
        P1, P2, P3 = P[..., 0], P[..., 1], P[..., 2]

        P1_2, P2_2, P3_2 = P1**2, P2**2, P3**2
        P1_4, P2_4, P3_4 = P1_2**2, P2_2**2, P3_2**2
        P1_6, P2_6, P3_6 = P1_2**3, P2_2**3, P3_2**3

        # 第二阶项: α₁(T)·Σ Pᵢ²
        alpha1_T = p.alpha1_temperature(T)
        f_2nd = alpha1_T * (P1_2 + P2_2 + P3_2)

        # 第四阶项: α₁₁·Σ Pᵢ⁴ + α₁₂·Σ Pᵢ²Pⱼ²
        f_4th = (p.alpha11 * (P1_4 + P2_4 + P3_4) +
                 p.alpha12 * (P1_2 * P2_2 + P2_2 * P3_2 + P1_2 * P3_2))

        # 第六阶项: α₁₁₁·Σ Pᵢ⁶ + α₁₁₂·Σ Pᵢ⁴(Pⱼ²+Pₖ²) + α₁₂₃·P₁²P₂²P₃²
        f_6th = (p.alpha111 * (P1_6 + P2_6 + P3_6) +
                 p.alpha112 * (P1_4 * (P2_2 + P3_2) +
                               P2_4 * (P1_2 + P3_2) +
                               P3_4 * (P1_2 + P2_2)) +
                 p.alpha123 * P1_2 * P2_2 * P3_2)

        return f_2nd + f_4th + f_6th

    # ============================================================
    # 梯度能 (需要空间导数)
    # ============================================================

    def f_gradient(self, P, dx, dy, fd_order=6):
        """
        计算极化梯度能密度 (J/m³).

        f_G = ½G₁₁[(∂P₁/∂x)² + (∂P₂/∂y)²]
            + ½G₁₂[(∂P₁/∂y)² + (∂P₂/∂x)²]
            + G₄₄·(∂P₁/∂y)(∂P₂/∂x)

        对于 2D 系统, P₃ 分量仅通过耦合间接影响.
        完整 3D 公式包含更多项.

        参数:
            P: 极化场, shape (nx, ny, 3)
            dx, dy: 网格间距 (m)
            fd_order: 有限差分阶数

        返回:
            f_G: 梯度能密度, shape (nx, ny)
        """
        from high_order_fd import compact_laplacian_2d, sixth_order_derivative

        p = self.params

        # 计算各分量的空间导数
        dP1_dx = sixth_order_derivative(P[..., 0], dx, axis=0, order=1)
        dP1_dy = sixth_order_derivative(P[..., 0], dy, axis=1, order=1)
        dP2_dx = sixth_order_derivative(P[..., 1], dx, axis=0, order=1)
        dP2_dy = sixth_order_derivative(P[..., 1], dy, axis=1, order=1)

        # 梯度能密度
        f_G = (0.5 * p.G11 * (dP1_dx**2 + dP2_dy**2) +
               0.5 * p.G12 * (dP1_dy**2 + dP2_dx**2) +
               p.G44 * dP1_dy * dP2_dx)

        return f_G

    # ============================================================
    # 磁能
    # ============================================================

    def f_magnetic(self, M, T):
        """
        计算磁自由能密度 (J/m³).

        采用 Landau 展开形式:
        f_M = -½β₁(T)·Σ Mᵢ² + ¼β₁₁·(Σ Mᵢ²)²
            + K₁·(M₁⁴M₂⁴ + M₂⁴M₃⁴ + M₁⁴M₃⁴) / M_s⁸

        其中 β₁(T) 类似 Curie-Weiss 律:
            β₁(T) = μ₀ · M_s² · T_N / (T_N - T + ε)

        参数:
            M: 磁化场, shape (nx, ny, 3)
            T: 温度 (K)

        返回:
            f_M: 磁自由能密度, shape (nx, ny)
        """
        from multiferroic_constants import MU_0, K_BOLTZMANN

        p = self.params
        M1, M2, M3 = M[..., 0], M[..., 1], M[..., 2]
        M2_sum = M1**2 + M2**2 + M3**2
        M_s = p.M_saturation
        T_N = 643.0  # Néel 温度 (K)

        # 有效第二阶磁系数 (Curie-Weiss 型)
        epsilon = 1.0  # 小量防止奇点
        beta1 = MU_0 * M_s**2 * T_N / (T_N - T + epsilon)
        beta11 = MU_0 * M_s**4 * 0.1  # 经验估计

        # Landau 展开
        f_M_2nd = -0.5 * beta1 * M2_sum
        f_M_4th = 0.25 * beta11 * M2_sum**2

        # 磁晶各向异性 (立方晶系, 四阶)
        M_s_8 = M_s**8 if M_s > 0 else 1.0
        f_anis = p.K1_mag * (M1**4 * M2**4 + M2**4 * M3**4 +
                              M1**4 * M3**4) / max(M_s_8, 1e-30)

        return f_M_2nd + f_M_4th + f_anis

    # ============================================================
    # 磁电耦合能
    # ============================================================

    def f_magnetoelectric(self, P, M):
        """
        计算磁电耦合能密度 (J/m³).

        f_ME = α_lin·(P₁M₁² + P₂M₂² + P₃M₃²)    [线性 ME]
             + γ_biq·((P₁M₁)² + (P₂M₂)² + (P₃M₃)²)  [双线性]

        物理含义:
            线性项: 极化与磁化平方直接耦合 → 导致电场诱导磁化
            双线性项: 极化-磁化分量同步调制 → 导致磁电畴壁关联

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)

        返回:
            f_ME: 磁电耦合能密度, shape (nx, ny)
        """
        p = self.params
        P1, P2, P3 = P[..., 0], P[..., 1], P[..., 2]
        M1, M2, M3 = M[..., 0], M[..., 1], M[..., 2]

        f_linear = p.alpha_ME_linear * (P1 * M1**2 + P2 * M2**2 + P3 * M3**2)
        f_biquadratic = p.gamma_ME_biquadratic * (
            (P1 * M1)**2 + (P2 * M2)**2 + (P3 * M3)**2
        )

        return f_linear + f_biquadratic

    # ============================================================
    # 总自由能密度
    # ============================================================

    def total_free_energy_density(self, P, M, T, dx, dy, fd_order=6):
        """
        计算 LGD 总自由能密度 (J/m³).

        f_total = f_landau(P, T) + f_gradient(P, dx, dy)
                + f_magnetic(M, T) + f_magnetoelectric(P, M)

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)
            T: 温度 (K)
            dx, dy: 网格间距 (m)
            fd_order: 有限差分阶数

        返回:
            f_total: 总自由能密度, shape (nx, ny)
        """
        f_L = self.f_landau(P, T)
        f_G = self.f_gradient(P, dx, dy, fd_order)
        f_M = self.f_magnetic(M, T)
        f_ME = self.f_magnetoelectric(P, M)

        return f_L + f_G + f_M + f_ME

    def total_free_energy(self, P, M, T, dx, dy, fd_order=6):
        """
        计算总自由能 (J/m).

        F = ∫∫ f_total dx dy

        对于 2D 系统, 单位为 J/m (沿 z 方向归一化).

        参数:
            P, M, T, dx, dy, fd_order: 同上

        返回:
            F: 总自由能 (J/m)
        """
        f_total = self.total_free_energy_density(P, M, T, dx, dy, fd_order)
        return np.sum(f_total) * dx * dy

    # ============================================================
    # 有效场 (变分导数)
    # ============================================================

    def effective_field_P(self, P, M, T, dx, dy, fd_order=6):
        """
        计算极化有效场 H_eff^P = -δF/δP.

        各分量的变分导数:
            δF/δP₁ = ∂f/∂P₁ - ∂/∂x(∂f/∂(∂P₁/∂x))
                            - ∂/∂y(∂f/∂(∂P₁/∂y))

        对于 Landau 部分:
            ∂f_L/∂P₁ = 2α₁P₁ + 4α₁₁P₁³ + 2α₁₂P₁(P₂²+P₃²)
                       + 6α₁₁₁P₁⁵ + 4α₁₁₂P₁³(P₂²+P₃²)
                       + 2α₁₁₂P₁(P₂⁴+P₃⁴) + 2α₁₂₃P₁P₂²P₃²

        梯度部分通过拉普拉斯算子:
            -G₁₁·∂²P₁/∂x² - G₁₂·∂²P₁/∂y² (简化形式)

        磁电耦合部分:
            ∂f_ME/∂P₁ = α_lin·M₁² + 2γ_biq·P₁M₁²

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)
            T: 温度 (K)
            dx, dy: 网格间距
            fd_order: 有限差分阶数

        返回:
            H_eff_P: 极化有效场, shape (nx, ny, 3)
        """
        from high_order_fd import compact_laplacian_2d

        p = self.params
        H_eff = np.zeros_like(P)

        P1, P2, P3 = P[..., 0], P[..., 1], P[..., 2]
        M1, M2, M3 = M[..., 0], M[..., 1], M[..., 2]

        P1_2, P2_2, P3_2 = P1**2, P2**2, P3**2
        P1_3, P2_3, P3_3 = P1**3, P2**3, P3**3
        P1_5, P2_5, P3_5 = P1**5, P2**5, P3**5

        alpha1_T = p.alpha1_temperature(T)

        # ---- ∂f_L/∂P₁ ----
        dL_dP1 = (2.0 * alpha1_T * P1 +
                  4.0 * p.alpha11 * P1_3 +
                  2.0 * p.alpha12 * P1 * (P2_2 + P3_2) +
                  6.0 * p.alpha111 * P1_5 +
                  4.0 * p.alpha112 * P1_3 * (P2_2 + P3_2) +
                  2.0 * p.alpha112 * P1 * (P2_2**2 + P3_2**2) +
                  2.0 * p.alpha123 * P1 * P2_2 * P3_2)

        # ---- ∂f_L/∂P₂ ----
        dL_dP2 = (2.0 * alpha1_T * P2 +
                  4.0 * p.alpha11 * P2_3 +
                  2.0 * p.alpha12 * P2 * (P1_2 + P3_2) +
                  6.0 * p.alpha111 * P2_5 +
                  4.0 * p.alpha112 * P2_3 * (P1_2 + P3_2) +
                  2.0 * p.alpha112 * P2 * (P1_2**2 + P3_2**2) +
                  2.0 * p.alpha123 * P2 * P1_2 * P3_2)

        # ---- ∂f_L/∂P₃ ----
        dL_dP3 = (2.0 * alpha1_T * P3 +
                  4.0 * p.alpha11 * P3_3 +
                  2.0 * p.alpha12 * P3 * (P1_2 + P2_2) +
                  6.0 * p.alpha111 * P3_5 +
                  4.0 * p.alpha112 * P3_3 * (P1_2 + P2_2) +
                  2.0 * p.alpha112 * P3 * (P1_2**2 + P2_2**2) +
                  2.0 * p.alpha123 * P3 * P1_2 * P2_2)

        # ---- 梯度能贡献 (拉普拉斯项) ----
        for k in range(3):
            lap = compact_laplacian_2d(P[..., k], dx, dy, fd_order)
            # ∂f_G/∂Pₖ 包含 G₁₁·∂²Pₖ/∂xₖ² + G₁₂·∂²Pₖ/∂xⱼ²
            if k == 0:
                grad_contrib = -(p.G11 * compact_laplacian_2d(
                    P[..., 0], dx, dy, fd_order, component='x') +
                                 p.G12 * compact_laplacian_2d(
                    P[..., 0], dx, dy, fd_order, component='y'))
            elif k == 1:
                grad_contrib = -(p.G12 * compact_laplacian_2d(
                    P[..., 1], dx, dy, fd_order, component='x') +
                                 p.G11 * compact_laplacian_2d(
                    P[..., 1], dx, dy, fd_order, component='y'))
            else:
                grad_contrib = -(p.G11 + p.G12) * lap

            H_eff[..., k] = -(np.array([dL_dP1, dL_dP2, dL_dP3])[k] +
                               grad_contrib)

        # ---- 磁电耦合贡献 ----
        dME_dP1 = p.alpha_ME_linear * M1**2 + \
            2.0 * p.gamma_ME_biquadratic * P1 * M1**2
        dME_dP2 = p.alpha_ME_linear * M2**2 + \
            2.0 * p.gamma_ME_biquadratic * P2 * M2**2
        dME_dP3 = p.alpha_ME_linear * M3**2 + \
            2.0 * p.gamma_ME_biquadratic * P3 * M3**2

        H_eff[..., 0] -= dME_dP1
        H_eff[..., 1] -= dME_dP2
        H_eff[..., 2] -= dME_dP3

        return H_eff

    def effective_field_M(self, P, M, T):
        """
        计算磁化有效场 H_eff^M = -δF/δM.

        δF/δM₁ = -β₁M₁ + β₁₁M₁(M₁²+M₂²+M₃²)
                 + 4K₁M₁³(M₂⁴+M₃⁴)/M_s⁸
                 + 2α_lin·P₁M₁ + 4γ_biq·P₁²M₁

        参数:
            P: 极化场, shape (nx, ny, 3)
            M: 磁化场, shape (nx, ny, 3)
            T: 温度 (K)

        返回:
            H_eff_M: 磁化有效场, shape (nx, ny, 3)
        """
        from multiferroic_constants import MU_0

        p = self.params
        M1, M2, M3 = M[..., 0], M[..., 1], M[..., 2]
        P1, P2, P3 = P[..., 0], P[..., 1], P[..., 2]
        M2_sum = M1**2 + M2**2 + M3**2
        M_s = p.M_saturation
        T_N = 643.0

        epsilon = 1.0
        beta1 = MU_0 * M_s**2 * T_N / (T_N - T + epsilon)
        beta11 = MU_0 * M_s**4 * 0.1

        H_eff = np.zeros_like(M)

        # Landau 磁项
        for k, (Mk, Pk) in enumerate(zip([M1, M2, M3], [P1, P2, P3])):
            # -β₁Mₖ + β₁₁Mₖ(M²_total)
            dM_dMk = -beta1 * Mk + beta11 * Mk * M2_sum

            # 磁晶各向异性导数 (简化)
            M_s_8 = max(M_s**8, 1e-30)
            Mk_3 = Mk**3
            other_indices = [j for j in range(3) if j != k]
            Mj, Ml = M[..., other_indices[0]], M[..., other_indices[1]]
            d_anis = 4.0 * p.K1_mag * Mk_3 * \
                (Mj**4 + Ml**4) / M_s_8

            # 磁电耦合
            dME_dMk = (2.0 * p.alpha_ME_linear * Pk * Mk +
                       2.0 * p.gamma_ME_biquadratic * Pk**2 * Mk)

            H_eff[..., k] = -(dM_dMk + d_anis + dME_dMk)

        return H_eff

    # ============================================================
    # 自由能分量提取 (对应种子项目 1270)
    # ============================================================

    def extract_energy_components(self, P, M, T, dx, dy, fd_order=6):
        """
        提取自由能各分量的空间积分值.

        对应种子项目 1270 (MD 轨迹能量提取):
        类似分子动力学中提取势能各分量,
        此处提取 LGD 泛函的 Landau/梯度/磁/磁电分量.

        返回:
            dict: {
                'f_landau': Landau 体能 (J/m),
                'f_gradient': 梯度能 (J/m),
                'f_magnetic': 磁能 (J/m),
                'f_magnetoelectric': 磁电耦合能 (J/m),
                'f_total': 总能 (J/m),
                'P_avg': 平均极化 (C/m²),
                'M_avg': 平均磁化 (A/m),
            }
        """
        dA = dx * dy
        f_L = np.sum(self.f_landau(P, T)) * dA
        f_G = np.sum(self.f_gradient(P, dx, dy, fd_order)) * dA
        f_M = np.sum(self.f_magnetic(M, T)) * dA
        f_ME = np.sum(self.f_magnetoelectric(P, M)) * dA

        P_avg = np.mean(P, axis=(0, 1))
        M_avg = np.mean(M, axis=(0, 1))

        return {
            'f_landau': f_L,
            'f_gradient': f_G,
            'f_magnetic': f_M,
            'f_magnetoelectric': f_ME,
            'f_total': f_L + f_G + f_M + f_ME,
            'P_avg': P_avg,
            'M_avg': M_avg,
            'P_magnitude': np.linalg.norm(P_avg),
            'M_magnitude': np.linalg.norm(M_avg),
        }
