"""
Landau-Ginzburg-Devonshire 自由能模块
融合来源: 523_hermite_product_display (Hermite 正交多项式展开)

功能:
- 构建多铁性材料（以 BiFeO3 为原型）的 Landau 自由能密度
- 铁电极化 P 与磁化 M 的耦合能量计算
- 利用 Hermite 多项式展开处理热涨落导致的自由能修正
- 提供各阶变分导数（TDGL 方程右端项）

核心物理公式:

总自由能密度:
    f = f_P + f_M + f_c + f_g

铁电部分 (LGD):
    f_P = α₁ (P_x² + P_y²) + α₁₁ (P_x² + P_y²)² + α₁₂ P_x² P_y²
        + (g₁₁/2)[(∂P_x/∂x)² + (∂P_y/∂y)²]
        + (g₁₂/2)[(∂P_x/∂y)² + (∂P_y/∂x)²]
        + g₄₄ (∂P_x/∂y)(∂P_y/∂x)

磁性部分:
    f_M = β₁ (M_x² + M_y²) + β₁₁ (M_x² + M_y²)² + β₁₂ M_x² M_y²
        + (A₁₁/2)[(∂M_x/∂x)² + (∂M_y/∂y)²]
        + (A₁₂/2)[(∂M_x/∂y)² + (∂M_y/∂x)²]

磁电耦合 (交换收缩型):
    f_c = γ (P_x M_y - P_y M_x)²

梯度耦合交叉项 (高阶):
    f_g = η [ (∂P_x/∂x)(∂M_y/∂y) - (∂P_y/∂y)(∂M_x/∂x) ]

其中参数符号约定:
    α₁ = α₀ (T - T_c)    (居里-外斯定律，T_c 为铁电居里温度)
    β₁ = β₀ (T - T_N)    (T_N 为奈尔温度)

Hermite 热涨落修正:
    对任一序参量场 Q，其热涨落可投影到概率 Hermite 多项式基:
    He_n(Q/σ) 满足:
        ∫ exp(-Q²/(2σ²)) He_m(Q/σ) He_n(Q/σ) dQ = √(2π) σ n! δ_{mn}
    自由能的二阶热修正:
        Δf = (k_B T / 2) * Σ_i λ_i
    其中 λ_i 为自由能 Hessian 矩阵的特征值。
"""

import numpy as np
from typing import Tuple, Optional


class MultiferroicMaterialParams:
    """多铁性材料参数容器（以 BiFeO3 近室温参数为参考）。"""

    def __init__(self, temperature: float = 300.0):
        self.T = temperature          # 温度 (K)
        self.Tc = 1103.0              # 铁电居里温度 (K)
        self.Tn = 643.0               # 反铁磁奈尔温度 (K)
        self.alpha0 = 1.0e5           # 单位: m/F (SI)
        self.alpha1 = self.alpha0 * (self.T - self.Tc)
        self.alpha11 = 7.0e8          # 四阶铁电系数
        self.alpha12 = 2.0e8
        self.beta0 = 5.0e3
        self.beta1 = self.beta0 * (self.T - self.Tn)
        self.beta11 = 1.0e6
        self.beta12 = 3.0e5
        self.gamma = 2.0e-3           # 磁电耦合系数 (m/F·?)
        self.g11 = 2.0e-10            # 铁电梯度系数 (m^4/C^2·?)
        self.g12 = 1.0e-10
        self.g44 = 1.0e-10
        self.A11 = 5.0e-12            # 磁性交换刚度 (J/m)
        self.A12 = 2.0e-12
        self.eta = 1.0e-15            # 高阶交叉梯度耦合
        self.sigma = 0.05             # Hermite 展开特征宽度 (C/m^2)

    def validate(self):
        """检查参数物理合理性。"""
        assert np.isfinite(self.alpha1), "alpha1 必须为有限值"
        assert np.isfinite(self.beta1), "beta1 必须为有限值"
        assert self.alpha11 > 0, "alpha11 必须为正（保证铁电相稳定）"
        assert self.beta11 > 0, "beta11 必须为正"


def hermite_probabilist(n: int, x: np.ndarray) -> np.ndarray:
    """
    计算概率学家 Hermite 多项式 He_n(x)。
    融合自 hermite_product_display 中 he_polynomial 的递推公式:
        He_0(x) = 1
        He_1(x) = x
        He_n(x) = x He_{n-1}(x) - (n-1) He_{n-2}(x)
    """
    x = np.asarray(x)
    if n < 0:
        raise ValueError("n 必须非负")
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return x.copy()
    H_prev2 = np.ones_like(x, dtype=float)   # He_0
    H_prev1 = x.copy()                        # He_1
    for j in range(2, n + 1):
        H_curr = x * H_prev1 - (j - 1) * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


def normalized_hermite_probabilist(n: int, x: np.ndarray) -> np.ndarray:
    """
    归一化概率 Hermite 多项式 Hen_n(x)，满足正交归一性:
        ∫ exp(-x^2/2) Hen_m(x) Hen_n(x) dx = √(2π) δ_{mn}
    """
    He = hermite_probabilist(n, x)
    norm = np.sqrt(np.math.factorial(n) * np.sqrt(2.0 * np.pi))
    return He / norm


def landau_free_energy_density(P: np.ndarray, M: np.ndarray,
                                dPdx: np.ndarray, dPdy: np.ndarray,
                                dMdx: np.ndarray, dMdy: np.ndarray,
                                params: MultiferroicMaterialParams) -> float:
    """
    计算单点总 Landau 自由能密度。

    参数:
        P:  铁电极化矢量 [P_x, P_y]    (单位: C/m^2)
        M:  磁化矢量 [M_x, M_y]        (单位: A/m)
        dPdx, dPdy: 极化空间梯度
        dMdx, dMdy: 磁化空间梯度
        params: 材料参数

    返回:
        f: 自由能密度 (J/m^3)
    """
    Px, Py = float(P[0]), float(P[1])
    Mx, My = float(M[0]), float(M[1])

    # 铁电部分
    P2 = Px * Px + Py * Py
    fP = (params.alpha1 * P2
          + params.alpha11 * P2 * P2
          + params.alpha12 * Px * Px * Py * Py)

    # 磁性部分
    M2 = Mx * Mx + My * My
    fM = (params.beta1 * M2
          + params.beta11 * M2 * M2
          + params.beta12 * Mx * Mx * My * My)

    # 磁电耦合 (ME coupling)
    fc = params.gamma * (Px * My - Py * Mx) ** 2

    # 铁电梯度
    f_grad_P = (0.5 * params.g11 * (dPdx[0] ** 2 + dPdy[1] ** 2)
                + 0.5 * params.g12 * (dPdx[1] ** 2 + dPdy[0] ** 2)
                + params.g44 * dPdx[1] * dPdy[0])

    # 磁性梯度
    f_grad_M = (0.5 * params.A11 * (dMdx[0] ** 2 + dMdy[1] ** 2)
                + 0.5 * params.A12 * (dMdx[1] ** 2 + dMdy[0] ** 2))

    # 高阶交叉梯度
    f_cross = params.eta * (dPdx[0] * dMdy[1] - dPdy[1] * dMdx[0])

    f_total = fP + fM + fc + f_grad_P + f_grad_M + f_cross

    # 边界鲁棒性: 若出现 NaN/Inf，返回大正值惩罚
    if not np.isfinite(f_total):
        return 1e20
    return f_total


def variational_derivative_P(P: np.ndarray, M: np.ndarray,
                              lapP: np.ndarray, params: MultiferroicMaterialParams) -> np.ndarray:
    """
    计算自由能对极化 P 的变分导数 δF/δP。
    用于 TDGL 方程:  ∂P/∂t = -Γ_P δF/δP

    公式:
        δF/δP_x = α₁ P_x + 4 α₁₁ P_x (P_x²+P_y²) + 2 α₁₂ P_x P_y²
                  + 2γ (P_x M_y - P_y M_x) M_y - g₁₁ ∂²P_x/∂x² - g₁₂ ∂²P_x/∂y²
        δF/δP_y = α₁ P_y + 4 α₁₁ P_y (P_x²+P_y²) + 2 α₁₂ P_y P_x²
                  - 2γ (P_x M_y - P_y M_x) M_x - g₁₁ ∂²P_y/∂y² - g₁₂ ∂²P_y/∂x²
    """
    # TODO: 实现自由能对极化 P 的变分导数 δF/δP
    # 公式:
    #   δF/δP_x = α₁ P_x + 4 α₁₁ P_x (P_x²+P_y²) + 2 α₁₂ P_x P_y²
    #             + 2γ (P_x M_y - P_y M_x) M_y - g₁₁ ∂²P_x/∂x² - g₁₂ ∂²P_x/∂y²
    #   δF/δP_y = α₁ P_y + 4 α₁₁ P_y (P_x²+P_y²) + 2 α₁₂ P_y P_x²
    #             - 2γ (P_x M_y - P_y M_x) M_x - g₁₁ ∂²P_y/∂y² - g₁₂ ∂²P_y/∂x²
    pass


def variational_derivative_M(P: np.ndarray, M: np.ndarray,
                              lapM: np.ndarray, params: MultiferroicMaterialParams) -> np.ndarray:
    """
    计算自由能对磁化 M 的变分导数 δF/δM。

    公式:
        δF/δM_x = β₁ M_x + 4 β₁₁ M_x (M_x²+M_y²) + 2 β₁₂ M_x M_y²
                  - 2γ (P_x M_y - P_y M_x) P_y - A₁₁ ∂²M_x/∂x² - A₁₂ ∂²M_x/∂y²
        δF/δM_y = β₁ M_y + 4 β₁₁ M_y (M_x²+M_y²) + 2 β₁₂ M_y M_x²
                  + 2γ (P_x M_y - P_y M_x) P_x - A₁₁ ∂²M_y/∂y² - A₁₂ ∂²M_y/∂x²
    """
    # TODO: 实现自由能对磁化 M 的变分导数 δF/δM
    # 公式:
    #   δF/δM_x = β₁ M_x + 4 β₁₁ M_x (M_x²+M_y²) + 2 β₁₂ M_x M_y²
    #             - 2γ (P_x M_y - P_y M_x) P_y - A₁₁ ∂²M_x/∂x² - A₁₂ ∂²M_x/∂y²
    #   δF/δM_y = β₁ M_y + 4 β₁₁ M_y (M_x²+M_y²) + 2 β₁₂ M_y M_x²
    #             + 2γ (P_x M_y - P_y M_x) P_x - A₁₁ ∂²M_y/∂y² - A₁₂ ∂²M_y/∂x²
    pass


def thermal_fluctuation_correction(P: np.ndarray, M: np.ndarray,
                                    params: MultiferroicMaterialParams,
                                    max_hermite_order: int = 6) -> float:
    """
    利用 Hermite 多项式展开估计局部热涨落对自由能的修正。
    融合 hermite_product_display 中 Hermite 多项式递推与正交性思想。

    方法:
    1. 计算自由能关于 P, M 的 Hessian 矩阵 H (4x4)
    2. 特征值 λ_i 代表涨落模式的有效刚度
    3. 热修正 Δf ≈ (k_B T / 2) Σ_i log(λ_i / λ_0) （谐波近似）

    返回:
        delta_f: 热涨落修正的自由能密度 (J/m^3)
    """
    kB = 1.380649e-23  # 玻尔兹曼常数 J/K
    Px, Py = P[0], P[1]
    Mx, My = M[0], M[1]

    # 构建 Hessian 矩阵（仅保留二阶主导项）
    H = np.zeros((4, 4), dtype=float)
    P2 = Px * Px + Py * Py
    M2 = Mx * Mx + My * My

    # d²f/dP_i dP_j
    H[0, 0] = params.alpha1 + 12.0 * params.alpha11 * Px * Px + 4.0 * params.alpha11 * Py * Py + 2.0 * params.alpha12 * Py * Py
    H[1, 1] = params.alpha1 + 12.0 * params.alpha11 * Py * Py + 4.0 * params.alpha11 * Px * Px + 2.0 * params.alpha12 * Px * Px
    H[0, 1] = H[1, 0] = 8.0 * params.alpha11 * Px * Py + 4.0 * params.alpha12 * Px * Py

    # d²f/dM_i dM_j
    H[2, 2] = params.beta1 + 12.0 * params.beta11 * Mx * Mx + 4.0 * params.beta11 * My * My + 2.0 * params.beta12 * My * My
    H[3, 3] = params.beta1 + 12.0 * params.beta11 * My * My + 4.0 * params.beta11 * Mx * Mx + 2.0 * params.beta12 * Mx * Mx
    H[2, 3] = H[3, 2] = 8.0 * params.beta11 * Mx * My + 4.0 * params.beta12 * Mx * My

    # 磁电耦合交叉项
    cross = Px * My - Py * Mx
    H[0, 2] = H[2, 0] = 2.0 * params.gamma * cross * (-Py) + 2.0 * params.gamma * My * My
    H[0, 3] = H[3, 0] = 2.0 * params.gamma * cross * Px + 2.0 * params.gamma * My * (-Mx)
    H[1, 2] = H[2, 1] = 2.0 * params.gamma * cross * My + 2.0 * params.gamma * (-Mx) * (-Py)
    H[1, 3] = H[3, 1] = 2.0 * params.gamma * cross * (-Mx) + 2.0 * params.gamma * (-Mx) * Px

    # 正则化确保正定
    H += np.eye(4) * 1e-10

    eigvals = np.linalg.eigvalsh(H)
    # 只取正特征值
    eigvals = np.clip(eigvals, 1e-20, None)

    # 谐波近似热修正
    delta_f = 0.5 * kB * params.T * np.sum(np.log(eigvals / 1.0))
    if not np.isfinite(delta_f):
        delta_f = 0.0
    return delta_f
