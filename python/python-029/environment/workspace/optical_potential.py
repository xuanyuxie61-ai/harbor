"""
optical_potential.py
==================
核反应光学模型复势构造模块

基于种子项目 020_artery_pde 的参数化思想，
为核反应光学模型构建 Wood–Saxon 型复势场，
包含实部体积势、虚部体积势、虚部表面势（导数形式）
以及 Thomas 型自旋-轨道耦合势。

核心公式
--------
实部 Woods–Saxon 势:
    V(r) = -V_0 / [1 + exp((r - R_v) / a_v)]

虚部体积势:
    W_v(r) = -W_0 / [1 + exp((r - R_w) / a_w)]

虚部表面势 (导数 Woods–Saxon):
    W_d(r) = -4 W_D * exp((r - R_d)/a_d) / [1 + exp((r - R_d)/a_d)]^2

Thomas 自旋-轨道势:
    V_{so}(r) = V_{so0} * (ħc)^2 / (m_π^2 c^4) * (1/r) * d/dr f_{so}(r) * L·S
    其中 f_{so}(r) = 1 / [1 + exp((r - R_so) / a_so)]

库仑势 (均匀带电球):
    V_C(r) = (Z_p Z_t e^2) / (2 R_C) * (3 - (r/R_C)^2)   (r <= R_C)
    V_C(r) = (Z_p Z_t e^2) / r                           (r > R_C)

总光学势:
    U(r) = V(r) + i [W_v(r) + W_d(r)] + V_{so}(r) + V_C(r)
"""

import numpy as np

# 物理常数 (自然单位制与常用单位转换)
HBAR_C = 197.3269804       # MeV·fm
MASS_NUCLEON = 939.5654133 # MeV/c^2 (中子平均质量)
ELEM_CHARGE2 = 1.43996448  # MeV·fm (e^2)


class OpticalPotentialParameters:
    """
    光学势参数容器类。
    参照种子项目 020_artery_pde 的参数封装思想，
    将核反应系统所有物理参数集中管理。
    """

    def __init__(self, projectile='n', target_A=56, target_Z=26, E_lab=14.0):
        """
        Parameters
        ----------
        projectile : str
            入射粒子类型: 'n' (中子), 'p' (质子), 'alpha'
        target_A : int
            靶核质量数 A
        target_Z : int
            靶核电荷数 Z
        E_lab : float
            实验室系入射能量 (MeV)
        """
        self.projectile = projectile
        self.target_A = target_A
        self.target_Z = target_Z
        self.E_lab = float(E_lab)

        # 入射粒子属性
        if projectile == 'n':
            self.proj_mass = 1.008665  # u
            self.proj_Z = 0
        elif projectile == 'p':
            self.proj_mass = 1.007276  # u
            self.proj_Z = 1
        elif projectile == 'alpha':
            self.proj_mass = 4.002603  # u
            self.proj_Z = 2
        else:
            raise ValueError(f"不支持的入射粒子类型: {projectile}")

        # 约化质量 μ (以核子质量为单位)
        self.reduced_mass = (self.proj_mass * target_A) / (self.proj_mass + target_A)
        # 约化质量 (MeV)
        self.mu_MeV = self.reduced_mass * MASS_NUCLEON

        # 波数 k (fm^{-1})
        # k = sqrt(2 μ E_lab) / ħc
        self.k = np.sqrt(2.0 * self.mu_MeV * self.E_lab) / HBAR_C

        # ---- Woods–Saxon 几何参数 (全局缩放) ----
        self.r0 = 1.25       # fm, 核半径参数
        self.r0_so = 1.25    # fm, 自旋-轨道半径参数
        self.rC = 1.25       # fm, 库仑半径参数
        self.R_v = self.r0 * (target_A ** (1.0 / 3.0))
        self.R_w = self.r0 * (target_A ** (1.0 / 3.0))
        self.R_d = self.r0 * (target_A ** (1.0 / 3.0))
        self.R_so = self.r0_so * (target_A ** (1.0 / 3.0))
        self.R_C = self.rC * (target_A ** (1.0 / 3.0))

        # 弥散参数 (fm)
        self.a_v = 0.65
        self.a_w = 0.65
        self.a_d = 0.47
        self.a_so = 0.65

        # ---- 势深参数 (MeV) ----
        # 基于 Koning-Delaroche 全局光学势参数化 (简化版)
        # 对于中子 on 56Fe at ~14 MeV:
        self.V0 = 51.5 - 0.3 * self.E_lab
        self.W0 = 2.5 + 0.15 * self.E_lab
        self.WD = 6.0 - 0.05 * self.E_lab
        self.Vso0 = 6.2

        # 边界检查
        self._validate_parameters()

    def _validate_parameters(self):
        """数值鲁棒性：参数边界检查。"""
        assert self.target_A > 0, "靶核质量数必须为正"
        assert self.target_Z >= 0, "靶核电荷数必须非负"
        assert self.E_lab > 0.0, "入射能量必须为正"
        assert self.a_v > 0.05, "弥散参数过小会导致数值不稳定"
        assert self.a_w > 0.05
        assert self.a_d > 0.05
        assert self.a_so > 0.05
        eps = 1e-12
        if abs(self.V0) < eps:
            self.V0 = eps
        if abs(self.W0) < eps:
            self.W0 = eps
        if abs(self.WD) < eps:
            self.WD = eps

    def __repr__(self):
        return (
            f"OpticalPotentialParameters("
            f"{self.projectile}+{self.target_A}{self._element_symbol()}, "
            f"E_lab={self.E_lab:.2f} MeV, k={self.k:.4f} fm^-1)"
        )

    def _element_symbol(self):
        """简单元素符号映射。"""
        symbols = {
            1: 'H', 2: 'He', 6: 'C', 8: 'O', 13: 'Al', 20: 'Ca',
            26: 'Fe', 28: 'Ni', 50: 'Sn', 82: 'Pb', 92: 'U'
        }
        return symbols.get(self.target_Z, f"Z{self.target_Z}")


def woods_saxon(r, V0, R, a):
    """
    标准 Woods–Saxon 形状因子。

    f_WS(r) = 1 / [1 + exp((r - R) / a)]

    当 r → -∞ 时 f → 1; r → +∞ 时 f → 0。
    在核物理中通常 V(r) = -V0 * f_WS(r)。
    """
    r = np.asarray(r, dtype=float)
    # 边界处理：避免指数溢出
    arg = (r - R) / a
    # 对极大的正 arg，exp → ∞，f → 0
    # 对极大的负 arg，exp → 0，f → 1
    f = np.empty_like(arg)
    # 使用分段处理防止溢出
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~mask_pos & ~mask_neg
    f[mask_pos] = 0.0
    f[mask_neg] = 1.0
    f[mask_mid] = 1.0 / (1.0 + np.exp(arg[mask_mid]))
    return f


def woods_saxon_derivative(r, V0, R, a):
    """
    Woods–Saxon 的导数形式，用于表面吸收势。

    g(r) = 4 * exp((r - R)/a) / [1 + exp((r - R)/a)]^2
         = -4a * d/dr f_WS(r)

    峰值出现在 r = R 处，g(R) = 1。
    """
    r = np.asarray(r, dtype=float)
    arg = (r - R) / a
    g = np.empty_like(arg)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~mask_pos & ~mask_neg
    g[mask_pos] = 0.0
    g[mask_neg] = 0.0
    e = np.exp(arg[mask_mid])
    g[mask_mid] = 4.0 * e / (1.0 + e) ** 2
    return g


def thomas_spin_orbit_factor(r, R_so, a_so):
    """
    Thomas 自旋-轨道形状因子的径向部分。

    f'_{so}(r) = (1/r) * d/dr [1 / (1 + exp((r - R_so)/a_so))]

    在 r → 0 时进行正则化，避免 1/r 奇点。
    """
    r = np.asarray(r, dtype=float)
    eps = 1e-15
    # 避免 r = 0 处的除零
    r_safe = np.where(np.abs(r) < eps, eps, r)
    arg = (r_safe - R_so) / a_so
    f = np.empty_like(arg)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~mask_pos & ~mask_neg
    f[mask_pos] = 0.0
    f[mask_neg] = 0.0
    e = np.exp(arg[mask_mid])
    # d/dr WS = -1/a * e / (1+e)^2
    f[mask_mid] = -(1.0 / (r_safe[mask_mid] * a_so)) * e / (1.0 + e) ** 2
    return f


def coulomb_potential(r, Zp, Zt, RC):
    """
    均匀带电球模型的库仑势。

    V_C(r) = (Zp * Zt * e^2) / (2 * R_C) * (3 - (r/R_C)^2)   (r <= R_C)
    V_C(r) = (Zp * Zt * e^2) / r                               (r > R_C)
    """
    r = np.asarray(r, dtype=float)
    VC = np.zeros_like(r)
    if Zp == 0 or Zt == 0:
        return VC
    prefactor = Zp * Zt * ELEM_CHARGE2
    mask_in = r <= RC
    mask_out = r > RC
    VC[mask_in] = prefactor / (2.0 * RC) * (3.0 - (r[mask_in] / RC) ** 2)
    # 边界处理：r 极小时避免除零
    r_out = r[mask_out]
    VC[mask_out] = prefactor / np.where(r_out > 0, r_out, 1e-15)
    return VC


def build_optical_potential(r, params, l=0, j=None):
    """
    构建给定径向网格 r 上的总光学势 U(r)。

    Parameters
    ----------
    r : array_like
        径向坐标 (fm)，要求 r > 0。
    params : OpticalPotentialParameters
        光学势参数对象。
    l : int
        轨道角动量量子数。
    j : float or None
        总角动量量子数 j = l ± 1/2。若为 None，则不包含自旋-轨道项。

    Returns
    -------
    U : ndarray
        复数数组，总光学势 (MeV)。
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("径向坐标 r 必须非负")

    # 实部体积势
    V = -params.V0 * woods_saxon(r, params.V0, params.R_v, params.a_v)

    # 虚部体积势
    Wv = -params.W0 * woods_saxon(r, params.W0, params.R_w, params.a_w)

    # 虚部表面势
    Wd = -params.WD * woods_saxon_derivative(r, params.WD, params.R_d, params.a_d)

    # 自旋-轨道势
    Vso = np.zeros_like(r)
    if j is not None and l > 0:
        # 自旋-轨道耦合常数
        # V_so(r) = Vso0 * (λ_π)^2 * f'_{so}(r) * <L·S>
        # 其中 λ_π = ħ/(m_π c) ≈ 1.414 fm
        lambda_pi = HBAR_C / 138.0  # ~1.43 fm (使用近似π介子质量)
        ls_coupling = 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.5 * 1.5)
        Vso = params.Vso0 * (lambda_pi ** 2) * thomas_spin_orbit_factor(r, params.R_so, params.a_so) * ls_coupling

    # 库仑势
    VC = coulomb_potential(r, params.proj_Z, params.target_Z, params.R_C)

    # 总势
    U = V + 1j * (Wv + Wd) + Vso + VC
    return U


def effective_potential(r, params, l=0, j=None):
    """
    构建等效势，包含离心势垒和光学势。

    V_eff(r) = (2μ/ħ²) * U(r) + l(l+1)/r²

    这是径向 Schrödinger 方程中的有效势项。
    """
    r = np.asarray(r, dtype=float)
    U = build_optical_potential(r, params, l, j)
    # HOLE 1: 请实现有效势公式 V_eff(r) = (2μ/ħ²) * U(r) + l(l+1)/r²
    # 提示：需考虑 r=0 处的正则化处理，避免除零
    return U  # 占位返回，不正确


if __name__ == "__main__":
    # 简单自检
    params = OpticalPotentialParameters('n', 56, 26, 14.0)
    print(params)
    r = np.linspace(0.01, 15.0, 200)
    U = build_optical_potential(r, params, l=2, j=2.5)
    print("势场实部范围:", np.real(U).min(), np.real(U).max())
    print("势场虚部范围:", np.imag(U).min(), np.imag(U).max())
