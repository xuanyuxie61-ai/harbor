# -*- coding: utf-8 -*-
"""
brillouin_integral.py — 布里渊区积分
========================================
核心科学问题: 在二维布里渊区上对能带结构进行积分,
计算载流子浓度、电流密度等物理量.

融合种子项目:
  - 179_circle_integrals: 圆上精确单积分 (Gamma函数解析解)
  - 542_histogram_pdf_2d_sample: 2D 采样方法

数学基础:
  六方晶格布里渊区 (TMDC):
    正六边形, K 点位于 |K| = 4π/(3a)
    面积: A_BZ = 8π²/(√3·a²)

  积分:
    <O> = (1/A_BZ) ∫_{BZ} O(k) d²k

  对于旋转对称的量:
    <O> = (1/(2π)²) ∫_0^{2π} ∫_0^{k_max} O(k) k dk dθ
"""

import numpy as np
from scipy.special import gamma as gamma_func


def circle_monomial_integral(e1, e2):
    """
    单位圆上 x^e1 * y^e2 的精确积分.
    融合 179_circle_integrals.

    公式:
      ∫_{S^1} x^e1 * y^e2 ds =
        2 * Γ((e1+1)/2) * Γ((e2+1)/2) / Γ((e1+e2+2)/2)
      若 e1 或 e2 为奇数, 积分为零.

    参数
    ----
    e1, e2 : int
        x, y 的幂次

    返回
    ----
    float
        积分值
    """
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0

    num = 2.0 * gamma_func((e1 + 1) / 2.0) * gamma_func((e2 + 1) / 2.0)
    den = gamma_func((e1 + e2 + 2) / 2.0)
    return num / den


def disk_monomial_integral(e1, e2):
    """
    单位圆盘上 x^e1 * y^e2 的积分:
      ∫_{D} x^e1 * y^e2 dA

    极坐标: x = r*cos(θ), y = r*sin(θ)
      = ∫_0^1 r^{e1+e2+1} dr · ∫_0^{2π} cos^e1(θ) sin^e2(θ) dθ
      = 1/(e1+e2+2) · circle_monomial_integral(e1, e2)
    """
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    radial = 1.0 / (e1 + e2 + 2)
    angular = circle_monomial_integral(e1, e2)
    return radial * angular


class BrillouinZoneIntegration:
    """
    二维布里渊区积分器.

    支持:
    1. 均匀 k 网格 (Monkhorst-Pack)
    2. 极坐标高斯积分
    3. 特殊点方法 (Chadi-Cohen)
    """

    def __init__(self, lattice_constant, method='uniform', n_k=20):
        """
        参数
        ----
        lattice_constant : float
            晶格常数 [Å]
        method : str
            积分方法 ('uniform', 'polar', 'special_points')
        n_k : int
            k 网格密度
        """
        self.a = lattice_constant * 1e-10  # → m
        self.method = method
        self.n_k = n_k

        # 布里渊区参数
        self.K_point = 4 * np.pi / (3 * self.a)  # K 点模 [1/m]
        self.A_BZ = 8 * np.pi**2 / (np.sqrt(3) * self.a**2)  # BZ 面积

    def uniform_grid(self):
        """
        Monkhorst-Pack 均匀 k 网格 (六角 BZ).

        生成方法: 在六角形 BZ 内生成均匀网格点.
        k_x = (2π/a) * (i/n_k)
        k_y = (2π/(a√3)) * (j/n_k)
        """
        n = self.n_k
        kx_max = 2 * np.pi / self.a
        ky_max = 2 * np.pi / (self.a * np.sqrt(3))

        kx = np.linspace(-kx_max, kx_max, n)
        ky = np.linspace(-ky_max, ky_max, n)
        KX, KY = np.meshgrid(kx, ky)

        # 筛选 BZ 内的点 (六角形)
        mask = np.ones_like(KX, dtype=bool)
        for angle in np.linspace(0, 2 * np.pi, 7)[:-1]:
            nx = np.cos(angle)
            ny = np.sin(angle)
            boundary = kx_max * np.cos(np.pi / 6)
            mask &= (KX * nx + KY * ny) <= boundary

        kx_flat = KX[mask].flatten()
        ky_flat = KY[mask].flatten()
        weights = np.ones(len(kx_flat)) / len(kx_flat)

        return kx_flat, ky_flat, weights

    def polar_grid(self):
        """
        极坐标 k 网格 (利用旋转对称性).

        k = (k_r, θ)
        径向: Gauss-Legendre 求积
        角向: 均匀分点
        """
        n_r = self.n_k // 2
        n_theta = self.n_k

        # 径向 Gauss-Legendre
        k_r, w_r = np.polynomial.legendre.leggauss(n_r)
        k_r = 0.5 * (k_r + 1) * self.K_point  # 映射到 [0, K]
        w_r = 0.5 * w_r * self.K_point

        # 角向均匀
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        w_theta = np.ones(n_theta) / n_theta * 2 * np.pi

        # 张量积
        kx = []
        ky = []
        weights = []
        for ir in range(n_r):
            for it in range(n_theta):
                kx.append(k_r[ir] * np.cos(theta[it]))
                ky.append(k_r[ir] * np.sin(theta[it]))
                weights.append(w_r[ir] * w_theta[it] * k_r[ir])

        return np.array(kx), np.array(ky), np.array(weights)

    def special_points(self):
        """
        Chadi-Cohen 特殊 k 点方法.
        对于六角 BZ, 使用高对称点:
          Γ, K, M, 以及中间点.
        """
        K = self.K_point

        # 特殊点 (六角晶格)
        k_points = np.array([
            [0, 0],            # Γ
            [K, 0],            # K
            [K/2, K*np.sqrt(3)/2],   # K'
            [K*np.sqrt(3)/2, K/2],   # M
            [-K*np.sqrt(3)/2, K/2],  # M'
            [K/2, -K*np.sqrt(3)/2],  # -K
        ])

        # 权重 (归一化)
        n_pts = len(k_points)
        weights = np.array([4, 6, 6, 3, 3, 6]) / 28.0

        return k_points[:, 0], k_points[:, 1], weights

    def integrate(self, integrand_func):
        """
        在布里渊区上积分:
            I = (1/A_BZ) ∫_{BZ} f(kx, ky) d²k

        参数
        ----
        integrand_func : callable
            f(kx, ky) → float

        返回
        ----
        float
            积分值
        """
        if self.method == 'uniform':
            kx, ky, w = self.uniform_grid()
        elif self.method == 'polar':
            kx, ky, w = self.polar_grid()
        elif self.method == 'special_points':
            kx, ky, w = self.special_points()
        else:
            kx, ky, w = self.uniform_grid()

        # 求积
        result = 0.0
        for i in range(len(kx)):
            result += w[i] * integrand_func(kx[i], ky[i])

        return result

    def effective_mass_tensor(self, energy_func, k0, dk=1e7):
        """
        从能带色散计算有效质量张量:
            1/m*_ij = (1/ℏ²) * ∂²E/∂k_i∂k_j

        数值差分:
            ∂²E/∂kx² ≈ (E(kx+dk,ky) - 2E(kx,ky) + E(kx-dk,ky))/dk²
        """
        E0 = energy_func(k0[0], k0[1])
        Ex_p = energy_func(k0[0] + dk, k0[1])
        Ex_m = energy_func(k0[0] - dk, k0[1])
        Ey_p = energy_func(k0[0], k0[1] + dk)
        Ey_m = energy_func(k0[0], k0[1] - dk)
        Exy_pp = energy_func(k0[0] + dk, k0[1] + dk)
        Exy_pm = energy_func(k0[0] + dk, k0[1] - dk)
        Exy_mp = energy_func(k0[0] - dk, k0[1] + dk)
        Exy_mm = energy_func(k0[0] - dk, k0[1] - dk)

        d2E_dkx2 = (Ex_p - 2*E0 + Ex_m) / dk**2
        d2E_dky2 = (Ey_p - 2*E0 + Ey_m) / dk**2
        d2E_dkxky = (Exy_pp - Exy_pm - Exy_mp + Exy_mm) / (4 * dk**2)

        # 有效质量张量 [1/m0]
        # 1/m*_ij = (1/ℏ²) ∂²E/∂k_i∂k_j
        # m* = ℏ² / (∂²E/∂k²)
        # 转换为 m0 单位
        prefactor = HBAR**2 / (M0 * E0) if E0 != 0 else 1.0
        # 简化: 返回 ∂²E/∂k² 的逆
        m_xx = 1.0 / max(abs(d2E_dkx2), 1e-30)
        m_yy = 1.0 / max(abs(d2E_dky2), 1e-30)
        m_xy = d2E_dkxky

        return np.array([[m_xx, m_xy], [m_xy, m_yy]])

    def fermi_surface_average(self, energy_func, E_fermi, property_func,
                               tolerance=0.01):
        """
        费米面上物理量的平均:
            <O>_FS = ∫_{FS} O(k) / |∇_k E(k)| dl / ∫_{FS} 1/|∇_k E| dl

        融合 circle_integrals 的解析积分思想.
        """
        # 在费米面上采样
        kx, ky, w = self.uniform_grid()
        E_vals = np.array([energy_func(kx[i], ky[i]) for i in range(len(kx))])

        # 费米面附近的点
        mask = np.abs(E_vals - E_fermi) < tolerance
        if not np.any(mask):
            return 0.0

        kx_fs = kx[mask]
        ky_fs = ky[mask]
        w_fs = w[mask]

        # 物理量平均
        prop_vals = np.array([property_func(kx_fs[i], ky_fs[i])
                              for i in range(len(kx_fs))])
        return np.sum(w_fs * prop_vals) / max(np.sum(w_fs), 1e-30)


# 从 high_order_fd 导入
HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
E0 = 1.602176634e-19
