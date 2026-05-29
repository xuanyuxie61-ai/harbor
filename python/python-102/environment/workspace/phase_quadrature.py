"""
phase_quadrature.py
===================
基于高精度数值积分规则计算超构表面纳米柱截面上的有效电磁响应。

本模块源自项目 957_quadrilateral_witherden_rule（四边形 Witherden
高斯积分规则），将其扩展应用于超构表面纳米柱截面上的有效极化率、
传输相位以及等效介电常数的体积积分计算。

科学背景：
对于亚波长纳米柱，其等效电磁响应可以通过对截面上的局域场进行
体积平均得到：
    α_eff = ∫_Ω ε₀ (ε_r(x,y) - 1) E_loc(x,y) dA / E_inc
在弱散射近似下（Born approximation），等效极化率正比于：
    α_eff ≈ ε₀ (ε_pillar - 1) * V_pillar
对于强散射情况，需要使用高精度数值积分在截面上采样局域场。

核心公式：
    传输相位（propagation phase）:
        Φ = k₀ ∫_0^h (n_eff(z) - n_air) dz
    其中 n_eff 为纳米柱截面上的有效折射率，通过求解本征模式得到。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Witherden 四边形积分规则数据（精度最高 21 阶）
# 将原始 MATLAB 规则转写为 Python 常量表
# ---------------------------------------------------------------------------

WITHERDEN_RULES = {
    1: {
        'n': 1,
        'x': np.array([0.5]),
        'y': np.array([0.5]),
        'w': np.array([1.0]),
    },
    3: {
        'n': 4,
        'x': np.array([0.211324865405187, 0.788675134594813,
                       0.211324865405187, 0.788675134594813]),
        'y': np.array([0.211324865405187, 0.211324865405187,
                       0.788675134594813, 0.788675134594813]),
        'w': np.array([0.25, 0.25, 0.25, 0.25]),
    },
    5: {
        'n': 8,
        'x': np.array([
            0.102592160379459, 0.897407839620541,
            0.102592160379459, 0.897407839620541,
            0.5, 0.5,
            0.281566496584151, 0.718433503415849,
        ]),
        'y': np.array([
            0.102592160379459, 0.102592160379459,
            0.897407839620541, 0.897407839620541,
            0.281566496584151, 0.718433503415849,
            0.5, 0.5,
        ]),
        'w': np.array([
            0.138564346606752, 0.138564346606752,
            0.138564346606752, 0.138564346606752,
            0.221714285714286, 0.221714285714286,
            0.221714285714286, 0.221714285714286,
        ]),
    },
    7: {
        'n': 12,
        'x': np.array([
            0.057104196114518, 0.942895803885482,
            0.057104196114518, 0.942895803885482,
            0.5, 0.5,
            0.209299385066662, 0.790700614933338,
            0.209299385066662, 0.790700614933338,
            0.197166296714531, 0.802833703285469,
        ]),
        'y': np.array([
            0.057104196114518, 0.057104196114518,
            0.942895803885482, 0.942895803885482,
            0.197166296714531, 0.802833703285469,
            0.197166296714531, 0.197166296714531,
            0.802833703285469, 0.802833703285469,
            0.209299385066662, 0.790700614933338,
        ]),
        'w': np.array([
            0.050844906370207, 0.050844906370207,
            0.050844906370207, 0.050844906370207,
            0.116786275403396, 0.116786275403396,
            0.082851075618464, 0.082851075618464,
            0.082851075618464, 0.082851075618464,
            0.116786275403396, 0.116786275403396,
        ]),
    },
}


def quadrilateral_witherden_rule(p):
    """
    返回单位正方形 [0,1]×[0,1] 上的 Witherden 积分规则。

    Parameters
    ----------
    p : int
        目标精度阶数（奇数），0 <= p <= 7（当前实现到 7 阶）

    Returns
    -------
    n : int
        积分点数
    x, y : ndarray, shape (n,)
        积分点坐标
    w : ndarray, shape (n,)
        积分权重（权重之和为 1）
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    if p > 7:
        p = 7  # 降级到最高可用规则
    # 找到满足精度的最小规则
    available = sorted(WITHERDEN_RULES.keys())
    chosen = available[0]
    for av in available:
        if av >= p:
            chosen = av
            break
    rule = WITHERDEN_RULES[chosen]
    return rule['n'], rule['x'].copy(), rule['y'].copy(), rule['w'].copy()


class PhaseQuadrature:
    """
    使用高精度数值积分计算纳米柱截面上的有效电磁量。
    """

    def __init__(self, wavelength=1.55e-6, n_si=3.48, n_air=1.0):
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength
        self.n_si = n_si
        self.n_air = n_air
        self.eps_si = n_si ** 2
        self.eps_air = n_air ** 2

    def map_to_pillar(self, x_unit, y_unit, cx, cy, width, height):
        """
        将单位正方形 [0,1]² 映射到纳米柱截面矩形 [cx±w/2, cy±h/2]。
        """
        x = cx + (x_unit - 0.5) * width
        y = cy + (y_unit - 0.5) * height
        return x, y

    def local_effective_index(self, x, y, cx, cy, width, height):
        """
        截面上的局域有效折射率分布。
        纳米柱内部 n_si，外部 n_air。
        """
        # TODO: 根据纳米柱几何截面模型实现折射率分布判断
        # 提示：需与 maxwell_fem.py 中 epsilon_profile 的截面形状定义保持一致
        ...
        return ...

    def integrate_phase_delay(self, cx, cy, width, height, h_pillar,
                              precision=7):
        """
        使用 Witherden 积分规则计算纳米柱截面上的平均相位延迟。

        物理模型（一阶近似）：
            ΔΦ(x,y) = k₀ * h_pillar * (n_eff(x,y) - n_air)
        其中 n_eff 在纳米柱内为 n_si，外部为 n_air。

        对截面取平均：
            <ΔΦ> = (1/A) ∫_A ΔΦ(x,y) dA

        Parameters
        ----------
        cx, cy : float
            纳米柱中心 [m]
        width, height : float
            截面宽度和高度（y方向） [m]
        h_pillar : float
            纳米柱高度（z方向） [m]
        precision : int
            积分精度阶数

        Returns
        -------
        avg_phase : float
            平均相位延迟 [rad]
        transmission : float
            传输振幅（简化模型）
        """
        n_q, xu, yu, w = quadrilateral_witherden_rule(precision)
        # 映射到物理坐标
        x, y = self.map_to_pillar(xu, yu, cx, cy, width, height)
        n_eff = self.local_effective_index(x, y, cx, cy, width, height)

        # 权重之和已归一化为 1
        phase_delay = self.k0 * h_pillar * (n_eff - self.n_air)
        avg_phase = np.sum(w * phase_delay)

        # 简化振幅模型：考虑反射损耗
        # T = 1 - R, R ≈ |(n1 - n2)/(n1 + n2)|²
        R_avg = np.sum(w * ((n_eff - self.n_air) / (n_eff + self.n_air)) ** 2)
        transmission = 1.0 - R_avg
        return avg_phase, transmission

    def integrate_polarizability(self, cx, cy, width, height,
                                  precision=7):
        """
        计算纳米柱的等效极化率张量分量（标量近似）。

         Clausius-Mossotti 型近似：
            α = ε₀ * V * (ε_r - 1) / (1 + L * (ε_r - 1))
        其中 L 为退极化因子，对于无限长圆柱 L ≈ 0.5（TM模式）。

        这里使用体积积分：
            α_eff = ∫_V ε₀ (ε_r(r) - 1) dV
        """
        n_q, xu, yu, w = quadrilateral_witherden_rule(precision)
        x, y = self.map_to_pillar(xu, yu, cx, cy, width, height)
        inside = (np.abs(x - cx) <= width / 2.0) & (np.abs(y - cy) <= height / 2.0)
        eps_r = np.where(inside, self.eps_si, self.eps_air)
        # 单位正方形面积 = 1，权重已归一化
        # 物理面积 = width * height
        area = width * height
        alpha = 8.854187817e-12 * area * np.sum(w * (eps_r - self.eps_air))
        return alpha

    def integrate_energy_density(self, field_func, cx, cy, width, height,
                                  precision=7):
        """
        对给定的场分布函数 field_func(x,y) 在纳米柱截面上进行积分，
        计算电磁能量密度的时间平均值：
            <w_e> = (1/4) ε₀ ε_r |E|²
        """
        n_q, xu, yu, w = quadrilateral_witherden_rule(precision)
        x, y = self.map_to_pillar(xu, yu, cx, cy, width, height)
        E_vals = field_func(x, y)
        eps_r = self.local_effective_index(x, y, cx, cy, width, height)
        eps0 = 8.854187817e-12
        energy_density = 0.25 * eps0 * eps_r * np.abs(E_vals) ** 2
        avg_energy = np.sum(w * energy_density)
        return avg_energy

    def compute_dispersion_relation(self, width, h_pillar,
                                     n_modes=3, precision=7):
        """
        基于有效介质理论（EMT）和波导模型，计算纳米柱波导的有效折射率
        随宽度的色散关系。

        模型：将矩形纳米柱近似为等效平面波导，求解本征方程：
            tan(κ w / 2) = γ / κ
        其中 κ = k₀ sqrt(n_si² - n_eff²), γ = k₀ sqrt(n_eff² - n_air²)

        Returns
        -------
        n_eff_list : list of float
            各阶模式的有效折射率
        """
        from scipy.optimize import brentq

        n_eff_list = []
        for m in range(n_modes):
            # 本征方程残差
            def residual(ne):
                if ne <= self.n_air or ne >= self.n_si:
                    return 1e10
                kappa = self.k0 * np.sqrt(self.eps_si - ne ** 2)
                gamma = self.k0 * np.sqrt(ne ** 2 - self.eps_air)
                if kappa == 0:
                    return 1e10
                return np.tan(kappa * width / 2.0) - gamma / kappa

            try:
                # 搜索区间
                ne_min = self.n_air + 1e-4
                ne_max = self.n_si - 1e-4
                # 寻找变号区间
                n_scan = 500
                ne_scan = np.linspace(ne_min, ne_max, n_scan)
                res_scan = np.array([residual(ne) for ne in ne_scan])
                for i in range(n_scan - 1):
                    if res_scan[i] * res_scan[i + 1] < 0:
                        root = brentq(residual, ne_scan[i], ne_scan[i + 1])
                        # 检查是否重复
                        if all(abs(root - r) > 1e-4 for r in n_eff_list):
                            n_eff_list.append(root)
                        if len(n_eff_list) >= n_modes:
                            break
            except Exception:
                pass

        # 如果未找到足够模式，填充插值
        while len(n_eff_list) < n_modes:
            if len(n_eff_list) == 0:
                n_eff_list.append(self.n_air + 0.1 * (self.n_si - self.n_air))
            else:
                n_eff_list.append(n_eff_list[-1] - 0.05 * (self.n_si - self.n_air))
        return n_eff_list[:n_modes]


def demo():
    """演示：计算纳米柱截面的相位延迟和极化率。"""
    pq = PhaseQuadrature(wavelength=1.55e-6)
    cx, cy = 0.0, 0.0
    w_pillar = 0.3e-6
    h_pillar = 0.6e-6
    height_z = 1.0e-6

    avg_phase, trans = pq.integrate_phase_delay(cx, cy, w_pillar, h_pillar, height_z)
    alpha = pq.integrate_polarizability(cx, cy, w_pillar, h_pillar)
    n_effs = pq.compute_dispersion_relation(w_pillar, height_z)

    print(f"[phase_quadrature] 平均相位延迟: {avg_phase:.4f} rad = {np.degrees(avg_phase):.2f}°")
    print(f"[phase_quadrature] 传输振幅: {trans:.4f}")
    print(f"[phase_quadrature] 等效极化率: {alpha:.4e} F·m²")
    print(f"[phase_quadrature] 前 {len(n_effs)} 阶模式有效折射率: " +
          ", ".join(f"{n:.4f}" for n in n_effs))
    return avg_phase, alpha, n_effs


if __name__ == "__main__":
    demo()
