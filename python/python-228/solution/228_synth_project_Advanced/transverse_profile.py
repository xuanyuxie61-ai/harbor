"""
transverse_profile.py — 横向簇射剖面建模 (Molière 理论)
========================================================

融合种子项目:
  - 176_circle_arc_grid : 圆弧网格 (横向曲率修正)
  - 678_line_fekete_rule : Fekete 点 (最优采样位置)

本模块实现电磁簇射的横向能量分布模型:

1. Molière 参数化:
   f(r) = (1/R_M^2) * [C1 * exp(-r/(R_M*s1)) + C2 * exp(-r/(R_M*s2))]
   其中 C1 ≈ 0.22, s1 ≈ 0.84, C2 ≈ 0.78, s2 ≈ 2.33

2. NKG (Nishimura-Kamata-Greisen) 函数:
   f(r) = C(s) * (r/R_M)^(s-2) * (1 + r/R_M)^(s-4.5)
   其中 s 为年龄参数 (shower age):
     s = 3*t / (t + 2*t_max)
     t = depth, t_max = shower maximum

3. 横向积分:
   F(R) = 2*pi * integral_0^R f(r) * r dr
   归一化: F(inf) = 1

4. 多能标修正 (高能时横向扩展更大):
   R_eff = R_M * (1 + 0.15 * ln(E/E_c))
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from material_properties import MaterialSpec


@dataclass
class TransverseProfileConfig:
    """横向剖面配置"""
    n_radial_points: int = 64
    r_max_moliere: float = 10.0    # 最大半径 (Molière 半径单位)
    age_parameter: Optional[float] = None  # None 表示自动计算
    nkg_s_parameter: float = 1.0    # NKG 年龄参数


class TransverseProfileModel:
    """横向簇射剖面模型"""

    # Molière 函数系数
    MOLIERE_C1 = 0.2219
    MOLIERE_S1 = 0.8394
    MOLIERE_C2 = 0.7781
    MOLIERE_S2 = 2.3286

    def __init__(self, material: MaterialSpec):
        self.mat = material

    def moliere_function(self, r_cm: float, E_MeV: float = 1000.0) -> float:
        """
        Molière 横向分布函数 f(r) [1/cm^2]

        f(r) = (1/R_M^2) * sum_i C_i/s_i * exp(-r / (s_i * R_eff))

        归一化: 2*pi * integral_0^inf f(r) * r dr = 1

        高能修正:
          R_eff = R_M * (1 + alpha * ln(E/Ec))
          alpha ≈ 0.08 (高能横向扩展增强)
        """
        R_M = self.mat.moliere_radius_cm
        alpha_correction = 1.0 + 0.08 * math.log(max(E_MeV / self.mat.Ec_MeV, 1.0))
        R_eff = R_M * alpha_correction

        if r_cm < 0:
            return 0.0
        x = r_cm / R_eff
        if x < 1e-10:
            # 中心极限: f(0) = (C1/s1 + C2/s2) / R_eff^2
            return (self.MOLIERE_C1 / self.MOLIERE_S1 +
                    self.MOLIERE_C2 / self.MOLIERE_S2) / (R_eff * R_eff)

        term1 = (self.MOLIERE_C1 / self.MOLIERE_S1) * math.exp(-x / self.MOLIERE_S1)
        term2 = (self.MOLIERE_C2 / self.MOLIERE_S2) * math.exp(-x / self.MOLIERE_S2)
        return (term1 + term2) / (R_eff * R_eff)

    def nkg_function(
        self, r_cm: float, age_s: float, E_MeV: float = 1000.0,
    ) -> float:
        """
        NKG (Nishimura-Kamata-Greisen) 横向分布函数

        f(r, s) = C(s) / R_M^2 * (r/R_M)^(s-2) * (1 + r/R_M)^(s-4.5)

        归一化常数:
          C(s) = Gamma(4.5-s) / (2*pi * Gamma(s) * Gamma(4.5-2*s))

        适用范围: 0.5 < s < 2.0
          s < 1: shower 早期 (前极大)
          s = 1: 极大值附近
          s > 1: shower 晚期 (后极大)

        对于 s → 2: f(r) → delta(r)/(2*pi*r) (完全准直)
        """
        R_M = self.mat.moliere_radius_cm
        if r_cm <= 0:
            if age_s > 2.0:
                return 0.0
            elif abs(age_s - 2.0) < 0.1:
                return 1.0 / (math.pi * R_M * R_M * 0.01)  # 正则化
            else:
                return 0.0

        x = r_cm / R_M
        s = max(0.3, min(2.5, age_s))

        # 计算归一化常数 C(s)
        try:
            log_C = (
                math.lgamma(4.5 - s)
                - math.log(2.0 * math.pi)
                - math.lgamma(s)
                - math.lgamma(max(4.5 - 2.0 * s, 0.01))
            )
        except (ValueError, OverflowError):
            log_C = -math.log(2.0 * math.pi * R_M * R_M)

        C = math.exp(log_C) / (R_M * R_M)

        # NKG 函数 (对数计算防止溢出)
        log_f = (
            math.log(max(C, 1e-300))
            + (s - 2.0) * math.log(x)
            + (s - 4.5) * math.log(1.0 + x)
        )

        if log_f > 500:
            return float('inf')
        if log_f < -700:
            return 0.0
        return math.exp(log_f)

    def compute_age_parameter(
        self, depth_X0: float, E0_MeV: float,
    ) -> float:
        """
        计算簇射年龄参数 s(t)

        s(t) = 3*t / (t + 2*t_max)

        其中 t_max = ln(E0/Ec) / ln(2) - 0.5 (光子) 或 - 1.0 (电子)

        物理含义:
          s → 0: t → 0 (极早期, 粒子数增长)
          s = 1: t = t_max (极大值)
          s → 3: t → inf (完全衰减)
        """
        if E0_MeV <= self.mat.Ec_MeV:
            return 1.0
        t_max = math.log(E0_MeV / self.mat.Ec_MeV) / math.log(2.0) - 1.0
        t_max = max(t_max, 0.5)
        return 3.0 * depth_X0 / (depth_X0 + 2.0 * t_max)

    def transverse_integral(
        self, R_cm: float, depth_X0: float, E0_MeV: float,
        n_quad: int = 64,
    ) -> float:
        """
        计算半径 R 内的横向积分 F(R) = 2*pi * integral_0^R f(r)*r*dr

        使用 Gauss-Legendre 求积 (从 quadrature 模块移植核心逻辑):
          F(R) ≈ 2*pi * sum_i w_i * f(r_i) * r_i
          r_i 为 GL 节点映射到 [0, R]
        """
        age_s = self.compute_age_parameter(depth_X0, E0_MeV)

        # Gauss-Legendre 节点和权重 (简化版, 使用 Chebyshev 近似初值)
        nodes, weights = self._gauss_legendre_nodes(n_quad)

        # 映射到 [0, R]
        total = 0.0
        for i in range(n_quad):
            r_i = R_cm * (nodes[i] + 1.0) / 2.0
            w_i = weights[i] * R_cm / 2.0
            f_val = self.nkg_function(r_i, age_s, E0_MeV)
            if math.isfinite(f_val):
                total += w_i * f_val * r_i

        return 2.0 * math.pi * total

    def lateral_moment(
        self, n: int, depth_X0: float, E0_MeV: float,
    ) -> float:
        """
        计算横向 n 阶矩 <r^n> = 2*pi * integral_0^inf f(r) * r^(n+1) dr

        解析结果 (Molière 函数):
          <r^n> = R_eff^n * n! * (C1 * s1^n + C2 * s2^n)

        对于 NKG 函数, 需数值积分。
        """
        R_eff = self.mat.moliere_radius_cm
        if n == 0:
            return 1.0  # 归一化
        elif n == 1:
            # 平均半径 <r>
            return R_eff * (
                self.MOLIERE_C1 * self.MOLIERE_S1 ** 2 +
                self.MOLIERE_C2 * self.MOLIERE_S2 ** 2
            )
        elif n == 2:
            # RMS 半径
            return R_eff * math.sqrt(2.0 * (
                self.MOLIERE_C1 * self.MOLIERE_S1 ** 3 +
                self.MOLIERE_C2 * self.MOLIERE_S2 ** 3
            ))
        else:
            # 一般阶: 解析公式
            return R_eff ** n * math.factorial(n) * (
                self.MOLIERE_C1 * self.MOLIERE_S1 ** (n + 1) +
                self.MOLIERE_C2 * self.MOLIERE_S2 ** (n + 1)
            )

    def generate_fekete_sampling_points(
        self, R_cm: float, n_points: int,
    ) -> List[Tuple[float, float, float]]:
        """
        生成 Fekete 采样点 (最优横向采样位置)

        Fekete 点最大化 Vandermonde 行列式, 在多项式插值中
        提供最优节点分布。

        对于圆盘区域, Fekete 点近似为:
          r_i = R * j_{0,i} / j_{0,n}  (Bessel 函数零点)
          theta_i = 2*pi*i/n_phi

        其中 j_{0,i} 为 J_0(x) 的第 i 个零点。
        """
        points = []
        # Bessel 函数零点近似 ( McMahon 展开)
        bessel_zeros = []
        for k in range(1, n_points + 1):
            # j_{0,k} ≈ (k - 0.25) * pi
            mu = k - 0.25
            j0k = mu * math.pi - 1.0 / (8.0 * mu * math.pi)
            bessel_zeros.append(j0k)

        if not bessel_zeros:
            return points

        j0_max = bessel_zeros[-1]

        # 角向分布
        n_phi = max(4, int(math.sqrt(n_points)))

        for k, j0k in enumerate(bessel_zeros):
            r = R_cm * j0k / j0_max
            # 每环的角向点数正比于半径
            n_phi_ring = max(1, int(n_phi * j0k / j0_max))
            for j in range(n_phi_ring):
                theta = 2.0 * math.pi * j / n_phi_ring + (k % 2) * math.pi / n_phi_ring
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                weight = 1.0 / (1.0 + k)  # Fekete 权重近似
                points.append((x, y, weight))

        return points

    def _gauss_legendre_nodes(self, n: int) -> Tuple[List[float], List[float]]:
        """计算 n 点 Gauss-Legendre 节点和权重 (Newton 迭代)"""
        nodes = []
        weights = []

        for i in range(n):
            # 初始猜测 (Chebyshev 节点)
            x = math.cos(math.pi * (i + 0.75) / (n + 0.5))

            # Newton 迭代求 P_n(x) = 0
            for _ in range(30):
                p0 = 1.0
                p1 = x
                for j in range(2, n + 1):
                    p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j
                    p0 = p1
                    p1 = p2
                # p1 = P_n(x), 导数:
                dp = n * (x * p1 - p0) / (x * x - 1.0 + 1e-30)
                dx = -p1 / (dp + 1e-30)
                x += dx
                if abs(dx) < 1e-15:
                    break

            nodes.append(x)
            w = 2.0 / ((1.0 - x * x) * dp * dp + 1e-30)
            weights.append(w)

        return nodes, weights
