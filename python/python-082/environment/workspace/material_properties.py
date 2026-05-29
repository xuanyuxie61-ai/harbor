"""
material_properties.py
复合材料力学性能与细观力学均匀化。
科学背景：
  纤维增强复合材料的宏观等效弹性性能通过细观力学模型由纤维和基体性能预测。
  采用 Halpin-Tsai 方程与 Mori-Tanaka 方法进行多尺度均匀化。
核心公式：
  纵向模量（混合律）：E_1 = E_f * V_f + E_m * (1 - V_f)
  横向模量（Halpin-Tsai）：E_2 = E_m * (1 + ξ η V_f) / (1 - η V_f)
    其中 η = (E_f / E_m - 1) / (E_f / E_m + ξ)
  剪切模量：G_12 = G_m * (1 + ξ η G V_f) / (1 - η G V_f)
    其中 η G = (G_f / G_m - 1) / (G_f / G_m + ξ)
  泊松比：ν_12 = ν_f * V_f + ν_m * (1 - V_f)
  面内剪切：G_12 同上
"""

import numpy as np
from utils import validate_positive


class CompositeMaterial:
    """纤维/基体两相复合材料性能容器。"""

    def __init__(self, E_f, nu_f, E_m, nu_m, fiber_volume_fraction):
        """
        E_f: 纤维弹性模量 (GPa)
        nu_f: 纤维泊松比
        E_m: 基体弹性模量 (GPa)
        nu_m: 基体泊松比
        fiber_volume_fraction: 纤维体积分数 V_f ∈ [0, 1]
        """
        validate_positive(E_f, "E_f")
        validate_positive(E_m, "E_m")
        if not (0.0 <= fiber_volume_fraction <= 1.0):
            raise ValueError("Fiber volume fraction must be in [0, 1].")
        self.E_f = float(E_f)
        self.nu_f = float(nu_f)
        self.E_m = float(E_m)
        self.nu_m = float(nu_m)
        self.V_f = float(fiber_volume_fraction)
        self.V_m = 1.0 - self.V_f

        # 剪切模量
        self.G_f = self.E_f / (2.0 * (1.0 + self.nu_f))
        self.G_m = self.E_m / (2.0 * (1.0 + self.nu_m))

        # 均匀化性能
        self._homogenize()

    def _homogenize(self):
        """细观力学均匀化计算。"""
        Vf = self.V_f
        Vm = self.V_m

        # 纵向模量（混合律/Voigt模型）
        self.E1 = self.E_f * Vf + self.E_m * Vm

        # Halpin-Tsai 形状因子（圆形纤维 ξ ≈ 2）
        xi = 2.0

        # 横向模量 E2
        eta_E = (self.E_f / self.E_m - 1.0) / (self.E_f / self.E_m + xi)
        self.E2 = self.E_m * (1.0 + xi * eta_E * Vf) / (1.0 - eta_E * Vf)

        # 面内剪切模量 G12
        eta_G = (self.G_f / self.G_m - 1.0) / (self.G_f / self.G_m + xi)
        self.G12 = self.G_m * (1.0 + xi * eta_G * Vf) / (1.0 - eta_G * Vf)

        # 泊松比
        self.nu12 = self.nu_f * Vf + self.nu_m * Vm

        # 横向泊松比（近似）
        self.nu21 = self.nu12 * self.E2 / self.E1

        # 平面应力柔度矩阵 S
        self.S = np.array([
            [1.0 / self.E1, -self.nu21 / self.E2, 0.0],
            [-self.nu12 / self.E1, 1.0 / self.E2, 0.0],
            [0.0, 0.0, 1.0 / self.G12]
        ])

        # 平面应力刚度矩阵 Q
        self.Q = np.linalg.inv(self.S)

        # 三维工程常数
        self.E3 = self.E2
        self.G13 = self.G12
        self.G23 = self.G_m / (1.0 - np.sqrt(Vf) * (1.0 - self.G_m / self.G_f))
        self.nu13 = self.nu12
        self.nu23 = 0.3  # 近似值

    def compute_transformed_stiffness(self, theta_deg):
        """
        计算偏轴刚度矩阵 Q̄(θ)。
        转换公式（经典层合板理论）：
          Q̄_11 = Q_11 c^4 + 2(Q_12 + 2Q_66)s^2c^2 + Q_22 s^4
          Q̄_12 = (Q_11 + Q_22 - 4Q_66)s^2c^2 + Q_12(s^4 + c^4)
          Q̄_22 = Q_11 s^4 + 2(Q_12 + 2Q_66)s^2c^2 + Q_22 c^4
          Q̄_16 = (Q_11 - Q_12 - 2Q_66)sc^3 + (Q_12 - Q_22 + 2Q_66)s^3c
          Q̄_26 = (Q_11 - Q_12 - 2Q_66)s^3c + (Q_12 - Q_22 + 2Q_66)sc^3
          Q̄_66 = (Q_11 + Q_22 - 2Q_12 - 2Q_66)s^2c^2 + Q_66(s^4 + c^4)
        """
        theta = np.radians(float(theta_deg))
        c = np.cos(theta)
        s = np.sin(theta)
        c2 = c * c
        s2 = s * s
        s4 = s2 * s2
        c4 = c2 * c2
        s2c2 = s2 * c2

        Q11, Q12 = self.Q[0, 0], self.Q[0, 1]
        Q22, Q66 = self.Q[1, 1], self.Q[2, 2]

        Q_bar = np.zeros((3, 3))
        Q_bar[0, 0] = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
        Q_bar[0, 1] = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)
        Q_bar[1, 0] = Q_bar[0, 1]
        Q_bar[1, 1] = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
        Q_bar[0, 2] = (Q11 - Q12 - 2.0 * Q66) * s * c2 * c + (Q12 - Q22 + 2.0 * Q66) * s2 * s * c
        Q_bar[2, 0] = Q_bar[0, 2]
        Q_bar[1, 2] = (Q11 - Q12 - 2.0 * Q66) * s2 * s * c + (Q12 - Q22 + 2.0 * Q66) * s * c2 * c
        Q_bar[2, 1] = Q_bar[1, 2]
        Q_bar[2, 2] = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)
        return Q_bar

    def compute_degraded_stiffness(self, d_f, d_m, d_s):
        """
        根据损伤变量计算退化刚度矩阵。
        基于连续损伤力学（CDM）的刚度退化：
          Ẽ_1 = (1 - d_f) * E_1    (纤维方向损伤)
          Ẽ_2 = (1 - d_m) * E_2    (基体方向损伤)
          G̃_12 = (1 - d_s) * G_12  (剪切损伤)
          ν̃_12 = ν_12 * (1 - d_f)  （泊松比退化，简化模型）
        返回退化后的平面应力刚度矩阵 Q_d。
        """
        d_f = np.clip(d_f, 0.0, 0.99)
        d_m = np.clip(d_m, 0.0, 0.99)
        d_s = np.clip(d_s, 0.0, 0.99)

        E1d = (1.0 - d_f) * self.E1
        E2d = (1.0 - d_m) * self.E2
        G12d = (1.0 - d_s) * self.G12
        nu12d = self.nu12 * (1.0 - d_f)
        nu21d = nu12d * E2d / E1d

        S_d = np.array([
            [1.0 / E1d, -nu21d / E2d, 0.0],
            [-nu12d / E1d, 1.0 / E2d, 0.0],
            [0.0, 0.0, 1.0 / G12d]
        ])

        # 边界处理：若退化后矩阵接近奇异，进行正则化
        det_S = np.linalg.det(S_d)
        if abs(det_S) < 1e-20:
            S_d += 1e-12 * np.eye(3)

        Q_d = np.linalg.inv(S_d)
        return Q_d

    def print_properties(self):
        """打印均匀化材料性能。"""
        print("-" * 50)
        print("Composite Homogenized Properties")
        print("-" * 50)
        print(f"  Fiber volume fraction V_f = {self.V_f:.4f}")
        print(f"  Longitudinal modulus E_1   = {self.E1:.4f} GPa")
        print(f"  Transverse modulus E_2     = {self.E2:.4f} GPa")
        print(f"  In-plane shear modulus G_12= {self.G12:.4f} GPa")
        print(f"  Major Poisson ratio nu_12  = {self.nu12:.4f}")
        print(f"  Minor Poisson ratio nu_21  = {self.nu21:.4f}")
        print("-" * 50)


def create_carbon_epoxy(V_f=0.6):
    """创建标准碳纤维/环氧树脂复合材料。"""
    # T300碳纤维 / 环氧树脂基体（典型值）
    E_f = 230.0   # GPa
    nu_f = 0.20
    E_m = 3.5     # GPa
    nu_m = 0.35
    return CompositeMaterial(E_f, nu_f, E_m, nu_m, V_f)
