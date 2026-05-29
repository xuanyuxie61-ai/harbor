"""
hyperelastic_material.py
博士级大变形非线性有限元分析 — 超弹性材料本构模型模块

融合原项目:
  - 981_r8ge: 张量运算（矩阵乘法、转置）

核心数学:
  1. 变形梯度 (Deformation Gradient):
     F = I + ∇u
     其中 u 为位移场，∇u 为位移梯度张量 (3x3)

  2. 右 Cauchy-Green 变形张量:
     C = F^T F

  3. Green-Lagrange 应变张量:
     E = 1/2 (C - I)

  4. 体积比:
     J = det(F)

  5. 可压缩 Neo-Hookean 超弹性模型:
     应变能密度函数:
       Ψ(C) = (μ/2)(I1_bar - 3) + (K/2)(ln(J))^2
     其中:
       I1 = tr(C)
       I1_bar = J^{-2/3} I1   (体积修正的第一不变量)
       μ = G = E / (2(1+ν))   (剪切模量)
       K = E / (3(1-2ν))      (体积模量)

  6. 第二 Piola-Kirchhoff 应力张量:
     S = 2 ∂Ψ/∂C
       = μ J^{-2/3} (I - (1/3) I1 C^{-1}) + K ln(J) C^{-1}

  7. 材料切线模量 (第四阶张量):
     C_mat = 4 ∂²Ψ/∂C∂C
     对于数值实现，使用 Voigt 记法将 S 和 C_mat 表示为向量和矩阵:
       S_voigt = [S11, S22, S33, S12, S23, S13]^T
       C_voigt: (6x6) 矩阵

  8. 一致切线刚度 (Consistent Tangent Modulus):
     在有限元实现中，单元刚度矩阵包含几何非线性项:
       K_mat = ∫ B_L^T C_voigt B_L dV
       K_geo = ∫ G^T S G dV   (几何刚度)
       K_T = K_mat + K_geo
     其中 B_L 为线性应变-位移矩阵，G 为形函数梯度矩阵
"""

import numpy as np


class NeoHookeanMaterial:
    def __init__(self, young_modulus, poisson_ratio, damage=0.0):
        """
        可压缩 Neo-Hookean 超弹性材料

        参数:
            young_modulus: 杨氏模量 E (MPa)
            poisson_ratio: 泊松比 ν
            damage: 损伤变量 D ∈ [0, 1]，用于刚度折减
        """
        self.E = float(young_modulus)
        self.nu = float(poisson_ratio)
        self.D = float(damage)

        # 剪切模量和体积模量
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        # 考虑损伤的折减
        self.mu_d = self.mu * (1.0 - self.D)
        self.K_d = self.K * (1.0 - self.D)

        # 边界检查
        if not (0.0 <= self.nu < 0.5):
            raise ValueError(f"Poisson ratio must be in [0, 0.5), got {self.nu}")
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if not (0.0 <= self.D <= 1.0):
            raise ValueError(f"Damage must be in [0,1], got {self.D}")

    def update_damage(self, D_new):
        """更新损伤变量并重新计算折减模量"""
        self.D = float(np.clip(D_new, 0.0, 1.0))
        self.mu_d = self.mu * (1.0 - self.D)
        self.K_d = self.K * (1.0 - self.D)

    def compute_stress_tangent(self, F):
        """
        基于变形梯度 F 计算第二 Piola-Kirchhoff 应力 S 和一致切线模量 C_voigt

        输入:
            F: (3,3) 变形梯度张量
        输出:
            S: (3,3) 第二 PK 应力
            C_voigt: (6,6) 材料切线模量（Voigt 记法）
            J: 体积比
        """
        F = np.array(F, dtype=float)
        J = np.linalg.det(F)

        # 数值鲁棒性: 避免 J 接近零、负或过大导致溢出
        J = float(np.clip(J, 0.01, 100.0))
        if J < 1e-8:
            J = 1e-8

        # C = F^T F
        C = F.T @ F

        # C^{-1}
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.eye(3)

        I1 = np.trace(C)
        I3 = max(J ** 2, 1e-8)  # det(C) = J^2
        I3_inv = 1.0 / I3

        # ln(J)
        lnJ = np.log(J)

        # 第二 PK 应力 S
        # S = μ_d * J^{-2/3} * (I - (1/3) I1 C^{-1}) + K_d * ln(J) * C^{-1}
        Jm23 = J ** (-2.0 / 3.0)
        S = self.mu_d * Jm23 * (np.eye(3) - (I1 / 3.0) * C_inv) + self.K_d * lnJ * C_inv

        # 材料切线模量 C_voigt (简化实现，基于解析公式)
        C_voigt = self._compute_material_tangent(C, C_inv, I1, J, lnJ)

        return S, C_voigt, J

    def _compute_material_tangent(self, C, C_inv, I1, J, lnJ):
        """
        计算材料切线模量的 Voigt 表示 (6x6)

        数学:
          对于大变形分析，使用 Saint Venant-Kirchhoff 近似:
          C_mat = 4 ∂²Ψ/∂C∂C ≈ 常数各向同性弹性张量

          λ = K_d - 2μ_d/3   (第一 Lamé 参数)
          μ = μ_d            (第二 Lamé 参数 / 剪切模量)

          Voigt 矩阵:
            [λ+2μ, λ,    λ,    0, 0, 0]
            [λ,    λ+2μ, λ,    0, 0, 0]
            [λ,    λ,    λ+2μ, 0, 0, 0]
            [0,    0,    0,    μ, 0, 0]
            [0,    0,    0,    0, μ, 0]
            [0,    0,    0,    0, 0, μ]
        """
        lam = self.K_d - 2.0 * self.mu_d / 3.0
        mu = self.mu_d

        C6 = np.zeros((6, 6))
        for i in range(3):
            C6[i, i] = lam + 2.0 * mu
            for j in range(3):
                if i != j:
                    C6[i, j] = lam
        for i in range(3, 6):
            C6[i, i] = mu

        return C6

    def compute_strain_energy(self, F):
        """
        计算应变能密度 Ψ
        """
        J = np.linalg.det(F)
        if J < 1e-8:
            J = 1e-8
        C = F.T @ F
        I1 = np.trace(C)
        Jm23 = J ** (-2.0 / 3.0)
        lnJ = np.log(J)
        Psi = 0.5 * self.mu_d * (Jm23 * I1 - 3.0) + 0.5 * self.K_d * (lnJ ** 2)
        return Psi


def compute_deformation_gradient(grad_u):
    """
    由位移梯度计算变形梯度

    数学:
      F = I + ∇u
      (∇u)_{ij} = ∂u_i / ∂X_j
    """
    grad_u = np.array(grad_u, dtype=float)
    if grad_u.shape != (3, 3):
        raise ValueError("grad_u must be 3x3")
    return np.eye(3) + grad_u


def green_lagrange_strain(F):
    """
    由变形梯度计算 Green-Lagrange 应变

    数学:
      E = 1/2 (F^T F - I)
    """
    C = F.T @ F
    return 0.5 * (C - np.eye(3))


def voigt_stress_tensor(S):
    """
    将 3x3 对称应力张量转换为 6x1 Voigt 向量
    [S11, S22, S33, S12, S23, S13]^T
    """
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[0, 2]])
