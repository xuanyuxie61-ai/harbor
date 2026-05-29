"""
超弹性本构模型模块
==================
基于种子项目:
  - 1229_test_zero: 非线性方程求根算法(Brent法、Newton法)

科学背景:
  在大变形非线性有限元中，Neo-Hookean和Mooney-Rivlin模型是描述
  橡胶类超弹性材料的经典本构关系。本模块实现：
  1. 可压缩Neo-Hookean模型的应变能密度、第二Piola-Kirchhoff应力 S
     与材料切线模量 C
  2. 大变形问题的Newton-Raphson局部迭代中，利用Brent法求解
     等效剪切模量的非线性方程(在特定加载路径下)

关键公式:
  - 变形梯度: F = I + ∇u
  - 右Cauchy-Green张量: C = F^T F
  - 第三不变量: J = det(F) = sqrt(det(C))
  - Neo-Hookean应变能密度:
      ψ(C) = (μ/2)(tr(C) - 3) - μ ln(J) + (λ/2)(ln(J))^2
  - 第二Piola-Kirchhoff应力:
      S = 2 ∂ψ/∂C = μ(I - C^{-1}) + λ ln(J) C^{-1}
  - 材料切线模量(第4阶张量):
      C_{IJKL} = λ C^{-1}_{IJ} C^{-1}_{KL} + (μ - λ ln(J))
                 (C^{-1}_{IK} C^{-1}_{JL} + C^{-1}_{IL} C^{-1}_{JK})
  - 等效应力(von Mises型):
      σ_vm = sqrt(3/2 s:s)  其中 s = dev(σ)
"""

import numpy as np
from typing import Tuple, Optional


def deformation_gradient(dN_dX: np.ndarray, u_e: np.ndarray) -> np.ndarray:
    """
    由节点位移计算单元级别的变形梯度 F。
    F = I + ∇u = I + Σ u_i ⊗ dN_i/dX

    参数:
        dN_dX: (4, 3) 形函数对参考坐标的导数
        u_e: (4, 3) 单元节点位移

    返回:
        F: (3, 3) 变形梯度
    """
    F = np.eye(3, dtype=np.float64)
    for i in range(4):
        F += np.outer(u_e[i], dN_dX[i])
    return F


def right_cauchy_green(F: np.ndarray) -> np.ndarray:
    """右Cauchy-Green变形张量 C = F^T F"""
    return F.T @ F


def compute_invariants(C: np.ndarray) -> Tuple[float, float, float]:
    """
    计算C的主不变量 I1, I2, I3。
    I1 = tr(C), I2 = 0.5[(tr(C))^2 - tr(C^2)], I3 = det(C)
    """
    I1 = float(np.trace(C))
    I2 = 0.5 * (I1 ** 2 - float(np.trace(C @ C)))
    I3 = float(np.linalg.det(C))
    return I1, I2, I3


def neo_hookean_pk2_stress(C: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """
    计算可压缩Neo-Hookean材料的第二Piola-Kirchhoff应力 S。

    S = μ (I - C^{-1}) + λ ln(J) C^{-1}
    其中 J = sqrt(det(C))

    参数:
        C: (3, 3) 右Cauchy-Green张量
        mu: 剪切模量
        lam: Lamé第一参数 (体积模量相关)

    返回:
        S: (3, 3) 第二Piola-Kirchhoff应力
    """
    # TODO: Hole 1 - 实现Neo-Hookean PK2应力公式
    raise NotImplementedError("Hole 1: 请实现neo_hookean_pk2_stress")


def neo_hookean_material_tangent(C: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """
    计算可压缩Neo-Hookean材料的第4阶材料切线模量 C_{IJKL}，
    并以 (6, 6) Voigt矩阵形式返回。

    公式:
      C_{IJKL} = λ C^{-1}_{IJ} C^{-1}_{KL}
                + (μ - λ ln(J)) (C^{-1}_{IK} C^{-1}_{JL} + C^{-1}_{IL} C^{-1}_{JK})

    Voigt映射: 11→0, 22→1, 33→2, 12→3, 13→4, 23→5
    """
    # TODO: Hole 1 - 实现Neo-Hookean材料切线模量的Voigt矩阵
    raise NotImplementedError("Hole 1: 请实现neo_hookean_material_tangent")


def green_lagrange_strain(C: np.ndarray) -> np.ndarray:
    """
    Green-Lagrange应变张量:
      E = 1/2 (C - I)
    """
    return 0.5 * (C - np.eye(3))


def voigt_strain(E: np.ndarray) -> np.ndarray:
    """
    将 (3,3) 应变张量转换为6维Voigt向量。
    [E11, E22, E33, 2*E12, 2*E13, 2*E23]
    """
    return np.array([
        E[0, 0], E[1, 1], E[2, 2],
        2.0 * E[0, 1], 2.0 * E[0, 2], 2.0 * E[1, 2]
    ], dtype=np.float64)


def voigt_stress(S: np.ndarray) -> np.ndarray:
    """
    将 (3,3) 应力张量转换为6维Voigt向量。
    [S11, S22, S33, S12, S13, S23]
    """
    return np.array([
        S[0, 0], S[1, 1], S[2, 2],
        S[0, 1], S[0, 2], S[1, 2]
    ], dtype=np.float64)


def cauchy_stress_from_pk2(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    由第二Piola-Kirchhoff应力计算Cauchy应力:
      σ = (1/J) F S F^T
    """
    J = float(np.linalg.det(F))
    if abs(J) < 1e-14:
        raise ValueError("Cauchy应力: J 接近零")
    return (F @ S @ F.T) / J


def von_mises_cauchy(sigma: np.ndarray) -> float:
    """
    由Cauchy应力张量计算von Mises等效应力。
      σ_vm = sqrt(3/2 s:s)
    其中 s = σ - tr(σ)/3 I 为偏应力张量。
    """
    tr = float(np.trace(sigma))
    s = sigma - (tr / 3.0) * np.eye(3)
    return float(np.sqrt(1.5 * np.sum(s * s)))


# ========================================================================
# 基于 test_zero 的非线性求根: 等效剪切模量迭代
# ========================================================================

def _effective_shear_residual(mu_eff: float, mu0: float, gamma: float,
                               alpha: float = 0.1) -> float:
    """
    非线性等效剪切模量的隐式方程残差:
      r(μ_eff) = μ_eff - μ0 * exp(-α * γ * μ_eff)
    其中 γ 为等效剪应变，α 为损伤耦合系数。
    此方程在大变形损伤力学中出现，描述模量随变形的退化。
    """
    return mu_eff - mu0 * np.exp(-alpha * gamma * mu_eff)


def brent_method(f, a: float, b: float, tol: float = 1e-12,
                 max_iter: int = 100) -> float:
    """
    Brent法求根：结合二分、线性插值与反二次插值的混合策略。
    要求 f(a) 和 f(b) 异号。

    参数:
        f: 目标函数
        a, b: 初始区间
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        root: 方程的根
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Brent法要求 f(a)*f(b) < 0")

    c = a
    fc = fa
    s = b
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b = b, c
            c = a
            fa, fb = fb, fc
            fc = fa

        tol_act = 2.0 * tol * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or abs(fb) < tol:
            return b

        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                # 线性插值(割线法)
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # 反二次插值
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < (min1 if min1 < min2 else min2):
                e = d
                d = p / q
            else:
                d = e = m
        a = b
        fa = fb
        if abs(d) > tol_act:
            b += d
        else:
            b += tol_act if m > 0 else -tol_act
        fb = f(b)
    return b


def solve_effective_shear_modulus(mu0: float, gamma: float,
                                   alpha: float = 0.1) -> float:
    """
    使用Brent法求解损伤耦合下的等效剪切模量。
    方程: μ_eff = μ0 * exp(-α * γ * μ_eff)

    参数:
        mu0: 初始剪切模量
        gamma: 等效剪应变
        alpha: 损伤耦合系数

    返回:
        mu_eff: 等效剪切模量
    """
    if gamma < 0:
        raise ValueError("等效剪应变 γ 必须非负")
    if mu0 <= 0:
        raise ValueError("初始剪切模量 μ0 必须为正")

    # 确定 bracket: r(0) = -mu0 < 0, r(mu0*5) > 0 (对于小gamma)
    f = lambda m: _effective_shear_residual(m, mu0, gamma, alpha)
    a = 1e-8
    b = mu0 * 5.0
    # 确保异号，必要时扩展区间
    for _ in range(20):
        if f(a) * f(b) <= 0:
            break
        b *= 2.0
        if b > mu0 * 1e6:
            # 极端情况直接返回线性近似
            return mu0 * np.exp(-alpha * gamma * mu0)
    if f(a) * f(b) > 0:
        return mu0 * np.exp(-alpha * gamma * mu0)
    return brent_method(f, a, b, tol=1e-14)
