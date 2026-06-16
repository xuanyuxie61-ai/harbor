"""
isobar_amplitude.py
-------------------
Isobar 模型下三体 B 衰变振幅的参数化, 包含:
  * 两体 Breit-Wigner 共振振幅
  * 角分布的 Legendre 多项式展开 (映射自 662_legendre_product)
  * 全振幅 = Σ_r a_r * BW_r(s) * F_L(p) * Z_L(Ω)
  * 最小曲面精确解作为振幅在共形极限下的解析参考 (映射自 768)

物理公式:
  对 B → R(→ h1 h2) h3, 其中 R 为自旋 J 的中间共振态:
    A_r(s_12, s_13) = a_r * e^{i delta_r} * BW_r(s_12)
                       * F_L^{Blatt}(p_B) * F_L^{Blatt}(p_R)
                       * Z_J^m(cos theta_12) * exp(i m phi_12)
  其中:
    - a_r, delta_r: 实数幅值和强相位 (待拟合参数)
    - BW_r(s): 相对论性 Breit-Wigner 线形, 带运行宽度
    - F_L: Blatt-Weisskopf 势垒因子 (L 为 B 衰变轨道角动量)
    - Z_J^m: 角函数 (与 Legendre 多项式 / 球谐函数相关)

  总衰变率:
    |A(s_12, s_13)|^2 = |Σ_r A_r|^2
                      = Σ_{r,s} A_r A_s^*

  Legendre 展开 (来自 662):
    对角分布 dΓ/dcosθ 做展开:
      f(cosθ) = Σ_l c_l P_l(cosθ),    c_l = (2l+1)/2 * ∫_{-1}^{1} f(x) P_l(x) dx
    对指数型 CP 相位因子 exp(i δ_weak cosθ) 的乘积积分
      T_{ij} = ∫_{-1}^{1} exp(i b x) P_i(x) P_j(x) dx
    用于构建角分布的耦合矩阵。

  最小曲面参考 (来自 768):
    在共形映射 (s12, s13) → (u, v) 下, 等幅线满足最小曲面方程
      (1 + A_u^2) A_{vv} - 2 A_u A_v A_{uv} + (1 + A_v^2) A_{uu} = 0
    其 catenoid 解 A(u,v) = acosh(a sqrt(u^2+v^2))/a
    可作为无共振背景下的解析参考振幅。
"""

from __future__ import annotations
import math
import cmath
from typing import List, Tuple

import numpy as np

from b_physics_constants import (
    breit_wigner_running, blatt_weisskopf, BLATT_RADIUS_DEFAULT,
)


# -------------------------------------------------------------------------- #
#                        Legendre 多项式 (来自 662)                          #
# -------------------------------------------------------------------------- #
def legendre_polynomial(p: int, x: float) -> np.ndarray:
    """
    返回 L(0), L(1), ..., L(p) 在 x 处的值。
    采用 Bonnet 递推:
      (n+1) L_{n+1}(x) = (2n+1) x L_n(x) - n L_{n-1}(x)
      L_0(x) = 1,  L_1(x) = x
    """
    if p < 0:
        raise ValueError("Legendre 阶数 p 必须非负")
    L = np.zeros(p + 1, dtype=float)
    L[0] = 1.0
    if p >= 1:
        L[1] = x
    for n in range(1, p):
        L[n + 1] = ((2 * n + 1) * x * L[n] - n * L[n - 1]) / (n + 1)
    return L


def legendre_quadrature(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    高斯-勒让德求积节点和权重 (在 [-1, 1] 上)。
    使用 Golub-Welsch 算法: 构造 Jacobi 三对角矩阵, 求其特征值与特征向量。
    """
    if order < 1:
        raise ValueError("求积阶数至少为 1")
    n = order
    i = np.arange(1, n, dtype=float)
    beta = i / np.sqrt(4.0 * i * i - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    weights = 2.0 * eigvecs[0, :] ** 2
    return eigvals, weights


def legendre_exponential_product_table(p: int, b: complex) -> np.ndarray:
    """
    计算指数加权 Legendre 乘积表 (仿 662):
        T_{ij} = ∫_{-1}^{1} exp(b x) L_i(x) L_j(x) dx
    其中 b 可以是复数, 对应 CP 破坏相位因子 exp(i δ_weak)。
    使用 Gauss-Legendre 求积 (非精确, 因含指数因子), 阶数 ≥ (3p+4)/2。
    """
    order = (3 * p + 4) // 2 + 1
    xq, wq = legendre_quadrature(order)
    table = np.zeros((p + 1, p + 1), dtype=complex)
    for k in range(order):
        x = xq[k]
        L = legendre_polynomial(p, x)
        # 外积: table += w_k * exp(b x) * L_i * L_j
        weight_factor = wq[k] * cmath.exp(b * x)
        for i in range(p + 1):
            for j in range(p + 1):
                table[i, j] += weight_factor * L[i] * L[j]
    return table


def legendre_linear_product_table(p: int, e: int) -> np.ndarray:
    """
    计算线性加权 Legendre 乘积表 (仿 662):
        T_{ij} = ∫_{-1}^{1} x^e L_i(x) L_j(x) dx
    当 e 为偶数且 e ≤ 2p 时, 可用 Gauss-Legendre 求积精确计算。
    """
    order = p + 1 + (e + 1) // 2 + 1
    xq, wq = legendre_quadrature(order)
    table = np.zeros((p + 1, p + 1), dtype=float)
    for k in range(order):
        x = xq[k]
        L = legendre_polynomial(p, x)
        weight_factor = wq[k] * (x ** e)
        for i in range(p + 1):
            for j in range(p + 1):
                table[i, j] += weight_factor * L[i] * L[j]
    return table


# -------------------------------------------------------------------------- #
#                        Isobar 共振子振幅                                  #
# -------------------------------------------------------------------------- #
class IsobarResonance:
    """
    一个中间共振态 R 的参数:
      - name:     共振态名称 (如 "rho(770)", "D*0(2007)")
      - m_r:      质量 (MeV)
      - gamma_r:  总宽度 (MeV)
      - J:        自旋
      - L_parent: B → R h 衰变中的轨道角动量
      - L_decay:  R → h1 h2 衰变中的轨道角动量 (= J)
      - a, delta: 实数幅值和强相位 (相对参考道)
      - daug1, daug2: 两个子粒子的质量 (MeV)
      - spectator: 旁观者粒子质量 (MeV)
    """
    def __init__(
            self,
            name: str,
            m_r: float,
            gamma_r: float,
            J: int,
            L_parent: int,
            daug1: float,
            daug2: float,
            spectator: float,
            m_parent: float,
            a: float = 1.0,
            delta: float = 0.0,
            radius: float = BLATT_RADIUS_DEFAULT,
    ):
        self.name = name
        self.m_r = m_r
        self.gamma_r = gamma_r
        self.J = J
        self.L_parent = L_parent
        self.L_decay = J   # R → h1 h2, 轨道角动量 = 自旋 J
        self.daug1 = daug1
        self.daug2 = daug2
        self.spectator = spectator
        self.m_parent = m_parent
        self.a = a
        self.delta = delta
        self.radius = radius

    def evaluate(self, s_res: complex, cos_theta: float) -> complex:
        """
        计算单一共振态对总振幅的贡献 (在给定 s_res 和 helicity 角 theta 下):
            A_r(s, cosθ) = a * exp(i delta) * BW_r(s) * F_L_parent(p_B)
                            * F_L_decay(p_R) * P_J(cosθ)
        其中:
            p_B = 两体动量在 B 静止系 (s_res 固定, 旁观者与 R 系统)
            p_R = 两体动量在 R 静止系 (s_res, m1, m2)
        """
        bw = breit_wigner_running(
            s=s_res,
            m_r=self.m_r,
            gamma_r=self.gamma_r,
            m_parent=self.m_r,   # 用于运行宽度参考
            m_daug1=self.daug1,
            m_daug2=self.daug2,
            L=self.L_decay,
            radius=self.radius,
        )
        # 计算 p_R (R → daug1 daug2 两体动量)
        p_R = _two_body_momentum(s_res, self.daug1, self.daug2)
        p_R_ref = _two_body_momentum(self.m_r * self.m_r, self.daug1, self.daug2)
        F_L_decay = blatt_weisskopf(p_R * self.radius, self.L_decay)
        F_L_decay_ref = blatt_weisskopf(p_R_ref * self.radius, self.L_decay)
        if F_L_decay_ref > 1.0e-30:
            F_L_decay_ratio = F_L_decay / F_L_decay_ref
        else:
            F_L_decay_ratio = 1.0

        # 计算 p_B (B → R + spectator 两体动量)
        M_R = math.sqrt(max(s_res.real, 0.0))
        p_B = _two_body_momentum(
            self.m_parent * self.m_parent, M_R, self.spectator
        )
        p_B_ref = _two_body_momentum(
            self.m_parent * self.m_parent, self.m_r, self.spectator
        )
        F_L_parent = blatt_weisskopf(p_B * self.radius, self.L_parent)
        F_L_parent_ref = blatt_weisskopf(p_B_ref * self.radius, self.L_parent)
        if F_L_parent_ref > 1.0e-30:
            F_L_parent_ratio = F_L_parent / F_L_parent_ref
        else:
            F_L_parent_ratio = 1.0

        # 角函数: Z_J(cosθ) = P_J(cosθ) (helicity = 0 简化)
        L_vals = legendre_polynomial(self.J, cos_theta)
        Z_J = L_vals[self.J]

        phase = cmath.exp(1j * self.delta)
        return self.a * phase * bw * F_L_parent_ratio * F_L_decay_ratio * Z_J


def _two_body_momentum(M_sq: float, m1: float, m2: float) -> float:
    """两体动量 |p| = sqrt(lambda(M^2, m1^2, m2^2)) / (2 M)。"""
    if M_sq <= (m1 + m2) ** 2:
        return 0.0
    lam = (M_sq - (m1 + m2) ** 2) * (M_sq - (m1 - m2) ** 2)
    if lam <= 0.0:
        return 0.0
    return math.sqrt(lam) / (2.0 * math.sqrt(M_sq))


# -------------------------------------------------------------------------- #
#                   总 Isobar 振幅 (含 CP 破坏)                             #
# -------------------------------------------------------------------------- #
class IsobarAmplitude:
    """
    总衰变振幅 = B 直接衰变道 (非共振) + Σ_r 共振贡献。
    对 B+ 和 B- 衰变, 弱相位变号:
        A^+(s, cosθ) = A_NR + Σ_r a_r e^{i(delta_r + gamma_3)} BW_r P_J
        A^-(s, cosθ) = A_NR + Σ_r a_r e^{i(delta_r - gamma_3)} BW_r P_J
    其中 gamma_3 为 CKM 角 gamma (phi_3)。
    直接 CP 不对称:
        A_CP^{dir}(s, cosθ) = (|A^+|^2 - |A^-|^2) / (|A^+|^2 + |A^-|^2)
    """
    def __init__(
            self,
            resonances: List[IsobarResonance],
            A_NR: complex = 0.0 + 0.0j,
            gamma_weak: float = 1.218,  # gamma ~ 70° ≈ 1.22 rad
    ):
        self.resonances = resonances
        self.A_NR = A_NR
        self.gamma_weak = gamma_weak

    def evaluate(self, s_res: complex, cos_theta: float, charge_sign: int = +1):
        """
        计算总振幅。charge_sign = +1 对应 B^-, -1 对应 B^+ (弱相位变号)。
        """
        total = self.A_NR + 0.0j
        for r in self.resonances:
            amp = r.evaluate(s_res, cos_theta)
            # 弱相位: 对 B- 为 +gamma, 对 B+ 为 -gamma
            weak_phase = cmath.exp(1j * charge_sign * self.gamma_weak)
            total += amp * weak_phase
        return total

    def cp_asymmetry_direct(self, s_res: complex, cos_theta: float) -> float:
        """
        直接 CP 不对称:
            A_CP^{dir} = (|A^-|^2 - |A^+|^2) / (|A^-|^2 + |A^+|^2)
        """
        A_minus = self.evaluate(s_res, cos_theta, charge_sign=+1)
        A_plus  = self.evaluate(s_res, cos_theta, charge_sign=-1)
        num = abs(A_minus) ** 2 - abs(A_plus) ** 2
        den = abs(A_minus) ** 2 + abs(A_plus) ** 2
        if den < 1.0e-30:
            return 0.0
        return num / den


# -------------------------------------------------------------------------- #
#          最小曲面参考解 (仿 768_minimal_surface_exact)                    #
# -------------------------------------------------------------------------- #
def minimal_surface_catenoid_amplitude(
        u: np.ndarray,
        v: np.ndarray,
        a_param: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    共形映射下的 Dalitz 振幅 catenoid 解 (仿 768):
        A(u, v) = acosh(a sqrt(u^2 + v^2)) / a
    返回 (A, A_u, A_v), 其中 A_u, A_v 为偏导数。
    用于: 在无共振背景下, 作为振幅的解析参考解,
    校验数值 Dalitz PDE 求解器的精度。
    """
    r2 = u * u + v * v
    r = np.sqrt(np.maximum(r2, 1.0e-30))
    arg = a_param * r
    # acosh(x) 仅在 x >= 1 有实值; 对 arg < 1 返回 NaN, 此处截断
    arg_safe = np.maximum(arg, 1.0 + 1.0e-12)
    A = np.arccosh(arg_safe) / a_param
    # A_u = u / (r * sqrt(a^2 r^2 - 1))
    sqrt_term = np.sqrt(np.maximum(arg_safe ** 2 - 1.0, 1.0e-30))
    denom = r * sqrt_term
    denom = np.where(np.abs(denom) < 1.0e-30, 1.0e-30, denom)
    A_u = u / denom
    A_v = v / denom
    return A, A_u, A_v


def minimal_surface_residual(
        A: np.ndarray,
        A_u: np.ndarray,
        A_v: np.ndarray,
        A_uu: np.ndarray,
        A_uv: np.ndarray,
        A_vv: np.ndarray,
) -> np.ndarray:
    """
    计算最小曲面方程的残差:
        R = (1 + A_u^2) A_{vv} - 2 A_u A_v A_{uv} + (1 + A_v^2) A_{uu}
    残差为零的解满足零平均曲率条件, 对应共形 Dalitz 振幅的极值解。
    """
    return (
        (1.0 + A_u * A_u) * A_vv
        - 2.0 * A_u * A_v * A_uv
        + (1.0 + A_v * A_v) * A_uu
    )


# -------------------------------------------------------------------------- #
#                    角分布 Legendre 展开系数                              #
# -------------------------------------------------------------------------- #
def angular_distribution_legendre_coefficients(
        amplitude_func,
        s_res: complex,
        p_max: int,
        b_weak: complex = 0.0 + 0.0j,
) -> np.ndarray:
    """
    计算角分布 |A(s, cosθ)|^2 的 Legendre 展开系数:
        |A|^2(cosθ) = Σ_l c_l P_l(cosθ)
    其中
        c_l = (2l+1)/2 ∫_{-1}^{1} |A|^2(x) P_l(x) dx
    若 b_weak != 0, 则积分带指数权 exp(b_weak x), 对应弱相位效应。
    使用 Gauss-Legendre 求积。
    """
    order = max(2 * p_max + 2, 12)
    xq, wq = legendre_quadrature(order)
    c = np.zeros(p_max + 1, dtype=complex)
    for k in range(order):
        x = xq[k]
        A_val = amplitude_func(s_res, x)
        mod2 = abs(A_val) ** 2
        L = legendre_polynomial(p_max, x)
        weight = wq[k]
        if abs(b_weak) > 1.0e-30:
            weight = weight * cmath.exp(b_weak * x).real
        for l in range(p_max + 1):
            c[l] += weight * mod2 * L[l]
    for l in range(p_max + 1):
        c[l] *= (2 * l + 1) / 2.0
    return c
