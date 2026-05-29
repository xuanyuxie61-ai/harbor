"""
pmns_matrix.py
PMNS 矩阵构造与味-质量本征态转换

本模块实现了 Pontecorvo–Maki–Nakagawa–Sakata (PMNS) 矩阵的完整构造,
包含三种味 (e, μ, τ) 与三种质量本征态 (1, 2, 3) 之间的幺正变换。

核心公式:
    |ν_α⟩ = Σ_i U_{αi} |ν_i⟩

PMNS 矩阵采用标准参数化:
    U = R₂₃(θ₂₃) · diag(1, e^{iδ_CP}, 1) · R₁₃(θ₁₃) · R₁₂(θ₁₂)

其中 R_{ij}(θ) 为实空间旋转矩阵, δ_CP 为 CP 破坏相位。
"""

import numpy as np
from constants import (
    THETA_12, THETA_23, THETA_13, DELTA_CP,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH
)


def rotation_12(theta12):
    """
    构造 1-2 扇区旋转矩阵 R₁₂(θ₁₂)。

    矩阵形式:
        [  c₁₂   s₁₂   0  ]
        [ -s₁₂   c₁₂   0  ]
        [   0     0    1  ]

    其中 c₁₂ = cos(θ₁₂), s₁₂ = sin(θ₁₂)。
    """
    c = np.cos(theta12)
    s = np.sin(theta12)
    R = np.array([
        [ c,  s,  0.0],
        [-s,  c,  0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.complex128)
    return R


def rotation_13(theta13, delta=0.0):
    """
    构造 1-3 扇区旋转矩阵 R₁₃(θ₁₃)。

    在标准参数化中, 1-3 旋转后引入 CP 相位:
        R₁₃ = diag(1, e^{iδ}, 1) · [纯 1-3 旋转]

    矩阵形式:
        [  c₁₃        0      s₁₃·e^{-iδ} ]
        [   0         1          0       ]
        [ -s₁₃·e^{iδ}  0       c₁₃      ]
    """
    c = np.cos(theta13)
    s = np.sin(theta13)
    exp_delta = np.exp(1j * delta)
    R = np.array([
        [ c,           0.0,  s / exp_delta],
        [ 0.0,         1.0,  0.0          ],
        [-s * exp_delta, 0.0,  c           ]
    ], dtype=np.complex128)
    return R


def rotation_23(theta23):
    """
    构造 2-3 扇区旋转矩阵 R₂₃(θ₂₃)。

    矩阵形式:
        [ 1   0    0  ]
        [ 0  c₂₃  s₂₃ ]
        [ 0 -s₂₃  c₂₃ ]
    """
    c = np.cos(theta23)
    s = np.sin(theta23)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s ],
        [0.0, -s,  c ]
    ], dtype=np.complex128)
    return R


def build_pmns_matrix(theta12=None, theta23=None, theta13=None, delta_cp=None):
    """
    构造完整的 PMNS 混合矩阵。

    标准参数化顺序 (PDG 约定):
        U = R₂₃(θ₂₃) · R₁₃(θ₁₃, δ_CP) · R₁₂(θ₁₂)

    参数:
        theta12:  θ₁₂ [rad], 默认使用常数模块值
        theta23:  θ₂₃ [rad], 默认使用常数模块值
        theta13:  θ₁₃ [rad], 默认使用常数模块值
        delta_cp: δ_CP [rad], 默认使用常数模块值

    返回:
        U: 3×3 复数幺正矩阵

    幺正性验证:
        U · U† = I
        |U_{αi}|² 满足每一行/列和为 1
    """
    t12 = THETA_12 if theta12 is None else float(theta12)
    t23 = THETA_23 if theta23 is None else float(theta23)
    t13 = THETA_13 if theta13 is None else float(theta13)
    dcp = DELTA_CP if delta_cp is None else float(delta_cp)

    # 边界检查
    eps = 1e-12
    if t12 < eps or t12 > np.pi / 2 - eps:
        raise ValueError(f"theta12 must be in (0, π/2), got {t12}")
    if t23 < eps or t23 > np.pi / 2 - eps:
        raise ValueError(f"theta23 must be in (0, π/2), got {t23}")
    if t13 < eps or t13 > np.pi / 2 - eps:
        raise ValueError(f"theta13 must be in (0, π/2), got {t13}")

    # === HOLE 1 ===
    # 请根据 PDG 标准参数化顺序构造 PMNS 矩阵 U
    # 提示: U = R₂₃(θ₂₃) · R₁₃(θ₁₃, δ_CP) · R₁₂(θ₁₂)
    # 可用辅助函数: rotation_12(t12), rotation_13(t13, dcp), rotation_23(t23)
    # === END HOLE 1 ===
    raise NotImplementedError("HOLE 1: build_pmns_matrix 核心构造尚未实现")


def build_mass_matrix(delta_m2_21=None, delta_m2_31=None, hierarchy='normal'):
    """
    构造真空中的质量平方矩阵 M² (对角形式)。

    在味基中, 有效哈密顿量 (低能极限) 为:
        H_vac = (1 / 2E) · U · diag(0, Δm²₂₁, Δm²₃₁) · U†

    我们定义对角质量平方矩阵为:
        M²_diag = diag(0, Δm²₂₁, Δm²₃₁)

    参数:
        delta_m2_21: Δm²₂₁ [eV²], 默认使用常数
        delta_m2_31: Δm²₃₁ [eV²], 默认根据 hierarchy 选择
        hierarchy:   'normal' 或 'inverted'

    返回:
        M2: 3×3 对角实矩阵 [eV²]
    """
    dm21 = DELTA_M2_21 if delta_m2_21 is None else float(delta_m2_21)

    if delta_m2_31 is None:
        if hierarchy.lower() == 'normal':
            dm31 = DELTA_M2_31
        elif hierarchy.lower() == 'inverted':
            dm31 = DELTA_M2_31_IH
        else:
            raise ValueError("hierarchy must be 'normal' or 'inverted'")
    else:
        dm31 = float(delta_m2_31)

    # 边界检查: Δm²₂₁ 必须为正 (太阳中微子数据)
    if dm21 <= 0:
        raise ValueError(f"delta_m2_21 must be positive, got {dm21}")

    M2 = np.diag([0.0, dm21, dm31])
    return M2


def check_unitarity(U, tol=1e-10):
    """
    验证 PMNS 矩阵的幺正性:
        U · U† = I   且   U† · U = I

    同时验证每行每列的模方和:
        Σ_i |U_{αi}|² = 1   (行和)
        Σ_α |U_{αi}|² = 1   (列和)

    返回:
        is_unitary: bool
        max_error:  float
    """
    identity = np.eye(3, dtype=np.complex128)
    udag = U.conj().T

    err1 = np.max(np.abs(U @ udag - identity))
    err2 = np.max(np.abs(udag @ U - identity))

    row_sums = np.sum(np.abs(U) ** 2, axis=1)
    col_sums = np.sum(np.abs(U) ** 2, axis=0)
    err3 = np.max(np.abs(row_sums - 1.0))
    err4 = np.max(np.abs(col_sums - 1.0))

    max_error = max(err1, err2, err3, err4)
    return max_error < tol, max_error


def pmns_to_mass_basis(U, flavor_state):
    """
    将味态 |ν_α⟩ 转换到质量本征态基 |ν_i⟩:
        |ν_i⟩ = Σ_α U_{αi}* |ν_α⟩ = U† · |ν_α⟩

    参数:
        U:            3×3 PMNS 矩阵
        flavor_state: 3 维味态向量

    返回:
        mass_state: 3 维质量本征态向量
    """
    flavor_state = np.asarray(flavor_state, dtype=np.complex128)
    if flavor_state.shape != (3,):
        raise ValueError("flavor_state must be a 3-element vector")
    return U.conj().T @ flavor_state


def mass_to_flavor_basis(U, mass_state):
    """
    将质量本征态 |ν_i⟩ 转换到味基 |ν_α⟩:
        |ν_α⟩ = Σ_i U_{αi} |ν_i⟩ = U · |ν_i⟩

    参数:
        U:          3×3 PMNS 矩阵
        mass_state: 3 维质量本征态向量

    返回:
        flavor_state: 3 维味态向量
    """
    mass_state = np.asarray(mass_state, dtype=np.complex128)
    if mass_state.shape != (3,):
        raise ValueError("mass_state must be a 3-element vector")
    return U @ mass_state


def get_initial_flavor_state(flavor='electron'):
    """
    返回初始纯味态向量。

    参数:
        flavor: 'electron', 'muon', 'tau'

    返回:
        psi: 3 维向量, 对应 |ν_e⟩, |ν_μ⟩ 或 |ν_τ⟩
    """
    flavor_map = {
        'electron': 0,
        'muon': 1,
        'tau': 2,
        'e': 0,
        'mu': 1,
        'tau': 2
    }
    idx = flavor_map.get(flavor.lower(), 0)
    psi = np.zeros(3, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def jarkslog_invariant(U):
    """
    计算 Jarlskog 不变量 J_CP, 它是 CP 破坏的度量。

    定义:
        J_CP = Im[ U_{e1} U_{μ2} U*_{e2} U*_{μ1} ]

    对于标准参数化:
        J_CP = sin(θ₁₂) sin(θ₂₃) sin(θ₁₃) cos(θ₁₂) cos(θ₂₃) cos²(θ₁₃) sin(δ_CP)

    返回:
        J: Jarlskog 不变量 (无量纲实数)
    """
    j = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    return float(j)
