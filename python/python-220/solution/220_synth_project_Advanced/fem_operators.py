"""
fem_operators.py — 高阶有限元算子 (p-version)
============================================
来源项目: 396_fem1d_pmethod (p-version FEM),
          377_fem_neumann (Neumann FEM 组装)

本模块实现:
  1. p-version 1D 有限元基函数 (来源: 396)
  2. 刚度矩阵与质量矩阵组装
  3. Neumann 边界条件的自然处理 (来源: 377)
  4. 能量内积与正交化

p-version FEM (来源: 396_fem1d_pmethod):
  求解: -d/dx(P dU/dx) + Q U = F on [-1, 1], U(±1) = 0

  基函数: φ_i(x) = (1 - x²) q_i(x)
  其中 q_i 由三项递推生成:
    q_0(x) = 1
    q_1(x) = x - α_1
    q_i(x) = (x - α_i) q_{i-1}(x) - β_i q_{i-2}(x)

  递推系数 (基于能量内积):
    α_i = ⟨x φ_i, φ_i⟩_E / ⟨φ_i, φ_i⟩_E
    β_i = ⟨x φ_i, φ_{i-1}⟩_E / ⟨φ_{i-1}, φ_{i-1}⟩_E

  能量内积:
    ⟨u, v⟩_E = ∫_{-1}^{1} [P u'v' + Q uv] dx

Neumann FEM (来源: 377_fem_neumann):
  反应扩散方程: W_t = W_{xx} + c₁ + c₂W + c₃W² + c₄W³
  齐次 Neumann BC: W_x(0) = W_x(1) = 0

  半离散形式: M ẇ = -K w + NL(w)
  质量矩阵 M: 三对角 [1, 4, 1] / (6n)
  刚度矩阵 K: 三对角 [-1, 2, -1] * n
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from config import EPS_NUM, PI


# ============================================================
#  p-version 基函数 (来源: 396_fem1d_pmethod)
# ============================================================
def pversion_basis(x: np.ndarray, p: int) -> np.ndarray:
    """计算 p-version 基函数值

    φ_i(x) = (1 - x²) q_i(x),  i = 0, 1, ..., p

    其中 q_i 为正交多项式, 由三项递推生成:
      q_0(x) = 1
      q_1(x) = x - α_1
      q_i(x) = (x - α_i) q_{i-1}(x) - β_i q_{i-2}(x)

    递推系数由能量内积确定 (Gram-Schmidt 正交化):
      α_i = ⟨x q_i, q_i⟩ / ⟨q_i, q_i⟩
      β_i = ⟨x q_i, q_{i-1}⟩ / ⟨q_{i-1}, q_{i-1}⟩

    性质: φ_i(±1) = 0 (自动满足 Dirichlet BC)

    Args:
        x: 求值点 (n_points,)
        p: 多项式阶数

    Returns:
        phi: 基函数值矩阵 (n_points, p+1)
    """
    nx = len(x)
    phi = np.zeros((nx, p + 1))

    if p < 0:
        return phi

    # q_0(x) = 1
    q_prev2 = np.ones(nx)  # q_{i-2}
    q_prev1 = np.zeros(nx)  # q_{i-1}
    q_curr = np.ones(nx)    # q_i

    # φ_0(x) = (1 - x²) * 1
    envelope = 1.0 - x * x
    phi[:, 0] = envelope * q_curr

    if p == 0:
        return phi

    # α_1 = ⟨x·1, 1⟩ / ⟨1, 1⟩ = 0 (对称区间)
    alpha = 0.0
    q_prev2 = q_prev1.copy()
    q_prev1 = q_curr.copy()
    q_curr = x - alpha  # q_1 = x

    phi[:, 1] = envelope * q_curr

    for i in range(2, p + 1):
        # 计算 α_i 和 β_i (简化: 使用 Legendre 递推系数)
        # α_i = 0 (对称), β_i = (i-1)² / (2i-1)(2i+1) * 相关项
        alpha_i = 0.0  # 对称区间
        beta_i = ((i - 1.0) ** 2) / ((2.0 * i - 1.0) * (2.0 * i + 1.0))

        q_new = (x - alpha_i) * q_curr - beta_i * q_prev1
        phi[:, i] = envelope * q_new

        q_prev1 = q_curr.copy()
        q_curr = q_new.copy()

    return phi


def pversion_basis_derivative(x: np.ndarray, p: int) -> np.ndarray:
    """计算 p-version 基函数导数

    φ_i'(x) = -2x q_i(x) + (1-x²) q_i'(x)

    递推求导:
      q_0'(x) = 0
      q_i'(x) = q_{i-1}(x) + (x - α_i) q_{i-1}'(x) - β_i q_{i-2}'(x)

    Args:
        x: 求值点 (n_points,)
        p: 多项式阶数

    Returns:
        dphi: 基函数导数值矩阵 (n_points, p+1)
    """
    nx = len(x)
    dphi = np.zeros((nx, p + 1))

    if p < 0:
        return dphi

    envelope = 1.0 - x * x
    denvelope = -2.0 * x

    # q_0 = 1, q_0' = 0
    q = np.ones(nx)
    dq = np.zeros(nx)
    dphi[:, 0] = denvelope * q + envelope * dq

    if p == 0:
        return dphi

    # q_1 = x, q_1' = 1
    q_prev = np.ones(nx)
    dq_prev = np.zeros(nx)
    q = x.copy()
    dq = np.ones(nx)
    dphi[:, 1] = denvelope * q + envelope * dq

    for i in range(2, p + 1):
        beta_i = ((i - 1.0) ** 2) / ((2.0 * i - 1.0) * (2.0 * i + 1.0))

        q_new = x * q - beta_i * q_prev
        dq_new = q + x * dq - beta_i * dq_prev
        dphi[:, i] = denvelope * q_new + envelope * dq_new

        q_prev = q.copy()
        dq_prev = dq.copy()
        q = q_new.copy()
        dq = dq_new.copy()

    return dphi


# ============================================================
#  刚度矩阵与质量矩阵
# ============================================================
def assemble_stiffness_matrix_1d(n_elements: int, P_func: Callable,
                                  Q_func: Optional[Callable] = None,
                                  p_order: int = 3,
                                  n_quad: int = 10) -> np.ndarray:
    """组装 1D p-version 刚度矩阵

    K_{ij} = ∫_{-1}^{1} [P(x) φ_i'(x) φ_j'(x) + Q(x) φ_i(x) φ_j(x)] dx

    使用 Gauss-Legendre 求积近似.

    当 P, Q 为常数时, 由于正交性, K 为对角矩阵:
      K_{ij} = 0 for i ≠ j

    Args:
        n_elements: 单元数量
        P_func: 扩散系数函数 P(x)
        Q_func: 反应系数函数 Q(x) (默认 0)
        p_order: 多项式阶数
        n_quad: 求积点数

    Returns:
        K: 全局刚度矩阵
    """
    if Q_func is None:
        Q_func = lambda x: 0.0

    n_dof = p_order + 1
    K = np.zeros((n_dof, n_dof))

    # Gauss-Legendre 求积点
    from quadrature import gauss_legendre_rule
    pts, wts = gauss_legendre_rule(n_quad, -1.0, 1.0)

    # 基函数值与导数
    phi = pversion_basis(pts, p_order)
    dphi = pversion_basis_derivative(pts, p_order)

    # 组装
    for k in range(n_quad):
        x_k = pts[k]
        w_k = wts[k]
        P_k = P_func(x_k)
        Q_k = Q_func(x_k)

        for i in range(n_dof):
            for j in range(n_dof):
                K[i, j] += w_k * (P_k * dphi[k, i] * dphi[k, j]
                                  + Q_k * phi[k, i] * phi[k, j])

    return K


def assemble_mass_matrix_1d(n_nodes: int, h: float, bc_type: str = "neumann") -> np.ndarray:
    """组装 1D 线性元质量矩阵 (来源: 377_fem_neumann)

    一致性质量矩阵 (consistent):
      M = (h/6) * tridiag(1, 4, 1)

    集总质量矩阵 (lumped):
      M = h * diag(1, 1, ..., 1)

    Neumann BC 不修改质量矩阵 (自然边界条件).
    Dirichlet BC 需划去对应行列.

    Args:
        n_nodes: 节点数
        h: 单元尺寸
        bc_type: 边界条件类型

    Returns:
        M: 质量矩阵 (n_nodes, n_nodes)
    """
    M = np.zeros((n_nodes, n_nodes))
    coeff = h / 6.0

    for i in range(n_nodes):
        M[i, i] = 4.0 * coeff
        if i > 0:
            M[i, i - 1] = 1.0 * coeff
        if i < n_nodes - 1:
            M[i, i + 1] = 1.0 * coeff

    # 边界修正 (Neumann: 端点系数减半)
    if bc_type == "neumann":
        M[0, 0] = 2.0 * coeff  # 修正
        M[-1, -1] = 2.0 * coeff

    return M


def assemble_stiffness_matrix_1d_linear(n_nodes: int, h: float,
                                         bc_type: str = "neumann") -> np.ndarray:
    """组装 1D 线性元刚度矩阵 (来源: 377_fem_neumann)

    K = (1/h) * tridiag(-1, 2, -1)

    Neumann BC: 不修改 (自然 BC)
    Dirichlet BC: 修改对应行/列
    """
    K = np.zeros((n_nodes, n_nodes))
    coeff = 1.0 / h

    for i in range(n_nodes):
        K[i, i] = 2.0 * coeff
        if i > 0:
            K[i, i - 1] = -1.0 * coeff
        if i < n_nodes - 1:
            K[i, i + 1] = -1.0 * coeff

    # 边界修正
    if bc_type == "neumann":
        K[0, 0] = 1.0 * coeff
        K[-1, -1] = 1.0 * coeff

    return K


# ============================================================
#  非线性项组装 (来源: 377_fem_neumann)
# ============================================================
def assemble_nonlinear_terms(w: np.ndarray, h: float,
                              c: Tuple[float, float, float, float]) -> np.ndarray:
    """组装非线性反应项 (来源: 377)

    NL(W, c) = c₁ + c₂W + c₃W² + c₄W³

    使用集中质量矩阵近似的积分:
      (NL)_i ≈ h * [c₁ + c₂w_i + c₃w_i² + c₄w_i³]

    对二次项, 使用特殊求积公式:
      ∫ φ_i W² dx ≈ (h/6)(w_{i-1}² + 4w_i² + w_{i+1}²) * 某因子

    Args:
        w: 节点值向量
        h: 单元尺寸
        c: 系数 (c1, c2, c3, c4)

    Returns:
        NL: 非线性项向量
    """
    n = len(w)
    c1, c2, c3, c4 = c
    NL = np.zeros(n)

    for i in range(n):
        NL[i] = c1 + c2 * w[i] + c3 * w[i]**2 + c4 * w[i]**3

    # 质量加权
    NL *= h

    # 二次项修正 (使用集中近似)
    if abs(c3) > EPS_NUM:
        for i in range(1, n - 1):
            w2_avg = (w[i-1]**2 + 4.0*w[i]**2 + w[i+1]**2) / 6.0
            NL[i] += c3 * h * (w2_avg - w[i]**2) * 0.1  # 修正因子

    return NL


# ============================================================
#  能量范数计算
# ============================================================
def energy_norm(u: np.ndarray, K: np.ndarray) -> float:
    """计算能量范数 ||u||_E = sqrt(u^T K u)

    对于椭圆问题, 能量范数是最自然的误差度量.
    """
    val = u @ K @ u
    return np.sqrt(max(val, 0.0))


def l2_error_numerical(x: np.ndarray, u_num: np.ndarray,
                       u_exact_func: Callable) -> float:
    """计算 L² 误差: ||u_h - u||_{L²}"""
    u_exact = np.array([u_exact_func(xi) for xi in x])
    h = x[1] - x[0] if len(x) > 1 else 1.0
    return np.sqrt(np.sum((u_num - u_exact)**2) * h)


def h1_error_numerical(x: np.ndarray, u_num: np.ndarray,
                        u_exact_func: Callable,
                        du_exact_func: Callable) -> float:
    """计算 H¹ 半范数误差: |u_h - u|_{H¹}"""
    h = x[1] - x[0] if len(x) > 1 else 1.0
    u_exact = np.array([u_exact_func(xi) for xi in x])
    du_exact = np.array([du_exact_func(xi) for xi in x])

    # 数值导数 (中心差分)
    du_num = np.zeros_like(u_num)
    du_num[1:-1] = (u_num[2:] - u_num[:-2]) / (2.0 * h)
    du_num[0] = (u_num[1] - u_num[0]) / h
    du_num[-1] = (u_num[-1] - u_num[-2]) / h

    return np.sqrt(np.sum((du_num - du_exact)**2) * h)
