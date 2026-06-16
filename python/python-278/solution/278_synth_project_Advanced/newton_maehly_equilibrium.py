"""
newton_maehly_equilibrium.py
=============================
Newton-Maehly 消去法求解 CALPHAD 相平衡。

将原始 Newton-Maehly 多项式求根方法推广到多变量非线性系统:
  F(x) = 0,  x ∈ R^n

原始算法 (种子项目 801_newton_maehly):
  对于多项式的每个根 x_k, 执行:
    x_k^{new} = x_k - p(x_k) / [p'(x_k) - sum_{j≠k} 1/(x_k - x_j)]
  其中分母中的求和项为 Maehly 消去因子。

推广到相平衡:
  求解化学势等式:
    mu_Fe^alpha(x_C^alpha, T) = mu_Fe^beta(x_C^beta, T)
    mu_C^alpha(x_C^alpha, T) = mu_C^beta(x_C^beta, T)

  Newton 步:
    J * delta = -F(x)
    x^{new} = x + delta

  Maehly 消去:
    在 Newton 步之后, 应用消去因子:
    delta_i -= sum_{j≠i} w_ij * delta_j / (x_i - x_j)
    以防止已收敛的解被后续迭代破坏。

同时实现:
  - 公共切线法 (Common Tangent) 求两相平衡
  - 多相平衡的全局搜索
  - LU 分解求解线性系统 (借鉴 linpack_d)
"""

import numpy as np
from gibbs_energy_calphad import (
    gibbs_substitutional,
    chemical_potential_C,
    chemical_potential_Fe,
)
from calphad_fec_constants import NM_MAX_ITER, NM_TOL, NM_DEF_TOL


def lu_factor(A):
    """
    LU 分解 (部分选主元): 借鉴 LINPACK dgefa。

    A = P * L * U
    其中 P 为置换矩阵, L 为单位下三角, U 为上三角。

    Parameters
    ----------
    A : np.ndarray
        n×n 矩阵

    Returns
    -------
    tuple
        (LU, piv) 紧凑存储的 LU 分解和主元索引
    """
    n = A.shape[0]
    LU = A.copy().astype(float)
    piv = np.arange(n)

    for k in range(n - 1):
        # 选主元: 找第 k 列绝对值最大的元素
        max_idx = k + np.argmax(np.abs(LU[k:, k]))
        if max_idx != k:
            LU[[k, max_idx]] = LU[[max_idx, k]]
            piv[[k, max_idx]] = piv[[max_idx, k]]

        if abs(LU[k, k]) < 1e-30:
            continue  # 奇异矩阵, 跳过

        # 消元
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            LU[i, k + 1:] -= LU[i, k] * LU[k, k + 1:]

    return LU, piv


def lu_solve(LU, piv, b):
    """
    使用 LU 分解求解线性系统: A*x = b。
    借鉴 LINPACK dgesl 的前代/回代过程。

    Parameters
    ----------
    LU : np.ndarray
        lu_factor 返回的紧凑 LU
    piv : np.ndarray
        主元索引
    b : np.ndarray
        右端向量

    Returns
    -------
    np.ndarray
        解向量 x
    """
    n = len(b)
    x = b[piv].copy()

    # 前代: 解 L*y = P*b
    for i in range(1, n):
        for j in range(i):
            x[i] -= LU[i, j] * x[j]

    # 回代: 解 U*x = y
    for i in range(n - 1, -1, -1):
        if abs(LU[i, i]) < 1e-30:
            x[i] = 0.0
            continue
        x[i] /= LU[i, i]
        for j in range(i):
            x[j] -= LU[j, i] * x[i]

    return x


def jacobian_phase_equilibrium(x_C_alpha, x_C_beta, T, phase_a, phase_b):
    """
    计算相平衡方程组的 Jacobian 矩阵:

    F = [mu_Fe^alpha - mu_Fe^beta]
        [mu_C^alpha  - mu_C^beta ]

    J = [[d(mu_Fe^alpha)/d(x_C^alpha), -d(mu_Fe^beta)/d(x_C^beta)],
         [d(mu_C^alpha)/d(x_C^alpha),  -d(mu_C^beta)/d(x_C^beta) ]]

    使用数值差分 (中心差分, O(h²)):

    Parameters
    ----------
    x_C_alpha : float
        alpha 相中碳的摩尔分数
    x_C_beta : float
        beta 相中碳的摩尔分数
    T : float
        温度 (K)
    phase_a : str
        alpha 相名称
    phase_b : str
        beta 相名称

    Returns
    -------
    tuple
        (F, J): 残差向量和 Jacobian 矩阵
    """
    eps = 1.0e-7

    mu_Fe_a = chemical_potential_Fe(x_C_alpha, T, phase_a)
    mu_Fe_b = chemical_potential_Fe(x_C_beta, T, phase_b)
    mu_C_a = chemical_potential_C(x_C_alpha, T, phase_a)
    mu_C_b = chemical_potential_C(x_C_beta, T, phase_b)

    F = np.array([
        mu_Fe_a - mu_Fe_b,
        mu_C_a - mu_C_b,
    ])

    # 数值 Jacobian (中心差分)
    xa_p = min(x_C_alpha + eps, 1.0 - 1e-10)
    xa_m = max(x_C_alpha - eps, 1e-10)
    xb_p = min(x_C_beta + eps, 1.0 - 1e-10)
    xb_m = max(x_C_beta - eps, 1e-10)

    dxa = xa_p - xa_m
    dxb = xb_p - xb_m

    # dF/d(x_C^alpha)
    mu_Fe_a_p = chemical_potential_Fe(xa_p, T, phase_a)
    mu_Fe_a_m = chemical_potential_Fe(xa_m, T, phase_a)
    mu_C_a_p = chemical_potential_C(xa_p, T, phase_a)
    mu_C_a_m = chemical_potential_C(xa_m, T, phase_a)

    # dF/d(x_C^beta)
    mu_Fe_b_p = chemical_potential_Fe(xb_p, T, phase_b)
    mu_Fe_b_m = chemical_potential_Fe(xb_m, T, phase_b)
    mu_C_b_p = chemical_potential_C(xb_p, T, phase_b)
    mu_C_b_m = chemical_potential_C(xb_m, T, phase_b)

    J = np.array([
        [(mu_Fe_a_p - mu_Fe_a_m) / dxa, -(mu_Fe_b_p - mu_Fe_b_m) / dxb],
        [(mu_C_a_p - mu_C_a_m) / dxa, -(mu_C_b_p - mu_C_b_m) / dxb],
    ])

    return F, J


def newton_maehly_solve(T, phase_a, phase_b, x_init=None):
    """
    Newton-Maehly 消去法求解两相平衡:

    对于 2×2 系统:
        F_1(x_C^alpha, x_C^beta) = mu_Fe^alpha - mu_Fe^beta = 0
        F_2(x_C^alpha, x_C^beta) = mu_C^alpha - mu_C^beta = 0

    Newton-Maehly 步:
        delta = -J^{-1} * F
        x_C^alpha -= delta_1 + NM_def * delta_2 / (x_C^alpha - x_C^beta)
        x_C^beta  -= delta_2 + NM_def * delta_1 / (x_C^beta  - x_C^alpha)

    Maehly 消去因子防止两个相的成分收敛到同一点
    (物理上两相必须共存, 成分必须不同)。

    Parameters
    ----------
    T : float
        温度 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    x_init : tuple, optional
        初始猜测 (x_C^alpha, x_C^beta)

    Returns
    -------
    dict
        {
            'x_C_alpha': float,    # alpha 相平衡成分
            'x_C_beta': float,     # beta 相平衡成分
            'converged': bool,     # 是否收敛
            'n_iter': int,         # 迭代次数
            'residual': float,     # 最终残差
        }
    """
    if x_init is None:
        # 基于 CALPHAD 经验的初始猜测
        if ('FCC' in phase_a and 'BCC' in phase_b) or \
           ('FCC' in phase_b and 'BCC' in phase_a):
            x_init = (0.005, 0.02)  # gamma/alpha 平衡
        elif 'LIQUID' in phase_a:
            x_init = (0.02, 0.05)  # 液相/固相
        else:
            x_init = (0.01, 0.03)

    xa, xb = x_init

    for iteration in range(NM_MAX_ITER):
        # 确保成分在物理范围内
        xa = np.clip(xa, 1.0e-10, 0.30)
        xb = np.clip(xb, 1.0e-10, 0.30)

        # 确保两相成分不同
        if abs(xa - xb) < 1.0e-10:
            xb = xa + 1.0e-6

        # 计算残差和 Jacobian
        F, J = jacobian_phase_equilibrium(xa, xb, T, phase_a, phase_b)

        residual = np.max(np.abs(F))
        if residual < NM_TOL:
            return {
                'x_C_alpha': float(xa),
                'x_C_beta': float(xb),
                'converged': True,
                'n_iter': iteration,
                'residual': float(residual),
            }

        # LU 分解求解 Newton 步
        try:
            LU, piv = lu_factor(J)
            delta = lu_solve(LU, piv, -F)
        except np.linalg.LinAlgError:
            # Jacobian 奇异, 使用正则化
            J_reg = J + 1e-8 * np.eye(2)
            delta = np.linalg.solve(J_reg, -F)

        # Maehly 消去修正
        diff = xa - xb
        if abs(diff) > NM_DEF_TOL:
            # 消去因子: 防止两相成分趋同
            defl_1 = NM_DEF_TOL * delta[1] / diff
            defl_2 = NM_DEF_TOL * delta[0] / (-diff)
        else:
            defl_1 = 0.0
            defl_2 = 0.0

        # 更新 (带阻尼)
        damp = min(1.0, 0.5 / max(np.abs(delta[0]), np.abs(delta[1]), 0.5))
        xa += damp * (delta[0] - defl_1)
        xb += damp * (delta[1] - defl_2)

    return {
        'x_C_alpha': float(xa),
        'x_C_beta': float(xb),
        'converged': False,
        'n_iter': NM_MAX_ITER,
        'residual': float(np.max(np.abs(F))),
    }


def common_tangent_search(T, phase_a, phase_b, n_scan=50):
    """
    公共切线法扫描搜索两相平衡成分。

    对于给定温度 T, 扫描 alpha 相的成分 x_a,
    对每个 x_a 找 beta 相的成分 x_b 使得:
        (G_beta(x_b) - G_alpha(x_a)) / (x_b - x_a) = dG_alpha/dx |_{x_a}

    这等价于化学势等式。

    Parameters
    ----------
    T : float
        温度 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    n_scan : int
        扫描点数

    Returns
    -------
    list of dict
        所有找到的平衡点
    """
    results = []
    x_a_grid = np.logspace(np.log10(1e-6), np.log10(0.20), n_scan)

    for xa in x_a_grid:
        # 计算 alpha 相在 xa 处的切线斜率
        dG_dxa = (chemical_potential_C(xa, T, phase_a)
                  - chemical_potential_Fe(xa, T, phase_a))

        # 在 beta 相中寻找切点
        x_b_grid = np.logspace(np.log10(1e-6), np.log10(0.25), n_scan)
        G_a = gibbs_substitutional(xa, T, phase_a)

        best_err = np.inf
        best_xb = None

        for xb in x_b_grid:
            if abs(xb - xa) < 1e-10:
                continue
            G_b = gibbs_substitutional(xb, T, phase_b)
            # 公共切线条件
            slope = (G_b - G_a) / (xb - xa)
            err = abs(slope - dG_dxa)
            if err < best_err:
                best_err = err
                best_xb = xb

        if best_xb is not None and best_err < 1e3:
            # 使用 Newton-Maehly 精炼
            result = newton_maehly_solve(T, phase_a, phase_b,
                                         x_init=(xa, best_xb))
            if result['converged']:
                results.append(result)

    return results
