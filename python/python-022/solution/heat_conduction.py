"""
heat_conduction.py
==================
电子热传导求解模块。

融合原项目 967_r83v（三对角线性系统共轭梯度求解）与
1135_spiral_pde_movie（五点 Laplacian 离散算子）的核心思想，
求解球坐标下的非线性电子热传导方程：

    rho * C_ve * dT_e/dt = (1/r^2) * d/dr (r^2 * K_e * dT_e/dr) + S

其中电子热导率采用 Spitzer-Harm 公式（或限制形式）：

    K_e = f * (n_e * k_B^2 * T_e * tau_e) / m_e
    tau_e = 3 * sqrt(m_e) * (k_B*T_e)^(3/2) / (4 * sqrt(pi) * Z * n_e * e^4 * lnLambda)

更常见的形式:
    K_SH = (1.84e-5 / Z*lnLambda) * T_e^(5/2)   [W/(m*K)]

热流限制（flux limiter）：
    q_max = f * n_e * k_B * T_e * sqrt(k_B*T_e / m_e)
    q = min(K_e * |dT_e/dr|, q_max)

数值方法：
- 空间离散：中心差分（基于 spiral_pde_movie 的 Laplacian 离散思想）
- 线性求解：R83V 三对角共轭梯度法（基于 r83v_cg）
- 非线性处理：Picard 迭代或线性化
"""

import numpy as np
from typing import Tuple
from icf_parameters import PC, NP, TP
from utils import log_mean, safe_divide, clamp_array


def spitzer_harm_conductivity(T_e: float, Z_eff: float, n_e: float) -> float:
    """
    Spitzer-Harm 电子热导率 [W/(m*K)]:
        K_SH = (1.84e-5) * T_e^(5/2) / (Z_eff * ln(Lambda))
    其中 Coulomb 对数 ln(Lambda) 近似为:
        lnLambda = 23.5 - ln(sqrt(n_e) / T_e)   [n_e in m^-3, T_e in K]
    """
    if T_e <= 0.0 or Z_eff <= 0.0 or n_e <= 0.0:
        return 0.0

    # Coulomb 对数
    ln_lambda = 23.5 - np.log(np.sqrt(n_e) / max(T_e, 1.0))
    ln_lambda = max(ln_lambda, 2.0)

    K = 1.84e-5 * T_e**2.5 / (Z_eff * ln_lambda)
    return max(K, 0.0)


def flux_limited_conductivity(T_e: float, Z_eff: float, n_e: float,
                              grad_T: float) -> float:
    """
    热流限制后的有效热导率:
        q = -K_eff * grad_T
        K_eff = q_max / |grad_T|   当 K_SH * |grad_T| > q_max
    """
    K_sh = spitzer_harm_conductivity(T_e, Z_eff, n_e)
    if abs(grad_T) < 1.0e-30:
        return K_sh

    # 自由流热流限制
    v_th = np.sqrt(PC.BOLTZMANN * T_e / PC.ELECTRON_MASS)
    q_max = NP.FLUX_LIMITER * n_e * PC.BOLTZMANN * T_e * v_th

    q_sh = K_sh * abs(grad_T)
    if q_sh > q_max:
        return q_max / abs(grad_T)
    return K_sh


def build_tridiag_heat_matrix(r: np.ndarray, K_eff: np.ndarray,
                              rho: np.ndarray, cv: np.ndarray,
                              dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建热传导隐式离散的三对角线性系统:
        (M + dt*A) * T_new = M * T_old + dt * S

    返回下对角线、主对角线、上对角线及右端项。
    """
    n_nodes = len(r)
    n_cells = n_nodes - 1

    lower = np.zeros(n_nodes - 1)
    main = np.zeros(n_nodes)
    upper = np.zeros(n_nodes - 1)
    rhs = np.zeros(n_nodes)

    # 质量项（节点 lumped mass）
    for i in range(1, n_nodes - 1):
        # 节点 i 对应左右两个半单元
        vol_left = 4.0 * np.pi * r[i]**2 * (r[i] - r[i - 1]) / 2.0 if i > 0 else 0.0
        vol_right = 4.0 * np.pi * r[i]**2 * (r[i + 1] - r[i]) / 2.0 if i < n_nodes - 1 else 0.0
        m_node = rho[i - 1] * vol_left + rho[min(i, n_cells - 1)] * vol_right
        cv_node = cv[i - 1] if i > 0 else cv[0]
        main[i] += m_node * cv_node
        rhs[i] += m_node * cv_node  # 乘以 T_old 在后续处理

    # 边界节点
    main[0] = 1.0
    main[-1] = 1.0
    rhs[0] = 0.0   # 球心温度固定或绝热
    rhs[-1] = 0.0  # 外边界

    # 扩散项（单元间通量）
    for i in range(n_cells):
        r_face = 0.5 * (r[i] + r[i + 1])
        dr = r[i + 1] - r[i]
        if dr < 1.0e-15:
            continue

        # 界面面积
        A_face = 4.0 * np.pi * r_face**2
        # 有效热导率取对数平均或算术平均
        K_face = 0.5 * (K_eff[i] + K_eff[i]) if i >= len(K_eff) else K_eff[i]

        # 通量系数
        coeff = dt * A_face * K_face / dr

        if i == 0:
            # 球心对称边界: 只影响节点 1
            main[1] += coeff
        elif i == n_cells - 1:
            # 外边界
            main[i] += coeff
        else:
            main[i] += coeff
            main[i + 1] += coeff
            lower[i] -= coeff
            upper[i] -= coeff

    return lower, main, upper, rhs


def r83v_cg_solve(lower: np.ndarray, main: np.ndarray, upper: np.ndarray,
                  b: np.ndarray, x0: np.ndarray = None,
                  tol: float = 1.0e-12, max_iter: int = None) -> np.ndarray:
    """
    三对角矩阵的共轭梯度法求解（基于原项目 967_r83v/r83v_cg）。

    矩阵形式: A = diag(main) + diag(upper, 1) + diag(lower, -1)
    """
    n = len(main)
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    def matvec(v):
        """三对角矩阵向量乘法。"""
        result = main * v
        if n > 1:
            result[:-1] += upper * v[1:]
            result[1:] += lower * v[:-1]
        return result

    # 初始化残差
    Ax = matvec(x)
    r = b - Ax
    p = r.copy()

    rs_old = float(np.dot(r, r))
    if rs_old < tol**2:
        return x

    for it in range(max_iter):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1.0e-30:
            break

        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap

        rs_new = float(np.dot(r, r))
        if np.sqrt(rs_new) < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def solve_heat_conduction(r: np.ndarray, T_old: np.ndarray,
                          rho: np.ndarray, Z_eff: np.ndarray,
                          n_e: np.ndarray, dt: float,
                          source: np.ndarray = None) -> np.ndarray:
    """
    求解一个时间步的电子热传导方程。

    参数
    ----
    r : np.ndarray
        节点坐标
    T_old : np.ndarray
        单元中心旧温度（插值到节点）
    rho : np.ndarray
        单元密度
    Z_eff : np.ndarray
        有效电离度
    n_e : np.ndarray
        电子数密度
    dt : float
        时间步
    source : np.ndarray
        热源项（单元中心）

    返回
    ----
    T_new_cells : np.ndarray
        新的单元中心温度
    """
    n_nodes = len(r)
    n_cells = n_nodes - 1

    if source is None:
        source = np.zeros(n_cells)

    # 单元温度插值到节点（线性）
    T_nodes = np.zeros(n_nodes)
    T_nodes[1:-1] = 0.5 * (T_old[:-1] + T_old[1:])
    T_nodes[0] = T_old[0]
    T_nodes[-1] = T_old[-1]

    # 计算单元梯度与有效热导率
    K_eff = np.zeros(n_cells)
    for i in range(n_cells):
        dr = r[i + 1] - r[i]
        if dr < 1.0e-15:
            grad_T = 0.0
        else:
            grad_T = (T_nodes[i + 1] - T_nodes[i]) / dr
        K_eff[i] = flux_limited_conductivity(T_old[i], Z_eff[i], n_e[i], grad_T)

    # 比热容（电子）
    cv_e = np.zeros(n_cells)
    for i in range(n_cells):
        cv_e[i] = 1.5 * PC.BOLTZMANN * n_e[i] / max(rho[i], 1.0e-30)

    # 构建线性系统
    lower, main, upper, rhs = build_tridiag_heat_matrix(r, K_eff, rho, cv_e, dt)

    # 右端项: M*T_old + dt*S
    for i in range(1, n_nodes - 1):
        rhs[i] *= T_nodes[i]
        # 加入热源（分配到节点）
        if i < n_cells:
            rhs[i] += dt * source[i] * 0.5
        if i > 0:
            rhs[i] += dt * source[i - 1] * 0.5

    # 边界处理
    rhs[0] = T_nodes[0]
    rhs[-1] = T_nodes[-1]

    # 共轭梯度求解
    T_new_nodes = r83v_cg_solve(lower, main, upper, rhs, x0=T_nodes)

    # 节点温度返回单元中心
    T_new_cells = 0.5 * (T_new_nodes[:-1] + T_new_nodes[1:])
    T_new_cells = np.maximum(T_new_cells, 1.0)

    return T_new_cells
