"""
topology_optimizer.py
=====================
基于 SIMP (Solid Isotropic Material with Penalization) 的拓扑优化模块。
整合自：
  - 998_r8st：稀疏矩阵 COO 格式与共轭梯度法 (CG) 求解思想

目标：在体积约束下最小化结构柔度
    min_{ρ}  C(ρ) = F^T U = U^T K(ρ) U
    s.t.     ∫_Ω ρ dΩ ≤ V_max
             0 < ρ_min ≤ ρ ≤ 1
             K(ρ) U = F

其中 SIMP 插值：
    E(ρ_e) = E_min + ρ_e^p (E_0 - E_min)
    典型惩罚因子 p = 3
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 1. 稀疏矩阵向量乘法 (r8st 思想)
# =============================================================================

def sparse_mv(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
              x: np.ndarray, n: int) -> np.ndarray:
    """
    COO 格式稀疏矩阵与向量乘法 y = A @ x。
    参数 n 为方阵维度。
    """
    y = np.zeros(n, dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        y[r] += v * x[c]
    return y


def sparse_sym_cg(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                   b: np.ndarray, n: int, tol: float = 1e-10,
                   max_iter: Optional[int] = None) -> np.ndarray:
    """
    对称正定稀疏线性系统的共轭梯度法 (CG)。
    整合 r8st_cg 思想，适用于大规模拓扑优化中的重复求解。

    算法：
        给定初值 x0，计算 r0 = b - A x0，p0 = r0
        for k = 0, 1, 2, ...:
            α_k = (r_k^T r_k) / (p_k^T A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            p_{k+1} = r_{k+1} + β_k p_k
    """
    if max_iter is None:
        max_iter = n
    x = np.zeros(n, dtype=np.float64)
    r = b - sparse_mv(rows, cols, vals, x, n)
    p = r.copy()
    rs_old = np.dot(r, r)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-14:
        norm_b = 1.0

    for _ in range(max_iter):
        Ap = sparse_mv(rows, cols, vals, p, n)
        alpha = rs_old / (np.dot(p, Ap) + 1e-20)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) / norm_b < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


# =============================================================================
# 2. 密度过滤与投影 (制造约束)
# =============================================================================

def density_filter(element_centers: np.ndarray, rho: np.ndarray,
                   r_min: float) -> np.ndarray:
    """
    密度过滤算子：消除棋盘格模式并控制最小特征尺寸。
    采用线性权重核：
        H_{ei} = max(0, r_min - dist(e, i))
        ρ̃_e = Σ_i H_{ei} ρ_i / Σ_i H_{ei}
    """
    n_elements = len(rho)
    # 向量化距离矩阵计算
    dx = element_centers[:, 0:1] - element_centers[:, 0].reshape(1, -1)
    dy = element_centers[:, 1:2] - element_centers[:, 1].reshape(1, -1)
    dist = np.sqrt(dx**2 + dy**2)
    H = np.maximum(0.0, r_min - dist)
    sum_H = np.sum(H, axis=1)
    rho_tilde = np.where(sum_H > 1e-14, (H @ rho) / sum_H, rho)
    return rho_tilde


def heaviside_projection(rho_tilde: np.ndarray, beta: float,
                         eta: float = 0.5) -> np.ndarray:
    """
    Heaviside 投影：将过滤后的灰度密度推向 0/1 离散值。
    连续可微近似：
        H̃(ρ) = [tanh(β·η) + tanh(β·(ρ-η))] / [tanh(β·η) + tanh(β·(1-η))]

    参数 beta 越大，投影越锐利（推荐从 1 逐渐增大到 64）。
    """
    num = np.tanh(beta * eta) + np.tanh(beta * (rho_tilde - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


# =============================================================================
# 3. SIMP 材料插值
# =============================================================================

def simp_interpolation(rho: np.ndarray, E0: float, E_min: float = 1e-9,
                        p: float = 3.0) -> np.ndarray:
    """
    SIMP 弹性模量插值：
        E_e = E_min + ρ_e^p (E0 - E_min)

    其中 E_min 避免刚度矩阵奇异，p 为惩罚因子。
    """
    return E_min + np.power(rho, p) * (E0 - E_min)


def simp_derivative(rho: np.ndarray, E0: float, E_min: float = 1e-9,
                     p: float = 3.0) -> np.ndarray:
    """
    dE/dρ = p · ρ^{p-1} · (E0 - E_min)
    """
    return p * np.power(rho, p - 1.0) * (E0 - E_min)


# =============================================================================
# 4. 灵敏度分析
# =============================================================================

def compute_compliance_sensitivity(element_node: np.ndarray, node_xy: np.ndarray,
                                    U: np.ndarray, E_e: np.ndarray,
                                    dE_drho: np.ndarray, nu: float,
                                    plane_stress: bool = True) -> np.ndarray:
    """
    计算柔度对密度的灵敏度（基于伴随法）。

    对于 SIMP 模型，柔度：
        C = U^T K U = Σ_e u_e^T k_e(E_e) u_e
    灵敏度：
        ∂C/∂ρ_e = - u_e^T (∂k_e/∂E_e · dE_e/dρ_e) u_e
                 = - (dE_e/dρ_e) · u_e^T k_e(E=1) u_e

    注意：由于求最小化柔度，优化方向与灵敏度符号相反；
          这里返回原始灵敏度值（负值表示增大密度可降低柔度）。
    """
    n_elements = element_node.shape[0]
    sens = np.zeros(n_elements, dtype=np.float64)
    from fem_core import elastic_d_matrix
    D0 = elastic_d_matrix(1.0, nu, plane_stress)

    # TODO Hole 2: 实现柔度对密度的灵敏度计算（伴随法）
    # 核心公式: ∂C/∂ρ_e = - (dE_e/dρ_e) · u_e^T k_e(E=1) u_e
    # 需要遍历每个单元，组装B矩阵，计算参考应变能，再乘以dE_drho
    raise NotImplementedError("Hole 2: compute_compliance_sensitivity 未实现")
    return sens


# =============================================================================
# 5. OC (Optimality Criteria) 更新
# =============================================================================

def oc_update(rho: np.ndarray, sens: np.ndarray, volfrac: float,
              move: float = 0.2, eta_oc: float = 0.5) -> np.ndarray:
    """
    Optimality Criteria (OC) 密度更新方法。

    更新规则：
        B_e = -sens_e / (λ · V_e)   （λ 为体积约束拉格朗日乘子）
        ρ^{new}_e = clamp( ρ_e · B_e^η_oc, ρ_e - move, ρ_e + move )
        ρ^{new}_e = clamp( ρ^{new}_e, ρ_min, 1.0 )

    其中 λ 通过二分搜索使得体积约束满足。
    """
    n_elements = len(rho)
    rho_min = 1e-3
    rho_new = np.zeros_like(rho)

    l1, l2 = 0.0, 1e6
    while (l2 - l1) / (l2 + l1 + 1e-10) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        # 注意 sens 已经是负值（增大密度降低柔度）
        # B_e 应正比于 |sens|，这里用 Be = (-sens) / lmid
        Be = np.zeros(n_elements, dtype=np.float64)
        for e in range(n_elements):
            if abs(sens[e]) < 1e-20:
                Be[e] = 1.0
            else:
                Be[e] = (-sens[e]) / lmid
                # 防止数值问题
                if Be[e] <= 0:
                    Be[e] = rho_min
        # OC 更新
        for e in range(n_elements):
            # 使用标准 OC 更新：ρ_new = ρ * sqrt(Be) 的变体
            factor = Be[e] ** eta_oc
            rnew = rho[e] * factor
            # 移动限制
            rnew = max(rho_min, max(rho[e] - move, min(1.0, min(rho[e] + move, rnew))))
            rho_new[e] = rnew

        if np.mean(rho_new) > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


# =============================================================================
# 6. 主优化循环
# =============================================================================

def simp_topology_optimization(node_xy: np.ndarray, element_node: np.ndarray,
                                F_ext: np.ndarray, bc_nodes: np.ndarray,
                                bc_values: np.ndarray, E0: float, nu: float,
                                volfrac: float, n_iter: int = 100,
                                r_min: float = 1.5, plane_stress: bool = True,
                                use_filter: bool = True,
                                use_projection: bool = False) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    SIMP 拓扑优化主循环。

    Returns
    -------
    rho : ndarray
        优化后的密度场。
    U : ndarray
        最终位移场。
    history_compliance : list
        柔度迭代历史。
    history_vol : list
        体积分数迭代历史。
    """
    n_elements = element_node.shape[0]
    n_dof = node_xy.shape[0] * 2

    # 初始化密度场
    rho = np.full(n_elements, volfrac, dtype=np.float64)
    rho_min = 1e-3
    rho = np.maximum(rho, rho_min)

    # 单元中心用于过滤
    element_centers = np.zeros((n_elements, 2), dtype=np.float64)
    for e in range(n_elements):
        enodes = element_node[e, :]
        element_centers[e] = np.mean(node_xy[enodes, :], axis=0)

    history_compliance = []
    history_vol = []

    for it in range(n_iter):
        # 过滤与投影
        if use_filter:
            rho_f = density_filter(element_centers, rho, r_min)
        else:
            rho_f = rho.copy()
        if use_projection:
            beta_proj = min(64.0, 1.0 + 0.5 * it)
            rho_f = heaviside_projection(rho_f, beta_proj)

        # SIMP 插值
        E_e = simp_interpolation(rho_f, E0)
        dE = simp_derivative(rho_f, E0)

        # 组装全局刚度矩阵
        from fem_core import assemble_global_stiffness
        rows, cols, vals = assemble_global_stiffness(
            node_xy, element_node, 1.0, nu, plane_stress, "T3")
        # 用 SIMP 材料插值缩放
        vals_scaled = np.zeros_like(vals)
        # 需要将 vals 按单元分组重新缩放（组装是按元素顺序的）
        idx = 0
        n_local = 3
        n_edof = n_local * 2
        for e in range(n_elements):
            scale = E_e[e]
            for _ in range(n_edof * n_edof):
                vals_scaled[idx] = vals[idx] * scale
                idx += 1

        # 构建稠密矩阵用于求解（小规模问题）
        K = np.zeros((n_dof, n_dof), dtype=np.float64)
        for r, c, v in zip(rows, cols, vals_scaled):
            K[r, c] += v

        K_mod, F_mod = apply_dirichlet_bcs(K, F_ext, bc_nodes, bc_values)
        cond_est = np.linalg.cond(K_mod)
        if cond_est > 1e16:
            K_mod += np.eye(n_dof) * 1e-8 * np.max(np.abs(K_mod))

        # 求解位移
        U = np.linalg.solve(K_mod, F_mod)

        # 计算柔度
        compliance = np.dot(F_ext, U)
        history_compliance.append(compliance)
        history_vol.append(np.mean(rho_f))

        # 灵敏度分析
        sens = compute_compliance_sensitivity(
            element_node, node_xy, U, E_e, dE, nu, plane_stress)

        # 灵敏度过滤（链式法则）
        if use_filter:
            dx = element_centers[:, 0:1] - element_centers[:, 0].reshape(1, -1)
            dy = element_centers[:, 1:2] - element_centers[:, 1].reshape(1, -1)
            dist = np.sqrt(dx**2 + dy**2)
            H = np.maximum(0.0, r_min - dist)
            sum_H = np.sum(H, axis=1)
            sum_H_safe = np.maximum(1e-14, sum_H)
            # 链式法则：∂C/∂ρ_i = Σ_e (∂C/∂ρ̃_e) · (H_{ei} / sum_H_e)
            # 矩阵形式: sens_f = H^T @ (sens / sum_H)
            sens_f = H.T @ (sens / sum_H_safe)
            sens = sens_f

        # OC 更新
        rho = oc_update(rho, sens, volfrac)

    return rho, U, history_compliance, history_vol


def apply_dirichlet_bcs(K_dense: np.ndarray, F: np.ndarray, bc_nodes: np.ndarray,
                         bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """同 fem_core 中的实现，避免循环导入。"""
    K_mod = K_dense.copy()
    F_mod = F.copy()
    penalty = 1e12 * np.max(np.abs(K_dense))
    if penalty == 0.0:
        penalty = 1e12
    for i, dof in enumerate(bc_nodes):
        val = bc_values[i]
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = penalty
        F_mod[dof] = val * penalty
    return K_mod, F_mod
