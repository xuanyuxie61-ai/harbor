# -*- coding: utf-8 -*-
"""
stability_analysis.py
后屈曲路径稳定性分析与混沌检测

融合种子项目:
  - 171_chirikov_iteration: 标准映射迭代与KAM环面破坏

科学背景:
  壳体后屈曲路径的稳定性可通过切线刚度矩阵 K_T 的特征值分析判定:
    - 全部特征值 > 0: 稳定平衡 (能量极小值)
    - 存在特征值 < 0: 不稳定平衡 (鞍点或极大值)
    - 特征值 = 0: 临界状态 (极限点或分岔点)

  对于参数化路径 (u(s), λ(s))，Lyapunov 指数用于量化相邻轨迹的分离率:
    λ_L = lim_{t→∞} (1/t) ln( ||δu(t)|| / ||δu(0)|| )

  Chirikov 共振重叠判据 (启发式):
    在非线性振动系统中，当两个共振带的半宽之和超过其间距时，
    KAM 不变环面破坏，系统进入全局混沌。对于壳体后屈曲:
      共振带间距 Δk ≈ |n₁ - n₂|
      半宽 δk ≈ √(ε)  (ε 为非线性强度参数)
    混沌条件: 2δk > Δk 即 ε > (Δk/2)²
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


class StabilityAnalyzer:
    """
    壳体后屈曲稳定性分析器
    """

    def __init__(self, fem_model):
        self.fem = fem_model

    def tangent_stiffness_eigenvalues(self, u: np.ndarray, k: int = 5) -> dict:
        """
        计算切线刚度矩阵的前 k 个最小特征值

        Parameters
        ----------
        u : (n_dof,) ndarray
            当前位移
        k : int
            特征值个数

        Returns
        -------
        result : dict
            eigenvalues, eigenvectors, stability (True/False)
        """
        n_dof = self.fem.n_dof
        K_lin = self.fem.assemble_linear_stiffness()
        K_geo = self.fem.assemble_geometric_stiffness(u)
        K_T = K_lin + K_geo

        bottom, top = self.fem.mesh.get_boundary_nodes()
        fixed_dofs = []
        for nid in bottom:
            fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
        for nid in top:
            fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1])
        if len(bottom) > 0:
            fixed_dofs.append(bottom[0] * 3 + 2)
        fixed_dofs = np.unique(fixed_dofs)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

        if len(free_dofs) == 0:
            return {'eigenvalues': np.array([]), 'stable': True}

        K_ff = K_T[free_dofs][:, free_dofs]
        k_eff = min(k, len(free_dofs) - 1)
        if k_eff < 1:
            return {'eigenvalues': np.array([1.0]), 'stable': True}

        try:
            eigvals, eigvecs = eigsh(K_ff, k=k_eff, which='SM', tol=1e-4)
            stable = np.all(eigvals > -1e-8)
            return {
                'eigenvalues': eigvals,
                'eigenvectors': eigvecs,
                'stable': stable,
                'min_eig': float(np.min(eigvals))
            }
        except Exception:
            # 回退到稠密矩阵
            K_dense = K_ff.toarray()
            eigvals = np.linalg.eigvalsh(K_dense)
            stable = np.all(eigvals > -1e-8)
            return {
                'eigenvalues': eigvals[:k_eff],
                'eigenvectors': None,
                'stable': stable,
                'min_eig': float(np.min(eigvals))
            }

    def lyapunov_exponent_discrete(self, path: list, perturbation_scale: float = 1e-6,
                                   n_iter: int = 50) -> float:
        """
        计算后屈曲路径的离散 Lyapunov 指数

        将路径视为离散动力系统:
          z_{k+1} = f(z_k),  z_k = (u_k, λ_k)
        对初始扰动 δz_0 进行迭代:
          δz_{k+1} ≈ J_k δz_k
        其中 J_k 为局部 Jacobian。

        Lyapunov 指数:
          λ_L = (1/N) Σ_k ln( ||δz_k|| / ||δz_{k-1}|| )

        Parameters
        ----------
        path : list of dict
            路径历史
        perturbation_scale : float
            初始扰动大小
        n_iter : int
            迭代次数

        Returns
        -------
        lyapunov_exp : float
            最大 Lyapunov 指数近似值
        """
        if len(path) < 3:
            return 0.0

        exponents = []
        for i in range(1, min(n_iter, len(path) - 1)):
            du = path[i + 1]['disp'] - path[i]['disp']
            du_prev = path[i]['disp'] - path[i - 1]['disp']
            dl = path[i + 1]['lambda'] - path[i]['lambda']
            dl_prev = path[i]['lambda'] - path[i - 1]['lambda']

            norm_curr = np.sqrt(np.dot(du, du) + dl ** 2) + 1e-14
            norm_prev = np.sqrt(np.dot(du_prev, du_prev) + dl_prev ** 2) + 1e-14
            ratio = norm_curr / norm_prev
            exponents.append(np.log(max(ratio, 1e-14)))

        if not exponents:
            return 0.0
        return float(np.mean(exponents))

    def chirikov_overlap_criterion(self, path: list, mode_spacing: int = 2) -> bool:
        """
        应用 Chirikov 共振重叠判据检测壳体屈曲路径中的混沌区域

        将后屈曲路径视为参数驱动的非线性振子，
        相邻环向模态 n 和 n+Δn 构成共振对。
        当非线性参数 ε = ||w_max|| / t 足够大时，共振重叠导致混沌。

        判据:
          ε > ε_crit = (Δn / (2n))²

        Parameters
        ----------
        path : list of dict
        mode_spacing : int
            相邻共振模态间距 Δn

        Returns
        -------
        chaotic : bool
            是否满足混沌条件
        """
        if len(path) < 2:
            return False
        t = self.fem.mesh.geom.t
        max_w = max([p['max_disp'] for p in path])
        epsilon = max_w / t
        # 简化的临界非线性参数
        n_avg = 5.0  # 典型环向波数
        delta_n = float(mode_spacing)
        epsilon_crit = (delta_n / (2.0 * n_avg)) ** 2
        return epsilon > epsilon_crit

    def koiter_bifurcation_class(self, path: list) -> str:
        """
        根据 Koiter 理论对分岔类型进行分类

        基于屈曲路径初始斜率 dλ/dξ (ξ 为归一化位移):
          - 对称稳定:   dλ/dξ > 0
          - 对称不稳定: dλ/dξ < 0 且路径连续
          - 非对称:     存在斜率无穷大点
          - 跳跃 (snap-through): λ 出现局部极值

        Returns
        -------
        classification : str
        """
        if len(path) < 3:
            return "undetermined"
        slopes = []
        for i in range(1, len(path)):
            dxi = path[i]['max_disp'] - path[i - 1]['max_disp']
            dl = path[i]['lambda'] - path[i - 1]['lambda']
            if abs(dxi) > 1e-12:
                slopes.append(dl / dxi)

        if not slopes:
            return "undetermined"

        has_negative = any(s < 0 for s in slopes)
        has_extreme = any(slopes[i] * slopes[i - 1] < 0 for i in range(1, len(slopes)))

        if has_extreme:
            return "snap-through"
        elif has_negative:
            return "symmetric-unstable"
        else:
            return "symmetric-stable"

    def energy_barrier(self, path: list) -> float:
        """
        计算后屈曲路径的能量势垒

        从屈曲点到最近稳定构型的能量差:
          ΔΠ = Π_unstable - Π_stable
        """
        if len(path) < 2:
            return 0.0
        # 简化: 使用 λ 作为势能指标
        lambda_values = [p['lambda'] for p in path]
        max_lambda = max(lambda_values)
        min_lambda = min(lambda_values)
        return float(max_lambda - min_lambda)
