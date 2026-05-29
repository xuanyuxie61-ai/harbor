# -*- coding: utf-8 -*-
"""
stability_analysis.py
高超声速边界层线性稳定性分析 (LST)

核心算法来源：
- jordan_matrix: Jordan 标准形分解，用于分析算子的非模态增长（transient growth）
- triangulation_neighbor: 邻居搜索思想，用于特征值谱的连通性分析与模态追踪

物理背景：
高超声速边界层转捩由小扰动的线性放大主导。
线性化 Navier-Stokes 方程在平行流假设下简化为 Orr-Sommerfeld 方程。

对于可压缩边界层，稳定性方程组为：

    i (α U - ω) û + α dU/dy v̂ + i α p̂/γ = 1/Re [d²û/dy² - α² û + (1/3) i α (i α û + dv̂/dy)]

    i (α U - ω) v̂ + d p̂/dy / γ = 1/Re [d²v̂/dy² - α² v̂ + (1/3) (i α û + dv̂/dy)']

    i (α U - ω) θ̂ + dT/dy v̂ + (γ-1) (i α û + dv̂/dy) = 1/(Re Pr) [d²θ̂/dy² - α² θ̂]

    i α û + dv̂/dy = i (α U - ω) p̂ - (γ-1) (i α û + dv̂/dy)

其中 α 为流向波数，ω 为角频率，û, v̂, θ̂, p̂ 为扰动速度、温度、压力。

本模块构建离散稳定性矩阵 A，求解广义特征值问题:
    A q = ω B q
并分析特征值谱的 Jordan 结构以评估非模态增长。
"""

import numpy as np
from utils import chebyshev_diff_matrix, chebyshev_nodes, safe_divide


class CompressibleLST:
    """
    可压缩边界层线性稳定性理论 (LST) 求解器。
    """

    def __init__(self, Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4, N=120):
        """
        参数:
            Ma (float): 马赫数
            Re (float): 雷诺数
            Pr (float): 普朗特数
            gamma (float): 比热比
            N (int): Chebyshev 多项式阶数
        """
        self.Ma = Ma
        self.Re = Re
        self.Pr = Pr
        self.gamma = gamma
        self.N = N

        # Chebyshev 节点与微分矩阵
        self.eta_cheb = chebyshev_nodes(N, a=0.0, b=12.0)
        self.D = chebyshev_diff_matrix(N, a=0.0, b=12.0)
        self.D2 = self.D @ self.D

    def set_baseflow(self, eta, u, T, mu):
        """
        设置基流剖面（需插值到 Chebyshev 节点）。

        参数:
            eta (np.ndarray): 原始 η 坐标
            u (np.ndarray): 速度剖面 u(η)
            T (np.ndarray): 温度剖面 T(η)
            mu (np.ndarray): 粘性剖面 μ(η)
        """
        self.u_base = np.interp(self.eta_cheb, eta, u)
        self.T_base = np.interp(self.eta_cheb, eta, T)
        self.mu_base = np.interp(self.eta_cheb, eta, mu)

        # 导数（通过 Chebyshev 微分）
        self.du_deta = self.D @ self.u_base
        self.dT_deta = self.D @ self.T_base
        self.dmu_deta = self.D @ self.mu_base

        # 密度（理想气体）
        self.rho_base = safe_divide(1.0, self.T_base, fill_value=1.0)

    def build_stability_operator(self, alpha, beta=0.0):
        """
        构建空间模式 (temporal) 稳定性算子矩阵。

        对于时间模式，求解特征值 ω：
            det(A - ω I) = 0

        其中 A 为 4N × 4N 复矩阵，由离散化后的 LST 方程构成。
        状态向量按 [u; v; T; p] 排列。

        参数:
            alpha (float): 流向波数
            beta (float): 展向波数

        返回:
            tuple: (A, B) 广义特征值问题的矩阵对
        """
        N = self.N + 1  # 节点数
        n_eq = 4 * N

        # 单位矩阵与微分算子
        I = np.eye(N)
        D1 = self.D
        D2 = self.D2

        # 基流对角矩阵
        U = np.diag(self.u_base)
        T = np.diag(self.T_base)
        Mu = np.diag(self.mu_base)
        Rho = np.diag(self.rho_base)
        dU = np.diag(self.du_deta)
        dT = np.diag(self.dT_deta)
        dMu = np.diag(self.dmu_deta)

        # 波数
        k2 = alpha**2 + beta**2

        # 算子矩阵初始化
        A = np.zeros((n_eq, n_eq), dtype=complex)

        # ---- 动量-x 方程 ----
        # i α ρ (U - c) u + ρ v dU/dy + i α p = μ (d²u/dy² - k² u) + dμ/dy (du/dy + i α v)
        # 这里直接写成 A q = ω B q 形式，设 ω = α c
        # 整理为: i α ρ U u + ρ dU v + i α p - μ (D² - k²) u - dμ (D u + i α v) = ω (i α ρ u)

        row_u = slice(0, N)
        row_v = slice(N, 2 * N)
        row_T = slice(2 * N, 3 * N)
        row_p = slice(3 * N, 4 * N)

        col_u = slice(0, N)
        col_v = slice(N, 2 * N)
        col_T = slice(2 * N, 3 * N)
        col_p = slice(3 * N, 4 * N)

        # 动量-x
        A[row_u, col_u] = 1j * alpha * Rho @ U - Mu @ (D2 - k2 * I) - dMu @ D1
        A[row_u, col_v] = Rho @ dU - 1j * alpha * dMu
        A[row_u, col_p] = 1j * alpha * I

        # TODO: 完成动量-y、能量与连续性方程的矩阵块组装
        # 提示: 需根据可压缩线性稳定性理论 (LST) 的以下方程，
        #       用 Chebyshev 微分矩阵 D1/D2 和基流对角矩阵组装 A 的子块:
        #
        #   动量-y:  iαρUv + dp/dη = μ(D²-k²)v + dμ/dη·Dv
        #   能量:    iαρUθ + ρv·dT/dη + (γ-1)(iαu+Dv) = 1/(Re·Pr)·μ(D²-k²)θ
        #   连续:    iαρu + D(ρv) = iαρU·p/γ - (γ-1)(iαu+Dv)ρ  (简化形式)
        #
        #   注意: 本模块的 D1/D2 来自 utils.chebyshev_diff_matrix，
        #         若该函数实现有误，此处即使公式正确也无法得到正确特征值。
        raise NotImplementedError("build_stability_operator: 请完成动量-y、能量与连续性方程的矩阵块组装")

        # 边界条件（通过行替换实现）
        # 壁面 (η=0): u=v=θ=0, dp/dy=0
        # 远场 (η=η_max): u=v=θ=p=0
        bc_rows = []

        # 壁面
        bc_rows.append((0, col_u, np.zeros(N), 1.0))       # u(0)=0
        bc_rows.append((N, col_v, np.zeros(N), 1.0))       # v(0)=0
        bc_rows.append((2 * N, col_T, np.zeros(N), 1.0))   # θ(0)=0
        # dp/dy(0)=0 -> D1[0,:] @ p = 0

        # 远场
        bc_rows.append((N - 1, col_u, np.zeros(N), 1.0))   # u(∞)=0
        bc_rows.append((2 * N - 1, col_v, np.zeros(N), 1.0))  # v(∞)=0
        bc_rows.append((3 * N - 1, col_T, np.zeros(N), 1.0))  # θ(∞)=0
        bc_rows.append((4 * N - 1, col_p, np.zeros(N), 1.0))  # p(∞)=0

        B = np.eye(n_eq, dtype=complex)
        # 仅保留时间导数项在 B 中
        B[row_u, col_u] = 1j * alpha * Rho
        B[row_v, col_v] = 1j * alpha * Rho
        B[row_T, col_T] = 1j * alpha * Rho
        B[row_p, col_p] = 1j * alpha * Rho @ U / self.gamma

        # 施加边界条件
        for r, c, vec, diag_val in bc_rows:
            A[r, :] = 0.0
            A[r, c] = vec
            A[r, r] = diag_val
            B[r, :] = 0.0
            B[r, r] = 1.0

        # dp/dy(0)=0
        r = 3 * N
        if r < n_eq:
            A[r, :] = 0.0
            A[r, row_p] = D1[0, :]
            B[r, :] = 0.0

        return A, B

    def temporal_eigenvalues(self, alpha, beta=0.0):
        """
        求解时间模式特征值 ω = α c。

        参数:
            alpha (float): 流向波数
            beta (float): 展向波数

        返回:
            np.ndarray: 特征值 ω（按虚部降序排列，最不稳定模态在前）
        """
        A, B = self.build_stability_operator(alpha, beta)
        try:
            eigvals, eigvecs = np.linalg.eig(np.linalg.solve(B, A))
        except np.linalg.LinAlgError:
            # 若 B 奇异，使用伪逆替代
            B_reg = B + 1e-12 * np.eye(B.shape[0], dtype=complex)
            eigvals = np.linalg.eigvals(np.linalg.solve(B_reg, A))

        # 筛选物理模态: Im(ω) > 0 表示时间增长（不稳定）
        # 按虚部降序
        idx = np.argsort(-np.imag(eigvals))
        return eigvals[idx]

    def spatial_eigenvalues(self, omega_real, beta=0.0, alpha_guess=0.5):
        """
        空间模式：固定 ω_r，求解 α。

        采用牛顿迭代在复平面上搜索：
            D(α; ω, Re) = 0

        参数:
            omega_real (float): 实频率
            beta (float): 展向波数
            alpha_guess (complex): 初始猜测

        返回:
            list[complex]: 空间特征值 α
        """
        # 简化处理：在多个 α 初值下进行局部搜索
        alphas = []
        for guess in [alpha_guess, alpha_guess * 1j, -alpha_guess, 0.1 + 0.1j]:
            alpha = self._newton_spatial(guess, omega_real, beta)
            if alpha is not None:
                alphas.append(alpha)
        return alphas

    def _newton_spatial(self, alpha0, omega_r, beta, max_iter=30, tol=1e-8):
        """
        牛顿迭代求解空间特征值。

        色散关系 D(α) = det(A(α) - ω I) = 0。
        采用特征值追踪而非行列式（数值稳定性更好）。
        """
        alpha = complex(alpha0)
        for _ in range(max_iter):
            A, B = self.build_stability_operator(alpha.real, beta)
            # 求最接近 omega_r 的特征值
            try:
                ev = np.linalg.eigvals(A, B)
            except Exception:
                return None
            distances = np.abs(ev - omega_r)
            k = np.argmin(distances)
            residual = ev[k] - omega_r
            if abs(residual) < tol:
                return alpha

            # 数值 Jacobian
            h = 1e-6
            Aph, Bph = self.build_stability_operator(alpha.real + h, beta)
            try:
                evp = np.linalg.eigvals(Aph, Bph)
            except Exception:
                return None
            dk_dar = (evp[np.argmin(np.abs(evp - omega_r))] - ev[np.argmin(distances)]) / h

            if abs(dk_dar) < 1e-12:
                break
            alpha = alpha - residual / dk_dar
        return None if abs(residual) >= tol else alpha

    def jordan_analysis(self, alpha, beta=0.0):
        """
        基于 jordan_matrix 思想的稳定性算子 Jordan 分析。

        对 LST 离散矩阵进行 Jordan 分解：
            A = P J P^{-1}

        Jordan 块大小反映特征值的几何重数缺陷，与非模态增长直接相关。
        高超声速边界层中，连续谱与离散模态的接近可导致显著的 transient growth。

        参数:
            alpha (float): 流向波数
            beta (float): 展向波数

        返回:
            dict: 包含 Jordan 块结构、条件数等信息
        """
        A, B = self.build_stability_operator(alpha, beta)
        M = np.linalg.solve(B + 1e-12 * np.eye(B.shape[0]), A)

        # 计算特征值与特征向量
        eigvals, eigvecs = np.linalg.eig(M)

        # 条件数（非模态增长度量）
        cond_num = np.linalg.cond(eigvecs)

        # 特征值聚类分析（识别 Jordan 块）
        tol = 1e-4
        clusters = []
        used = set()
        for i in range(len(eigvals)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(eigvals)):
                if j not in used and abs(eigvals[i] - eigvals[j]) < tol:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)

        # 最大 Jordan 块大小估计
        max_block_size = max(len(c) for c in clusters) if clusters else 1

        # 瞬态增长上界: G_max ≤ ||V|| ||V^{-1}|| exp(Re(ω_max) t)
        omega_max = np.max(np.imag(eigvals))

        return {
            'eigenvalues': eigvals,
            'condition_number': cond_num,
            'clusters': clusters,
            'max_jordan_block': max_block_size,
            'transient_growth_bound': cond_num,
            'max_temporal_growth_rate': omega_max
        }


def track_eigenvalue_mode(alpha_list, lst_solver, beta=0.0):
    """
    基于 triangulation_neighbor 思想的特征值模态追踪。

    随波数 α 变化时，通过最近邻搜索追踪同一物理模态的演化，
    绘制中性曲线（neutral curve）。

    参数:
        alpha_list (list[float]): 波数列表
        lst_solver (CompressibleLST): LST 求解器实例
        beta (float): 展向波数

    返回:
        list[complex]: 追踪到的特征值序列
    """
    tracked = []
    prev_omega = None

    for alpha in alpha_list:
        omegas = lst_solver.temporal_eigenvalues(alpha, beta)
        if len(omegas) == 0:
            tracked.append(np.nan)
            continue

        if prev_omega is None:
            # 取最不稳定模态
            choice = omegas[0]
        else:
            # 最近邻追踪
            distances = np.abs(omegas - prev_omega)
            choice = omegas[np.argmin(distances)]

        tracked.append(choice)
        prev_omega = choice

    return tracked
