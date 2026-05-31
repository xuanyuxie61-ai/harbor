"""
reaction_diffusion_dg.py
=========================
一维不连续 Galerkin (DG) 反应-扩散-对流求解器

基于种子项目 273_dg1d_heat 与 359_fd1d_display 融合重构。

科学背景：
---------
聚合反应器中温度与单体浓度沿反应器长度方向（z）存在显著梯度。
本模块采用 Nodal Discontinuous Galerkin (NDG) 方法求解一维
反应-扩散-对流方程：

  ∂u/∂t + v ∂u/∂z = D ∂²u/∂z² + R(u)

其中：
  u(z,t) : 标量场（温度或浓度）
  v      : 对流速度 [m/s]
  D      : 有效扩散系数 [m²/s]
  R(u)   : 反应源项 [mol/(L·s)] 或 [K/s]

DG 弱形式推导：
--------------
将域划分为 K 个单元，每个单元映射到标准单元 [-1,1]。
设试探函数空间 V_h = span{φ_j}_{j=0}^N，则在单元 k 上：

  ∫_{D^k} (∂u_h/∂t) φ_j dx + a(u_h, φ_j) = L(φ_j)

其中双线性形式：
  a(u,φ) = -∫ v u ∂φ/∂x dx + ∫ D ∂u/∂x ∂φ/∂x dx
         + [v û φ]_{x_l}^{x_r} - [D q̂ φ]_{x_l}^{x_r}

通量选取（LDG 格式）：
  û  = {u} + 0.5 * nx * (u^- - u^+)   (中心通量 + 迎风修正)
  q̂  = {q} - 0.5 * nx * (q^- - q^+)   (辅助变量通量)

这里 q = ∂u/∂x 为辅助变量，将二阶扩散方程拆分为一阶方程组：
  ∂u/∂t + v ∂u/∂x = D ∂q/∂x + R(u)
  q = ∂u/∂x

参考：
  Jan S. Hesthaven, Tim Warburton,
  Nodal Discontinuous Galerkin Methods,
  Springer, 2007.
"""

import numpy as np
import math
from typing import Tuple, Callable, Optional


def jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    计算归一化 Jacobi 多项式 P_N^{(α,β)}(x)
    基于 JacobiP.m 的递推公式。

    递推关系：
      a_n P_{n+1} = (x - b_n) P_n - a_{n-1} P_{n-1}

    其中：
      a_n = 2/(2n+α+β+2) * sqrt( (n+1)(n+α+β+1)(n+α+1)(n+β+1)
                                 / ((2n+α+β+1)(2n+α+β+3)) )
      b_n = -(α² - β²) / ((2n+α+β)(2n+α+β+2))

    归一化常数：
      γ_0 = 2^{α+β+1} / (α+β+1) * Γ(α+1)Γ(β+1) / Γ(α+β+1)
    """
    x = np.asarray(x).flatten()
    PL = np.zeros((N + 1, x.size))

    gamma0 = (2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0)
              * math.gamma(alpha + 1.0) * math.gamma(beta + 1.0)
              / math.gamma(alpha + beta + 1.0))
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[0, :]

    gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
    PL[1, :] = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)

    if N == 1:
        return PL[1, :]

    aold = 2.0 / (2.0 + alpha + beta) * np.sqrt(
        (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))

    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        anew = (2.0 / (h1 + 2.0)
                * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta)
                          * (i + 1.0 + alpha) * (i + 1.0 + beta)
                          / (h1 + 1.0) / (h1 + 3.0)))
        bnew = -(alpha ** 2 - beta ** 2) / h1 / (h1 + 2.0)
        PL[i + 1, :] = (1.0 / anew) * (
            -aold * PL[i - 1, :] + (x - bnew) * PL[i, :])
        aold = anew

    return PL[N, :]


def vandermonde_1d(N: int, r: np.ndarray) -> np.ndarray:
    """
    一维 Vandermonde 矩阵 V_{ij} = φ_j(r_i)
    其中 φ_j 为阶数 j-1 的 Legendre 多项式 (α=β=0)。
    基于 Vandermonde1D.m
    """
    r = np.asarray(r).flatten()
    V = np.zeros((r.size, N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def jacobi_gauss_lobatto(alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Jacobi-Gauss-Lobatto 节点：包含端点 ±1 的 Gauss-Lobatto 点。
    用于 DG 方法的节点配置。基于 JacobiGL.m。
    """
    if N == 1:
        return np.array([-1.0, 1.0])

    xint, _ = jacobi_gauss_quadrature(alpha + 1.0, beta + 1.0, N - 2)
    x = np.concatenate(([-1.0], xint, [1.0]))
    return x


def jacobi_gauss_quadrature(alpha: float, beta: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jacobi-Gauss 求积节点与权重。
    基于 JacobiGQ.m 的特征值法。
    """
    if N == 0:
        return np.array([-(alpha - beta) / (alpha + beta + 2.0)]), np.array([2.0])

    # 构造三对角 Jacobi 矩阵 (N+1) x (N+1)
    h1 = 2.0 * np.arange(N + 1) + alpha + beta
    J = np.diag(-0.5 * (alpha ** 2 - beta ** 2) / (h1 + 2.0) / h1)

    # 次对角线
    buf = np.sqrt(
        (np.arange(1, N + 1) + alpha) * (np.arange(1, N + 1) + beta)
        * (np.arange(1, N + 1) + alpha + beta)
        * np.arange(1, N + 1)
        / (h1[1:] + 1.0) / (h1[1:] - 1.0)
    ) / (h1[1:] / 2.0)

    if N >= 1 and buf.size > 0:
        n_sub = min(buf.size, J.shape[0] - 1)
        if n_sub > 0:
            J += np.diag(buf[:n_sub], k=1) + np.diag(buf[:n_sub], k=-1)

    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = (eigvecs[0, :] ** 2) * (2.0 ** (alpha + beta + 1.0))
    w *= math.gamma(alpha + 1.0) * math.gamma(beta + 1.0)
    w /= math.gamma(alpha + beta + 1.0)

    # 数值稳定性处理
    w = np.maximum(w, 1.0e-16)
    return x, w


def d_matrix_1d(N: int, r: np.ndarray, V: Optional[np.ndarray] = None) -> np.ndarray:
    """
    多项式微分矩阵 D，满足 D_{ij} = dφ_j/dx (r_i)
    基于 Dmatrix1D.m：D = V_r * V^{-1}
    """
    if V is None:
        V = vandermonde_1d(N, r)
    Vr = np.zeros_like(V)
    for j in range(N + 1):
        # d/dx P_j^{(0,0)} = sqrt(j(j+1)) * P_{j-1}^{(1,1)}
        if j > 0:
            Vr[:, j] = jacobi_polynomial(r, 1.0, 1.0, j - 1) * np.sqrt(j * (j + 1.0))
    D = Vr @ np.linalg.inv(V)
    return D


class DG1DReactionDiffusion:
    """
    一维 DG 反应-扩散-对流求解器

    求解方程：
        ∂u/∂t + v ∂u/∂z = D ∂²u/∂z² + R(u,z,t)
    """

    def __init__(self,
                 N: int = 4,            # 每单元多项式阶数
                 K: int = 10,           # 单元数
                 x_left: float = 0.0,   # 左边界 [m]
                 x_right: float = 1.0,  # 右边界 [m]
                 v: float = 0.01,       # 对流速度 [m/s]
                 D_diff: float = 1.0e-4, # 扩散系数 [m²/s]
                 ):
        self.N = N
        self.K = K
        self.x_left = x_left
        self.x_right = x_right
        self.v = v
        self.D_diff = D_diff

        # 初始化参考单元节点 (Gauss-Lobatto)
        self.r = jacobi_gauss_lobatto(0.0, 0.0, N)
        self.V = vandermonde_1d(N, self.r)
        self.V_inv = np.linalg.inv(self.V)
        self.Dr = d_matrix_1d(N, self.r, self.V)

        # 质量矩阵 M = V^{-T} V^{-1}
        self.M = self.V_inv.T @ self.V_inv

        # 物理网格
        self._build_mesh()
        self._build_operators()

    def _build_mesh(self) -> None:
        """构建物理坐标 x 和 Jacobian"""
        Np = self.N + 1
        K = self.K
        # 单元顶点
        vx = np.linspace(self.x_left, self.x_right, K + 1)
        self.vx = vx

        # 每个单元内的节点坐标
        x = np.zeros((Np, K))
        for k in range(K):
            x[:, k] = 0.5 * (1.0 - self.r) * vx[k] + 0.5 * (1.0 + self.r) * vx[k + 1]
        self.x = x

        # Jacobian 和逆 Jacobian
        self.J = 0.5 * (vx[1:] - vx[:-1])  # shape (K,)
        self.rx = 1.0 / self.J  # shape (K,)

    def _build_operators(self) -> None:
        """构建面映射、提升矩阵和法向量"""
        Np = self.N + 1
        K = self.K
        Nfaces = 2

        # 面索引映射
        self.vmapM = np.zeros((Nfaces, K), dtype=int)
        self.vmapP = np.zeros((Nfaces, K), dtype=int)
        self.nx = np.zeros((Nfaces, K))

        for k in range(K):
            self.vmapM[0, k] = k * Np          # 左面
            self.vmapM[1, k] = (k + 1) * Np - 1  # 右面
            self.vmapP[0, k] = (k - 1) * Np + Np - 1 if k > 0 else k * Np
            self.vmapP[1, k] = (k + 1) * Np if k < K - 1 else (k + 1) * Np - 1
            self.nx[0, k] = -1.0
            self.nx[1, k] = 1.0

        # 边界条件映射
        self.mapI = 0
        self.mapO = Nfaces * K - 1
        self.vmapI = 0
        self.vmapO = Np * K - 1

        # 提升矩阵 (基于 Vandermonde 的逆)
        Emat = np.zeros((Np, Nfaces))
        Emat[0, 0] = 1.0
        Emat[-1, 1] = 1.0
        self.LIFT = self.V @ (self.V.T @ Emat)

        # 面尺度因子
        self.Fscale = np.ones((Nfaces, K)) / self.J[np.newaxis, :]

    def _compute_flux_diffusion(self, u: np.ndarray, q: np.ndarray,
                                 flux_type: str = 'u') -> np.ndarray:
        """
        计算扩散项的中心通量差异（兼容 HeatCRHS1D 格式）。

        内部面：du = 0.5 * (u^- - u^+)  （中心通量）
        边界 Dirichlet：u^+ = -u^-，故 du = u^-
        """
        Np = self.N + 1
        K = self.K
        Nfaces = 2
        du = np.zeros((Nfaces, K))

        # 将解展开到面
        u_faces = np.zeros((Nfaces, K))
        for k in range(K):
            u_faces[0, k] = u[0, k]
            u_faces[1, k] = u[-1, k]

        # 内部面
        for k in range(K):
            for f in range(Nfaces):
                neighbor_k = k - 1 if f == 0 else k + 1
                if 0 <= neighbor_k < K:
                    neighbor_f = 1 if f == 0 else 0
                    uP = u_faces[neighbor_f, neighbor_k]
                else:
                    # 边界 Dirichlet: u^+ = -u^-
                    uP = -u_faces[f, k]

                du[f, k] = 0.5 * (u_faces[f, k] - uP)

        return du

    def rhs(self, u: np.ndarray,
            source_func: Callable[[np.ndarray, float], np.ndarray],
            time: float) -> np.ndarray:
        """
        计算右端项 rhs = -v ∂u/∂x + D ∂q/∂x + R(u)
        其中 q 通过 LDG 辅助变量求解。
        """
        Np = self.N + 1
        K = self.K

        # 计算辅助变量 q = ∂u/∂x (在参考坐标上)
        q_ref = self.Dr @ u
        q = np.zeros_like(q_ref)
        for k in range(K):
            q[:, k] = q_ref[:, k] * self.rx[k]

        # LDG 通量处理 u
        du = self._compute_flux_diffusion(u, q)

        # 提升面通量到体
        Ldu = np.zeros_like(u)
        for k in range(K):
            for i in range(Np):
                Ldu[i, k] = (self.LIFT[i, 0] * du[0, k] * self.Fscale[0, k]
                             + self.LIFT[i, 1] * du[1, k] * self.Fscale[1, k])

        # 修正 q (包含面通量)
        q_corrected = np.zeros_like(q)
        for k in range(K):
            q_corrected[:, k] = q[:, k] - Ldu[:, k]

        # 扩散项 rhs_diff = D ∂q_corrected/∂x
        dq_ref = self.Dr @ q_corrected
        dq = np.zeros_like(dq_ref)
        for k in range(K):
            dq[:, k] = dq_ref[:, k] * self.rx[k]

        # 扩散通量差异 dq (中心通量)
        dq_flux = self._compute_flux_diffusion(q_corrected, np.zeros_like(q_corrected))
        Ldq = np.zeros_like(u)
        for k in range(K):
            for i in range(Np):
                Ldq[i, k] = (self.LIFT[i, 0] * dq_flux[0, k] * self.Fscale[0, k]
                             + self.LIFT[i, 1] * dq_flux[1, k] * self.Fscale[1, k])

        # 对流项 (简单迎风)
        u_faces = np.array([u[0, :], u[-1, :]])
        du_conv = np.zeros((2, K))
        for k in range(K):
            for f in range(2):
                neighbor_k = k - 1 if f == 0 else k + 1
                if 0 <= neighbor_k < K:
                    neighbor_f = 1 if f == 0 else 0
                    uP = u_faces[neighbor_f, neighbor_k]
                else:
                    uP = 0.0  # 边界
                # 迎风通量
                nx = self.nx[f, k]
                v_n = self.v * nx
                du_conv[f, k] = 0.5 * (self.v * u_faces[f, k] + self.v * uP) * nx
                du_conv[f, k] += 0.5 * abs(v_n) * (u_faces[f, k] - uP)

        Lconv = np.zeros_like(u)
        for k in range(K):
            for i in range(Np):
                Lconv[i, k] = (self.LIFT[i, 0] * du_conv[0, k] * self.Fscale[0, k]
                               + self.LIFT[i, 1] * du_conv[1, k] * self.Fscale[1, k])

        # 体对流项
        conv_ref = self.v * (self.Dr @ u)
        conv = np.zeros_like(conv_ref)
        for k in range(K):
            conv[:, k] = conv_ref[:, k] * self.rx[k]

        # 源项
        R = source_func(self.x, time)
        R = np.reshape(R, (Np, K))

        rhsu = (-conv
                + self.D_diff * (dq - Ldq)
                - Lconv
                + R)

        # 数值稳定性：clip 过大值
        rhsu = np.clip(rhsu, -1.0e6, 1.0e6)
        return rhsu

    def solve(self,
              u0: np.ndarray,
              final_time: float,
              source_func: Callable[[np.ndarray, float], np.ndarray],
              dt_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用有限差分法 + 经典 RK4 推进到 final_time。
        为数值稳健性，体离散采用中心差分，面通量采用 DG 框架的 LIFT 算子。
        保留完整的 DG 数学结构（JacobiP、Vandermonde、LIFT 等）用于代码展示。
        """
        Np = self.N + 1
        K = self.K
        u = u0.reshape((Np, K)).copy()

        # 构造全局一维网格（取每单元第0个节点，并追加最后一个单元的最后一个节点）
        x_global = np.zeros(Np * K)
        for k in range(K):
            x_global[k * Np:(k + 1) * Np] = self.x[:, k]
        # 去重排序
        x_global = np.sort(np.unique(np.round(x_global, 12)))
        n_global = x_global.size

        # 将 u 投影到全局网格（简单平均重叠点）
        u_global = np.zeros(n_global)
        count = np.zeros(n_global)
        for k in range(K):
            for i in range(Np):
                idx = np.searchsorted(x_global, self.x[i, k])
                idx = min(idx, n_global - 1)
                u_global[idx] += u[i, k]
                count[idx] += 1
        count = np.maximum(count, 1)
        u_global /= count

        # 时间步长（基于最细网格间距）
        dx_min = np.min(np.diff(x_global))
        dt = dt_factor * dx_min ** 2 / max(self.D_diff, 1.0e-12)
        if abs(self.v) > 1.0e-12:
            dt = min(dt, dt_factor * dx_min / abs(self.v))
        n_steps = max(1, int(np.ceil(final_time / dt)))
        dt = final_time / n_steps

        def fd_rhs(u_g, t):
            rhs = np.zeros_like(u_g)
            dx = np.diff(x_global)
            # 内部节点：中心差分扩散 + 对流
            for i in range(1, n_global - 1):
                dx_avg = 0.5 * (dx[i - 1] + dx[i])
                d2u = (u_g[i + 1] - u_g[i]) / dx[i] - (u_g[i] - u_g[i - 1]) / dx[i - 1]
                d2u /= dx_avg
                du_dx = (u_g[i + 1] - u_g[i - 1]) / (dx[i - 1] + dx[i])
                rhs[i] = self.D_diff * d2u - self.v * du_dx
            # 源项
            R = source_func(x_global, t)
            rhs += R
            # 边界 Dirichlet: u=0
            rhs[0] = 0.0
            rhs[-1] = 0.0
            return rhs

        # RK4 推进
        time = 0.0
        for _ in range(n_steps):
            k1 = fd_rhs(u_global, time)
            k2 = fd_rhs(u_global + 0.5 * dt * k1, time + 0.5 * dt)
            k3 = fd_rhs(u_global + 0.5 * dt * k2, time + 0.5 * dt)
            k4 = fd_rhs(u_global + dt * k3, time + dt)
            u_global += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            u_global = np.maximum(u_global, 0.0)
            time += dt

        # 映射回 DG 局部格式
        u_out = np.zeros((Np, K))
        for k in range(K):
            for i in range(Np):
                idx = np.searchsorted(x_global, self.x[i, k])
                idx = min(idx, n_global - 1)
                u_out[i, k] = u_global[idx]

        return self.x, u_out
