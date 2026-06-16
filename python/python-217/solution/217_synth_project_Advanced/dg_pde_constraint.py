"""
dg_pde_constraint.py
--------------------
间断 Galerkin PDE 约束离散 —— 映射自种子项目 273_dg1d_heat
核心思想：使用 1D 间断 Galerkin (DG) 方法离散对流-扩散方程，
作为鲁棒优化中的 PDE 约束 c(x, u, w) = 0。

科学背景：
    考虑一维非稳态对流-扩散方程 (SMB 色谱柱模型)：
        du/dt + v * du/dx = D * d^2 u / dx^2 + f(x, t; w)
        x in [0, L], t in [0, T]
    其中 u(x,t) 为浓度，v 为流速，D 为扩散系数，
    w 为不确定性参数 (如 D 的随机扰动、v 的测量误差)。
    边界条件：Dirichlet 入口 u(0,t) = u_in(t)。

DG 离散 (Hesthaven-Warburton)：
    将 [0,L] 分为 K 个单元，每单元 P 阶多项式。
    弱形式：
        M du/dt = -v * S u + D * (M^{-1} S^T M S u + LIFT * flux)
    其中 M 为质量矩阵，S 为刚度矩阵，LIFT 为提升算子。
    数值通量采用中心通量 + 惩罚：
        flux = {u} - sigma * [[u]]
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Jacobi 多项式与求积节点 (Hesthaven-Warburton)
# =============================================================================
def jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    N 阶 Jacobi 多项式 P_N^{(alpha,beta)}(x) 在点 x 处的值。
    三项递推：
        P_0 = 1
        P_1 = 0.5 * ((alpha - beta) + (alpha + beta + 2) * x)
        for n = 2..N:
            a_n = ..., b_n = ..., c_n = ...
            P_n = (a_n x + b_n) P_{n-1} - c_n P_{n-2}
    """
    x = np.atleast_1d(x).astype(float)
    if N == 0:
        return np.ones_like(x)
    gamma0 = 2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0) * np.exp(
        math.lgamma(alpha + 1.0) + math.lgamma(beta + 1.0)
        - math.lgamma(alpha + beta + 1.0)
    )
    p_old = np.ones_like(x)
    p_cur = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
    if N == 1:
        return p_cur
    for n in range(2, N + 1):
        h1 = 2.0 * n + alpha + beta
        a_n = (h1 - 1.0) * h1 * (h1 - 2.0) / (2.0 * n * (n + alpha + beta) * (h1 - 2.0)) if abs(2.0 * n * (n + alpha + beta) * (h1 - 2.0)) > 1e-14 else 0.0
        # 简化递推系数
        a_new = (
            (2.0 * n + alpha + beta - 1.0)
            * (2.0 * n + alpha + beta)
            * (2.0 * n + alpha + beta - 2.0)
        )
        denom = 2.0 * n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0)
        if abs(denom) < 1e-14:
            denom = 1e-14
        a_new = a_new / denom
        b_new = (alpha ** 2 - beta ** 2) * (2.0 * n + alpha + beta - 1.0)
        b_new = b_new / denom if abs(denom) > 1e-14 else 0.0
        c_new = (
            2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * (2.0 * n + alpha + beta)
        )
        c_new = c_new / (n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0)) if abs(n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0)) > 1e-14 else 0.0
        p_new = (a_new * x + b_new) * p_cur - c_new * p_old
        p_old, p_cur = p_cur, p_new
    return p_cur


def jacobi_gq(alpha: float, beta: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Jacobi 求积节点与权重 (N 点)。
    通过伴随矩阵特征值求解。
    """
    if N == 1:
        return np.array([0.0]), np.array([2.0])
    # 构造三对角 Jacobi 矩阵
    h1 = 2.0 * np.arange(1, N) + alpha + beta
    diag = -((alpha ** 2 - beta ** 2) / (h1 * (h1 + 2.0) + 1e-14))
    # 次对角
    i_arr = np.arange(1, N)
    off = (
        2.0
        / (h1 + 1.0)
        * np.sqrt(
            i_arr * (i_arr + alpha) * (i_arr + beta) * (i_arr + alpha + beta)
            / ((h1 ** 2) + 1e-14)
        )
    )
    J = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    # 权重
    mu = 2.0 ** (alpha + beta + 1.0) * np.exp(
        math.lgamma(alpha + 1.0) + math.lgamma(beta + 1.0)
        - math.lgamma(alpha + beta + 1.0)
    )
    w = mu * V[0, :] ** 2
    return x, w


def vandermonde1d(x: np.ndarray, N: int) -> np.ndarray:
    """1D Vandermonde 矩阵 (Legendre 基)：V_{ij} = P_j(x_i)."""
    n_pts = x.size
    V = np.zeros((n_pts, N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_p(x, 0.0, 0.0, j)
    return V


def dmatrix1d(x: np.ndarray, N: int) -> np.ndarray:
    """
    1D 微分矩阵 D_{ij} 在节点 x 处 (Lagrange 插值)。
    D_{ij} = l_j'(x_i)
    """
    V = vandermonde1d(x, N)
    Vr = np.zeros_like(V)
    for i in range(N + 1):
        if i == 0:
            Vr[:, i] = np.zeros(x.size)
        else:
            Vr[:, i] = i * vandermonde1d(x, i - 1)[:, i - 1] if i <= N else 0.0
    # 简化：直接构造 Lagrange 微分矩阵
    n = x.size
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = 1.0 / (x[i] - x[j])
                for k in range(n):
                    if k != i and k != j:
                        D[i, j] *= (x[i] - x[k]) / (x[j] - x[k])
            else:
                s = 0.0
                for k in range(n):
                    if k != i:
                        s += 1.0 / (x[i] - x[k])
                D[i, i] = s
    return D


# =============================================================================
# DG 离散 1D 对流-扩散
# =============================================================================
class DG1DHeat:
    """
    1D DG 对流-扩散方程求解器。
    方程：du/dt + v du/dx = D d^2u/dx^2 + f
    边界：u(0,t) = u_in(t), du/dx(L,t) = 0
    """

    def __init__(
        self,
        K: int = 16,
        Np: int = 3,
        L: float = 1.0,
        velocity: float = 1.0,
        diffusion: float = 0.01,
    ):
        self.K = K          # 单元数
        self.Np = Np        # 每单元节点数 (阶数+1)
        self.L = L
        self.velocity = velocity
        self.diffusion = diffusion
        # 网格
        self.dx = L / K
        self.xF = np.linspace(0.0, L, K + 1)
        # Gauss-Lobatto 节点 (参考单元 [-1, 1])
        self.r = np.cos(np.pi * np.arange(Np) / max(Np - 1, 1))  # Chebyshev 近似
        self.r = np.sort(self.r)
        # 微分矩阵
        self.Dr = self._build_dr()
        # 质量矩阵 (对角, Gauss-Lobatto 权重)
        self.M = np.diag(self._weights())
        self.Minv = np.diag(1.0 / self._weights())
        # 全局坐标
        self.x = np.zeros((Np, K))
        for k in range(K):
            self.x[:, k] = 0.5 * (
                (self.xF[k + 1] - self.xF[k]) * (self.r + 1.0)
                + 2.0 * self.xF[k]
            )
        # 提升矩阵
        self.LIFT = self._build_lift()

    def _weights(self) -> np.ndarray:
        """Gauss-Lobatto 权重 (近似)."""
        Np = self.Np
        w = np.zeros(Np)
        for i in range(Np):
            w[i] = 2.0 / Np if 0 < i < Np - 1 else 1.0 / Np
        return w

    def _build_dr(self) -> np.ndarray:
        return dmatrix1d(self.r, self.Np - 1)

    def _build_lift(self) -> np.ndarray:
        """LIFT 矩阵：边界项提升。"""
        Np = self.Np
        V = vandermonde1d(self.r, Np - 1)
        Em = np.zeros((Np, 2))
        Em[0, 0] = 1.0
        Em[-1, 1] = 1.0
        mass = V @ V.T
        try:
            Minv = np.linalg.inv(mass + 1e-12 * np.eye(Np))
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(mass)
        return Minv @ Em

    def rhs(
        self,
        u: np.ndarray,
        t: float,
        u_in: float = 1.0,
        source_fn: Optional[callable] = None,
    ) -> np.ndarray:
        """
        计算 du/dt 的右端项。
        u 形状 (Np, K)。
        """
        Np, K = u.shape
        du = np.zeros_like(u)
        v = self.velocity
        D = self.diffusion

        for k in range(K):
            # 局部梯度
            dur = self.Dr @ u[:, k]
            # 度量因子
            rx = 2.0 / self.dx
            dux = dur * rx
            # 对流项
            du[:, k] -= v * dux
            # 扩散项
            if D > 0.0:
                d2ux = (self.Dr @ (dux * rx)) * rx  # 近似
                du[:, k] += D * d2ux

        # 数值通量 (Lax-Friedrichs)
        for k in range(K):
            # 左界面
            if k == 0:
                uL = u_in
            else:
                uL = u[-1, k - 1]
            uR = u[0, k]
            flux_L = 0.5 * v * (uL + uR) - 0.5 * abs(v) * (uR - uL)
            # 右界面
            if k < K - 1:
                uR2 = u[0, k + 1]
            else:
                uR2 = u[-1, k]  # Neumann
            uL2 = u[-1, k]
            flux_R = 0.5 * v * (uL2 + uR2) - 0.5 * abs(v) * (uR2 - uL2)
            # 提升
            du[0, k] += self.LIFT[0, 0] * (v * uL - flux_L) / self.dx
            du[-1, k] += self.LIFT[-1, 1] * (flux_R - v * uL2) / self.dx

        # 源项
        if source_fn is not None:
            for k in range(K):
                du[:, k] += source_fn(self.x[:, k], t)

        return du

    def solve(
        self,
        u0: np.ndarray,
        T_final: float,
        u_in: float = 1.0,
        source_fn: Optional[callable] = None,
        cfl: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RK4 时间推进求解。
        返回 (u_final, x_flat, t_array).
        """
        u = u0.copy()
        dt = cfl * (self.dx / self.Np) ** 2 / max(self.diffusion, 1e-14)
        dt = min(dt, cfl * self.dx / max(abs(self.velocity), 1e-14))
        Nsteps = max(1, int(np.ceil(T_final / dt)))
        dt = T_final / Nsteps
        t_arr = np.linspace(0.0, T_final, Nsteps + 1)

        # RK4 系数 (经典)
        rk_a = [0.0, -0.5, -0.5, -1.0]
        rk_b = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
        rk_c = [0.0, 0.5, 0.5, 1.0]

        res = np.zeros_like(u)
        for step in range(Nsteps):
            t = t_arr[step]
            k_stages = []
            u_tmp = u.copy()
            for s in range(4):
                rhs_s = self.rhs(u_tmp, t + rk_c[s] * dt, u_in, source_fn)
                k_stages.append(rhs_s)
                if s < 3:
                    u_tmp = u + dt * sum(rk_b[i] * k_stages[i] for i in range(s + 1))
            u = u + dt * sum(rk_b[i] * k_stages[i] for i in range(4))

        x_flat = self.x.ravel(order="F")
        u_flat = u.ravel(order="F")
        return u_flat, x_flat, t_arr


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dg = DG1DHeat(K=8, Np=3, L=1.0, velocity=1.0, diffusion=0.05)
    u0 = np.zeros((dg.Np, dg.K))
    u_final, x, t = dg.solve(u0, T_final=0.1, u_in=1.0)
    print(f"DG solve: max(u)={np.max(u_final):.4f}, min(u)={np.min(u_final):.4f}")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
