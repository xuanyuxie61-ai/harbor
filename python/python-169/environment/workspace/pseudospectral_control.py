"""
Gauss-Legendre 伪谱法轨迹优化模块
====================================
基于种子项目:
  - 663_legendre_product_display : Gauss-Legendre节点/权重计算与张量积网格

核心数学模型:
  1. 一维Gauss-Legendre求积:
     ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)
     其中 x_i 是 n 阶Legendre多项式 P_n(x) 的根，
     权重 w_i = 2 / [(1-x_i^2) (P_n'(x_i))^2]。

  2. 根的初始猜测与Newton修正:
     x_0^{(0)} = cos((4i-1)π / (4n+2)) · (1 - (1-1/n)/(8n^2))
     然后用Newton迭代精化至机器精度。

  3. 伪谱微分矩阵 D:
     D_{ij} = (P_N'(x_j) / P_N'(x_i)) · 1/(x_j - x_i)   (i≠j)
     D_{ii} = x_i / (1 - x_i^2)
     状态导数:  \dot{x}(τ_k) = Σ_{j=0}^{N} D_{kj} x(τ_j)

  4. 配点法约束:
     在Legendre-Radau节点上要求
       Σ_j D_{kj} x_j - (t_f - t_0)/2 · f(x_k, u_k) = 0
     其中缩放因子 (t_f - t_0)/2 来自时间域变换 τ ∈ [-1,1] → t ∈ [t_0, t_f]。

  5. 张量积高维积分:
     对于 d 维积分，节点和权重为各维度的一维规则的笛卡尔积:
       point = (x_{i1}, ..., x_{id})
       weight = w_{i1} · ... · w_{id}
"""

import numpy as np
from typing import Tuple, Optional


def legendre_compute(norder: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    计算n阶Gauss-Legendre积分的节点 xtab 和权重 weight。
    采用Newton迭代精化，初始猜测基于余弦分布。

    Legendre多项式递推:
      P_0(x) = 1
      P_1(x) = x
      (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    """
    if norder < 1:
        raise ValueError("阶数必须至少为1")
    n = norder
    xtab = np.zeros(n, dtype=float)
    weight = np.zeros(n, dtype=float)
    eps = np.finfo(float).eps

    # 只计算正半轴的根，利用对称性
    for i in range(1, n + 1):
        if i <= n // 2:
            # 初始猜测
            theta = np.pi * (4.0 * i - 1.0) / (4.0 * n + 2.0)
            z = np.cos(theta) * (1.0 - (1.0 - 1.0 / n) / (8.0 * n * n))
        else:
            # 对称映射
            z = -xtab[n - i]

        if i > n // 2:
            xtab[i - 1] = z
            weight[i - 1] = weight[n - i]
            continue

        # Newton迭代
        z1 = 0.0
        while abs(z - z1) > eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z) > 1.0:
                z = np.sign(z) * 0.999999999

        xtab[i - 1] = z
        weight[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp)

    # 对称复制另一半
    for i in range(1, n // 2 + 1):
        xtab[n - i] = -xtab[i - 1]
        weight[n - i] = weight[i - 1]

    # 排序
    idx = np.argsort(xtab)
    return xtab[idx], weight[idx]


def pseudospectral_differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    r"""
    计算Legendre-Gauss-Lobatto（或Gauss）节点的伪谱微分矩阵 D。
    这里使用标准Gauss节点的微分矩阵公式（Barycentric形式）。

    对于节点 {x_j}_{j=0}^{N}，定义重心权重:
      w_j = 1 / Π_{k≠j} (x_j - x_k)
    则微分矩阵:
      D_{ij} = w_j / (w_i (x_i - x_j))   (i≠j)
      D_{ii} = -Σ_{j≠i} D_{ij}
    """
    N = nodes.size - 1
    x = np.asarray(nodes, dtype=float)
    # 计算重心权重（对于Legendre节点）
    # 更稳定的方法：使用多项式导数
    n = x.size
    # 使用标准的重心权重
    # TODO: Hole 1 - 实现伪谱微分矩阵的核心计算
    # 需要计算重心权重 w 和微分矩阵 D
    # D_{ij} = w_j / (w_i * (x_i - x_j))   (i≠j)
    # D_{ii} = -Σ_{j≠i} D_{ij}
    raise NotImplementedError("Hole 1: 请实现伪谱微分矩阵计算")


class PseudospectralCollocation:
    r"""
    Gauss-Legendre伪谱法配点优化器。
    用于将连续时间最优控制问题离散化为非线性规划（NLP）。
    """

    def __init__(self, n_nodes: int = 16):
        self.n_nodes = n_nodes
        self.nodes, self.weights = legendre_compute(n_nodes)
        # 微分矩阵（注意：Gauss节点的微分矩阵需要特殊处理边界条件）
        # 这里使用Legendre-Gauss-Radau节点更合适（包含右端点）
        # 为简化，我们在Gauss节点上构造，并通过多项式插值处理边界
        self.D = pseudospectral_differentiation_matrix(self.nodes)

    def scale_time(self, t0: float, tf: float):
        r"""
        时间域变换:
          t = (tf - t0)/2 * τ + (tf + t0)/2,   τ ∈ [-1, 1]
          dt/dτ = (tf - t0)/2
        因此微分矩阵需乘以 2/(tf - t0)。
        """
        self.t0 = t0
        self.tf = tf
        self.scale = (tf - t0) / 2.0
        self.D_scaled = self.D / self.scale

    def collocation_constraints(self, state_mat: np.ndarray,
                                dynamics_func) -> np.ndarray:
        r"""
        计算配点约束残差:
          res_k = Σ_j D_{kj} x_j - f(x_k, u_k)   （已包含时间缩放）
        state_mat: (n_nodes, n_state) 状态矩阵
        dynamics_func: f(x_k) -> \dot{x}_k  （控制已隐含在状态中）
        """
        state_mat = np.asarray(state_mat, dtype=float)
        n_state = state_mat.shape[1]
        residuals = np.zeros((self.n_nodes, n_state), dtype=float)
        for k in range(self.n_nodes):
            dx_dt_approx = self.D_scaled[k, :] @ state_mat
            dx_dt_exact = dynamics_func(state_mat[k])
            residuals[k] = dx_dt_approx - dx_dt_exact
        return residuals

    def integrate_cost(self, integrand_vals: np.ndarray) -> float:
        r"""
        在配点节点上积分代价函数:
          J = ∫_{t0}^{tf} L(t) dt ≈ scale * Σ_i w_i L(t_i)
        """
        integrand_vals = np.asarray(integrand_vals, dtype=float).reshape(-1)
        if integrand_vals.size != self.n_nodes:
            raise ValueError("integrand_vals长度必须与节点数相同")
        return self.scale * np.sum(self.weights * integrand_vals)

    def interpolate_state(self, state_mat: np.ndarray, t_query: float) -> np.ndarray:
        r"""
        使用Lagrange插值在任意时刻查询状态值:
          x(t) = Σ_j x_j · L_j(τ)
          L_j(τ) = Π_{k≠j} (τ - τ_k) / (τ_j - τ_k)
        """
        # 将 t 映射到 τ
        if self.scale < 1e-14:
            return state_mat[0]
        tau = (t_query - (self.t0 + self.tf) / 2.0) / self.scale
        tau = np.clip(tau, -1.0, 1.0)
        x = np.asarray(state_mat, dtype=float)
        n = x.shape[0]
        # Lagrange基
        L = np.ones(n, dtype=float)
        for j in range(n):
            for k in range(n):
                if k != j:
                    denom = self.nodes[j] - self.nodes[k]
                    if abs(denom) < 1e-14:
                        denom = 1e-14
                    L[j] *= (tau - self.nodes[k]) / denom
        return L @ x


def tensor_product_quadrature_3d(orders: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    构造三维张量积Gauss-Legendre积分规则。
    返回:
      points : (n1*n2*n3, 3)
      weights: (n1*n2*n3,)
    积分公式:
      ∫∫∫ f(x,y,z) dx dy dz ≈ Σ_i w_i f(x_i, y_i, z_i)
    """
    nodes_list = []
    weights_list = []
    for order in orders:
        n, w = legendre_compute(order)
        nodes_list.append(n)
        weights_list.append(w)
    n1, n2, n3 = orders
    pts = np.zeros((n1 * n2 * n3, 3), dtype=float)
    wts = np.zeros(n1 * n2 * n3, dtype=float)
    idx = 0
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                pts[idx] = [nodes_list[0][i], nodes_list[1][j], nodes_list[2][k]]
                wts[idx] = weights_list[0][i] * weights_list[1][j] * weights_list[2][k]
                idx += 1
    return pts, wts
