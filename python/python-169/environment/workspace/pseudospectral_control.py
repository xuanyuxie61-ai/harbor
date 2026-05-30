
import numpy as np
from typing import Tuple, Optional


def legendre_compute(norder: int) -> Tuple[np.ndarray, np.ndarray]:
    if norder < 1:
        raise ValueError("阶数必须至少为1")
    n = norder
    xtab = np.zeros(n, dtype=float)
    weight = np.zeros(n, dtype=float)
    eps = np.finfo(float).eps


    for i in range(1, n + 1):
        if i <= n // 2:

            theta = np.pi * (4.0 * i - 1.0) / (4.0 * n + 2.0)
            z = np.cos(theta) * (1.0 - (1.0 - 1.0 / n) / (8.0 * n * n))
        else:

            z = -xtab[n - i]

        if i > n // 2:
            xtab[i - 1] = z
            weight[i - 1] = weight[n - i]
            continue


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


    for i in range(1, n // 2 + 1):
        xtab[n - i] = -xtab[i - 1]
        weight[n - i] = weight[i - 1]


    idx = np.argsort(xtab)
    return xtab[idx], weight[idx]


def pseudospectral_differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    N = nodes.size - 1
    x = np.asarray(nodes, dtype=float)


    n = x.size





    raise NotImplementedError("Hole 1: 请实现伪谱微分矩阵计算")


class PseudospectralCollocation:

    def __init__(self, n_nodes: int = 16):
        self.n_nodes = n_nodes
        self.nodes, self.weights = legendre_compute(n_nodes)



        self.D = pseudospectral_differentiation_matrix(self.nodes)

    def scale_time(self, t0: float, tf: float):
        self.t0 = t0
        self.tf = tf
        self.scale = (tf - t0) / 2.0
        self.D_scaled = self.D / self.scale

    def collocation_constraints(self, state_mat: np.ndarray,
                                dynamics_func) -> np.ndarray:
        state_mat = np.asarray(state_mat, dtype=float)
        n_state = state_mat.shape[1]
        residuals = np.zeros((self.n_nodes, n_state), dtype=float)
        for k in range(self.n_nodes):
            dx_dt_approx = self.D_scaled[k, :] @ state_mat
            dx_dt_exact = dynamics_func(state_mat[k])
            residuals[k] = dx_dt_approx - dx_dt_exact
        return residuals

    def integrate_cost(self, integrand_vals: np.ndarray) -> float:
        integrand_vals = np.asarray(integrand_vals, dtype=float).reshape(-1)
        if integrand_vals.size != self.n_nodes:
            raise ValueError("integrand_vals长度必须与节点数相同")
        return self.scale * np.sum(self.weights * integrand_vals)

    def interpolate_state(self, state_mat: np.ndarray, t_query: float) -> np.ndarray:

        if self.scale < 1e-14:
            return state_mat[0]
        tau = (t_query - (self.t0 + self.tf) / 2.0) / self.scale
        tau = np.clip(tau, -1.0, 1.0)
        x = np.asarray(state_mat, dtype=float)
        n = x.shape[0]

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
