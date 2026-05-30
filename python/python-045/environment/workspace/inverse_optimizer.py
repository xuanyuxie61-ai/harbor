#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def dijkstra_priority_map(n_nodes, adjacency_list, source_nodes, sensitivity_weights):
    INF = 1e18
    dist = np.full(n_nodes, INF, dtype=np.float64)
    connected = np.zeros(n_nodes, dtype=bool)


    for s in source_nodes:
        dist[s] = 0.0


    for _ in range(n_nodes):

        min_dist = INF
        mv = -1
        for i in range(n_nodes):
            if not connected[i] and dist[i] < min_dist:
                min_dist = dist[i]
                mv = i

        if mv == -1:
            break

        connected[mv] = True


        for neighbor, edge_weight in adjacency_list[mv]:
            if not connected[neighbor]:

                effective_weight = edge_weight / (sensitivity_weights[neighbor] + 1e-10)
                if dist[mv] + effective_weight < dist[neighbor]:
                    dist[neighbor] = dist[mv] + effective_weight

    return dist


def ifs_chaos_perturbation(x, scale=0.1, n_maps=4):
    x = np.asarray(x, dtype=np.float64)
    dim = len(x)



    i = np.random.randint(0, n_maps)


    angle = 2.0 * np.pi * i / n_maps
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    scale_factor = 0.5 + 0.3 * np.random.rand()

    A = scale_factor * np.array([[cos_a, -sin_a],
                                  [sin_a, cos_a]])
    if dim > 2:

        A_full = np.eye(dim) * 0.7
        idx = np.random.choice(dim, min(2, dim), replace=False)
        for k in range(len(idx)):
            for l in range(len(idx)):
                A_full[idx[k], idx[l]] = A[k % 2, l % 2] if k < 2 and l < 2 else A_full[idx[k], idx[l]]
        A = A_full

    b = scale * (np.random.rand(dim) - 0.5)

    x_perturbed = A @ x + b
    return x_perturbed


class OccamInversion:

    def __init__(self, forward_func, n_model, data_errors=None,
                 m_ref=None, lambda_init=1.0, max_iter=30,
                 target_misfit=1.0, lambda_factor=2.0):
        self.forward_func = forward_func
        self.n_model = n_model
        self.data_errors = data_errors
        self.m_ref = m_ref if m_ref is not None else np.zeros(n_model)
        self.lambda_param = lambda_init
        self.max_iter = max_iter
        self.target_misfit = target_misfit
        self.lambda_factor = lambda_factor
        self.history = []

    def _build_roughness_operator(self, n_model):
        if n_model <= 1:
            return np.zeros((1, n_model))
        R = np.zeros((n_model - 1, n_model), dtype=np.float64)
        for i in range(n_model - 1):
            R[i, i] = 1.0
            R[i, i + 1] = -1.0
        return R

    def _compute_jacobian(self, m, dm=0.01):
        d0 = self.forward_func(m)
        n_data = len(d0)
        J = np.zeros((n_data, self.n_model), dtype=np.float64)

        for j in range(self.n_model):
            m_plus = m.copy()
            m_minus = m.copy()

            delta = dm * max(abs(m[j]), 1.0)
            m_plus[j] += delta
            m_minus[j] -= delta

            d_plus = self.forward_func(m_plus)
            d_minus = self.forward_func(m_minus)
            J[:, j] = (d_plus - d_minus) / (2.0 * delta)

        return J

    def _solve_linear_system(self, J, Wd, R, lambda_param, rhs):
        Wd2 = (Wd ** 2)[:, np.newaxis]
        lhs = J.T @ (Wd2 * J) + lambda_param * (R.T @ R)

        lhs += 1e-6 * np.eye(self.n_model)

        try:
            delta_m = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta_m = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return delta_m

    def invert(self, d_obs, m_initial):
        d_obs = np.asarray(d_obs, dtype=np.float64)
        m = np.asarray(m_initial, dtype=np.float64)
        n_data = len(d_obs)

        if self.data_errors is None:
            Wd = np.ones(n_data)
        else:
            Wd = 1.0 / np.asarray(self.data_errors)

        R = self._build_roughness_operator(self.n_model)

        m_best = m.copy()
        best_misfit = np.inf
        lambda_best = self.lambda_param

















        raise NotImplementedError("Hole 3: Occam 反演核心迭代算法待实现")


class MultiObjectiveOptimizer:

    def __init__(self, forward_func, n_model, n_data,
                 dijkstra_sources=None, adjacency=None):
        self.forward_func = forward_func
        self.n_model = n_model
        self.n_data = n_data
        self.dijkstra_sources = dijkstra_sources or [0]
        self.adjacency = adjacency

    def compute_priorities(self, sensitivity):
        if self.adjacency is None:

            adj = [[] for _ in range(self.n_model)]
            for i in range(self.n_model - 1):
                adj[i].append((i + 1, 1.0))
                adj[i + 1].append((i, 1.0))
            self.adjacency = adj

        priorities = dijkstra_priority_map(
            self.n_model, self.adjacency,
            self.dijkstra_sources, sensitivity
        )
        return priorities

    def weighted_update(self, d_obs, m_current, sensitivity, learning_rate=0.1):
        priorities = self.compute_priorities(sensitivity)

        p_max = np.max(priorities)
        if p_max > 0:
            weights = 1.0 - priorities / p_max
        else:
            weights = np.ones(self.n_model)

        d_pred = self.forward_func(m_current)
        residual = d_obs - d_pred



        J = np.zeros((self.n_data, self.n_model))
        dm = 0.01
        for j in range(self.n_model):
            m_plus = m_current.copy()
            m_plus[j] += dm
            d_plus = self.forward_func(m_plus)
            J[:, j] = (d_plus - d_pred) / dm

        gradient = -2.0 * J.T @ residual
        gradient = gradient * weights

        m_new = m_current - learning_rate * gradient
        m_new = np.maximum(m_new, 0.1)
        return m_new


if __name__ == "__main__":

    def linear_forward(m):
        return 2.0 * m + 1.0

    inv = OccamInversion(linear_forward, n_model=3, max_iter=20)
    d_obs = np.array([5.0, 7.0, 9.0])
    m_init = np.ones(3)
    m_best, lam = inv.invert(d_obs, m_init)
    print(f"Occam 反演结果: m = {m_best}, λ = {lam:.4f}")
    print(f"预测: {linear_forward(m_best)}")


    x = np.array([1.0, 2.0, 3.0])
    for _ in range(3):
        x = ifs_chaos_perturbation(x, scale=0.5)
        print(f"IFS 扰动后: {x}")


    adj = [[(1, 1.0), (2, 2.0)], [(0, 1.0), (2, 1.0), (3, 1.0)],
           [(0, 2.0), (1, 1.0), (3, 1.0)], [(1, 1.0), (2, 1.0)]]
    sens = np.array([1.0, 0.5, 0.8, 0.3])
    prio = dijkstra_priority_map(4, adj, [0], sens)
    print(f"Dijkstra 优先级: {prio}")
