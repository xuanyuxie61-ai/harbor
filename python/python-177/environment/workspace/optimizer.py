# -*- coding: utf-8 -*-

import numpy as np
import itertools


class MatrixChainOptimizer:

    @staticmethod
    def matrix_chain_dp(dims):
        n = len(dims) - 1
        m = np.zeros((n, n), dtype=np.int64)
        s = np.zeros((n, n), dtype=np.int32)

        for chain_len in range(2, n + 1):
            for i in range(n - chain_len + 1):
                j = i + chain_len - 1
                m[i, j] = np.iinfo(np.int64).max
                for k in range(i, j):
                    cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                    if cost < m[i, j]:
                        m[i, j] = cost
                        s[i, j] = k

        return m[0, n - 1], s

    @staticmethod
    def matrix_chain_brute(dims):
        n = len(dims) - 1
        if n <= 1:
            return 0, []

        def brute_cost(i, j):
            if i == j:
                return 0, []
            min_cost = np.iinfo(np.int64).max
            best_k = i
            for k in range(i, j):
                left_cost, left_split = brute_cost(i, k)
                right_cost, right_split = brute_cost(k + 1, j)
                cost = left_cost + right_cost + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < min_cost:
                    min_cost = cost
                    best_k = k
            return min_cost, [best_k]

        cost, _ = brute_cost(0, n - 1)
        return cost

    @staticmethod
    def get_optimal_order(s, i, j):
        if i == j:
            return f"M{i+1}"
        k = s[i, j]
        left = MatrixChainOptimizer.get_optimal_order(s, i, k)
        right = MatrixChainOptimizer.get_optimal_order(s, k + 1, j)
        return f"({left} × {right})"


class ConstraintSatisfier:

    @staticmethod
    def brute_force_satisfy(n_vars, formula_func, max_solutions=1000):
        solutions = []
        total = 2 ** n_vars
        count = 0
        for idx in range(total):
            bvec = [(idx >> j) & 1 for j in range(n_vars)]
            if formula_func(bvec):
                count += 1
                if len(solutions) < max_solutions:
                    solutions.append(bvec)
        return solutions, count

    @staticmethod
    def young_equation_solver(sigma12, sigma13, sigma23, tol=1e-4, n_grid=200):
        solutions = []
        for i in range(n_grid):
            theta1 = np.pi * i / n_grid
            for j in range(n_grid):
                theta2 = np.pi * j / n_grid
                theta3 = 2.0 * np.pi - theta1 - theta2
                if theta3 < 0 or theta3 > 2.0 * np.pi:
                    continue
                lhs = sigma12 * np.cos(theta3)
                rhs = sigma13 * np.cos(theta2) + sigma23 * np.cos(theta1)
                if abs(lhs - rhs) < tol:
                    solutions.append((theta1, theta2, theta3))
        return solutions


class OperatorSequenceOptimizer:

    @staticmethod
    def estimate_flops(dims, order):
        total = 0

        for i in range(len(order) - 1):
            total += dims[order[i]] * dims[order[i + 1]]
        return total

    @staticmethod
    def optimize_preconditioner_chain(n_ops, dim_in, dim_mid, dim_out):
        dims = [dim_in, dim_mid, dim_mid, dim_out, dim_out]
        cost, s = MatrixChainOptimizer.matrix_chain_dp(dims)
        return cost, s
