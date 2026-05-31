"""
离散催化剂负载整数规划 (基于多维丢番图方程)
===========================================
在微反应器设计中，催化剂常以离散颗粒或涂层块形式加载。
需要在整数约束下优化催化剂在各区域的分配，使得总成本最低
且满足转化率要求。

数学模型：
    设有 n 个微反应器区域，区域 i 的单位催化剂负载量为 a_i [kg/m²]。
    总催化剂预算为 B [kg]。
    各区域需要装载整数倍 x_i 的单位量，满足：

        Σ_{i=1}^n a_i x_i = B

        x_i ∈ ℤ_{
        ≥ 0}

    目标：最小化总压降或最大化均匀性。

    本模块实现非负整数解的枚举与筛选，适用于小规模离散优化。
    对大规模问题，提供贪心启发式近似解。

算法复杂度：
    精确枚举的最坏情况为 O( (B/a_min)^{n-1} )，通过分支限界可大幅剪枝。
"""

import numpy as np
from typing import Tuple, List, Optional


class DiscreteCatalystLoadingOptimizer:
    """
    离散催化剂负载整数规划求解器。
    """

    def __init__(self, a_coeffs: np.ndarray, budget: float):
        """
        输入:
            a_coeffs: 各区域单位负载量数组 (正整数或正实数)
            budget:   总预算 B
        """
        self.a = np.asarray(a_coeffs, dtype=float).flatten()
        self.n = len(self.a)
        if np.any(self.a <= 0.0):
            raise ValueError("所有系数必须为正")
        if budget < 0.0:
            raise ValueError("预算必须非负")
        self.B = budget

    def solve_exact_nonnegative(
        self, max_solutions: int = 1000
    ) -> Tuple[np.ndarray, int]:
        """
        枚举所有非负整数解 x 满足 Σ a_i x_i = B。

        采用递归回溯枚举（对 B 进行缩放取整后处理）。
        返回:
            solutions: (k, n) 数组，每行为一个解
            num_solutions: 解的数量
        """
        # 将问题缩放为整数：乘以一个公共因子
        scale = 1.0
        for i in range(self.n):
            # 寻找使得 a_i * scale 接近整数的最小 scale
            pass
        # 简单处理：直接对缩放后的整数进行枚举
        a_int = np.round(self.a * 1000.0).astype(int)
        B_int = int(round(self.B * 1000.0))
        # 保证 gcd
        g = int(np.gcd.reduce(a_int))
        if g > 1:
            a_int = a_int // g
            B_int = B_int // g

        solutions = []
        y = np.zeros(self.n, dtype=int)
        j = 0
        r = B_int

        while True:
            # 计算剩余量
            r = B_int
            for i in range(j):
                r -= a_int[i] * y[i]

            if j < self.n:
                j += 1
                y[j - 1] = r // a_int[j - 1]
            else:
                if r == 0:
                    solutions.append(y.copy())
                    if len(solutions) >= max_solutions:
                        break
                # 回溯
                while j > 0:
                    if y[j - 1] > 0:
                        y[j - 1] -= 1
                        break
                    j -= 1
                if j == 0:
                    break

        if len(solutions) == 0:
            return np.empty((0, self.n)), 0
        sol_array = np.array(solutions)
        return sol_array, len(sol_array)

    def solve_bounded(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        max_solutions: int = 500,
    ) -> Tuple[np.ndarray, int]:
        """
        求解带上下界的丢番图方程：
            l_i ≤ x_i ≤ u_i
        """
        lower = np.asarray(lower_bounds, dtype=int).flatten()
        upper = np.asarray(upper_bounds, dtype=int).flatten()
        if len(lower) != self.n or len(upper) != self.n:
            raise ValueError("边界维度与系数维度不匹配")

        # 变量替换：x_i' = x_i - l_i，转化为非负问题
        a_int = np.round(self.a * 1000.0).astype(int)
        B_int = int(round(self.B * 1000.0))
        B_prime = B_int - np.dot(a_int, lower)

        if B_prime < 0:
            return np.empty((0, self.n)), 0

        # 重新定义上界
        u_prime = upper - lower
        solutions = []
        y = np.zeros(self.n, dtype=int)
        j = 0

        while True:
            r = B_prime
            for i in range(j):
                r -= a_int[i] * y[i]

            if j < self.n:
                j += 1
                max_val = min(r // a_int[j - 1], u_prime[j - 1])
                y[j - 1] = max_val
            else:
                if r == 0:
                    sol = y + lower
                    solutions.append(sol.copy())
                    if len(solutions) >= max_solutions:
                        break
                while j > 0:
                    if y[j - 1] > 0:
                        y[j - 1] -= 1
                        if y[j - 1] > u_prime[j - 1]:
                            y[j - 1] = u_prime[j - 1]
                        break
                    j -= 1
                if j == 0:
                    break

        if len(solutions) == 0:
            return np.empty((0, self.n)), 0
        return np.array(solutions), len(solutions)

    def greedy_heuristic_solution(
        self, objective_weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        贪心启发式求解：
            按单位催化剂的效益比 w_i/a_i 降序填充，直到预算耗尽。

        返回:
            x_greedy: 贪心解
            objective: 目标函数值 Σ w_i x_i
        """
        weights = np.asarray(objective_weights, dtype=float)
        if len(weights) != self.n:
            raise ValueError("权重维度不匹配")

        ratios = weights / self.a
        order = np.argsort(-ratios)
        x_greedy = np.zeros(self.n, dtype=int)
        remaining = self.B
        a_int = np.round(self.a * 1000.0).astype(int)
        B_int = int(round(remaining * 1000.0))
        w_int = np.round(weights * 1000.0).astype(int)

        for idx in order:
            if a_int[idx] <= 0:
                continue
            max_units = B_int // a_int[idx]
            x_greedy[idx] = max_units
            B_int -= max_units * a_int[idx]

        objective = float(np.dot(weights, x_greedy))
        return x_greedy, objective

    def select_optimal_loading(
        self,
        solutions: np.ndarray,
        objective_func: Optional[callable] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        从可行解集中选择最优解（按目标函数最小化）。
        默认目标为最大化负载均匀度（最小化方差）。
        """
        if solutions.shape[0] == 0:
            return np.zeros(self.n), float("inf")

        if objective_func is None:
            # 默认：最小化负载方差
            def objective_func(x):
                loads = self.a * x
                mu = np.mean(loads)
                if mu < 1.0e-12:
                    return 0.0
                return np.std(loads) / mu

        best_obj = float("inf")
        best_sol = solutions[0]
        for sol in solutions:
            obj = objective_func(sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = sol
        return best_sol, best_obj

    def compute_loading_efficiency(
        self, x: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算负载效率指标：
            utilization = Σ a_i x_i / B
            uniformity = 1 - std(a_i x_i) / mean(a_i x_i)
        """
        loads = self.a * x
        total = np.sum(loads)
        utilization = total / max(self.B, 1.0e-12)
        mu = np.mean(loads)
        if mu < 1.0e-12:
            uniformity = 0.0
        else:
            uniformity = max(0.0, 1.0 - np.std(loads) / mu)
        return utilization, uniformity
