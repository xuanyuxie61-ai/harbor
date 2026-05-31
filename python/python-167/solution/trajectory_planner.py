"""
trajectory_planner.py
足端轨迹规划与优化模块。
融入种子项目：
  - 1363_tsp_brute（旅行商问题暴力求解 → 映射为多足机器人足端落点排序优化）
  - 1004_r8vm（Vandermonde 多项式插值 → 足端摆动轨迹多项式拟合）

科学背景：
多足机器人在摆动相需要规划足端从抬起点到落点的光滑轨迹。
对多个候选落点，需优化访问顺序（类似 TSP），同时用高阶多项式
保证轨迹的连续性与加速度有界性。
"""

import numpy as np
from typing import List, Tuple
from numerical_solver import VandermondeSolver


class TSPBruteForce:
    """
    源自 tsp_brute.m 的旅行商问题暴力求解器。

    在多足机器人步态中，TSP 映射为：给定一组候选足端落点
    P = {p_1, p_2, ..., p_n} ⊂ R^3，寻找访问所有点的最短闭合路径，
    使得机器人重心偏移最小且能量消耗最低。

    数学模型：
        min_{π ∈ S_n}  Σ_{i=1}^{n} || p_{π(i)} - p_{π(i+1)} ||
    其中 S_n 为 n 个点的排列群，π(n+1) ≡ π(1)。
    """

    def __init__(self):
        pass

    def path_cost(self, distance: np.ndarray, perm: Tuple[int, ...]) -> float:
        """
        计算排列 perm 对应的路径总长度。
        """
        n = len(perm)
        cost = 0.0
        for i in range(n):
            j = (i + 1) % n
            cost += distance[perm[i], perm[j]]
        return cost

    def _next_permutation(self, perm: List[int]) -> bool:
        """
        字典序生成下一个排列（Trotter 算法思想）。
        返回 False 表示已穷尽所有排列。
        """
        n = len(perm)
        i = n - 2
        while i >= 0 and perm[i] >= perm[i + 1]:
            i -= 1
        if i < 0:
            return False
        j = n - 1
        while perm[j] <= perm[i]:
            j -= 1
        perm[i], perm[j] = perm[j], perm[i]
        perm[i + 1:] = reversed(perm[i + 1:])
        return True

    def solve(self, points: np.ndarray) -> Tuple[Tuple[int, ...], float, float, float]:
        """
        对给定点集求解最短闭合路径。
        返回：(最优排列, 最短长度, 平均长度, 最长长度)。
        """
        n = points.shape[0]
        if n < 2:
            return tuple(range(n)), 0.0, 0.0, 0.0
        # 构建欧氏距离矩阵
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distance = np.linalg.norm(diff, axis=2)

        perm = list(range(n))
        min_cost = float('inf')
        max_cost = -float('inf')
        sum_cost = 0.0
        count = 0
        best_perm = tuple(perm)

        while True:
            cost = self.path_cost(distance, tuple(perm))
            sum_cost += cost
            count += 1
            if cost < min_cost:
                min_cost = cost
                best_perm = tuple(perm)
            if cost > max_cost:
                max_cost = cost
            if not self._next_permutation(perm):
                break

        avg_cost = sum_cost / count if count > 0 else 0.0
        return best_perm, min_cost, avg_cost, max_cost


class PolynomialSwingTrajectory:
    """
    源自 r8vm 的 Vandermonde 多项式插值，用于足端摆动相轨迹生成。

    科学要求：足端轨迹需满足边界条件
        p(0) = p_start,    p(T) = p_end
        ṗ(0) = v_start,    ṗ(T) = v_end
        p̈(0) = a_start,    p̈(T) = a_end
    共 6 个约束，对应 5 次多项式（6 个系数）：
        p(t) = c_0 + c_1·t + c_2·t^2 + c_3·t^3 + c_4·t^4 + c_5·t^5

    将约束写成关于 c = [c_0, ..., c_5]^T 的线性系统 A·c = b，
    其中 A 为约束 Vandermonde 型矩阵。
    """

    def __init__(self):
        self.vandermonde = VandermondeSolver()

    def fit_quintic(self, T: float,
                    p_start: float, p_end: float,
                    v_start: float, v_end: float,
                    a_start: float, a_end: float) -> np.ndarray:
        """
        拟合满足 6 个边界条件的 5 次多项式系数 c[0..5]。
        返回系数向量。
        """
        if T <= 1e-9:
            T = 1e-6
        # 构造约束矩阵（非标准 Vandermonde，但思想一致）
        A = np.array([
            [1.0, 0.0, 0.0,       0.0,         0.0,         0.0],
            [1.0, T,   T**2,     T**3,        T**4,        T**5],
            [0.0, 1.0, 0.0,       0.0,         0.0,         0.0],
            [0.0, 1.0, 2.0*T,   3.0*T**2,    4.0*T**3,    5.0*T**4],
            [0.0, 0.0, 2.0,       0.0,         0.0,         0.0],
            [0.0, 0.0, 2.0,     6.0*T,      12.0*T**2,   20.0*T**3],
        ], dtype=float)
        b = np.array([p_start, p_end, v_start, v_end, a_start, a_end], dtype=float)
        # 使用 NumPy 的线性求解（比 Vandermonde 专用算法更通用）
        c = np.linalg.solve(A, b)
        return c

    def evaluate(self, coeffs: np.ndarray, t: float) -> Tuple[float, float, float]:
        """
        用 Horner 法则求 p(t), ṗ(t), p̈(t)。
        """
        c = coeffs
        # p(t)
        p = c[5]
        for i in range(4, -1, -1):
            p = p * t + c[i]
        # ṗ(t): 导数系数
        dc = np.array([c[1], 2.0*c[2], 3.0*c[3], 4.0*c[4], 5.0*c[5]], dtype=float)
        v = dc[4]
        for i in range(3, -1, -1):
            v = v * t + dc[i]
        # p̈(t): 二阶导数系数
        d2c = np.array([2.0*c[2], 6.0*c[3], 12.0*c[4], 20.0*c[5]], dtype=float)
        a = d2c[3]
        for i in range(2, -1, -1):
            a = a * t + d2c[i]
        return p, v, a


class FootfallPlanner:
    """
    综合落点规划器：结合 TSP 排序与多项式轨迹，为六足机器人规划足端运动。
    """

    def __init__(self, swing_height: float = 0.05, swing_period: float = 0.3):
        self.swing_h = swing_height
        self.swing_T = swing_period
        self.tsp = TSPBruteForce()
        self.traj = PolynomialSwingTrajectory()

    def plan_footholds(self, candidate_points: np.ndarray) -> Tuple[Tuple[int, ...], np.ndarray]:
        """
        对候选落点进行 TSP 优化排序。
        返回最优排列与对应的重排序后的点集。
        """
        best_perm, min_cost, avg_cost, max_cost = self.tsp.solve(candidate_points)
        sorted_points = candidate_points[list(best_perm), :]
        return best_perm, sorted_points

    def generate_swing_trajectory(self, p_start: np.ndarray, p_end: np.ndarray,
                                  n_samples: int = 30) -> np.ndarray:
        """
        生成单条腿的 3D 摆动轨迹（xy 平面直线 + z 方向抛物线提升）。
        返回 (n_samples, 3) 的轨迹点。
        """
        traj_points = np.zeros((n_samples, 3))
        for dim in range(3):
            if dim == 2:
                # z 方向：从 p_start[2] 提升到最高点再落到 p_end[2]
                c = self.traj.fit_quintic(
                    self.swing_T,
                    p_start[2], p_end[2],
                    0.0, 0.0,
                    0.0, 0.0
                )
                # 修改中间点使轨迹呈拱形：在 t = T/2 处强制 z = max(p_start[2], p_end[2]) + swing_h
                # 这里简化处理：直接在 z 系数中加入拱形偏移
                # 更精确做法是用 6 约束 + 1 中间点约束 = 7 次多项式，但 5 次已足够演示
                # 使用简单抛物线修正
                t_vals = np.linspace(0, self.swing_T, n_samples)
                for i, t in enumerate(t_vals):
                    z, _, _ = self.traj.evaluate(c, t)
                    # 叠加拱形
                    arch = 4.0 * self.swing_h * (t / self.swing_T) * (1.0 - t / self.swing_T)
                    traj_points[i, dim] = z + arch
            else:
                # x, y 方向：线性插值
                t_vals = np.linspace(0, 1, n_samples)
                traj_points[:, dim] = p_start[dim] + t_vals * (p_end[dim] - p_start[dim])
        return traj_points
