# -*- coding: utf-8 -*-
"""
optimization.py
===============
全局参数优化与离散失效序列动态规划模块。

源自种子项目：
  - 471_glomin（Brent 全局单变量优化）
  - 156_change_dynamic（动态规划求解最优组合）

科学背景：
---------
1. 材料参数标定（全局优化）：
   复合材料损伤模型的关键参数（如临界能量释放率 G_c、
   粘性参数 τ、S-N 曲线常数 C 和 m）需要通过实验数据标定。
   定义目标泛函（最小二乘残差）：
     J(θ) = Σ_{i=1}^{N_exp} [σ_num(x_i, t_i; θ) - σ_exp,i]² / σ_ref²
   其中 θ = [G_c, τ, C, m] 为待标定参数。
   由于损伤演化高度非线性，目标函数具有多峰特性，
   需要全局优化算法避免陷入局部极小。

   Brent 全局优化思想（单变量）：
     利用二阶导数上界 M 构造下界包络：
       L(x) = f(c) + f'(c)(x-c) - M/2 * (x-c)²
     通过不断细分区间并剔除不可能包含全局最优的子区间，
     保证收敛到全局最小值。

2. 层合板铺层失效序列优化（动态规划）：
   对于 [θ_1/θ_2/.../θ_n]_s 铺层的层合板，在单调载荷下
   各层按不同顺序发生失效（基体开裂、纤维断裂、分层）。
   每种失效序列对应不同的能量耗散路径和最终承载能力。
   动态规划用于寻找"最小能量"或"最大延性"的失效序列。

   状态定义：
     dp[s] = 达到失效状态 s（位掩码表示已失效铺层集合）
             所需的最小外部功。
   状态转移：
     dp[s ∪ {k}] = min(dp[s ∪ {k}],
                         dp[s] + W_fail(k | s))
   其中 W_fail(k | s) 是在已有损伤状态 s 下第 k 层失效所需功：
     W_fail(k | s) = A * h_k * ∫_0^{ε_f(s)} σ_k(ε; s) dε
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


class GlobalParameterCalibration:
    """
    基于 Brent 思想的全局参数标定器（单变量版本，可逐维应用）。
    """

    def __init__(self, f: Callable[[float], float],
                 a: float, b: float,
                 M_bound: Optional[float] = None,
                 tol: float = 1e-6, max_iter: int = 100):
        """
        Parameters
        ----------
        f : callable
            目标函数 f(x)（需标定误差，要求最小化）。
        a, b : float
            搜索区间。
        M_bound : float or None
            |f''(x)| 的上界估计；None 时自动估计。
        tol : float
            收敛容差。
        max_iter : int
            最大迭代次数。
        """
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        if M_bound is None:
            # 自动估计二阶导数上界：在区间上采样有限差分
            self.M = self._estimate_second_derivative_bound()
        else:
            self.M = M_bound

    def _estimate_second_derivative_bound(self, num_samples: int = 20) -> float:
        """通过中心差分估计 |f''(x)| 的上界。"""
        x_samples = np.linspace(self.a, self.b, num_samples)
        h = (self.b - self.a) / (num_samples * 10.0)
        max_d2 = 0.0
        for x in x_samples[1:-1]:
            if x - h < self.a or x + h > self.b:
                continue
            f_pp = (self.f(x + h) - 2.0 * self.f(x) + self.f(x - h)) / (h ** 2)
            max_d2 = max(max_d2, abs(f_pp))
        return max(max_d2, 1.0)

    def _lower_bound(self, c: float, fc: float, dfdc: float,
                     x: float) -> float:
        """
        在点 c 处的二次下界包络（利用二阶导数上界 M）：
          L(x) = f(c) + f'(c)(x-c) - M/2 * (x-c)²
        """
        dx = x - c
        return fc + dfdc * dx - 0.5 * self.M * dx ** 2

    def minimize(self) -> Tuple[float, float]:
        """
        全局最小化 f(x) 在 [a,b] 上。
        采用改进的区间分割 + 二次下界剪枝策略。

        Returns
        -------
        x_opt, f_opt
        """
        # 初始采样
        intervals = [(self.a, self.b)]
        best_x = (self.a + self.b) / 2.0
        best_f = self.f(best_x)

        for _ in range(self.max_iter):
            new_intervals = []
            for a_int, b_int in intervals:
                if b_int - a_int < self.tol:
                    new_intervals.append((a_int, b_int))
                    continue

                c = (a_int + b_int) / 2.0
                fc = self.f(c)
                h = min(self.tol * 10.0, (b_int - a_int) * 0.01)
                h = max(h, 1e-12)
                dfdc = (self.f(c + h) - self.f(c - h)) / (2.0 * h)

                if fc < best_f:
                    best_f = fc
                    best_x = c

                # 计算下界在区间端点的值
                lb_a = self._lower_bound(c, fc, dfdc, a_int)
                lb_b = self._lower_bound(c, fc, dfdc, b_int)
                lb_min = min(lb_a, lb_b)

                # 如果下界高于当前最优，剪枝
                if lb_min > best_f:
                    continue

                # 否则分割区间
                mid = (a_int + b_int) / 2.0
                new_intervals.append((a_int, mid))
                new_intervals.append((mid, b_int))

            intervals = new_intervals
            if not intervals:
                break
            # 移除过小区间
            intervals = [(a_i, b_i) for a_i, b_i in intervals
                         if b_i - a_i > self.tol]

        return best_x, best_f

    @staticmethod
    def multivariable_search(f: Callable, bounds: List[Tuple[float, float]],
                             num_grid: int = 10) -> Tuple[np.ndarray, float]:
        """
        多变量全局优化的粗网格搜索 + 局部精细搜索策略。
        先在各维度上均匀采样，找到最优网格点，再逐维 Brent 优化。

        Parameters
        ----------
        f : callable
            f(x_vec) -> float，x_vec 为 numpy 数组。
        bounds : list of (low, high)
            各变量搜索边界。
        num_grid : int
            每维网格点数。

        Returns
        -------
        x_opt, f_opt
        """
        ndim = len(bounds)
        # 粗网格采样（随机 Latin Hypercube 风格）
        best_x = None
        best_f = np.inf
        samples = np.random.rand(num_grid * ndim, ndim)
        for i in range(ndim):
            samples[:, i] = bounds[i][0] + samples[:, i] * (bounds[i][1] - bounds[i][0])

        for s in samples:
            try:
                val = f(s)
            except Exception:
                val = np.inf
            if val < best_f:
                best_f = val
                best_x = s.copy()

        # 逐维精细搜索（坐标下降）
        for _ in range(3):
            for i in range(ndim):
                def f_1d(xi):
                    x_temp = best_x.copy()
                    x_temp[i] = xi
                    return f(x_temp)
                calib = GlobalParameterCalibration(f_1d, bounds[i][0], bounds[i][1],
                                                    tol=1e-4, max_iter=50)
                xi_opt, _ = calib.minimize()
                best_x[i] = xi_opt
                best_f = f(best_x)

        return best_x, best_f


class FailureSequenceOptimizer:
    """
    层合板铺层失效序列的动态规划优化器（源自 change_dynamic）。
    """

    def __init__(self, num_plies: int,
                 ply_strengths: np.ndarray,
                 ply_thicknesses: np.ndarray,
                 E_ply: np.ndarray,
                 area: float = 1.0):
        """
        Parameters
        ----------
        num_plies : int
            铺层总数。
        ply_strengths : np.ndarray
            各层极限强度 [Pa]。
        ply_thicknesses : np.ndarray
            各层厚度 [m]。
        E_ply : np.ndarray
            各层弹性模量 [Pa]。
        area : float
            横截面积 [m²]。
        """
        self.n = num_plies
        self.sigma_ult = np.asarray(ply_strengths)
        self.h = np.asarray(ply_thicknesses)
        self.E = np.asarray(E_ply)
        self.A = area

        if len(self.sigma_ult) != self.n:
            raise ValueError("Length of ply_strengths must match num_plies.")

    def _failure_work(self, ply_index: int, damaged_set: int) -> float:
        """
        计算在已有损伤状态 damaged_set 下，第 ply_index 层失效所需外部功。

        简化模型：
          失效功 W = A * h_k * ∫_0^{ε_ult} E_eff(s) * ε dε
                   = 0.5 * A * h_k * E_eff * ε_ult²
          ε_ult = σ_ult / E_eff
        其中 E_eff 为剩余未失效层的等效模量（串联模型）：
          1/E_eff = Σ_{j∉damaged_set} h_j / (E_j * h_total_remaining)
        """
        # 找出未失效层
        remaining = [j for j in range(self.n) if not (damaged_set & (1 << j))]
        if ply_index not in remaining:
            return np.inf  # 已失效

        h_total = sum(self.h[j] for j in remaining)
        if h_total < 1e-30:
            return 0.0

        # 等效模量（按厚度加权平均，简化模型）
        E_eff = sum(self.E[j] * self.h[j] for j in remaining) / h_total
        eps_ult = self.sigma_ult[ply_index] / (E_eff + 1e-30)
        W = 0.5 * self.A * self.h[ply_index] * E_eff * (eps_ult ** 2)
        return W

    def optimize_min_work(self) -> Tuple[float, List[int]]:
        """
        动态规划求解最小总失效功的失效序列。

        状态：dp[mask] = 达到 mask 对应失效集合所需最小功。
        转移：对每一个未失效层 k，尝试将其作为下一个失效层。

        Returns
        -------
        min_total_work : float
            最小总失效功 [J]。
        sequence : list of int
            最优失效顺序（铺层索引列表）。
        """
        total_states = 1 << self.n
        INF = 1e30
        dp = np.full(total_states, INF)
        parent = np.full(total_states, -1, dtype=int)
        dp[0] = 0.0

        for mask in range(total_states):
            if dp[mask] >= INF:
                continue
            for k in range(self.n):
                if not (mask & (1 << k)):
                    new_mask = mask | (1 << k)
                    work = self._failure_work(k, mask)
                    if dp[mask] + work < dp[new_mask]:
                        dp[new_mask] = dp[mask] + work
                        parent[new_mask] = k

        # 回溯最优序列
        sequence = []
        mask = total_states - 1
        while mask > 0:
            k = parent[mask]
            if k < 0:
                break
            sequence.append(k)
            mask ^= (1 << k)
        sequence.reverse()

        return dp[total_states - 1], sequence

    def optimize_max_ductility(self) -> Tuple[float, List[int]]:
        """
        动态规划求解最大延性（即最大化失效步数之间的功增量均匀性）。
        目标改为最小化相邻失效步功增量的方差。
        """
        total_states = 1 << self.n
        INF = 1e30
        # dp[mask] = (min_variance, total_work, sequence_length)
        dp_var = np.full(total_states, INF)
        dp_work = np.zeros(total_states)
        parent = np.full(total_states, -1, dtype=int)
        dp_var[0] = 0.0

        for mask in range(total_states):
            if dp_var[mask] >= INF:
                continue
            for k in range(self.n):
                if not (mask & (1 << k)):
                    new_mask = mask | (1 << k)
                    work = self._failure_work(k, mask)
                    new_total = dp_work[mask] + work
                    # 用增量方差作为目标（简化：最大化最小增量）
                    min_increment = work
                    if dp_work[mask] > 0:
                        num_steps = bin(mask).count('1') + 1
                        avg = new_total / num_steps
                        # 目标：使各步功接近平均
                        penalty = abs(work - avg)
                    else:
                        penalty = 0.0

                    score = dp_var[mask] + penalty
                    if score < dp_var[new_mask]:
                        dp_var[new_mask] = score
                        dp_work[new_mask] = new_total
                        parent[new_mask] = k

        sequence = []
        mask = total_states - 1
        while mask > 0:
            k = parent[mask]
            if k < 0:
                break
            sequence.append(k)
            mask ^= (1 << k)
        sequence.reverse()

        return dp_work[total_states - 1], sequence


if __name__ == "__main__":
    # 自测试 1：全局优化
    def test_func(x):
        return (x - 0.3) ** 2 + 0.1 * np.sin(20 * np.pi * x)

    calib = GlobalParameterCalibration(test_func, 0.0, 1.0, tol=1e-5)
    x_opt, f_opt = calib.minimize()
    print("Global optimization result:", x_opt, f_opt)

    # 自测试 2：多变量优化
    def f2d(x):
        return (x[0] - 0.5) ** 2 + (x[1] + 0.2) ** 2 + 0.05 * np.sin(10 * x[0]) * np.cos(10 * x[1])

    x_opt2, f_opt2 = GlobalParameterCalibration.multivariable_search(
        f2d, [(-1.0, 1.0), (-1.0, 1.0)], num_grid=20)
    print("Multivariable optimization result:", x_opt2, f_opt2)

    # 自测试 3：失效序列优化
    n_plies = 4
    strengths = np.array([1500e6, 1200e6, 1200e6, 1500e6])
    thicknesses = np.array([0.125e-3, 0.125e-3, 0.125e-3, 0.125e-3])
    E_plies = np.array([181e9, 10.3e9, 10.3e9, 181e9])
    optimizer = FailureSequenceOptimizer(n_plies, strengths, thicknesses, E_plies, area=1e-4)
    min_work, seq = optimizer.optimize_min_work()
    print("Minimum work failure sequence:", seq, "work=", min_work)
