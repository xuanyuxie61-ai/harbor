"""
linpack_benchmark.py — LINPACK 性能基准与精度验证
===================================================

融合种子项目:
  - 687_linpack_bench : LINPACK 基准测试

本模块实现:
  1. LU 分解性能测试 (高斯消元 + 部分主元)
  2. 线性系统求解精度验证
  3. 迭代求解器收敛性测试
  4. 条件数估计

用于评估级联方程求解器的计算效率和数值精度。

LINPACK 基准:
  操作计数: (2*n^3)/3 + 2*n^2  (LU 分解)
  MFLOPS = ops / time_seconds / 1e6

残差检查:
  r = b - A*x
  r_normalized = ||r|| / (||A|| * ||x|| * eps)

条件数:
  kappa(A) = ||A|| * ||A^{-1}||
  对于病态系统, kappa >> 1
"""

import math
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    matrix_size: int
    lu_time_seconds: float
    solve_time_seconds: float
    mflops: float
    residual_norm: float
    residual_normalized: float
    condition_estimate: float
    is_accurate: bool


class LinpackBenchmark:
    """LINPACK 基准测试器"""

    def __init__(self, seed: int = 12345):
        self.seed = seed

    def run_benchmark(self, n: int = 100) -> BenchmarkResult:
        """
        运行 n×n LINPACK 基准测试

        步骤:
        1. 生成随机矩阵 A 和右端项 b
        2. LU 分解 (带部分主元)
        3. 前代/回代求解
        4. 计算残差和 MFLOPS
        """
        # 生成测试矩阵
        A, b = self._generate_test_system(n)

        # LU 分解计时
        t_start = time.perf_counter()
        LU, pivots = self._lu_factor(A)
        t_lu = time.perf_counter() - t_start

        # 求解计时
        t_start = time.perf_counter()
        x = self._lu_solve(LU, pivots, b)
        t_solve = time.perf_counter() - t_start

        # 计算残差 r = b - A*x
        r = self._compute_residual(A, b, x)
        r_norm = self._vector_norm(r)
        A_norm = self._matrix_norm_inf(A)
        x_norm = self._vector_norm(x)

        eps = 2.2e-16  # float64 机器精度
        r_normalized = r_norm / max(A_norm * x_norm * eps, 1e-30)

        # MFLOPS 计算
        ops = (2.0 * n ** 3) / 3.0 + 2.0 * n ** 2
        total_time = t_lu + t_solve
        mflops = ops / max(total_time, 1e-10) / 1e6

        # 条件数估计 (1-范数)
        cond_est = self._estimate_condition(LU, n)

        is_accurate = r_normalized < 100.0  # 合理阈值

        return BenchmarkResult(
            matrix_size=n,
            lu_time_seconds=t_lu,
            solve_time_seconds=t_solve,
            mflops=mflops,
            residual_norm=r_norm,
            residual_normalized=r_normalized,
            condition_estimate=cond_est,
            is_accurate=is_accurate,
        )

    def _generate_test_system(self, n: int) -> Tuple[List[List[float]], List[float]]:
        """
        生成随机测试系统 A*x = b

        A: 对角占优随机矩阵 (保证非奇异)
        b: 随机右端项
        精确解: x_true = [1, 1, ..., 1]
        """
        rng_state = self.seed
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n

        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                # 伪随机 (线性同余)
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                val = (rng_state / 0x7FFFFFFF) * 2.0 - 1.0
                A[i][j] = val
                row_sum += abs(val)
            # 对角占优
            A[i][i] += row_sum * 1.5
            b[i] = sum(A[i])  # 对应 x = [1, 1, ..., 1]

        return A, b

    def _lu_factor(
        self, A: List[List[float]],
    ) -> Tuple[List[List[float]], List[int]]:
        """
        LU 分解 (带部分主元)

        A = P * L * U
        其中:
          P: 置换矩阵 (由 pivots 表示)
          L: 下三角 (单位对角)
          U: 上三角

        结果存储在 LU 矩阵中:
          LU[i][j] = U[i][j]  for j >= i
          LU[i][j] = L[i][j]  for j < i
        """
        n = len(A)
        LU = [row[:] for row in A]
        pivots = list(range(n))

        for k in range(n):
            # 部分主元: 找到列 k 中绝对值最大的元素
            max_val = abs(LU[k][k])
            max_row = k
            for i in range(k + 1, n):
                if abs(LU[i][k]) > max_val:
                    max_val = abs(LU[i][k])
                    max_row = i

            if max_val < 1e-30:
                continue  # 奇异

            # 行交换
            if max_row != k:
                LU[k], LU[max_row] = LU[max_row], LU[k]
                pivots[k], pivots[max_row] = pivots[max_row], pivots[k]

            # 消元
            pivot = LU[k][k]
            for i in range(k + 1, n):
                factor = LU[i][k] / pivot
                LU[i][k] = factor  # 存储 L
                for j in range(k + 1, n):
                    LU[i][j] -= factor * LU[k][j]

        return LU, pivots

    def _lu_solve(
        self,
        LU: List[List[float]],
        pivots: List[int],
        b: List[float],
    ) -> List[float]:
        """
        前代/回代求解

        1. Pb = P * b (置换)
        2. L * y = Pb (前代)
        3. U * x = y (回代)
        """
        n = len(b)
        x = list(b)

        # 置换
        for i in range(n):
            if pivots[i] != i:
                x[i], x[pivots[i]] = x[pivots[i]], x[i]

        # 前代 (L * y = Pb)
        for i in range(1, n):
            for j in range(i):
                x[i] -= LU[i][j] * x[j]

        # 回代 (U * x = y)
        for i in range(n - 1, -1, -1):
            if abs(LU[i][i]) < 1e-30:
                continue
            for j in range(i + 1, n):
                x[i] -= LU[i][j] * x[j]
            x[i] /= LU[i][i]

        return x

    def _compute_residual(
        self,
        A: List[List[float]],
        b: List[float],
        x: List[float],
    ) -> List[float]:
        """计算残差 r = b - A*x"""
        n = len(b)
        r = list(b)
        for i in range(n):
            for j in range(n):
                r[i] -= A[i][j] * x[j]
        return r

    def _vector_norm(self, v: List[float]) -> float:
        """2-范数"""
        return math.sqrt(sum(x * x for x in v))

    def _matrix_norm_inf(self, A: List[List[float]]) -> float:
        """无穷范数 (最大行绝对值之和)"""
        return max(sum(abs(a) for a in row) for row in A)

    def _estimate_condition(
        self, LU: List[List[float]], n: int,
    ) -> float:
        """
        条件数估计 (Hager 算法简化版)

        kappa_inf(A) ≈ ||A||_inf * ||A^{-1}||_inf
        通过迭代估计 ||A^{-1}||_inf
        """
        # 用 LU 求解 A * x = e (e 为全1向量)
        e = [1.0] * n
        pivots = list(range(n))
        x = self._lu_solve(LU, pivots, e)
        x_norm = sum(abs(xi) for xi in x)

        # 条件数上界
        A_norm = self._matrix_norm_inf(LU)
        return A_norm * x_norm

    def convergence_study(
        self, sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """
        收敛性研究: 不同矩阵大小的性能

        期望:
          time ~ n^3
          MFLOPS ~ 常数 (对于缓存友好的大小)
          residual ~ eps * kappa(A)
        """
        if sizes is None:
            sizes = [20, 40, 60, 80, 100]

        results = []
        for n in sizes:
            result = self.run_benchmark(n)
            results.append(result)

        return results


class IterativeSolver:
    """迭代求解器 (用于大规模稀疏系统)"""

    @staticmethod
    def conjugate_gradient(
        A: List[List[float]],
        b: List[float],
        x0: Optional[List[float]] = None,
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> Tuple[List[float], int, List[float]]:
        """
        共轭梯度法 (CG)

        适用于对称正定矩阵 A。
        收敛速率: ||e_k||_A <= 2 * ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k * ||e_0||_A

        算法:
          r_0 = b - A*x_0
          p_0 = r_0
          for k = 0, 1, ...:
            alpha_k = (r_k^T r_k) / (p_k^T A p_k)
            x_{k+1} = x_k + alpha_k * p_k
            r_{k+1} = r_k - alpha_k * A * p_k
            if ||r_{k+1}|| < tol: break
            beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            p_{k+1} = r_{k+1} + beta_k * p_k
        """
        n = len(b)
        x = x0 if x0 is not None else [0.0] * n

        # r = b - A*x
        r = list(b)
        for i in range(n):
            for j in range(n):
                r[i] -= A[i][j] * x[j]

        p = list(r)
        rs_old = sum(ri * ri for ri in r)
        residuals = [math.sqrt(rs_old)]

        for k in range(max_iter):
            # Ap = A * p
            Ap = [0.0] * n
            for i in range(n):
                for j in range(n):
                    Ap[i] += A[i][j] * p[j]

            pAp = sum(pi * Api for pi, Api in zip(p, Ap))
            if abs(pAp) < 1e-30:
                break

            alpha = rs_old / pAp

            # x = x + alpha * p
            for i in range(n):
                x[i] += alpha * p[i]

            # r = r - alpha * Ap
            for i in range(n):
                r[i] -= alpha * Ap[i]

            rs_new = sum(ri * ri for ri in r)
            residuals.append(math.sqrt(rs_new))

            if math.sqrt(rs_new) < tol:
                return x, k + 1, residuals

            beta = rs_new / max(rs_old, 1e-30)
            for i in range(n):
                p[i] = r[i] + beta * p[i]

            rs_old = rs_new

        return x, max_iter, residuals

    @staticmethod
    def jacobi_iteration(
        A: List[List[float]],
        b: List[float],
        x0: Optional[List[float]] = None,
        tol: float = 1e-8,
        max_iter: int = 500,
        omega: float = 1.0,
    ) -> Tuple[List[float], int, List[float]]:
        """
        加权 Jacobi 迭代

        x_i^{k+1} = (1-omega)*x_i^k + (omega/A_ii) * (b_i - sum_{j!=i} A_ij * x_j^k)

        收敛条件: rho(I - D^{-1}*A) < 1
        对于 omega = 2/3: 对 M-矩阵总是收敛
        """
        n = len(b)
        x = x0 if x0 is not None else [0.0] * n
        residuals = []

        for k in range(max_iter):
            x_new = list(x)
            for i in range(n):
                if abs(A[i][i]) < 1e-30:
                    continue
                sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x_new[i] = (1.0 - omega) * x[i] + omega * (b[i] - sigma) / A[i][i]

            # 残差
            r_norm = 0.0
            for i in range(n):
                r_i = b[i] - sum(A[i][j] * x_new[j] for j in range(n))
                r_norm += r_i * r_i
            r_norm = math.sqrt(r_norm)
            residuals.append(r_norm)

            x = x_new
            if r_norm < tol:
                return x, k + 1, residuals

        return x, max_iter, residuals
