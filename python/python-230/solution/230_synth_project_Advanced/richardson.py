"""
richardson.py
=============

Richardson 外推 —— 有限差分精度提升核心。

种子项目 1435 (zoomin, 高阶求根) 与 166 (chebyshev2_exactness) 启发:
Richardson 外推利用不同步长的低阶近似构造高阶近似, 思想类似
Chebyshev 正交展开的高阶精确性检验。

Richardson 外推基本原理:

设 A(h) 为 f 的近似, 误差展开为:
    A(h) = A* + a_1 h^{p_1} + a_2 h^{p_2} + ...  (p_1 < p_2 < ...)

一次 Richardson 外推消去首项:
    A_1(h) = [r^{p_1} A(h/r) - A(h)] / (r^{p_1} - 1)

Romberg 表 (迭代外推):
    A_{k,j} = [r^{p_j} A_{k,j-1}(h/r) - A_{k,j-1}(h)] / (r^{p_j} - 1)

对中心差分 (偶数阶展开):
    p_k = 2k, r = 2
    A(h) = f'(x) + c_2 h² + c_4 h⁴ + c_6 h⁶ + ...

收敛判定:
    |A_{k,k} - A_{k,k-1}| / |A_{k,k}| < tol
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, List, Optional


class RichardsonExtrapolator:
    """
    Richardson 外推表构造器。

    给定函数 A(h) 返回 f 的近似, 利用步长逐次减半构造 Romberg 表,
    获得 f 的最高精度估计。

    公式 (Romberg 递推, r=2, p_k = m + 2k 对中心差分):
        T[k][0] = A(h/2^k) · (h/2^k)^{-m} · h^m  (保持量纲)
        T[k][j] = [2^{p_j} T[k][j-1] - T[k-1][j-1]] / (2^{p_j} - 1)

    其中 p_j = m + 2j (中心差分的偶数阶误差).

    Attributes
    ----------
    table : list of list of float
        Romberg 表
    converged : bool
        是否达到目标精度
    best_estimate : float
        最佳估计值 (表中最精确的元素)
    estimated_error : float
        估计误差
    optimal_order : int
        最优外推阶数
    """

    def __init__(
        self,
        deriv_order: int = 1,
        max_level: int = 10,
        tol: float = 1e-12,
        r: float = 2.0,
    ):
        """
        Parameters
        ----------
        deriv_order : int
            所求导数阶数 m
        max_level : int
            最大外推层数 (步长减半次数)
        tol : float
            收敛容差
        r : float
            步长缩减因子 (默认 2)
        """
        self.m = deriv_order
        self.max_level = max_level
        self.tol = tol
        self.r = r
        self.table: List[List[float]] = []
        self.converged = False
        self.best_estimate = 0.0
        self.estimated_error = float("inf")
        self.optimal_order = 0

    def extrapolate(
        self,
        fd_evaluator: Callable[[float], float],
        h0: float,
    ) -> float:
        """
        执行 Richardson 外推。

        Parameters
        ----------
        fd_evaluator : callable
            fd_evaluator(h) 返回步长 h 下的有限差分近似值
            (已除以 h^m 归一化)
        h0 : float
            初始步长

        Returns
        -------
        float
            最佳估计值
        """
        self.table = []
        self.converged = False

        # 第一列: 直接有限差分近似
        for k in range(self.max_level + 1):
            hk = h0 / (self.r ** k)
            Ah = fd_evaluator(hk)
            self.table.append([Ah])

        # Romberg 递推
        # 对 m 阶中心差分: 误差阶 p_j = m + 2j
        # 但这里我们使用通用公式: p_j = 2j (偶数阶)
        for j in range(1, self.max_level + 1):
            r_pj = self.r ** (2 * j)  # r^{p_j}
            for k in range(j, self.max_level + 1):
                val = (
                    r_pj * self.table[k][j - 1] - self.table[k - 1][j - 1]
                ) / (r_pj - 1.0)
                self.table[k].append(val)

        # 选择最优: 沿对角线找最小变化
        best_err = float("inf")
        best_k = 0
        for k in range(1, self.max_level + 1):
            if k < len(self.table[k]):
                diag_val = self.table[k][-1] if len(self.table[k]) > k else self.table[k][k]
            else:
                continue
            # 估计误差: 与前一阶的差
            if k >= 2 and len(self.table[k - 1]) >= k:
                err = abs(self.table[k][min(k, len(self.table[k]) - 1)]
                          - self.table[k - 1][min(k - 1, len(self.table[k - 1]) - 1)])
            else:
                err = abs(self.table[k][-1] - self.table[k][-2]) if len(self.table[k]) >= 2 else float("inf")
            if err < best_err:
                best_err = err
                best_k = k
            if err < self.tol:
                self.converged = True
                break

        self.best_estimate = self.table[best_k][-1]
        self.estimated_error = best_err
        self.optimal_order = best_k
        return self.best_estimate

    def get_table_array(self) -> np.ndarray:
        """返回 Romberg 表 (numpy 数组, 未填满位置为 NaN)。"""
        n = len(self.table)
        arr = np.full((n, n), np.nan)
        for i, row in enumerate(self.table):
            for j, val in enumerate(row):
                if j < n:
                    arr[i, j] = val
        return arr

    def convergence_rates(self) -> List[float]:
        """
        估计逐阶收敛速率:
            rate_k = log2(|e_{k-1}| / |e_k|)
        """
        rates = []
        for k in range(1, len(self.table)):
            if len(self.table[k]) >= 2 and len(self.table[k - 1]) >= 2:
                e_prev = abs(self.table[k - 1][-1] - self.table[k - 1][-2])
                e_curr = abs(self.table[k][-1] - self.table[k][-2])
                if e_curr > 1e-300 and e_prev > 1e-300:
                    rates.append(math.log2(e_prev / e_curr))
        return rates


# ---------------------------------------------------------------------------
# 自适应步长选择 (基于 Richardson 外推)
# ---------------------------------------------------------------------------
def optimal_step_size(
    f: Callable[[float], float],
    x: float,
    m: int = 1,
    eps: float = 1e-16,
    f_scale: float = 1.0,
) -> float:
    """
    最优步长估计 (数值分析标准结果):

        h_opt ≈ (ε · f_scale / |f^{(m)}(x)|)^{1/(p+m)}

    对 p 阶中心差分求 m 阶导数:
        h_opt ≈ ε^{1/(p+m)} · (f_scale)^{1/(p+m)}

    通常 p=4 (四阶差分), m=1 (一阶导):
        h_opt ≈ ε^{1/5} ≈ 1e-3 (双精度)

    Parameters
    ----------
    f : callable
    x : float
    m : int
        导数阶
    eps : float
        机器精度 (~1e-16 对 float64)
    f_scale : float
        函数尺度估计

    Returns
    -------
    float
        推荐步长
    """
    # 对 p=4 中心差分, m 阶导: h ~ eps^(1/(4+m))
    p = 4
    exponent = 1.0 / (p + m)
    h = (eps * max(abs(f_scale), 1e-10)) ** exponent
    # 安全边界
    return max(h, 1e-14)


# ---------------------------------------------------------------------------
# Richardson 增强的有限差分
# ---------------------------------------------------------------------------
def richardson_fd(
    f: Callable[[float], float],
    x: float,
    m: int = 1,
    h0: float = 0.1,
    max_level: int = 8,
    tol: float = 1e-12,
) -> Tuple[float, float, bool]:
    """
    Richardson 外推增强的高精度有限差分。

    返回 (best_estimate, error_estimate, converged)。
    """
    from finite_diff import FiniteDiff

    fd = FiniteDiff(m=m, fd_order=4)

    def evaluator(h):
        return fd(f, x, h) * (h ** m)  # 未归一化的 A(h)

    # 实际应返回 fd(f,x,h) 本身 (已含 h^{-m})
    def evaluator_proper(h):
        return fd(f, x, h)

    ext = RichardsonExtrapolator(deriv_order=m, max_level=max_level, tol=tol)
    best = ext.extrapolate(evaluator_proper, h0)
    return best, ext.estimated_error, ext.converged


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    f = lambda x: math.sin(x)
    x0 = 0.5

    # Richardson 外推求一阶导
    best, err, conv = richardson_fd(f, x0, m=1, h0=0.1, max_level=8)
    exact = math.cos(x0)
    print(f"Richardson f'(0.5) = {best:.15f}")
    print(f"精确值 cos(0.5)  = {exact:.15f}")
    print(f"估计误差 = {err:.2e}, 收敛 = {conv}")
    print(f"实际误差 = {abs(best - exact):.2e}")

    # 收敛速率
    ext = RichardsonExtrapolator(deriv_order=1, max_level=8)
    fd_eval = lambda h: (f(x0 + h) - f(x0 - h)) / (2 * h)
    ext.extrapolate(fd_eval, 0.1)
    rates = ext.convergence_rates()
    print(f"收敛速率: {[f'{r:.2f}' for r in rates]}")
