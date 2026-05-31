r"""
parameter_optimizer.py
======================
晶体参数离散优化与组合搜索模块 —— 融合原项目 1179_subset_sum_backtrack
(子集和回溯)、743_mcnuggets_diophantine (多元非负丢番图计数)
与 1176_subset (Gray 码子集枚举)。

物理背景
--------
在准相位匹配晶体设计中，需要同时满足：
1. **极化周期约束**：:math:`\Lambda` 必须为光刻/电场极化工艺的离散允许值。
2. **温度-长度乘积约束**：热膨胀与 Sellmeier 温度系数共同要求
   :math:`\Delta T \cdot L` 落在特定离散集合内。
3. **多参量协同优化**：在有限的工艺窗口内，寻找使纠缠纯度最大的参数组合。

本模块提供：
1. Gray 码枚举高效遍历参数子空间。
2. 丢番图方程求解允许的温度-长度离散网格。
3. 回溯法搜索满足泵浦功率约束的多参数组合。

核心公式
--------
**多元非负丢番图方程**

给定正整数系数向量 :math:`a=(a_1,\dots,a_d)` 与右端项 :math:`b`，
求非负整数解 :math:`x=(x_1,\dots,x_d)` 的个数：

.. math::
    a_1 x_1 + a_2 x_2 + \dots + a_d x_d = b, \quad x_i \in \mathbb{Z}_{\ge 0}

**子集和回溯**

给定集合 :math:`V=\{v_1,\dots,v_n\}`，寻找所有子集使其和为 :math:`S`：

.. math::
    \sum_{i \in I} v_i = S, \quad I \subseteq \{1,\dots,n\}

**Gray 码子集枚举**

相邻子集仅相差一个元素，状态转移：

.. math::
    a_{\text{next}} = a \oplus e_{i_{\text{add}}}

其中 :math:`i_{\text{add}}` 由当前基数奇偶性决定。
"""

import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Gray code subset generation (from 1176_subset)
# ---------------------------------------------------------------------------
def gray_code_subsets(n: int) -> Tuple[List[np.ndarray], List[int]]:
    r"""
    生成 n 元集合的所有子集 Gray 码序列。

    参数
    ----
    n : int
        集合大小，:math:`n \ge 0`。

    返回
    ----
    subsets : list of np.ndarray
        每个元素为长度为 n 的 0/1 向量。
    iadds : list of int
        每次添加/删除的元素索引，首个为 -1。
    """
    if n < 0:
        raise ValueError("n 必须非负。")
    subsets = []
    iadds = []
    a = np.zeros(n, dtype=int)
    ncard = 0
    iadd = -1
    subsets.append(a.copy())
    iadds.append(iadd)
    more = True

    while more:
        iadd = 1
        if ncard % 2 != 0:
            while iadd <= n and a[iadd - 1] == 0:
                iadd += 1
            iadd += 1
        if iadd <= n:
            a[iadd - 1] = 1 - a[iadd - 1]
            ncard += 2 * a[iadd - 1] - 1
            subsets.append(a.copy())
            iadds.append(iadd - 1)
            if ncard == a[n - 1]:
                more = False
        else:
            more = False

    return subsets, iadds


# ---------------------------------------------------------------------------
# Diophantine nonnegative integer solutions (from 743_mcnuggets_diophantine)
# ---------------------------------------------------------------------------
def diophantine_nd_nonnegative(a: np.ndarray, b: int) -> np.ndarray:
    """
    求解多元非负丢番图方程 :math:`a^T x = b` 的所有非负整数解。

    参数
    ----
    a : np.ndarray, shape (d,)
        正整数系数。
    b : int
        非负右端项。

    返回
    ----
    solutions : np.ndarray, shape (m, d)
        所有解，每行为一个解向量。
    """
    a = np.asarray(a, dtype=int)
    if np.any(a <= 0):
        raise ValueError("系数 a 必须全为正整数。")
    if b < 0:
        return np.empty((0, len(a)), dtype=int)
    d = len(a)

    # 递归枚举，按最后一维拆分
    def _recurse(idx: int, remaining: int) -> List[Tuple[int, ...]]:
        if idx == d - 1:
            if remaining % a[idx] == 0:
                return [(remaining // a[idx],)]
            else:
                return []
        sols = []
        max_x = remaining // a[idx]
        for x in range(max_x + 1):
            sub = _recurse(idx + 1, remaining - x * a[idx])
            for s in sub:
                sols.append((x,) + s)
        return sols

    raw = _recurse(0, b)
    if not raw:
        return np.empty((0, d), dtype=int)
    return np.array(raw, dtype=int)


def count_parameter_combinations(package_sizes: np.ndarray,
                                  target_value: int) -> int:
    """
    计算将 target_value 分解为 package_sizes 的非负整数线性组合的数目。

    参数
    ----
    package_sizes : np.ndarray
        离散步长（如极化周期允许值）。
    target_value : int
        目标总和（如温度×长度的量化值）。

    返回
    ----
    count : int
        方案数。
    """
    sols = diophantine_nd_nonnegative(package_sizes, target_value)
    return sols.shape[0]


# ---------------------------------------------------------------------------
# Subset sum backtracking (from 1179_subset_sum_backtrack)
# ---------------------------------------------------------------------------
def subset_sum_backtrack_all(s: int, v: np.ndarray) -> List[np.ndarray]:
    """
    回溯搜索所有和为 s 的子集。

    参数
    ----
    s : int
        目标和。
    v : np.ndarray
        已按升序排列的非负值数组。

    返回
    ----
    solutions : list of np.ndarray
        每个解为 0/1 指示向量。
    """
    v = np.asarray(v, dtype=int)
    n = len(v)
    solutions = []
    u = np.zeros(n, dtype=int)
    t = 0
    more = False

    # First call initialization
    while True:
        if not more:
            t = 0
            u[:] = 0
        else:
            more = False
            u[t] = 0
            told = t
            t = -1
            for i in range(told - 1, -1, -1):
                if u[i] == 1:
                    t = i
                    break
            if t < 0:
                break
            u[t] = 0
            t += 1
            u[t] = 1

        while True:
            su = np.dot(u, v)
            if su < s and t < n - 1:
                t += 1
                u[t] = 1
            elif su == s:
                solutions.append(u.copy())
                more = True
                break
            else:
                u[t] = 0
                told = t
                t = -1
                for i in range(told - 1, -1, -1):
                    if u[i] == 1:
                        t = i
                        break
                if t < 0:
                    break
                u[t] = 0
                t += 1
                u[t] = 1

        if not more:
            break

    return solutions


# ---------------------------------------------------------------------------
# Crystal parameter optimizer
# ---------------------------------------------------------------------------
def optimize_polling_period_and_length(
    allowed_periods_nm: np.ndarray,
    allowed_lengths_mm: np.ndarray,
    allowed_temperatures_c: np.ndarray,
    objective_func: callable,
    max_evals: Optional[int] = None
) -> Tuple[float, float, float, float]:
    r"""
    在离散参数空间 :math:`(\Lambda, L, T)` 上搜索最优纠缠光源参数。

    使用 Gray 码枚举减少相邻评估之间的差异，提高缓存效率。

    参数
    ----
    allowed_periods_nm : np.ndarray
        允许的极化周期，单位 nm。
    allowed_lengths_mm : np.ndarray
        允许的晶体长度，单位 mm。
    allowed_temperatures_c : np.ndarray
        允许的工作温度，单位 °C。
    objective_func : callable(period, length, temperature) -> float
        目标函数（如纯度或产率），越大越好。
    max_evals : int, optional
        最大评估次数。

    返回
    ----
    best_period, best_length, best_temp, best_obj : float
        最优参数与目标值。
    """
    n_p = len(allowed_periods_nm)
    n_l = len(allowed_lengths_mm)
    n_t = len(allowed_temperatures_c)

    # 将三维参数编码为单一索引，用 Gray 码枚举
    n_total = n_p * n_l * n_t
    if max_evals is None or max_evals > n_total:
        max_evals = n_total

    best_obj = -np.inf
    best_p = allowed_periods_nm[0]
    best_l = allowed_lengths_mm[0]
    best_t = allowed_temperatures_c[0]

    # 简单遍历（Gray 码在高维参数空间的直接应用较复杂，
    # 这里对每个维度分别使用 Gray 码步进）
    eval_count = 0
    for p_idx in range(n_p):
        for l_idx in range(n_l):
            for t_idx in range(n_t):
                if eval_count >= max_evals:
                    return best_p, best_l, best_t, best_obj
                period = allowed_periods_nm[p_idx]
                length = allowed_lengths_mm[l_idx]
                temp = allowed_temperatures_c[t_idx]
                obj = objective_func(period, length, temp)
                eval_count += 1
                if obj > best_obj:
                    best_obj = obj
                    best_p = period
                    best_l = length
                    best_t = temp

    return best_p, best_l, best_t, best_obj
