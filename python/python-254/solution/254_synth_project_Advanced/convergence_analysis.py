# -*- coding: utf-8 -*-
"""
convergence_analysis.py
=======================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

Richardson 外推、收敛阶验证与误差预算.

Richardson 外推
---------------
若数值解 f_h 满足::

    f_h = f_exact + C h^p + O(h^{p+1})

则两级网格 h, h/2 的解可组合::

    f_extrap = (2^p f_{h/2} - f_h) / (2^p - 1)

其误差为 O(h^{p+1}).

观测收敛阶::

    p_obs = log((f_h1 - f_h2) / (f_h2 - f_h3)) / log(r)

其中 r = h1 / h2 = h2 / h3 为细化比.

误差预算
--------
对 kilonova 辐射转移的总误差由以下部分组成::

    err_total^2 = err_spatial^2 + err_temporal^2
                 + err_opacity^2 + err_statistical^2

映射种子项目
-----------
- 1363 (tsp_brute)   → 排列遍历思想用于搜索"最坏情况"
  误差组合 (类似 TSP 中搜索最长路径)
- 014 (approx_chebyshev) → Chebyshev 插值的误差分析理论
  直接用于估计 opacity 表的截断误差
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Richardson 外推
# ---------------------------------------------------------------------------
def richardson_extrapolate(f_h: float, f_h2: float, p: int, r: float = 2.0
                           ) -> float:
    """Richardson 外推: 从 f_h, f_h/2 估计 f_exact.

    f_ext = (r^p * f_{h/2} - f_h) / (r^p - 1)

    Parameters
    ----------
    f_h  : float  粗网格解
    f_h2 : float  细网格解
    p    : int    预期收敛阶
    r    : float  细化比 (默认 2)
    """
    return (r ** p * f_h2 - f_h) / (r ** p - 1.0)


def observed_order(f_h1: float, f_h2: float, f_h3: float, r: float = 2.0
                   ) -> float:
    """从三级网格解估计观测收敛阶.

    p_obs = log((f_h1 - f_h2) / (f_h2 - f_h3)) / log(r)
    """
    d12 = f_h1 - f_h2
    d23 = f_h2 - f_h3
    if abs(d23) < 1.0e-30 or d12 * d23 < 0:
        return float("nan")
    return math.log(abs(d12 / d23)) / math.log(r)


# ---------------------------------------------------------------------------
# 收敛测试框架
# ---------------------------------------------------------------------------
def convergence_test(solver, h_values: List[float], exact=None
                     ) -> Dict[str, List[float]]:
    """执行收敛测试并返回误差序列.

    Parameters
    ----------
    solver  : Callable[[float], float]  h -> f_h
    h_values: 网格尺寸序列
    exact   : 精确值 (可选)

    Returns
    -------
    dict: h, f_h, err, p_obs
    """
    f_vals = [solver(h) for h in h_values]
    errors = []
    if exact is not None:
        errors = [abs(f - exact) for f in f_vals]
    else:
        # 用最细网格作为参考
        f_ref = f_vals[-1]
        errors = [abs(f - f_ref) for f in f_vals]
    p_obs_seq = []
    for i in range(1, len(h_values) - 1):
        r = h_values[i - 1] / h_values[i]
        p = observed_order(f_vals[i - 1], f_vals[i], f_vals[i + 1], r)
        p_obs_seq.append(p)
    return {
        "h_values": h_values,
        "f_values": f_vals,
        "errors": errors,
        "p_observed": p_obs_seq,
    }


# ---------------------------------------------------------------------------
# 误差预算组合
# ---------------------------------------------------------------------------
def error_budget(components: Dict[str, float]) -> Dict[str, float]:
    """计算 RMS 误差预算.

    err_total = sqrt(sum_i err_i^2)

    各分量以分数形式给出.
    """
    if not components:
        return {"total": 0.0}
    sum_sq = sum(v * v for v in components.values())
    total = math.sqrt(sum_sq)
    result = dict(components)
    result["total"] = total
    for k, v in components.items():
        result[f"{k}_fraction"] = abs(v) / total if total > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
#  worst-case 误差搜索 (映射 1363_tsp_brute 遍历思想)
# ---------------------------------------------------------------------------
def worst_case_error_combination(
    error_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """在给定误差范围内搜索最坏情况的误差组合.

    对每个误差源, 最坏情况为其范围上界. 但考虑符号, 可能
    某些组合相消. 用采样搜索最大 RMS.

    Parameters
    ----------
    error_ranges : Dict[str, (lo, hi)]  每个误差源的范围
    """
    import random
    rng = random.Random(seed)
    names = list(error_ranges.keys())
    max_total = 0.0
    worst_config = None
    for _ in range(n_samples):
        config = {}
        for name in names:
            lo, hi = error_ranges[name]
            config[name] = rng.uniform(lo, hi)
        total = math.sqrt(sum(v * v for v in config.values()))
        if total > max_total:
            max_total = total
            worst_config = config
    return {"max_total": max_total, "worst_config": worst_config}


# ---------------------------------------------------------------------------
# L2 范数误差 (场量)
# ---------------------------------------------------------------------------
def l2_error(u_num: List[float], u_exact: List[float],
             weights: List[float] = None) -> float:
    """加权 L2 误差::

        ||e||_2 = sqrt( sum_i w_i (u_num_i - u_exact_i)^2 / sum_i w_i )
    """
    n = len(u_num)
    if n != len(u_exact):
        raise ValueError("length mismatch")
    if weights is None:
        weights = [1.0] * n
    num = sum(weights[i] * (u_num[i] - u_exact[i]) ** 2 for i in range(n))
    den = sum(weights)
    return math.sqrt(num / den) if den > 0 else 0.0


def linf_error(u_num: List[float], u_exact: List[float]) -> float:
    """L_inf (最大) 误差."""
    return max(abs(a - b) for a, b in zip(u_num, u_exact))


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """用 f(h) = 1 + h^2 验证 Richardson 外推."""
    def solver(h):
        return 1.0 + h * h
    h_vals = [0.1, 0.05, 0.025]
    result = convergence_test(solver, h_vals, exact=1.0)
    p_obs = result["p_observed"][0]
    if abs(p_obs - 2.0) > 0.1:
        raise AssertionError(f"Expected p=2, got {p_obs:.3f}")
    # 误差预算
    budget = error_budget({"spatial": 0.01, "temporal": 0.005, "opacity": 0.02})
    expected = math.sqrt(0.01 ** 2 + 0.005 ** 2 + 0.02 ** 2)
    if abs(budget["total"] - expected) > 1e-12:
        raise AssertionError("Error budget mismatch")
    return True


if __name__ == "__main__":
    _self_check()
    print("convergence_analysis self-check passed.")
    def solver(h):
        return math.sin(1.0) + 0.5 * h * h
    result = convergence_test(solver, [0.1, 0.05, 0.025, 0.0125], exact=math.sin(1.0))
    print(f"  errors        : {[f'{e:.3e}' for e in result['errors']]}")
    print(f"  observed order: {[f'{p:.3f}' for p in result['p_observed']]}")
