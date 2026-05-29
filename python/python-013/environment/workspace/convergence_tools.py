"""
convergence_tools.py

基于 collatz (196_collatz) 与 lebesgue (658_lebesgue) 的
收敛监控与稳定性分析模块。

在自洽场计算 (如 DMFT、Hartree-Fock) 中，迭代序列的收敛性至关重要。
本模块提供:
1. 迭代路径长度监控 (Collatz 序列类比)
2. 自洽迭代的混频与收敛判据
3. 插值稳定性的 Lebesgue 常数分析
4. 自能迭代的自适应阻尼
"""

import numpy as np
from typing import List, Callable, Optional, Tuple


# ---------------------------------------------------------------------------
# collatz: 迭代序列监控
# ---------------------------------------------------------------------------

def collatz_sequence(start: int) -> List[int]:
    """
    Collatz 序列: 若 T 为偶数则 T/2，若为奇数则 3T+1。
    返回从 start 到 1 的完整序列。
    
    物理类比: 将迭代步数映射为"停止时间"，用于监控
    自洽迭代是否陷入周期振荡或缓慢收敛。
    """
    if start <= 0:
        return []
    seq = [start]
    t = start
    max_steps = 10000
    steps = 0
    while t != 1 and steps < max_steps:
        if t % 2 == 0:
            t = t // 2
        else:
            t = 3 * t + 1
        seq.append(t)
        steps += 1
    return seq


def collatz_stopping_time(start: int) -> int:
    """Collatz 停止时间 (到达 1 的步数)。"""
    return len(collatz_sequence(start)) - 1


def iteration_complexity_index(residuals: List[float]) -> float:
    """
    计算迭代序列的"复杂度指数":
        C = Σ_n |r_{n+1} - r_n| / |r_n|
    
    类比 Collatz 序列的总变差，用于评估迭代的"崎岖程度"。
    值越大表示收敛越不稳定。
    """
    if len(residuals) < 2:
        return 0.0
    C = 0.0
    for i in range(len(residuals) - 1):
        r = abs(residuals[i])
        if r < 1e-14:
            r = 1e-14
        C += abs(residuals[i + 1] - residuals[i]) / r
    return C


# ---------------------------------------------------------------------------
# 自洽迭代工具
# ---------------------------------------------------------------------------

def simple_mixing(old: np.ndarray, new: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    简单线性混频:
        x_{out} = α x_{new} + (1-α) x_{old}
    
    α 为混频因子，0 < α <= 1。
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha 必须在 (0, 1] 内")
    return alpha * new + (1.0 - alpha) * old


def pulay_mixing(history: List[np.ndarray], residuals: List[np.ndarray], n_keep: int = 5) -> np.ndarray:
    """
    Pulay/DIIS 混频: 利用历史迭代构造最优线性组合。
    
    最小化 |Σ c_i r_i|^2，约束 Σ c_i = 1。
    
    参数:
        history: 历史解向量列表
        residuals: 对应残差列表
        n_keep: 保留历史步数
    
    返回:
        混频后的新解
    """
    m = min(len(history), len(residuals), n_keep)
    if m < 2:
        return history[-1]
    # 构造残差重叠矩阵 A_{ij} = <r_i | r_j>
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            idx_i = -(m - i)
            idx_j = -(m - j)
            A[i, j] = np.vdot(residuals[idx_i], residuals[idx_j]).real
    # 添加正则化
    A += 1e-10 * np.eye(m)
    # 解约束最小二乘
    rhs = np.zeros(m + 1)
    rhs[m] = 1.0
    M = np.zeros((m + 1, m + 1))
    M[:m, :m] = A
    M[m, :m] = 1.0
    M[:m, m] = 1.0
    try:
        c = np.linalg.solve(M, rhs)[:m]
    except np.linalg.LinAlgError:
        # 回退到简单混频
        return history[-1]
    # 构造混频解
    x_mix = np.zeros_like(history[-1])
    for i in range(m):
        x_mix += c[i] * history[-(m - i)]
    return x_mix


def self_consistent_iteration(update_func: Callable, x0: np.ndarray,
                               tol: float = 1e-6, max_iter: int = 100,
                               mixing: str = "simple", alpha: float = 0.5,
                               n_keep: int = 5) -> Tuple[np.ndarray, int, List[float]]:
    """
    通用自洽迭代框架。
    
    参数:
        update_func: x_new = f(x_old)
        x0: 初始猜测
        tol: 收敛容差
        max_iter: 最大迭代次数
        mixing: "simple" 或 "pulay"
        alpha: 混频因子
        n_keep: Pulay 历史长度
    
    返回:
        x_final, n_iter, residual_history
    """
    if tol <= 0:
        raise ValueError("tol > 0")
    if max_iter < 1:
        raise ValueError("max_iter >= 1")
    x = x0.copy()
    residuals = []
    history = []
    res_history = []
    for it in range(max_iter):
        x_new = update_func(x)
        r = x_new - x
        res_norm = float(np.linalg.norm(r))
        residuals.append(res_norm)
        res_history.append(res_norm)
        history.append(x.copy())
        if res_norm < tol:
            return x_new, it + 1, res_history
        if mixing == "pulay" and len(history) >= 2:
            # 计算残差向量列表
            res_vecs = [history[i + 1] - history[i] for i in range(len(history) - 1)]
            res_vecs.append(r)
            x = pulay_mixing(history, res_vecs, n_keep)
        else:
            x = simple_mixing(x, x_new, alpha)
    return x, max_iter, res_history


# ---------------------------------------------------------------------------
# Lebesgue 稳定性分析 (复用 matsubara_green 中的函数)
# ---------------------------------------------------------------------------

def interpolation_stability_analysis(x_nodes: np.ndarray, x_test: np.ndarray) -> dict:
    """
    对给定的插值节点进行稳定性分析，返回 Lebesgue 常数等。
    """
    from matsubara_green import lebesgue_constant_estimate
    n = len(x_nodes)
    lmax = lebesgue_constant_estimate(n, x_nodes, x_test)
    # 节点间距分析
    dx = np.diff(np.sort(x_nodes))
    min_spacing = float(np.min(dx)) if len(dx) > 0 else 0.0
    max_spacing = float(np.max(dx)) if len(dx) > 0 else 0.0
    return {
        "lebesgue_constant": lmax,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "node_count": n,
    }


# ---------------------------------------------------------------------------
# 自适应阻尼
# ---------------------------------------------------------------------------

def adaptive_damping(residuals: List[float], alpha_min: float = 0.05,
                      alpha_max: float = 0.8) -> float:
    """
    根据残差变化趋势自适应调整混频因子:
        - 残差单调递减 → 增大 α (加速)
        - 残差震荡 → 减小 α (稳定)
    """
    if len(residuals) < 3:
        return alpha_max
    # 计算最近三步的变化趋势
    dr1 = residuals[-2] - residuals[-3]
    dr2 = residuals[-1] - residuals[-2]
    if dr1 < 0 and dr2 < 0:
        # 单调下降，加速
        return min(alpha_max, alpha_max * 1.1)
    elif dr1 * dr2 < 0:
        # 震荡，减速
        return max(alpha_min, alpha_max * 0.5)
    else:
        return alpha_max * 0.7


if __name__ == "__main__":
    seq = collatz_sequence(27)
    print(f"Collatz(27) stopping time = {len(seq)-1}")
    x0 = np.array([1.0, 2.0])
    def f(x):
        return np.array([0.5 * x[0] + 1.0, 0.3 * x[1] + 0.5])
    x, it, res = self_consistent_iteration(f, x0, tol=1e-8, max_iter=50)
    print(f"Converged in {it} iterations, residual={res[-1]:.2e}")
