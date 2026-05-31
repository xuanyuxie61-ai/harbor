"""
support_optimizer.py
====================
基于回溯与假位法的稀疏支持集优化模块

科学背景：
---------
在压缩感知中，稀疏信号的支持集（support set）定义为非零系数的位置：
    supp(x) = {i : x_i != 0}

支持集的准确估计对重建质量至关重要。本模块结合两种经典算法：

1. 回溯子集搜索（Backtracking Subset Search，来自项目 1179_subset_sum_backtrack）：
   用于在候选支持集上搜索满足约束的最优子集。

   给定候选集 V = {v_1, v_2, ..., v_n} 和目标值 S，寻找子集使得和恰好为 S。
   在 CS 中，V 对应各候选系数的贡献，S 对应残差能量。

   回溯算法框架：
       若当前和 < S 且可继续添加：添加下一个元素
       若当前和 == S：找到一个解
       若当前和 > S：回溯，移除最后添加的元素

2. 假位法（Regula Falsi，来自项目 809_nonlin_regula）：
   用于寻找最优正则化参数 lambda，使得重建支持集大小等于预期稀疏度。

   假位法迭代格式：
       c = (a * f(b) - b * f(a)) / (f(b) - f(a))
       若 sign(f(c)) == sign(f(a))：a = c
       否则：b = c

   收敛阶：超线性（约 1.618）。

支持集优化策略：
---------------
Step 1: 利用回溯搜索确定候选支持集
Step 2: 利用假位法优化阈值，使支持集大小满足稀疏约束
Step 3: 在确定的支持集上求解最小二乘问题
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


def subset_sum_backtrack(s: float, values: np.ndarray,
                         more: bool = False,
                         u: Optional[np.ndarray] = None,
                         t: int = 0) -> Tuple[bool, np.ndarray, int]:
    """
    回溯子集搜索（来自项目 1179_subset_sum_backtrack）。

    寻找和恰好为 s 的值的子集。

    参数:
        s: 目标和
        values: 候选值数组（已排序，非负）
        more: 是否继续搜索（首次调用为 False）
        u: 当前选择状态向量
        t: 当前最高选中索引
    返回:
        (more, u, t): more=True 表示找到新解
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if not more:
        t = 0
        u = np.zeros(n, dtype=int)
    else:
        more = False
        if t > 0:
            u[t - 1] = 0

        told = t
        t = -1
        for i in range(told - 1, 0, -1):
            if u[i - 1] == 1:
                t = i
                break

        if t < 1:
            return False, u, 0

        u[t - 1] = 0
        t = t + 1
        u[t - 1] = 1

    while True:
        su = float(np.dot(u, values))

        if su < s and t < n:
            t = t + 1
            u[t - 1] = 1
        elif abs(su - s) < 1e-10 * max(1.0, abs(s)):
            more = True
            return more, u, t
        else:
            u[t - 1] = 0
            told = t
            t = -1
            for i in range(told - 1, 0, -1):
                if u[i - 1] == 1:
                    t = i
                    break
            if t < 1:
                return False, u, 0
            u[t - 1] = 0
            t = t + 1
            u[t - 1] = 1


def regula_falsi(f: Callable, a: float, b: float,
                 tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, int]:
    """
    假位法（Regula Falsi）求根（来自项目 809_nonlin_regula）。

    要求 f(a) 和 f(b) 异号。

    参数:
        f: 目标函数
        a, b: 初始区间端点
        tol: 容差
        max_iter: 最大迭代次数
    返回:
        (x_root, iters): 根近似值和迭代次数
    """
    fa = f(a)
    fb = f(b)

    if np.sign(fa) == np.sign(fb):
        raise ValueError(f"f(a)={fa:.3e} 和 f(b)={fb:.3e} 同号，假位法要求异号")

    it = 0
    while abs(b - a) > tol and it < max_iter:
        # 假位公式
        if abs(fb - fa) < 1e-20:
            break
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        it += 1

        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    return 0.5 * (a + b), it


def optimize_threshold_for_sparsity(coefficients: np.ndarray,
                                    target_sparsity: int,
                                    lambda_min: float = 1e-6,
                                    lambda_max: float = 1.0) -> float:
    """
    利用假位法优化软阈值参数，使支持集大小接近目标稀疏度。

    目标函数：
        g(lambda) = |supp(S_lambda(c))| - target_sparsity
    其中 S_lambda 为软阈值算子。

    参数:
        coefficients: 系数向量
        target_sparsity: 目标支持集大小
        lambda_min, lambda_max: 搜索区间
    返回:
        最优阈值 lambda
    """
    coeffs = np.asarray(coefficients, dtype=float)

    def g(lam: float) -> float:
        thresholded = np.sign(coeffs) * np.maximum(np.abs(coeffs) - lam, 0.0)
        support_size = np.count_nonzero(np.abs(thresholded) > 1e-10)
        return float(support_size - target_sparsity)

    # 二分搜索确定初始区间
    g_min = g(lambda_min)
    g_max = g(lambda_max)

    # 扩展区间直到异号
    while np.sign(g_min) == np.sign(g_max) and lambda_max < 1e6:
        lambda_max *= 2.0
        g_max = g(lambda_max)

    if np.sign(g_min) == np.sign(g_max):
        # 若无法找到异号区间，返回使支持集最接近目标值的阈值
        if abs(g_min) < abs(g_max):
            return lambda_min
        else:
            return lambda_max

    try:
        lam_opt, _ = regula_falsi(g, lambda_min, lambda_max, tol=1e-4, max_iter=50)
    except ValueError:
        # 回退到简单搜索
        lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 100)
        best_lam = lambda_min
        best_err = abs(g(lambda_min))
        for lam in lambdas:
            err = abs(g(lam))
            if err < best_err:
                best_err = err
                best_lam = lam
        lam_opt = best_lam

    return max(lambda_min, min(lambda_max, lam_opt))


def backtracking_support_recovery(correlations: np.ndarray,
                                  target_energy: float,
                                  max_support_size: int = 100) -> np.ndarray:
    """
    利用回溯法从相关性向量中恢复支持集。

    策略：
        将各位置的相关性视为候选值，寻找和接近目标能量的最小子集。

    参数:
        correlations: 各位置的相关性/能量值（非负，已排序）
        target_energy: 目标累积能量
        max_support_size: 最大支持集大小
    返回:
        支持集索引数组
    """
    correlations = np.asarray(correlations, dtype=float)
    n = len(correlations)

    if n == 0:
        return np.array([], dtype=int)

    # 按降序排列
    sorted_idx = np.argsort(-correlations)
    sorted_vals = correlations[sorted_idx]

    # 贪心策略 + 回溯微调
    cumulative = np.cumsum(sorted_vals)
    cutoff = np.searchsorted(cumulative, target_energy, side='right') + 1
    cutoff = min(cutoff, n, max_support_size)

    support = sorted_idx[:cutoff]

    # 若能量超调，尝试回溯移除最小元素
    current_energy = np.sum(correlations[support])
    while current_energy > target_energy * 1.05 and len(support) > 1:
        # 找到支持集中贡献最小的元素
        min_idx_in_support = np.argmin(correlations[support])
        support = np.delete(support, min_idx_in_support)
        current_energy = np.sum(correlations[support])

    return np.sort(support)


def refined_support_reconstruction(A: np.ndarray, y: np.ndarray,
                                   target_sparsity: int,
                                   Psi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    结合支持集优化的压缩感知重建流程。

    步骤：
        1. 使用 OMP 获得初始支持集估计
        2. 使用假位法优化阈值
        3. 在优化后的支持集上求解最小二乘

    参数:
        A: 感知矩阵
        y: 测量向量
        target_sparsity: 目标稀疏度
        Psi: 稀疏基（可选，若 A 已包含 Psi 则不需）
    返回:
        (x_recon, support): 重建向量和支持集
    """
    from cs_detector import orthogonal_matching_pursuit

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    # Step 1: OMP 初始估计
    x_omp, support_omp = orthogonal_matching_pursuit(A, y, target_sparsity)

    # Step 2: 计算初始系数的相关性（用于阈值优化）
    residual = y - A @ x_omp
    correlations = np.abs(A.T @ residual)

    # Step 3: 阈值优化
    lambda_opt = optimize_threshold_for_sparsity(x_omp, target_sparsity)

    # Step 4: 应用优化阈值
    x_thresholded = np.sign(x_omp) * np.maximum(np.abs(x_omp) - lambda_opt, 0.0)
    support = np.where(np.abs(x_thresholded) > 1e-10)[0]

    # Step 5: 在支持集上求解最小二乘
    if len(support) > 0:
        A_support = A[:, support]
        try:
            x_support, _, _, _ = np.linalg.lstsq(A_support, y, rcond=None)
        except np.linalg.LinAlgError:
            x_support = np.zeros(len(support))
        x_recon = np.zeros(A.shape[1], dtype=float)
        x_recon[support] = x_support
    else:
        x_recon = x_thresholded

    return x_recon, support
