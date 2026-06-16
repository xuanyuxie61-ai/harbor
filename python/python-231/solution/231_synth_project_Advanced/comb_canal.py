"""
comb_canal.py — PDF 特征向量退化度计数 (映射自 nondegenerate-canalization)
=============================================================
本模块将 Kadelka Lab 的非退化管化 (canalization) 枚举理论
映射到 PDF 误差分析的特征向量退化度计数.

原始理论 (Kadelka Lab):
    布尔函数的管化深度 (canalizing depth) 描述了函数中
    "嵌套" 的管化层的数量. 非退化函数是那些所有输入都
    对输出有影响的函数.

PDF 误差分析映射:
    Hessian 矩阵的特征值 {λ_k} 描述参数空间的各向异性.
    - "管化" (canalized) 特征方向: λ_k >> 1 (强约束方向)
    - "非管化" 特征方向: λ_k ≈ 0 (弱约束/退化方向)
    - 退化度: 近似零特征值的数量

核心公式 (退化度计数):
    D = |{k : λ_k < ε_threshold}|

核心公式 (管化深度, 映射自 number_k_canalizing_depth):
    对 N 个参数方向, 管化深度为:
    K = max {k : λ_{(k)} > threshold × λ_max}
    其中 λ_{(k)} 为第 k 大特征值.

核心公式 (组合计数, 映射自 nchoosek):
    C(N, k) = N! / (k! (N-k)!)
    用于计算退化子空间的维度组合数.

核心公式 (Stirling 数, 映射自 sterling_times_fak_r):
    S(n, r) = Σ_{i=0}^r (−1)^i C(r, i) (r−i)^n
    用于计算参数空间划分的组合数.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict

from phys_consts import EPS_NUMERICAL


# ============================================================
# 1. 组合数学工具
# ============================================================
def nchoosek(n: int, k: int) -> float:
    """组合数 C(n, k) = n! / (k! (n-k)!)"""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result *= (n - i) / (i + 1)
    return result


def stirling_second_kind(n: int, r: int) -> float:
    """
    第二类 Stirling 数 S(n, r):
        S(n, r) = (1/r!) Σ_{i=0}^r (−1)^{r−i} C(r, i) i^n

    组合意义: 将 n 个元素划分为 r 个非空子集的方法数.
    """
    if r < 0 or r > n:
        return 0.0
    if n == 0 and r == 0:
        return 1.0
    result = 0.0
    for i in range(r + 1):
        sign = (-1) ** (r - i)
        result += sign * nchoosek(r, i) * (i ** n)
    # 除以 r!
    r_fact = 1.0
    for i in range(1, r + 1):
        r_fact *= i
    return result / r_fact if r_fact > 0 else 0.0


def partition_count(n: int) -> int:
    """
    整数 n 的分划数 p(n).
    p(0) = 1, p(1) = 1, p(2) = 2, p(3) = 3, p(4) = 5, ...
    """
    if n <= 0:
        return 1 if n == 0 else 0
    dp = [0] * (n + 1)
    dp[0] = 1
    for k in range(1, n + 1):
        for i in range(k, n + 1):
            dp[i] += dp[i - k]
    return dp[n]


# ============================================================
# 2. PDF 特征向量退化度分析
# ============================================================
def count_degenerate_directions(
    eigenvalues: List[float],
    threshold_ratio: float = 1e-4
) -> Tuple[int, int, float]:
    """
    计算 Hessian 特征值中的退化方向数:

    参数:
        eigenvalues:    Hessian 特征值列表
        threshold_ratio: 退化阈值 (相对于最大特征值)
    返回:
        (n_degenerate, n_non_degenerate, degeneracy_fraction)
    """
    if not eigenvalues:
        return 0, 0, 0.0
    lam_max = max(abs(l) for l in eigenvalues)
    if lam_max < EPS_NUMERICAL:
        return len(eigenvalues), 0, 1.0
    threshold = lam_max * threshold_ratio
    n_degen = sum(1 for l in eigenvalues if abs(l) < threshold)
    n_total = len(eigenvalues)
    return n_degen, n_total - n_degen, n_degen / n_total


def canalizing_depth(
    eigenvalues: List[float],
    threshold_ratio: float = 0.01
) -> int:
    """
    管化深度 (映射自 number_k_canalizing_depth):

    管化深度 = 强约束方向的数量
        K = |{k : λ_{(k)} > threshold_ratio × λ_max}|

    物理意义: 参数空间中有多少个方向被数据强约束.
    """
    if not eigenvalues:
        return 0
    lam_max = max(abs(l) for l in eigenvalues)
    if lam_max < EPS_NUMERICAL:
        return 0
    threshold = lam_max * threshold_ratio
    sorted_eigs = sorted([abs(l) for l in eigenvalues], reverse=True)
    depth = 0
    for lam in sorted_eigs:
        if lam > threshold:
            depth += 1
        else:
            break
    return depth


def degenerate_subspace_dimension(
    eigenvalues: List[float],
    threshold_ratio: float = 1e-4
) -> int:
    """
    退化子空间维度 = 近似零特征值的数量.

    物理意义: 参数空间中有多少个方向完全不被数据约束.
    这些方向对应于 PDF 参数的"管化" (canalized) 组合.
    """
    if not eigenvalues:
        return 0
    lam_max = max(abs(l) for l in eigenvalues)
    threshold = lam_max * threshold_ratio
    return sum(1 for l in eigenvalues if abs(l) < threshold)


# ============================================================
# 3. 退化模式枚举
# ============================================================
def enumerate_degenerate_modes(
    n_params: int, n_degenerate: int,
    max_modes: int = 20
) -> List[Tuple[int, ...]]:
    """
    枚举退化模式 (映射自 partitions):

    将 n_degenerate 个退化方向分配到 n_params 个参数中.
    每种分配模式对应一种"简并"的参数组合.

    返回:
        模式列表, 每个模式是一个元组 (d_1, d_2, ..., d_n_params)
        其中 d_i 是第 i 个参数方向上的退化度.
    """
    if n_degenerate <= 0 or n_params <= 0:
        return [(0,) * n_params]
    if n_degenerate == 1:
        return [tuple(1 if i == k else 0 for i in range(n_params))
                for k in range(min(n_params, max_modes))]

    # 简化: 返回 n_degenerate 的分划
    result = []
    for p in _generate_partitions(n_degenerate, n_params):
        result.append(p)
        if len(result) >= max_modes:
            break
    return result


def _generate_partitions(n: int, max_parts: int):
    """生成 n 的分划 (最多 max_parts 个部分)"""
    if n == 0:
        yield (0,) * max_parts
        return
    if max_parts == 1:
        yield (n,)
        return
    for first in range(n, -1, -1):
        for rest in _generate_partitions(n - first, max_parts - 1):
            yield (first,) + rest


# ============================================================
# 4. 非退化管化函数计数
# ============================================================
def number_non_degenerate_functions(n: int, p: int = 2) -> Tuple[float, float, float]:
    """
    非退化管化函数计数 (映射自 number_ncfs):

    在 PDF 语境中, 这对应于:
    - n: 参数维度
    - p: 每个参数的"状态数" (2 = 正/负方向)

    返回:
        (total, N1, N2): 总数, 第一类非退化数, 第二类非退化数
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (2.0, 0.0, 2.0)

    # B*(n) 的简化计算
    B_star = 2 ** (2 ** min(n, 8))  # 防止溢出

    # N1: 第一类非退化数
    N1 = 0.0
    if p > 2:
        sum_term = 0.0
        for r in range(2, min(n + 1, 10)):
            term = ((p - 1) ** (r - 1) * n
                    * stirling_second_kind(n - 1, r - 1) * math.factorial(r - 1))
            sum_term += term
        N1 = (2 ** (n - 1)) * p * (p - 2) * ((p - 1) ** n) * sum_term

    # N2: 第二类非退化数
    N2 = 0.0
    for r in range(1, min(n, 10)):
        # sterling_difference 简化
        diff = r ** n
        for i in range(1, min(r + 1, 10)):
            term = ((-1) ** i * nchoosek(r - 1, i - 1)
                    * ((r - i) ** (n - 1)) * (r * r / i - r + n))
            diff += term
        N2 += (p - 1) ** r * diff
    N2 *= (2 ** n) * p * ((p - 1) ** n)

    return N1 + N2, N1, N2


def pdf_degeneracy_report(eigenvalues: List[float], n_params: int) -> str:
    """
    生成 PDF 退化度分析报告.
    """
    n_degen, n_non_degen, frac = count_degenerate_directions(eigenvalues)
    depth = canalizing_depth(eigenvalues)
    dim_degen = degenerate_subspace_dimension(eigenvalues)
    total, n1, n2 = number_non_degenerate_functions(min(n_params, 6))

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║   PDF 特征向量退化度分析报告                     ║",
        "╠══════════════════════════════════════════════════╣",
        f"║ 参数维度 N = {n_params}",
        f"║ 管化深度 K = {depth} (强约束方向数)",
        f"║ 退化方向数 D = {n_degen} (弱约束方向数)",
        f"║ 非退化方向数 = {n_non_degen}",
        f"║ 退化分数 = {frac:.4f}",
        f"║ 退化子空间维度 = {dim_degen}",
        f"║ 非退化函数数 ≈ {total:.2e}",
        "╚══════════════════════════════════════════════════╝",
    ]
    return '\n'.join(lines)
