"""
utils.py
通用工具与组合数学模块

融入原项目:
- 202_combo: 组合数学算法（排列、子集、贝尔数、斯特林数）
- 338_errors: 数值误差分析

功能:
1. 组合数学工具用于离子通道状态枚举
2. 数值误差估计与收敛性分析
"""

import numpy as np
from math import factorial, comb, exp, log, sqrt


# ============================================================================
# 组合数学工具（源自 202_combo）
# ============================================================================

def stirling_numbers_2(n, k):
    """
    计算第二类斯特林数 S(n,k)
    S(n,k) = (1/k!) * sum_{j=0}^{k} (-1)^(k-j) * C(k,j) * j^n
    表示将 n 个不同元素划分为 k 个非空子集的方式数
    """
    if n < 0 or k < 0 or k > n:
        return 0
    if n == 0 and k == 0:
        return 1
    result = 0
    for j in range(k + 1):
        sign = 1 if (k - j) % 2 == 0 else -1
        result += sign * comb(k, j) * (j ** n)
    return result // factorial(k)


def bell_numbers(n):
    """
    计算贝尔数 B_n
    B_n = sum_{k=0}^{n} S(n,k)
    表示 n 个元素的集合划分总数
    """
    if n < 0:
        return 0
    total = 0
    for k in range(n + 1):
        total += stirling_numbers_2(n, k)
    return total


def subset_lex_rank(n, a):
    """
    子集的字典序排名
    将子集 a（以二进制列表表示）映射为其在字典序中的排名
    """
    if n <= 0:
        return 0
    rank = 0
    for i in range(n):
        if a[i]:
            rank += 2 ** (n - 1 - i)
    return rank


def subset_lex_unrank(n, rank):
    """
    子集的字典序解排名
    """
    if n <= 0 or rank < 0 or rank >= 2 ** n:
        return [False] * n
    a = [False] * n
    for i in range(n - 1, -1, -1):
        if rank >= 2 ** i:
            a[n - 1 - i] = True
            rank -= 2 ** i
    return a


def perm_lex_rank(n, p):
    """
    排列的字典序排名
    rank = sum_{i=0}^{n-1} c_i * (n-1-i)!
    其中 c_i 是在位置 i 右侧且小于 p[i] 的元素个数
    """
    if n <= 0:
        return 0
    rank = 0
    used = [False] * (n + 1)
    for i in range(n):
        count = 0
        for j in range(1, p[i]):
            if not used[j]:
                count += 1
        rank += count * factorial(n - 1 - i)
        used[p[i]] = True
    return rank


def perm_lex_unrank(n, rank):
    """
    排列的字典序解排名
    """
    if n <= 0 or rank < 0 or rank >= factorial(n):
        return list(range(1, n + 1))
    p = [0] * n
    used = [False] * (n + 1)
    for i in range(n):
        fi = factorial(n - 1 - i)
        c = rank // fi
        rank %= fi
        count = 0
        for j in range(1, n + 1):
            if not used[j]:
                if count == c:
                    p[i] = j
                    used[j] = True
                    break
                count += 1
    return p


def ion_channel_state_enumeration(n_states, n_open):
    """
    离子通道状态枚举
    计算 n_states 个状态中恰好 n_open 个开放状态的组合数 C(n_states, n_open)
    并返回所有可能的状态配置
    
    用于心肌细胞离子通道（如Na+、K+通道）的门控状态分析
    """
    if n_states < 0 or n_open < 0 or n_open > n_states:
        return 0, []
    total = comb(n_states, n_open)
    configs = []
    
    def backtrack(start, current, ones):
        if ones == n_open:
            configs.append(current[:])
            return
        if start >= n_states:
            return
        for i in range(start, n_states - (n_open - ones) + 1):
            current[i] = 1
            backtrack(i + 1, current, ones + 1)
            current[i] = 0
    
    backtrack(0, [0] * n_states, 0)
    return total, configs


# ============================================================================
# 数值误差分析（源自 338_errors）
# ============================================================================

def compute_relative_error(exact, approx):
    """
    计算相对误差
    E_rel = |exact - approx| / |exact|
    """
    if exact == 0:
        return abs(approx) if approx != 0 else 0.0
    return abs(exact - approx) / abs(exact)


def compute_roundoff_error(x, y, operation):
    """
    分析基本运算的舍入误差
    浮点运算满足: fl(x op y) = (x op y)(1 + δ), |δ| ≤ ε_mach
    """
    eps_mach = np.finfo(float).eps
    if operation == 'add':
        return eps_mach * abs(x + y)
    elif operation == 'mul':
        return eps_mach * abs(x * y)
    elif operation == 'div':
        if y == 0:
            return float('inf')
        return eps_mach * abs(x / y)
    return 0.0


def convergence_rate(errors, resolutions):
    """
    估计数值方法的收敛速率
    假设 error ≈ C * h^p，通过对数线性回归估计 p
    """
    if len(errors) < 2 or len(resolutions) < 2:
        return 0.0
    log_h = np.log(resolutions)
    log_e = np.log(errors)
    n = len(log_h)
    # 线性回归: log_e = a + p * log_h
    mean_x = np.mean(log_h)
    mean_y = np.mean(log_e)
    numerator = np.sum((log_h - mean_x) * (log_e - mean_y))
    denominator = np.sum((log_h - mean_x) ** 2)
    if denominator == 0:
        return 0.0
    p = numerator / denominator
    return p


def condition_number_analysis(A):
    """
    矩阵条件数分析
    κ(A) = ||A|| * ||A^{-1}||
    使用谱范数（最大奇异值）
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return float('inf')
    try:
        sigma = np.linalg.svd(A, compute_uv=False)
        if sigma[-1] == 0:
            return float('inf')
        return sigma[0] / sigma[-1]
    except Exception:
        return float('inf')


def estimate_truncation_error(f, x, h, order=2):
    """
    估计数值微分的截断误差
    对于中心差分 f'(x) ≈ (f(x+h) - f(x-h))/(2h)
    截断误差 ≈ (h^2/6) * f'''(ξ)
    """
    if h <= 0:
        return float('inf')
    # 用高阶差分估计三阶导数
    f3_approx = (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
    return abs(h ** 2 * f3_approx / 6.0)


# ============================================================================
# 数值稳定性测试
# ============================================================================

def catastrophic_cancellation_test():
    """
    灾难性抵消测试
    例: P^2 - 2*Q^2，其中 P/Q ≈ sqrt(2)
    """
    p = 665857.0
    q = 470832.0
    exact = 1.0  # P^2 - 2*Q^2 的理论值
    computed = p ** 2 - 2.0 * q ** 2
    rel_err = compute_relative_error(exact, computed)
    return {
        'p': p,
        'q': q,
        'exact': exact,
        'computed': computed,
        'relative_error': rel_err
    }


def generate_gray_code(n):
    """
    生成 n 位格雷码
    g_i = i XOR (i >> 1)
    用于减少离子通道状态转换中的数值振荡
    """
    if n < 0 or n > 20:
        return []
    return [i ^ (i >> 1) for i in range(2 ** n)]
