"""
utils.py — 高性能外排序系统的数值工具库
============================================
融合来源: 915_prime_plot (素数生成与因子分析)

提供素数生成、哈希种子构造、边界条件检查、数值鲁棒性保障等功能。
素数在分布式哈希表和伪随机采样中具有关键作用，可确保分区函数的
均匀性和低碰撞率。
"""

import math
import random
from typing import List, Tuple, Optional


def miller_rabin_prime(n: int, k: int = 10) -> bool:
    """
    Miller-Rabin 素性测试，基于费马小定理的扩展：

        若 n 为素数，则对任意 a ∈ [2, n-2]，有 a^{n-1} ≡ 1 (mod n)。

    将 n−1 分解为 n−1 = 2^r · d（d 为奇数），则要么 a^d ≡ 1 (mod n)，
    要么存在某个 s ∈ [0, r−1] 使得 a^{2^s · d} ≡ −1 (mod n)。

    参数:
        n: 待测试的整数
        k: 测试轮数，错误概率 ≤ 4^{-k}

    返回:
        True 若 n 极大概率是素数，False 若确定是合数
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # 分解 n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_primes(start: int, count: int) -> List[int]:
    """
    生成从 start 开始的 count 个素数列表。

    在分布式外排序系统中，素数用于构造双哈希函数族：
        h_i(x) = (a_i * x + b_i) mod p
    其中 p 为大素数，a_i, b_i 为随机系数。素数取模保证哈希值的
    最大周期和最小碰撞率。
    """
    primes = []
    candidate = max(start, 2)
    while len(primes) < count:
        if miller_rabin_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def hash_family_seed(prime_idx: int, num_hashes: int = 4) -> List[Tuple[int, int, int]]:
    """
    构造基于素数的通用哈希函数族种子。

    对第 prime_idx 个素数 p，生成 num_hashes 组 (p, a, b) 参数，
    满足通用哈希性质：对任意 x ≠ y，
        P[h(x) = h(y)] ≤ 1 / m
    其中 m 为哈希表大小。
    """
    base_primes = generate_primes(1000 + prime_idx * 50, num_hashes + 1)
    p = base_primes[-1]
    seeds = []
    for i in range(num_hashes):
        a = base_primes[i] % (p - 1) + 1
        b = (base_primes[i] * 7 + 13) % (p - 1)
        seeds.append((p, a, b))
    return seeds


def robust_division(a: float, b: float, fallback: float = 0.0) -> float:
    """
    数值鲁棒除法，避免除以零和溢出。
    """
    if abs(b) < 1e-300:
        return fallback
    result = a / b
    if math.isinf(result) or math.isnan(result):
        return fallback
    return result


def safe_sqrt(x: float, fallback: float = 0.0) -> float:
    """
    安全的平方根，处理负数输入。
    """
    if x < 0:
        if x > -1e-12:
            return 0.0
        return fallback
    return math.sqrt(x)


def check_boundary(value: float, lower: float, upper: float, name: str = "value") -> float:
    """
    边界条件检查与裁剪，确保数值在合法区间内。
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} is NaN or Inf: {value}")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def compute_gcd(a: int, b: int) -> int:
    """
    欧几里得算法求最大公约数。

    在数据分块对齐中，GCD 用于确定最优块大小：
        block_size = gcd(total_size, memory_limit)
    以保证内存页对齐和最小 I/O 碎片。
    """
    while b:
        a, b = b, a % b
    return a


def entropy_of_distribution(probs: List[float]) -> float:
    """
    计算离散概率分布的香农熵：
        H(P) = - Σ p_i · log_2(p_i)

    在数据分区评估中，熵衡量数据分布的均匀程度。理想均匀分布时
    H_max = log_2(n)，熵越接近最大值表示分区越均衡。
    """
    h = 0.0
    for p in probs:
        if p > 1e-15:
            h -= p * math.log2(p)
    return h


def kldivergence(p: List[float], q: List[float]) -> float:
    """
    KL 散度：D_KL(P || Q) = Σ p_i · log(p_i / q_i)

    用于衡量采样分布与真实分布之间的差异，指导外排序自适应采样。
    """
    d = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-15 and qi > 1e-15:
            d += pi * math.log(pi / qi)
    return d
