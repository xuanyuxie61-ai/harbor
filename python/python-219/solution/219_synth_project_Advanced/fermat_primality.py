"""
fermat_primality.py
===================

Fermat 素性检验与模幂运算模块。

融合种子项目:
  - 912_prime_fermat : Fermat 素性检验 (概率性素数判定)

在最优控制框架中, 该模块用于:
  1. 验证空间离散化网格维度是否为素数 (用于某些谱方法的最优选择)
  2. 在构造循环群上的 Fourier 变换时, 需要素数长度
  3. 为随机采样器提供素数模数的哈希函数
  4. 验证时间步数 N 是否为素数 (用于某些高精度积分格式)

数学公式:
---------
1. Fermat 小定理:
     若 p 为素数且 gcd(a, p) = 1, 则  a^{p-1} ≡ 1 (mod p)

2. 快速模幂 (平方-乘法):
     a^n mod m = product of a^{2^k} mod m  (对 n 的二进制位)

3. 概率性检验:
     重复 k 次, 每次随机选取 a in [2, n-2],
     若所有 a^{n-1} mod n == 1, 则 n 为"可能素数"

4. Carmichael 数:
     某些合数 n 对所有 a 满足 a^{n-1} ≡ 1 (mod n),
     如 561 = 3 × 11 × 17, 65161 等.
"""

from __future__ import annotations

import math
from typing import List, Tuple


# ===========================================================================
# 1. 快速模幂 (来自 912_prime_fermat/prime_power)
# ===========================================================================
def prime_power(a: int, n: int, m: int) -> int:
    """快速模幂: 计算 a^n mod m.

    算法: 平方-乘法 (binary exponentiation)
        结果 = 1
        base = a mod m
        while n > 0:
            if n & 1:  result = (result * base) mod m
            base = (base * base) mod m
            n >>= 1

    Parameters
    ----------
    a : int  底数.
    n : int  指数 (非负).
    m : int  模数 (正).

    Returns
    -------
    int
        a^n mod m.
    """
    if m <= 0:
        raise ValueError(f"模数 m={m} 必须为正")
    if n < 0:
        raise ValueError(f"指数 n={n} 必须非负")
    if m == 1:
        return 0

    result = 1
    base = a % m
    while n > 0:
        if n & 1:
            result = (result * base) % m
        base = (base * base) % m
        n >>= 1
    return result


# ===========================================================================
# 2. 最大公约数 (Euclidean)
# ===========================================================================
def gcd(a: int, b: int) -> int:
    """Euclidean 算法求最大公约数."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


# ===========================================================================
# 3. Fermat 素性检验 (来自 912_prime_fermat/is_prime_fermat)
# ===========================================================================
def is_prime_fermat(n: int, k: int = 10, seed_offset: int = 0) -> bool:
    """Fermat 素性检验 (概率性).

    算法:
        1. 边界情况: n <= 1 返回 False, n <= 3 返回 True
        2. 重复 k 次:
            a. 随机选取 a in [2, n-2]  (使用确定性伪随机)
            b. 若 gcd(n, a) != 1, 返回 False
            c. 若 a^{n-1} mod n != 1, 返回 False (Fermat 检验失败)
        3. 若所有 k 次检验通过, 返回 True (n 为"可能素数")

    Parameters
    ----------
    n : int
        待检验的整数.
    k : int
        检验次数 (越大越可靠, 但越慢).
    seed_offset : int
        伪随机种子偏移 (用于可重复性).

    Returns
    -------
    bool
        True 表示 n 通过 Fermat 检验 (可能素数),
        False 表示 n 为合数.
    """
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n == 4:
        return False

    # 确定性伪随机 (简单 LCG)
    state = (n * 1103515245 + 12345 + seed_offset) & 0x7FFFFFFF

    for _ in range(k):
        # 生成 [2, n-2] 内的伪随机数
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        a = 2 + (state % (n - 3))

        if gcd(n, a) != 1:
            return False

        if prime_power(a, n - 1, n) != 1:
            return False

    return True


# ===========================================================================
# 4. 确定性 Miller-Rabin (增强版)
# ===========================================================================
def is_prime_miller_rabin(n: int, k: int = 10) -> bool:
    """Miller-Rabin 素性检验 (比 Fermat 更可靠).

    算法:
        1. 写 n - 1 = 2^s * d, 其中 d 为奇数
        2. 重复 k 次:
            a. 随机选取 a in [2, n-2]
            b. 计算 x = a^d mod n
            c. 若 x == 1 or x == n-1, 继续下一轮
            d. 重复 s-1 次: x = x^2 mod n
               若 x == n-1, 跳出内层循环
            e. 若未跳出, 返回 False (合数)
        3. 返回 True (可能素数)

    对 n < 3,317,044,064,679,887,385,961,981,
    使用前 13 个素数作为基底可确定性判定.
    """
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # 分解 n - 1 = 2^s * d
    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    # 小素数基底 (确定性)
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases[: min(k, len(bases))]:
        if a >= n:
            continue
        x = prime_power(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = prime_power(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ===========================================================================
# 5. 素数生成器
# ===========================================================================
def next_prime(n: int) -> int:
    """返回 >= n 的最小素数."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not is_prime_miller_rabin(n, k=5):
        n += 2
    return n


def primes_in_range(lo: int, hi: int) -> List[int]:
    """返回 [lo, hi] 内的所有素数 (简单试除法)."""
    if lo > hi:
        return []
    result = []
    for n in range(max(2, lo), hi + 1):
        if is_prime_miller_rabin(n, k=5):
            result.append(n)
    return result


# ===========================================================================
# 6. 在最优控制中的应用: 网格维度验证
# ===========================================================================
def validate_grid_dimension(m: int, require_prime: bool = False) -> Tuple[bool, str]:
    """验证空间离散化网格维度.

    Parameters
    ----------
    m : int
        网格每边的节点数.
    require_prime : bool
        是否要求 m 为素数 (某些谱方法需要).

    Returns
    -------
    (valid, message) : (bool, str)
    """
    if m <= 1:
        return False, f"网格维度 m={m} 必须 >= 2"
    if require_prime and not is_prime_miller_rabin(m, k=10):
        return False, f"网格维度 m={m} 不是素数 (谱方法要求)"
    return True, f"网格维度 m={m} 合法"


def suggest_prime_grid(target_m: int) -> int:
    """建议的素数网格维度 (>= target_m)."""
    return next_prime(target_m)


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """Fermat 素性模块自检."""
    test_cases = [
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (17, True),
        (561, False),  # Carmichael 数 (Fermat 会误判, Miller-Rabin 不会)
        (104729, True),  # 第 10000 个素数
    ]
    print("[Fermat Primality] 测试:")
    for n, expected in test_cases:
        fermat_result = is_prime_fermat(n, k=10)
        mr_result = is_prime_miller_rabin(n, k=10)
        status = "OK" if mr_result == expected else "FAIL"
        print(
            f"  n={n:6d}  expected={expected}  "
            f"fermat={fermat_result}  miller-rabin={mr_result}  [{status}]"
        )

    # 网格验证
    valid, msg = validate_grid_dimension(7, require_prime=True)
    print(f"[Grid Validation] m=7: {msg} (valid={valid})")
    valid, msg = validate_grid_dimension(8, require_prime=True)
    print(f"[Grid Validation] m=8: {msg} (valid={valid})")
    suggested = suggest_prime_grid(10)
    print(f"[Grid Suggestion] target=10 -> suggested={suggested}")


if __name__ == "__main__":
    self_check()
