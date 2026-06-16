# -*- coding: utf-8 -*-
"""
prime_grid.py
=============

素数网格与共振 L 壳层选择模块.

本模块利用素数筛法识别磁层中的共振 L 壳层:
  - 素数筛 (Eratosthenes): 筛选素数 L 壳层
  - Goldbach 分区: 寻找 L 壳层的素数分解
  - 共振条件: 素数 L 壳层的特殊共振特性

物理背景:

磁层中的波-粒子共振条件涉及有理数比例:
  omega / Omega_ce = p / q  (p, q 为整数)

对于特定的谐波结构, 共振 L 壳层呈现离散分布.
我们使用素数筛法来识别这些"特殊"L 壳层.

数学性质:
  - 素数定理: pi(x) ~ x / ln(x)
  - 孪生素数: p, p+2 同时为素数
  - Goldbach 猜想: 每个大于 2 的偶数可表示为两个素数之和

参考文献:
  [1] Hardy, G.H. & Wright, E.M., "An Introduction to the Theory of
      Numbers", Oxford (2008)
  [2] Thorne, R.M. et al., "Energy dependence of radiation belt electrons",
      JGR (2013)
"""

import numpy as np
import physical_constants as pc


def sieve_of_eratosthenes(n_max):
    """
    Eratosthenes 素数筛.

    算法:
      1. 创建布尔数组 is_prime[2..n_max] = True
      2. 对于 p = 2, 3, ..., sqrt(n_max):
         若 is_prime[p] 为真, 标记所有 p^2, p^2+p, ... 为假
      3. 返回所有 is_prime[i] 为真的 i

    时间复杂度: O(n * log(log(n)))

    参数
    ----
    n_max : int
        筛选上限

    返回
    -------
    primes : list
        素数列表
    is_prime : ndarray
        素数标志数组
    """
    if n_max < 2:
        return [], np.array([])

    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(np.sqrt(n_max)) + 1):
        if is_prime[p]:
            # 标记 p 的所有倍数为非素数
            is_prime[p*p::p] = False

    primes = list(np.where(is_prime)[0])
    return primes, is_prime


def twin_primes(n_max):
    """
    寻找孪生素数对 (p, p+2).

    参数
    ----
    n_max : int
        筛选上限

    返回
    -------
    twins : list of tuple
        孪生素数对列表
    """
    primes, _ = sieve_of_eratosthenes(n_max)
    twins = []
    for i in range(len(primes) - 1):
        if primes[i+1] - primes[i] == 2:
            twins.append((primes[i], primes[i+1]))
    return twins


def goldbach_partition(n):
    """
    将偶数 n 分解为两个素数之和.

    Goldbach 猜想 (未证明, 但对所有已知偶数成立):
      n = p1 + p2  (p1, p2 为素数)

    参数
    ----
    n : int
        偶数

    返回
    -------
    partitions : list of tuple
        所有可能的素数对
    """
    if n < 4 or n % 2 != 0:
        return []

    primes, is_prime = sieve_of_eratosthenes(n)
    partitions = []
    for p in primes:
        if p > n // 2:
            break
        if is_prime[n - p]:
            partitions.append((p, n - p))
    return partitions


class PrimeResonanceGrid:
    """
    素数共振 L 壳层网格.

    选择满足素数条件的 L 壳层作为精细网格区域:
      - 素数 L: 高精度计算区域
      - 孪生 L: 特殊共振区域
      - 合数 L: 标准精度区域

    参数
    ----
    L_min, L_max : float
        L 范围
    n_base : int
        基础网格数
    """

    def __init__(self, L_min=1.5, L_max=7.5, n_base=61):
        self.L_min = L_min
        self.L_max = L_max
        self.n_base = n_base

        # 基础网格
        self.L_base = np.linspace(L_min, L_max, n_base)

        # 整数化 L 值 (用于素数判断)
        self.L_int = np.round(self.L_base).astype(int)
        self.L_int = np.clip(self.L_int, 2, 100)

        # 素数判断
        max_int = int(np.max(self.L_int))
        _, self.is_prime_array = sieve_of_eratosthenes(max_int)

        # 分类
        self.prime_mask = np.array([self.is_prime_array[int(L)] for L in self.L_int])
        self.twin_mask = self._identify_twin_resonances()

    def _identify_twin_resonances(self):
        """识别孪生共振位置."""
        twin_mask = np.zeros(self.n_base, dtype=bool)
        twins = twin_primes(100)
        twin_set = set()
        for p1, p2 in twins:
            twin_set.add(p1)
            twin_set.add(p2)

        for i, L in enumerate(self.L_int):
            if int(L) in twin_set:
                twin_mask[i] = True
        return twin_mask

    def get_refinement_mask(self):
        """
        获取加密网格掩码.

        返回
        -------
        mask : ndarray
            需要加密的网格点标志
        """
        # 素数 L 和孪生 L 需要加密
        return self.prime_mask | self.twin_mask

    def get_resonance_weights(self):
        """
        计算共振权重.

        素数 L 壳层获得更高的计算权重.

        返回
        -------
        weights : ndarray
            权重数组
        """
        weights = np.ones(self.n_base)
        weights[self.prime_mask] = 2.0  # 素数: 2 倍权重
        weights[self.twin_mask] = 3.0   # 孪生: 3 倍权重
        return weights

    def compute_resonance_spectrum(self, B_field=None):
        """
        计算共振频谱.

        对于每个 L 壳层, 计算其共振频率:
          omega_res = n * Omega_ce(L) / gamma

        参数
        ----
        B_field : ndarray, optional
            磁场强度

        返回
        -------
        omega_res : ndarray
            共振频率
        """
        if B_field is None:
            B_field = pc.dipole_field_magnitude(self.L_base)

        # 典型电子 (E = 1 MeV)
        gamma = pc.kinetic_to_lorentz(1.0)
        Omega_ce = pc.gyrofrequency(B_field)

        # n = 1 基频共振
        omega_res = Omega_ce / gamma

        # 素数 L 壳层增强
        weights = self.get_resonance_weights()
        omega_res *= weights

        return omega_res

    def summary(self):
        """打印摘要."""
        n_prime = np.sum(self.prime_mask)
        n_twin = np.sum(self.twin_mask)
        print(f"素数共振网格:")
        print(f"  L 范围: [{self.L_min:.1f}, {self.L_max:.1f}]")
        print(f"  总网格点: {self.n_base}")
        print(f"  素数 L 壳层: {n_prime}")
        print(f"  孪生 L 壳层: {n_twin}")


def self_test():
    """自检验证."""
    print("=" * 60)
    print("素数网格模块自检验证")
    print("=" * 60)

    # 素数筛
    primes, _ = sieve_of_eratosthenes(30)
    print(f"  30 以内的素数: {primes}")

    # 孪生素数
    twins = twin_primes(30)
    print(f"  30 以内的孪生素数: {twins}")

    # Goldbach 分解
    parts = goldbach_partition(20)
    print(f"  20 的 Goldbach 分解: {parts}")

    # 共振网格
    grid = PrimeResonanceGrid()
    grid.summary()

    return True


if __name__ == "__main__":
    self_test()
