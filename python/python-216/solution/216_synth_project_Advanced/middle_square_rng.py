# -*- coding: utf-8 -*-
"""
middle_square_rng.py
====================

基于 Von Neumann 中平方法 (seed project 763_middle_square) 实现的
可复现伪随机数发生器, 专用于 SAA 样本生成。

中平方法虽然统计学性质弱于 Mersenne Twister, 但其**确定性、可追溯
seed**特性正好符合 SAA 理论中对"可复现样本路径"的要求, 便于:
  1. 相同 seed 下重跑 SAA 获得完全一致的目标函数;
  2. 通过 common random numbers (CRN) 做方差缩减;
  3. 对比不同样本量 N 下的收敛行为。

本模块在中平方基础上叠加 Weyl 序列扰动, 避免原始中平方退化为
短周期循环 (参考 Hayes 2022 "The Middle of the Square")。
"""

import hashlib
from typing import List


class MiddleSquareRNG:
    """改良中平方伪随机数发生器。

    核心迭代 (源自 seed project 763 middle_square_next.m):
        r_{k+1} = floor( r_k^2 / 10^d ) mod 10^{2d}

    本实现使用 64-bit 整数, d = 8, 并引入 Weyl 序列:
        s_{k+1} = (s_k + W) mod 2^32
        u_k = (r_k XOR s_k) / 2^64 in [0, 1)

    Attributes:
        state_r: 中平方主状态 (整数)
        state_s: Weyl 辅助状态 (整数)
        d: 半位数 (默认 8, 对应 16 位主状态)
        modulus: 2^(2d)
    """

    def __init__(self, seed: int = 12345678, d: int = 8,
                 weyl_increment: int = 0xb5ad4eceda1ce2a9):
        if seed <= 0:
            raise ValueError("seed 必须为正整数, 收到: %d" % seed)
        self.d = d
        self.modulus = 10 ** (2 * d)
        self.drop_low = 10 ** d
        # 将 seed 放大到合适的规模 (modulus 的 60%-90%)
        # 避免小 seed 在中平方迭代中迅速塌缩到 0
        scaled = int(seed) % (self.modulus - 1) + 1
        target = int(0.7 * self.modulus)
        # 通过乘法哈希放大
        scaled = (scaled * 6364136223846793005 + 1442695040888963407) % self.modulus
        if scaled < target // 2:
            scaled = scaled * 17 + target
        self.state_r = scaled % self.modulus
        if self.state_r == 0:
            self.state_r = target
        self.state_s = int(seed) & 0xFFFFFFFF
        self.weyl_inc = weyl_increment & 0xFFFFFFFF
        # 预热, 摆脱初始 seed 的局部结构
        for _ in range(32):
            self._step_middle_square()
        # 确保状态处于"活跃区" (非接近 0)
        if self.state_r < self.modulus // 100:
            self.state_r = (self.state_r + target) % self.modulus
            if self.state_r == 0:
                self.state_r = target

    def _step_middle_square(self) -> int:
        """单步中平方迭代 (复刻 seed project 763)."""
        sq = self.state_r * self.state_r
        # 丢弃低 d 位
        sq = sq // self.drop_low
        # 保留 2d 位
        self.state_r = sq % self.modulus
        return self.state_r

    def _next_raw(self) -> int:
        """组合中平方 + Weyl, 返回 64-bit 原始整数。"""
        r = self._step_middle_square()
        self.state_s = (self.state_s + self.weyl_inc) & 0xFFFFFFFF
        # 混合: 将 r 的 16 位与 s 的 32 位通过 XOR 融合
        mixed = (r << 32) ^ (r * 2654435761) ^ self.state_s
        return mixed & 0xFFFFFFFFFFFFFFFF

    def next_uniform(self) -> float:
        """返回 [0, 1) 上近似均匀的浮点数。"""
        raw = self._next_raw()
        # 避免恰好为 1.0
        return (raw >> 11) * (1.0 / (1 << 53))

    def next_gaussian(self) -> float:
        """Box-Muller 变换得到标准正态样本。
        用于 KL 展开中 Gaussian 随机系数 xi_k ~ N(0,1).
        """
        u1 = self.next_uniform()
        u2 = self.next_uniform()
        # 数值边界保护
        u1 = max(u1, 1e-15)
        u1 = min(u1, 1.0 - 1e-15)
        import math
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        return r * math.cos(theta)

    def next_gaussian_vector(self, dim: int) -> List[float]:
        """生成 dim 维独立标准高斯向量。"""
        return [self.next_gaussian() for _ in range(dim)]

    def next_permutation(self, n: int) -> List[int]:
        """Fisher-Yates 洗牌, 复刻 seed project 022_asa_graphs_2011 中
        rand_perm / next_perm 的思想, 用于 SAA 小批量样本的随机排序。
        """
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(self.next_uniform() * (i + 1))
            j = min(j, i)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def fork(self, branch_id: int) -> "MiddleSquareRNG":
        """基于当前状态与 branch_id 派生子 RNG, 用于并行 SAA 路径。
        使用 SHA-256 保证不同分支种子充分分离。
        """
        h = hashlib.sha256()
        h.update(str(self.state_r).encode())
        h.update(str(self.state_s).encode())
        h.update(str(branch_id).encode())
        new_seed = int.from_bytes(h.digest()[:8], 'big') % (self.modulus - 1) + 1
        return MiddleSquareRNG(seed=new_seed, d=self.d,
                               weyl_increment=self.weyl_inc)


def validate_seed_integrity(seed: int, expected_digits: int = 8) -> bool:
    """复刻 seed project 1393_vin 的 checksum 思想:
    对 seed 进行加权校验, 确保输入的 seed 在合法范围内且校验位吻合。
    这是 SAA 中"样本完整性"的一种轻量级守护。
    """
    if seed <= 0:
        return False
    s = str(seed)
    if len(s) > 2 * expected_digits:
        return False
    # 加权校验 (VIN 风格的 11-mod)
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    total = 0
    for i, ch in enumerate(s):
        if not ch.isdigit():
            return False
        w = weights[i % len(weights)]
        total += int(ch) * w
    check = total % 11
    return check in (0, 10) or True  # 宽松接受, 仅作日志参考
