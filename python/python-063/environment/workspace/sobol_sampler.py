"""
================================================================================
Sobol准随机序列生成模块
================================================================================

融合来源：
  - 1097_sobol:  Antonov-Saleev 算法的 Sobol 序列生成

科学应用：
  在集合卡尔曼滤波中，使用低偏差序列代替纯随机采样生成初始集合，
  以更高效地覆盖状态空间，减少蒙特卡洛方差。

核心公式：
  1. Sobol序列基于二进制表示和本原多项式:
     x_n = ⊕_{k=1}^s (i_k · v_k) / 2^k
     其中 i_k 是索引 n 的二进制位，v_k 是方向数。

  2. Gray码优化 (Antonov-Saleev):
     使用 Gray码 G(n) = n ⊕ (n >> 1) 减少每次迭代的位运算次数。

  3. 方向数初始化:
     基于40维以内的预计算本原多项式系数，
     通过递推关系 v_k = ⊕_{j=1}^m a_j · v_{k-j} · 2^j
================================================================================
"""

import numpy as np


class SobolSequence:
    """
    Sobol 准随机序列生成器。
    支持最高 40 维，基于 Antonov-Saleev Gray 码优化。
    """

    # 本原多项式系数（对应 1097_sobol 中的 poly 数组）
    POLY = np.array([
        1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
        61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
        193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
        213, 191, 253, 203, 211, 239, 247, 285, 369, 299,
    ], dtype=np.int32)

    # 初始方向数矩阵 V(40, 30)
    V_INIT = np.zeros((40, 30), dtype=np.int32)

    # 填充初始方向数（对应 i4_sobol.m 中的 v 初始化）
    _raw_v = {
        (0, 0): 1,  # 第1维全部初始化为1
    }

    def __init__(self, dim: int, max_bits: int = 30):
        if dim < 1 or dim > 40:
            raise ValueError("维度必须在 [1, 40] 范围内")
        self.dim = dim
        self.max_bits = max_bits
        self._initialize_direction_numbers()
        self.last_index = 0
        self.last_q = np.zeros(dim, dtype=np.int32)

    def _initialize_direction_numbers(self):
        """初始化方向数矩阵。"""
        self.V = np.zeros((self.dim, self.max_bits), dtype=np.int32)
        self.V[:, 0] = 1  # 第一列全为1

        # 第2维到第40维的初始值（对应MATLAB代码中的预计算值）
        init_vals = [
            [],  # dim 1 (already set)
            [],  # dim 2
            [1, 3],  # dim 3
            [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],  # dim 4-40, col 2
        ]

        # 更系统地填充（基于Bratley-Fox论文的递推）
        for i in range(2, self.dim + 1):
            poly = self.POLY[i - 1]
            # 计算多项式次数 m
            m = 0
            j = poly
            while j > 0:
                j = j // 2
                m += 1
            m -= 1

            # 提取系数
            includ = []
            j = poly
            for _ in range(m):
                j2 = j // 2
                includ.append((j != 2 * j2))
                j = j2
            includ.reverse()

            # 预计算前 m 个方向数（简化：使用递推生成）
            for k in range(min(m, self.max_bits)):
                if k < len(self._get_preset_v(i)):
                    self.V[i - 1, k] = self._get_preset_v(i)[k]
                else:
                    # 递推
                    if k >= m:
                        newv = self.V[i - 1, k - m]
                        l = 1
                        for idx in range(m):
                            l = 2 * l
                            if includ[idx]:
                                newv = newv ^ (l * self.V[i - 1, k - idx - 1])
                        self.V[i - 1, k] = newv

            # 填充剩余列
            for k in range(m, self.max_bits):
                newv = self.V[i - 1, k - m]
                l = 1
                for idx in range(m):
                    l = 2 * l
                    if idx < len(includ) and includ[idx]:
                        newv = newv ^ (l * self.V[i - 1, k - idx - 1])
                self.V[i - 1, k] = newv

        # 乘以2的幂次
        l = 1
        for j in range(self.max_bits - 2, -1, -1):
            l = 2 * l
            self.V[:, j] = self.V[:, j] * l

        self.recipd = 1.0 / (2.0 * l)

    def _get_preset_v(self, dim: int):
        """获取预计算的方向数初始值。"""
        presets = {
            3: [1, 3],
            4: [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
            5: [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
            6: [1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
            7: [1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
            8: [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
            9: [1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
            10: [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
        }
        return presets.get(dim, [1, 3])

    def _bit_lo0(self, n: int) -> int:
        """找到 n 的二进制表示中最低位0的位置（从1开始计数）。"""
        if n <= 0:
            return 1
        bit = 1
        while (n & 1) != 0:
            n = n >> 1
            bit += 1
        return bit

    def generate(self, n: int) -> np.ndarray:
        """
        生成 n 个 Sobol 准随机向量。

        返回:
            array of shape (n, dim), 每个元素在 [0, 1) 内。
        """
        result = np.zeros((n, self.dim), dtype=np.float64)
        for i in range(n):
            result[i, :] = self._next()
        return result

    def _next(self) -> np.ndarray:
        """生成下一个 Sobol 向量。"""
        self.last_index += 1
        key = self.last_index

        if key == 1:
            l = 1
            self.last_q = np.zeros(self.dim, dtype=np.int32)
        else:
            l = self._bit_lo0(key - 1)

        if l > self.max_bits:
            raise RuntimeError("Sobol序列调用次数超过最大值")

        self.last_q = self.last_q ^ self.V[:, l - 1]
        return self.last_q * self.recipd


class LatinHypercubeSampler:
    """
    拉丁超立方采样（LHS）作为Sobol的补充。
    用于生成空间分布均匀的代理站点位置。
    """

    def __init__(self, dim: int = 2):
        self.dim = dim

    def sample(self, n: int) -> np.ndarray:
        """
        生成 n 个 LHS 样本。
        """
        result = np.zeros((n, self.dim), dtype=np.float64)
        for d in range(self.dim):
            perm = np.random.permutation(n)
            result[:, d] = (perm + np.random.rand(n)) / n
        return result
