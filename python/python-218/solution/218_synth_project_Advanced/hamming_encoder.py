"""
hamming_encoder.py
==================
基于 Hamming 编码思想的互补 pivot 跟踪与编码.

数学背景
--------
Hamming(7,4) 码: 将 4 位信息编码为 7 位码字, 可纠正 1 位错误.
    生成矩阵 G ∈ R^{7×4}: c = G · m (mod 2)
    校验矩阵 H ∈ R^{3×7}: H · c^T = 0 (mod 2)

在互补问题中的应用 (创新):
    将 pivot 操作的序列编码为二进制串, 利用 Hamming 距离
    来度量两条 pivot 路径的差异. 这在参数化 VI 中用于:
        1. 跟踪解路径的分支点 (bifurcation)
        2. 检测 pivot 序列的循环 (cycling detection)
        3. 编码活动集的变化历史

关键公式
--------
Hamming 距离: d_H(a, b) = Σ_i |a_i - b_i|  (逐位异或的权重)
最小距离:     d_min = min_{c≠c'} d_H(c, c') = 3  (对 Hamming(7,4))
纠错能力:     t = floor((d_min - 1) / 2) = 1

伴随 VI 的参数编码:
    将参数 t ∈ [0,1] 离散化为 k 位二进制, 用 Hamming 码保护
    防止数值扰动导致的分支选择错误.

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


class HammingCode:
    """
    Hamming(7,4) 码的实现.

    生成矩阵 G (7×4):
        G = [1 1 0 1]
            [1 0 1 1]
            [1 0 0 0]
            [0 1 1 1]
            [0 1 0 0]
            [0 0 1 0]
            [0 0 0 1]

    校验矩阵 H (3×7):
        H = [1 0 1 0 1 0 1]
            [0 1 1 0 0 1 1]
            [0 0 0 1 1 1 1]

    编码: c = G · m (mod 2)
    校验: s = H · c (mod 2)  (伴随式 syndrome)
    纠错: 若 s ≠ 0, s 的二进制表示指出错误位.
    """

    # 生成矩阵
    G = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=int)

    # 校验矩阵
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=int)

    @classmethod
    def encode(cls, message: np.ndarray) -> np.ndarray:
        """
        编码 4 位信息为 7 位码字.

        c = G · m (mod 2)

        Parameters
        ----------
        message : ndarray (4,)
            4 位二进制信息

        Returns
        -------
        codeword : ndarray (7,)
            7 位码字
        """
        m = np.asarray(message, dtype=int) % 2
        return (cls.G @ m) % 2

    @classmethod
    def syndrome(cls, received: np.ndarray) -> np.ndarray:
        """
        计算伴随式 (syndrome).

        s = H · r (mod 2)

        若 s = 0, 无错误;
        否则 s 给出错误位置的二进制指示.
        """
        r = np.asarray(received, dtype=int) % 2
        return (cls.H @ r) % 2

    @classmethod
    def decode(cls, received: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """
        译码: 纠正最多 1 位错误.

        Returns
        -------
        message : ndarray (4,)
            译码后的信息
        error_position : int
            错误位置 (-1 表示无错误)
        corrected : bool
            是否成功纠正
        """
        r = np.asarray(received, dtype=int) % 2
        s = cls.syndrome(r)
        error_pos = -1
        corrected = True

        if np.any(s != 0):
            # 将 syndrome 转为位置索引
            # H 的列对应位置: 列 j 的二进制值 = j+1
            pos_val = s[0] * 4 + s[1] * 2 + s[2] * 1
            if 1 <= pos_val <= 7:
                error_pos = pos_val - 1
                r[error_pos] = (r[error_pos] + 1) % 2
            else:
                corrected = False  # 多位错误, 无法纠正

        # 提取信息位 (G 的前 4 行中, 单位阵的列对应信息位)
        # 信息位在第 2, 4, 5, 6 位 (0-indexed)
        message = r[[2, 4, 5, 6]]
        return message, error_pos, corrected

    @classmethod
    def hamming_distance(cls, a: np.ndarray, b: np.ndarray) -> int:
        """计算两个码字的 Hamming 距离."""
        return int(np.sum(np.asarray(a) != np.asarray(b)))

    @classmethod
    def all_codewords(cls) -> np.ndarray:
        """生成所有 16 个合法码字."""
        codewords = []
        for i in range(16):
            m = np.array([(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
            codewords.append(cls.encode(m))
        return np.array(codewords)


class PivotSequenceEncoder:
    """
    互补 pivot 序列的编码器.

    在 Lemke 算法或参数化 VI 中, pivot 操作序列可以编码为:
        - 选择哪个变量进入基 (entering variable): 2 位
        - 选择哪个变量离开基 (leaving variable): 2 位
    每个 pivot 步用 4 位表示, 然后用 Hamming(7,4) 编码为 7 位.

    通过比较两条 pivot 路径的 Hamming 距离, 可以:
        1. 检测分支点 (距离突增)
        2. 检测循环 (距离为 0 的周期)
        3. 度量参数扰动的敏感度
    """

    def __init__(self, max_pivots: int = 100):
        self.max_pivots = max_pivots
        self.history: List[np.ndarray] = []
        self.encoded_history: List[np.ndarray] = []

    def record_pivot(self, enter_var: int, leave_var: int, n_vars: int = 16) -> np.ndarray:
        """
        记录一个 pivot 操作.

        Parameters
        ----------
        enter_var : int
            进入基的变量索引
        leave_var : int
            离开基的变量索引
        n_vars : int
            变量总数 (用于归一化)

        Returns
        -------
        encoded : ndarray (7,)
            Hamming 编码后的 pivot
        """
        # 将变量索引量化为 2 位 (0-3)
        enter_bits = np.array([(enter_var >> 1) & 1, enter_var & 1], dtype=int)
        leave_bits = np.array([(leave_var >> 1) & 1, leave_var & 1], dtype=int)
        message = np.concatenate([enter_bits, leave_bits]) % 2

        encoded = HammingCode.encode(message)
        self.history.append(message)
        self.encoded_history.append(encoded)
        return encoded

    def detect_cycling(self, window_size: int = 10) -> bool:
        """
        检测 pivot 序列是否出现循环.

        方法: 计算滑动窗口内的自 Hamming 距离, 若存在零距离
        的周期子序列, 则判定为循环.
        """
        if len(self.encoded_history) < 2 * window_size:
            return False
        recent = self.encoded_history[-window_size:]
        for lag in range(1, window_size // 2):
            is_cyclic = True
            for i in range(window_size - lag):
                d = HammingCode.hamming_distance(recent[i], recent[i + lag])
                if d > 0:
                    is_cyclic = False
                    break
            if is_cyclic:
                return True
        return False

    def path_distance(self, other_history: List[np.ndarray]) -> int:
        """
        计算两条 pivot 路径的总 Hamming 距离.
        """
        min_len = min(len(self.encoded_history), len(other_history))
        total_dist = 0
        for i in range(min_len):
            total_dist += HammingCode.hamming_distance(
                self.encoded_history[i], other_history[i]
            )
        return total_dist

    def bifurcation_detection(self, threshold: int = 3) -> List[int]:
        """
        检测分支点: 相邻 pivot 的 Hamming 距离突增.

        Returns
        -------
        bifurcation_indices : list of int
            分支点的索引
        """
        bifurcations = []
        for i in range(1, len(self.encoded_history)):
            d = HammingCode.hamming_distance(
                self.encoded_history[i - 1], self.encoded_history[i]
            )
            if d >= threshold:
                bifurcations.append(i)
        return bifurcations
