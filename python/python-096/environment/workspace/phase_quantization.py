"""
phase_quantization.py
=====================
数字移相器量化与编码控制模块

核心算法来源：
  - 485_gray_code_display：格雷码与汉明距离
  - 429_file_name_sequence：文件名序列递增

在电磁学波束赋形中的角色：
  1. 格雷码用于数字移相器的相位状态编码，确保相邻相位状态切换时
     仅改变最少的控制位，降低开关瞬态干扰（glitch）
  2. 汉明距离分析量化相位编码的连续性
  3. 文件名序列工具管理多组波束码本（codebook）输出
"""

import numpy as np
from typing import List, Tuple


def int_to_binary_digits(n: int, m: int) -> np.ndarray:
    """
    将整数 n 转换为其二进制表示（最低位在前）。

    来源：485_gray_code_display
    """
    n_copy = abs(int(n))
    c = np.zeros(m, dtype=int)
    for j in range(m):
        c[j] = n_copy % 2
        n_copy //= 2
    return c


def int_to_gray_digits(n: int, m: int) -> np.ndarray:
    """
    将整数 n 转换为 m 位格雷码（最低位在前）。

    来源：485_gray_code_display

    数学定义：
      设 n 的二进制为 b_{m-1} ... b_1 b_0，则格雷码 g_i 为：
        g_{m-1} = b_{m-1}
        g_i = b_i \oplus b_{i+1},  i = 0, ..., m-2

      性质：相邻整数的格雷码仅有一位不同（Hamming distance = 1）。
    """
    b = int_to_binary_digits(n, m)
    g = np.zeros(m, dtype=int)
    if m > 0:
        g[m - 1] = b[m - 1]
    for i in range(m - 2, -1, -1):
        g[i] = b[i] ^ b[i + 1]
    return g


def gray_to_int(gray: np.ndarray) -> int:
    """格雷码转整数。"""
    m = gray.size
    b = np.zeros(m, dtype=int)
    b[m - 1] = gray[m - 1]
    for i in range(m - 2, -1, -1):
        b[i] = b[i + 1] ^ gray[i]
    val = 0
    for i in range(m - 1, -1, -1):
        val = val * 2 + b[i]
    return val


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    计算两个二进制向量之间的汉明距离。

    来源：485_gray_code_display
    """
    return int(np.sum(np.abs(a - b)))


def hamming_distance_matrix_gray(n: int) -> np.ndarray:
    """
    计算 0..n 的格雷码汉明距离矩阵。

    来源：485_gray_code_display

    在天线阵列中，这用于分析：当相位从状态 i 切换到状态 j 时，
    有多少个移相器控制位需要翻转。格雷码保证相邻状态汉明距离为 1。
    """
    if n < 0:
        return np.zeros((0, 0), dtype=int)
    m = int(np.floor(np.log2(max(n, 1)))) + 1
    dg = np.zeros((n + 1, n + 1), dtype=int)
    for i in range(n + 1):
        gi = int_to_gray_digits(i, m)
        for j in range(n + 1):
            gj = int_to_gray_digits(j, m)
            dg[i, j] = hamming_distance(gi, gj)
    return dg


def hamming_distance_matrix_binary(n: int) -> np.ndarray:
    """计算二进制编码的汉明距离矩阵（用于对比）。"""
    if n < 0:
        return np.zeros((0, 0), dtype=int)
    m = int(np.floor(np.log2(max(n, 1)))) + 1
    db = np.zeros((n + 1, n + 1), dtype=int)
    for i in range(n + 1):
        bi = int_to_binary_digits(i, m)
        for j in range(n + 1):
            bj = int_to_binary_digits(j, m)
            db[i, j] = hamming_distance(bi, bj)
    return db


class DigitalPhaseShifter:
    """
    数字移相器模型。

    物理背景：
      相控阵中的数字移相器通常具有 B 位分辨率，提供 2^B 个离散相位状态：
        \phi_k = 2\pi k / 2^B,  k = 0, 1, ..., 2^B - 1

      使用格雷码编码控制状态可最小化状态切换时的瞬态跳变。
    """

    def __init__(self, bits: int = 6):
        if bits < 1 or bits > 16:
            raise ValueError("bits 必须在 [1, 16] 范围内")
        self.bits = bits
        self.num_states = 2 ** bits
        self.phase_resolution = 2.0 * np.pi / self.num_states

    def quantize_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        将连续相位量化为离散状态。

        参数：
            phase: 连续相位（弧度），任意形状
        返回：
            quantized: 量化后相位（弧度）
        """
        phase = np.asarray(phase, dtype=float)
        # 归一化到 [0, 2pi)
        phase_mod = np.mod(phase, 2.0 * np.pi)
        state_idx = np.rint(phase_mod / self.phase_resolution).astype(int) % self.num_states
        return state_idx * self.phase_resolution

    def get_gray_code(self, state_idx: int) -> np.ndarray:
        """获取给定状态的格雷码控制字。"""
        return int_to_gray_digits(state_idx, self.bits)

    def get_state_from_gray(self, gray_code: np.ndarray) -> int:
        """由格雷码反解状态索引。"""
        return gray_to_int(gray_code) % self.num_states

    def switch_hamming_distance(self, state_a: int, state_b: int) -> int:
        """计算两个状态间切换需要翻转的控制位数。"""
        ga = self.get_gray_code(state_a)
        gb = self.get_gray_code(state_b)
        return hamming_distance(ga, gb)

    def quantize_and_code(self, phase: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        量化相位并返回格雷码控制字。

        返回：
            quantized_phase: 量化相位
            gray_codes: 控制字列表
        """
        phase = np.asarray(phase, dtype=float)
        phase_mod = np.mod(phase, 2.0 * np.pi)
        state_idx = np.rint(phase_mod / self.phase_resolution).astype(int) % self.num_states
        quantized_phase = state_idx * self.phase_resolution
        gray_codes = [self.get_gray_code(int(s)) for s in state_idx.flatten()]
        return quantized_phase, gray_codes


def generate_codebook_sequence(base_name: str, num_beams: int) -> List[str]:
    """
    生成波束码本文件名序列。

    来源：429_file_name_sequence
    """
    names = []
    curr = base_name
    for _ in range(num_beams):
        names.append(curr)
        # 简单数字递增
        parts = curr.rsplit('.', 1)
        if len(parts) == 2:
            stem, ext = parts
            # 在 stem 末尾找数字
            import re
            m = re.search(r'(\d+)$', stem)
            if m:
                num_str = m.group(1)
                new_num = str(int(num_str) + 1).zfill(len(num_str))
                stem = stem[:m.start()] + new_num
                curr = f"{stem}.{ext}"
            else:
                curr = f"{stem}1.{ext}"
        else:
            m = re.search(r'(\d+)$', curr)
            if m:
                num_str = m.group(1)
                new_num = str(int(num_str) + 1).zfill(len(num_str))
                curr = curr[:m.start()] + new_num
            else:
                curr = curr + "1"
    return names
