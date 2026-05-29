"""
data_compression.py
===================
高频金融数据的字典编码压缩与模式识别

本模块基于以下种子项目融合:
- 278_dictionary_code: 字典编码 → 高频 tick 数据的模式压缩与索引

核心数学模型:
--------------
1.  字典编码 (Dictionary Encoding):
    对高频交易数据流 {d_1, d_2, ..., d_N}, 其中每个 d_i 为离散化的市场状态向量:
        d_i = (ΔP_i, V_i, side_i, type_i)
    将唯一的状态模式收集为字典 D = {w_1, w_2, ..., w_M}, M ≪ N.
    原始数据编码为索引序列:
        encoded = [ idx(d_i) ]_{i=1}^N
    压缩比:
        CR = (N * |d|) / (M * |w| + N * log_2(M))
    其中 |d| 为原始记录大小, |w| 为字典条目大小.

2.  游程编码 (Run-Length Encoding) 增强:
    高频数据中相邻事件往往相同 (如连续限价单).
    对索引序列应用 RLE:
        (value, count) 对序列
    进一步压缩.

3.  信息熵分析:
    编码后的熵率:
        H = - Σ_j p_j log_2(p_j)
    其中 p_j 为字典项 j 的出现频率.
    理论最小编码长度:  N * H bits.
    字典大小 M 的选择权衡:
        M 过小 → 模式区分度低, 信息损失大
        M 过大 → 字典开销大, 压缩比下降

4.  模式距离度量:
    对两个模式 w_a, w_b, 采用加权 L1 距离:
        d(w_a, w_b) = Σ_k α_k |w_a^{(k)} - w_b^{(k)}|
    其中 α_k 为各维度的权重 (价格变化权重高, 时间戳权重低).
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TickPattern:
    """离散化的市场状态模式."""
    price_change_bucket: int
    volume_bucket: int
    side_code: int
    type_code: int

    def __hash__(self) -> int:
        return hash((self.price_change_bucket, self.volume_bucket,
                     self.side_code, self.type_code))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TickPattern):
            return False
        return (self.price_change_bucket == other.price_change_bucket and
                self.volume_bucket == other.volume_bucket and
                self.side_code == other.side_code and
                self.type_code == other.type_code)


class DictionaryEncoder:
    """
    高频数据字典编码器.
    """

    def __init__(self,
                 price_tick_size: float = 0.01,
                 volume_bucket_size: int = 5,
                 max_price_buckets: int = 21,
                 max_volume_buckets: int = 20):
        self.price_tick_size = price_tick_size
        self.volume_bucket_size = volume_bucket_size
        self.max_price_buckets = max_price_buckets
        self.max_volume_buckets = max_volume_buckets

        self.dictionary: Dict[TickPattern, int] = {}
        self.reverse_dict: Dict[int, TickPattern] = {}
        self.frequencies: Dict[int, int] = {}
        self.next_id = 1

    def _quantize(self, price_change: float, volume: int,
                  side: int, order_type: int) -> TickPattern:
        """将原始数据量化为模式."""
        pc_bucket = int(np.clip(
            np.round(price_change / self.price_tick_size),
            -self.max_price_buckets // 2,
            self.max_price_buckets // 2
        ))
        vol_bucket = int(np.clip(
            volume // self.volume_bucket_size,
            0,
            self.max_volume_buckets - 1
        ))
        return TickPattern(
            price_change_bucket=pc_bucket,
            volume_bucket=vol_bucket,
            side_code=int(side),
            type_code=int(order_type)
        )

    def build_dictionary(self,
                         price_changes: np.ndarray,
                         volumes: np.ndarray,
                         sides: np.ndarray,
                         types: np.ndarray) -> List[int]:
        """
        构建字典并编码数据.

        Returns
        -------
        encoded : List[int]
            编码后的索引序列.
        """
        n = len(price_changes)
        encoded = []

        for i in range(n):
            pattern = self._quantize(price_changes[i], volumes[i],
                                     sides[i], types[i])
            if pattern not in self.dictionary:
                idx = self.next_id
                self.next_id += 1
                self.dictionary[pattern] = idx
                self.reverse_dict[idx] = pattern
                self.frequencies[idx] = 0
            else:
                idx = self.dictionary[pattern]

            self.frequencies[idx] += 1
            encoded.append(idx)

        return encoded

    def encode_sequence(self,
                        price_changes: np.ndarray,
                        volumes: np.ndarray,
                        sides: np.ndarray,
                        types: np.ndarray) -> List[int]:
        """使用已有字典编码新序列."""
        n = len(price_changes)
        encoded = []
        for i in range(n):
            pattern = self._quantize(price_changes[i], volumes[i],
                                     sides[i], types[i])
            idx = self.dictionary.get(pattern, 0)
            encoded.append(idx)
        return encoded

    def run_length_encode(self, indices: List[int]) -> List[Tuple[int, int]]:
        """
        游程编码.
        将 [1,1,1,2,2,3,1,1] 编码为 [(1,3), (2,2), (3,1), (1,2)].
        """
        if not indices:
            return []

        rle = []
        current = indices[0]
        count = 1
        for idx in indices[1:]:
            if idx == current:
                count += 1
            else:
                rle.append((current, count))
                current = idx
                count = 1
        rle.append((current, count))
        return rle

    def compute_entropy(self) -> float:
        """
        计算当前编码的 Shannon 熵 (bits/symbol).
        """
        total = sum(self.frequencies.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for freq in self.frequencies.values():
            if freq > 0:
                p = freq / total
                entropy -= p * np.log2(p)
        return entropy

    def compression_ratio(self, original_size_bytes: int) -> float:
        """
        估算压缩比.
        original_size_bytes: 原始数据字节数.
        """
        dict_size = len(self.dictionary)
        # 字典条目大小: 4个int = 16 bytes
        dict_bytes = dict_size * 16
        # 编码序列: N * log2(M) bits = N * log2(M) / 8 bytes
        total_symbols = sum(self.frequencies.values())
        encoded_bytes = total_symbols * max(1, int(np.ceil(np.log2(dict_size + 1) / 8.0)))
        compressed = dict_bytes + encoded_bytes
        if compressed == 0:
            return 1.0
        return original_size_bytes / compressed

    def pattern_distance(self, idx_a: int, idx_b: int) -> float:
        """
        两个模式之间的加权 L1 距离.
        """
        if idx_a not in self.reverse_dict or idx_b not in self.reverse_dict:
            return np.inf

        pa = self.reverse_dict[idx_a]
        pb = self.reverse_dict[idx_b]

        weights = [1.0, 0.3, 0.5, 0.5]
        dist = (weights[0] * abs(pa.price_change_bucket - pb.price_change_bucket)
                + weights[1] * abs(pa.volume_bucket - pb.volume_bucket)
                + weights[2] * abs(pa.side_code - pb.side_code)
                + weights[3] * abs(pa.type_code - pb.type_code))
        return dist
