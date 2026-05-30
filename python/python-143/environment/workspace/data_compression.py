
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TickPattern:
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
        n = len(price_changes)
        encoded = []
        for i in range(n):
            pattern = self._quantize(price_changes[i], volumes[i],
                                     sides[i], types[i])
            idx = self.dictionary.get(pattern, 0)
            encoded.append(idx)
        return encoded

    def run_length_encode(self, indices: List[int]) -> List[Tuple[int, int]]:
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
        dict_size = len(self.dictionary)

        dict_bytes = dict_size * 16

        total_symbols = sum(self.frequencies.values())
        encoded_bytes = total_symbols * max(1, int(np.ceil(np.log2(dict_size + 1) / 8.0)))
        compressed = dict_bytes + encoded_bytes
        if compressed == 0:
            return 1.0
        return original_size_bytes / compressed

    def pattern_distance(self, idx_a: int, idx_b: int) -> float:
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
