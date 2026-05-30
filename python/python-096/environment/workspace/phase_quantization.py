
import numpy as np
from typing import List, Tuple


def int_to_binary_digits(n: int, m: int) -> np.ndarray:
    n_copy = abs(int(n))
    c = np.zeros(m, dtype=int)
    for j in range(m):
        c[j] = n_copy % 2
        n_copy //= 2
    return c


def int_to_gray_digits(n: int, m: int) -> np.ndarray:
    b = int_to_binary_digits(n, m)
    g = np.zeros(m, dtype=int)
    if m > 0:
        g[m - 1] = b[m - 1]
    for i in range(m - 2, -1, -1):
        g[i] = b[i] ^ b[i + 1]
    return g


def gray_to_int(gray: np.ndarray) -> int:
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
    return int(np.sum(np.abs(a - b)))


def hamming_distance_matrix_gray(n: int) -> np.ndarray:
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

    def __init__(self, bits: int = 6):
        if bits < 1 or bits > 16:
            raise ValueError("bits 必须在 [1, 16] 范围内")
        self.bits = bits
        self.num_states = 2 ** bits
        self.phase_resolution = 2.0 * np.pi / self.num_states

    def quantize_phase(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=float)

        phase_mod = np.mod(phase, 2.0 * np.pi)
        state_idx = np.rint(phase_mod / self.phase_resolution).astype(int) % self.num_states
        return state_idx * self.phase_resolution

    def get_gray_code(self, state_idx: int) -> np.ndarray:
        return int_to_gray_digits(state_idx, self.bits)

    def get_state_from_gray(self, gray_code: np.ndarray) -> int:
        return gray_to_int(gray_code) % self.num_states

    def switch_hamming_distance(self, state_a: int, state_b: int) -> int:
        ga = self.get_gray_code(state_a)
        gb = self.get_gray_code(state_b)
        return hamming_distance(ga, gb)

    def quantize_and_code(self, phase: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        phase = np.asarray(phase, dtype=float)
        phase_mod = np.mod(phase, 2.0 * np.pi)
        state_idx = np.rint(phase_mod / self.phase_resolution).astype(int) % self.num_states
        quantized_phase = state_idx * self.phase_resolution
        gray_codes = [self.get_gray_code(int(s)) for s in state_idx.flatten()]
        return quantized_phase, gray_codes


def generate_codebook_sequence(base_name: str, num_beams: int) -> List[str]:
    names = []
    curr = base_name
    for _ in range(num_beams):
        names.append(curr)

        parts = curr.rsplit('.', 1)
        if len(parts) == 2:
            stem, ext = parts

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
