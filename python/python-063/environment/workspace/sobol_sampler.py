
import numpy as np


class SobolSequence:


    POLY = np.array([
        1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
        61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
        193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
        213, 191, 253, 203, 211, 239, 247, 285, 369, 299,
    ], dtype=np.int32)


    V_INIT = np.zeros((40, 30), dtype=np.int32)


    _raw_v = {
        (0, 0): 1,
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
        self.V = np.zeros((self.dim, self.max_bits), dtype=np.int32)
        self.V[:, 0] = 1


        init_vals = [
            [],
            [],
            [1, 3],
            [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3],
        ]


        for i in range(2, self.dim + 1):
            poly = self.POLY[i - 1]

            m = 0
            j = poly
            while j > 0:
                j = j // 2
                m += 1
            m -= 1


            includ = []
            j = poly
            for _ in range(m):
                j2 = j // 2
                includ.append((j != 2 * j2))
                j = j2
            includ.reverse()


            for k in range(min(m, self.max_bits)):
                if k < len(self._get_preset_v(i)):
                    self.V[i - 1, k] = self._get_preset_v(i)[k]
                else:

                    if k >= m:
                        newv = self.V[i - 1, k - m]
                        l = 1
                        for idx in range(m):
                            l = 2 * l
                            if includ[idx]:
                                newv = newv ^ (l * self.V[i - 1, k - idx - 1])
                        self.V[i - 1, k] = newv


            for k in range(m, self.max_bits):
                newv = self.V[i - 1, k - m]
                l = 1
                for idx in range(m):
                    l = 2 * l
                    if idx < len(includ) and includ[idx]:
                        newv = newv ^ (l * self.V[i - 1, k - idx - 1])
                self.V[i - 1, k] = newv


        l = 1
        for j in range(self.max_bits - 2, -1, -1):
            l = 2 * l
            self.V[:, j] = self.V[:, j] * l

        self.recipd = 1.0 / (2.0 * l)

    def _get_preset_v(self, dim: int):
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
        if n <= 0:
            return 1
        bit = 1
        while (n & 1) != 0:
            n = n >> 1
            bit += 1
        return bit

    def generate(self, n: int) -> np.ndarray:
        result = np.zeros((n, self.dim), dtype=np.float64)
        for i in range(n):
            result[i, :] = self._next()
        return result

    def _next(self) -> np.ndarray:
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

    def __init__(self, dim: int = 2):
        self.dim = dim

    def sample(self, n: int) -> np.ndarray:
        result = np.zeros((n, self.dim), dtype=np.float64)
        for d in range(self.dim):
            perm = np.random.permutation(n)
            result[:, d] = (perm + np.random.rand(n)) / n
        return result
