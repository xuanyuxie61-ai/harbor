"""
randomness_engine.py
基于项目 1373_uniform (Park-Miller LCRG) 与 1374_unstable_ode 的
量子计算专用随机数引擎与伪随机序列生成器。

核心数学模型:
1. Park-Miller 线性同余生成器 (LCRG):
   X_{n+1} = (a * X_n) mod m,  a=16807, m=2^{31}-1
   Schrage 分解防溢出: a*X mod m = a(X mod q) - r*floor(X/q) mod m
   其中 q = floor(m/a) = 127773, r = m mod a = 2836

2. LCRG 跳跃公式 (Skip-ahead):
   X_N = a^N * X_0 + (a^N - 1)/(a-1) * b  (mod m)
   通过 power_mod (二进制快速幂, O(log N)) 与扩展欧几里得求解模逆元

3. 复数单位圆盘均匀采样 (用于量子振幅):
   z = r * exp(i*theta), r = sqrt(u), u ~ U[0,1], theta ~ U[0, 2*pi]

4. n维单位球面均匀采样 (用于量子态布洛赫球面):
   w = g / ||g||_2,  g ~ N(0, I_n)
"""

import numpy as np
from typing import Tuple, Optional


class QuantumRandomnessEngine:
    """
    量子计算专用伪随机数引擎。
    基于 Park-Miller 最小标准 LCRG，支持序列跳跃与并行子序列分割。
    """

    # Park-Miller 参数
    IA = 16807
    IM = 2147483647  # 2^31 - 1, 梅森素数
    IQ = 127773      # floor(IM / IA)
    IR = 2836        # IM mod IA
    AM = 1.0 / IM

    def __init__(self, seed: int = 1):
        if seed <= 0 or seed >= self.IM:
            raise ValueError(f"Seed must be in [1, {self.IM - 1}], got {seed}")
        self._seed = int(seed)
        self._initial_seed = int(seed)

    def reset(self) -> None:
        """重置到初始种子。"""
        self._seed = self._initial_seed

    def _advance(self) -> int:
        """Schrage 分解法前进一步，避免 32 位整数溢出。"""
        k = self._seed // self.IQ
        self._seed = self.IA * (self._seed - k * self.IQ) - k * self.IR
        if self._seed < 0:
            self._seed += self.IM
        return self._seed

    def uniform_01(self) -> float:
        """返回 [0,1] 区间上的伪随机双精度浮点数。"""
        return self._advance() * self.AM

    def uniform_ab(self, a: float, b: float) -> float:
        """返回 [a, b] 区间上的均匀分布随机数。"""
        if a >= b:
            raise ValueError(f"Invalid interval: a={a} >= b={b}")
        return a + (b - a) * self.uniform_01()

    def uniform_int(self, low: int, high: int) -> int:
        """返回 [low, high] 区间上的均匀分布整数。"""
        if low > high:
            raise ValueError(f"Invalid range: low={low} > high={high}")
        return low + int(self.uniform_01() * (high - low + 1)) % (high - low + 1)

    def uniform_disk(self) -> complex:
        """
        单位圆盘内均匀分布的复数采样。
        数学原理: 面积元 dA = r dr dθ, 为使分布均匀, 需 P(R<=r) = r^2,
        故 r = sqrt(u), u ~ U[0,1]。
        """
        u = self.uniform_01()
        theta = 2.0 * np.pi * self.uniform_01()
        r = np.sqrt(u)
        return complex(r * np.cos(theta), r * np.sin(theta))

    def uniform_sphere_nd(self, n: int) -> np.ndarray:
        """
        n维单位球面上均匀分布的向量采样。
        数学原理: n维标准正态分布具有球面对称性，归一化后即为球面均匀分布。
        w = g / ||g||_2, g ~ N(0, I_n)
        """
        if n <= 0:
            raise ValueError(f"Dimension n must be positive, got {n}")
        # 使用 Box-Muller 变换思想：通过 LCRG 生成正态分布近似
        # 这里为了效率直接用 numpy，但种子由本引擎控制
        np.random.seed(self._seed % (2**32))
        g = np.random.randn(n)
        self._advance()  # 同步状态
        norm = np.linalg.norm(g)
        if norm < 1e-15:
            g[0] = 1.0
            norm = 1.0
        return g / norm

    def power_mod(self, a: int, n: int, m: int) -> int:
        """
        二进制快速幂算法计算 (a^n) mod m。
        时间复杂度 O(log n)。
        """
        if m <= 0:
            raise ValueError("Modulus m must be positive")
        if n < 0:
            raise ValueError("Exponent n must be non-negative")
        result = 1 % m
        base = a % m
        exp = n
        while exp > 0:
            if exp & 1:
                result = (result * base) % m
            base = (base * base) % m
            exp >>= 1
        return result

    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        扩展欧几里得算法。
        返回 (g, x, y) 使得 a*x + b*y = g = gcd(a, b)。
        """
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = self.extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return (g, x, y)

    def mod_inverse(self, a: int, m: int) -> Optional[int]:
        """
        计算 a 在模 m 下的乘法逆元。
        若 gcd(a, m) != 1 则逆元不存在，返回 None。
        """
        g, x, _ = self.extended_gcd(a % m, m)
        if g != 1:
            return None
        return x % m

    def jump_ahead(self, n_steps: int) -> None:
        """
        LCRG 跳跃: 直接计算第 N 个状态而不迭代 N 次。
        X_N = a^N * X_0 * (a^N - 1)/(a-1) * b (mod m)
        对于 Park-Miller, b = 0, 故 X_N = a^N * X_0 mod m。
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        an = self.power_mod(self.IA, n_steps, self.IM)
        self._seed = (an * self._seed) % self.IM
        if self._seed == 0:
            self._seed = self._initial_seed

    def generate_sequence(self, length: int) -> np.ndarray:
        """生成指定长度的 [0,1] 伪随机序列。"""
        if length < 0:
            raise ValueError("Length must be non-negative")
        return np.array([self.uniform_01() for _ in range(length)], dtype=np.float64)


def box_muller_transform(u1: float, u2: float) -> Tuple[float, float]:
    """
    Box-Muller 变换: 将两个 [0,1] 均匀随机数转换为两个独立标准正态随机数。
    z1 = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    z2 = sqrt(-2*ln(u1)) * sin(2*pi*u2)
    边界处理: 当 u1 接近 0 时，用极小值 epsilon 替代避免 log(0)。
    """
    eps = 1e-15
    u1 = max(u1, eps)
    u1 = min(u1, 1.0 - eps)
    magnitude = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    z1 = magnitude * np.cos(angle)
    z2 = magnitude * np.sin(angle)
    return z1, z2


def quantum_random_hermitian(n: int, engine: QuantumRandomnessEngine) -> np.ndarray:
    """
    生成 n x n 的随机厄米矩阵 (用于量子哈密顿量模拟)。
    H = (A + A^dagger) / 2, 其中 A 的每个元素服从复高斯分布。
    """
    if n <= 0:
        raise ValueError("Matrix dimension must be positive")
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            u1, u2 = engine.uniform_01(), engine.uniform_01()
            z1, z2 = box_muller_transform(u1, u2)
            A[i, j] = complex(z1, z2) / np.sqrt(2.0)
    H = (A + A.conj().T) / 2.0
    return H


def quantum_random_unitary(n: int, engine: QuantumRandomnessEngine) -> np.ndarray:
    """
    生成 n x n 的随机酉矩阵 (用于量子电路模拟)。
    基于 QR 分解: 对复高斯矩阵做 QR 分解，再对角化 R 的相位。
    """
    if n <= 0:
        raise ValueError("Matrix dimension must be positive")
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            u1, u2 = engine.uniform_01(), engine.uniform_01()
            z1, z2 = box_muller_transform(u1, u2)
            A[i, j] = complex(z1, z2) / np.sqrt(2.0)
    Q, R = np.linalg.qr(A)
    # 调整相位使 R 的对角元为正实数
    D = np.diag(np.diag(R))
    D = np.diag(np.exp(1j * np.angle(np.diag(D))))
    U = Q @ D
    return U
