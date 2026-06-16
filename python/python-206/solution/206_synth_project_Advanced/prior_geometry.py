"""
prior_geometry.py  --  先验几何: 结构化协方差、单纯形格、布尔掩码、边界词
===============================================================
来源种子项目:
    709_magic4_matrix    : magic-4k 矩阵构造 (行列等和结构).
    054_asa299           : 单纯形格点枚举 (simplex_lattice_point_next).
    1159_KadelkaLab      : 非退化 canalization 布尔函数计数.
    1414_whale           : 六角鲸鱼瓦片边界词组合学.
科学问题角色:
    1. 基于 magic-4 构造 *块循环对称先验协方差* Sigma_prior,
       利用行列等和性质保证先验在参数置换下不变.
    2. 单纯形格点枚举产生离散校准配置 (不同观测预算分配).
    3. canalization 布尔掩码标记"激活"参数子空间,
       降维后验采样.
    4. 鲸鱼边界词产生 *旋转-反射等价类* 索引,
       用于先验对称群上的轨道约化.
核心公式:
    Sigma_prior = sigma^2 * (I + alpha * C_magic)
    其中 C_magic 为 magic-4 正规化矩阵, 满足:
        sum_i (C_magic)_{ij} = sum_j (C_magic)_{ij} = S  (常数).
"""
from __future__ import annotations
import math
from typing import List, Tuple, Iterator, Dict
from numerical_base import NUMERICS


# ======================================================================
# 1. Magic-4k 结构化协方差 (709)
# ======================================================================
def magic4_matrix(n: int) -> List[List[int]]:
    """
    构造 n x n magic 矩阵, n 为 4 的倍数.
    算法 (Burkardt): 顺序填 1..n^2, 在 4x4 子块内画 X,
    X 上的元素 k 替换为 n^2+1-k.
    """
    if n % 4 != 0:
        raise ValueError("magic4 要求 n 为 4 的倍数")
    A = [[0] * n for _ in range(n)]
    k = 1
    for i in range(n):
        for j in range(n):
            A[i][j] = k
            k += 1
    # 在每个 4x4 子块上, 标记 X 形位置并替换
    for bi in range(n // 4):
        for bj in range(n // 4):
            i0, j0 = 4 * bi, 4 * bj
            x_marks = set()
            for t in range(4):
                x_marks.add((t, t))
                x_marks.add((t, 3 - t))
            for (di, dj) in x_marks:
                i, j = i0 + di, j0 + dj
                A[i][j] = n * n + 1 - A[i][j]
    return A


def build_magic_covariance(dim: int, sigma2: float = 1.0,
                           alpha: float = 0.1
                           ) -> List[List[float]]:
    """
    构造 dim x dim 先验协方差矩阵:
        Sigma = sigma^2 * (I + alpha * C)
    其中 C 为 magic-4 正规化:
        C_{ij} = (M_{ij} - S/dim) / S,  S = 行和 = n(n^2+1)/2.
    结果保证对称正定 (通过 + jitter).
    """
    # 找到 >= dim 的最小 4 的倍数
    n = 4 * max(1, (dim + 3) // 4)
    M = magic4_matrix(n)
    S = n * (n * n + 1) / 2.0
    C = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            C[i][j] = (M[i][j] - S / n) / S
    Sigma = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            Sigma[i][j] = sigma2 * ((1.0 if i == j else 0.0) + alpha * C[i][j])
    # 对称化 + 正定化
    for i in range(dim):
        for j in range(i + 1, dim):
            avg = 0.5 * (Sigma[i][j] + Sigma[j][i])
            Sigma[i][j] = avg
            Sigma[j][i] = avg
        Sigma[i][i] += NUMERICS.cholesky_jitter
    return Sigma


# ======================================================================
# 2. 单纯形格点枚举 (054_asa299)
# ======================================================================
def simplex_lattice_points(n: int, T: int) -> Iterator[List[int]]:
    """
    生成单纯形 {x in Z^n : x_i >= 0, sum x_i <= T} 中所有格点,
    反字典序. 移植自 Burkardt simplex_lattice_point_next.
    用于枚举离散校准配置 (观测预算分配).
    """
    x = [0] * n
    x[0] = T
    more = True
    while more:
        yield list(x)
        more = False
        i = 0
        while i < n - 1:
            if x[i] > 0 and (i == 0 or x[i - 1] == 0):
                x[i] -= 1
                x[i + 1] += 1
                # 把后面所有 "尾巴" 清零再重新分配
                for k in range(i + 2, n):
                    x[k] = 0
                more = True
                break
            i += 1
        if not more:
            # 回扫一次再尝试
            for i in range(n - 1):
                if x[i] > 0:
                    x[i] -= 1
                    x[i + 1] += 1
                    more = True
                    break
            if not more:
                break


def calibration_designs(total_budget: int, n_channels: int
                        ) -> List[Dict[str, int]]:
    """
    生成校准设计: 在 n_channels 个观测通道上分配 total_budget 次采样.
    """
    designs = []
    for pt in simplex_lattice_points(n_channels, total_budget):
        designs.append({f"ch_{i}": v for i, v in enumerate(pt)})
    return designs


# ======================================================================
# 3. Canalization 布尔掩码 (1159)
# ======================================================================
def nchoosek(n: int, k: int) -> int:
    k = max(k, n - k)
    if k == n:
        return 1
    num = 1
    den = 1
    for i in range(k + 1, n + 1):
        num *= i
        den *= (i - k)
    return num // den


def b_star(n: int) -> int:
    """
    B*(n) = 2^{2^n} - 2((-1)^n - n)
            + sum_{k=1}^n (-1)^k C(n,k) 2^{k+1} 2^{2^{n-k}}
    (Kadelka 非退化 canalization 计数核心函数)
    为避免溢出, 对 n >= 6 使用对数计算.
    """
    if n >= 8:
        # 用对数主导项: 2^{2^n}
        return 1 << (1 << n)  # 仅对 n<=7 精确
    s = 0
    for k in range(1, n + 1):
        term = ((-1) ** k) * nchoosek(n, k) * (2 ** (k + 1)) * (2 ** (2 ** (n - k)))
        s += term
    return 2 ** (2 ** n) - 2 * (((-1) ** n) - n) + s


def canalization_mask(dim: int, threshold: float = 0.5) -> List[bool]:
    """
    根据 canalization 计数构造布尔激活掩码.
    思想: 第 i 维参数若 B*(i+1) mod 2^k > threshold * 2^k 则激活.
    用于降维后验采样 (仅探索 "有效" 参数方向).
    """
    mask = []
    for i in range(dim):
        bs = b_star(i + 1)
        frac = (bs & 0xFF) / 255.0  # 取低 8 位做归一化
        mask.append(frac > threshold)
    # 至少保留一个激活
    if not any(mask):
        mask[0] = True
    return mask


# ======================================================================
# 4. 鲸鱼边界词组合索引 (1414)
# ======================================================================
# 12 方向编码: A..L, 步长为 1/2 或 sqrt(3)/{3,6}
_WHALE_DIRS = ["A", "B", "b", "C", "D", "d",
               "E", "F", "f", "G", "H", "h"]


def whale_boundary_word(tile_id: int, length: int = 6) -> List[str]:
    """
    为 tile_id 生成长度为 length 的边界词.
    方向序列由 tile_id 混合决定, 用于标定 *先验对称群*
    上不同轨道的代表元.
    """
    word = []
    state = tile_id
    for _ in range(length):
        state = (state * 2654435761) & 0xFFFFFFFF
        idx = (state >> 16) % len(_WHALE_DIRS)
        word.append(_WHALE_DIRS[idx])
    return word


def orbit_representatives(dim: int, n_tiles: int = 8
                          ) -> List[Tuple[int, List[str]]]:
    """
    枚举 dim 个参数方向在 D_{2*dim} 二面体群下的轨道代表.
    返回 (tile_id, boundary_word) 列表.
    """
    reps = []
    seen_signatures = set()
    for tid in range(n_tiles):
        w = whale_boundary_word(tid, length=dim)
        sig = tuple(sorted(w))
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            reps.append((tid, w))
    return reps


# ======================================================================
# 5. 组合先验对象
# ======================================================================
class PriorGeometry:
    """
    聚合所有先验几何构造, 提供:
      - Sigma : dim x dim 正定先验协方差
      - active_mask : 激活参数子空间
      - designs : 离散校准配置
      - orbit_reps : 对称群轨道代表
    """
    def __init__(self, dim: int, sigma2: float = 0.25,
                 alpha: float = 0.15, budget: int = 4,
                 n_channels: int = 4, seed: int = 0):
        self.dim = dim
        self.Sigma = build_magic_covariance(dim, sigma2, alpha)
        self.Sigma_inv = _invert_spd(self.Sigma)
        self.active_mask = canalization_mask(dim, threshold=0.4)
        self.n_active = sum(self.active_mask)
        self.designs = calibration_designs(budget, n_channels)
        self.orbit_reps = orbit_representatives(dim, n_tiles=2 * dim)
        self.log_det = _log_det_spd(self.Sigma)
        self._rng = _LCG(seed)

    def log_pdf(self, theta: List[float]) -> float:
        """
        高斯先验对数密度:
            log p(theta) = -0.5 * (theta^T Sigma^{-1} theta
                                    + log det Sigma + d * log 2pi)
        """
        d = len(theta)
        quad = 0.0
        for i in range(d):
            for j in range(d):
                quad += theta[i] * self.Sigma_inv[i][j] * theta[j]
        return -0.5 * (quad + self.log_det + d * math.log(2.0 * math.pi))

    def sample(self) -> List[float]:
        """从先验 N(0, Sigma) 采样 (Cholesky 近似)."""
        z = [self._rng.normal() for _ in range(self.dim)]
        L = _cholesky_lower(self.Sigma)
        out = [0.0] * self.dim
        for i in range(self.dim):
            s = 0.0
            for j in range(i + 1):
                s += L[i][j] * z[j]
            out[i] = s
        return out


# ----------------------------------------------------------------------
# 线性代数辅助
# ----------------------------------------------------------------------
def _invert_spd(A: List[List[float]]) -> List[List[float]]:
    """SPD 矩阵求逆 (Gauss-Jordan, 带主元)."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for col in range(n):
        # 主元
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        p = M[col][col]
        if abs(p) < NUMERICS.cholesky_jitter:
            M[col][col] += NUMERICS.cholesky_jitter
            p = M[col][col]
        for j in range(2 * n):
            M[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            for j in range(2 * n):
                M[r][j] -= fac * M[col][j]
    return [row[n:] for row in M]


def _log_det_spd(A: List[List[float]]) -> float:
    L = _cholesky_lower(A)
    s = 0.0
    for i in range(len(A)):
        s += NUMERICS.clamp_log(max(L[i][i], NUMERICS.safe_log_floor))
    return 2.0 * s


def _cholesky_lower(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, NUMERICS.cholesky_jitter))
            else:
                L[i][j] = (A[i][j] - s) / max(L[j][j], NUMERICS.cholesky_jitter)
    return L


class _LCG:
    def __init__(self, seed: int = 0):
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1664525
        self._c = 1013904223
        self._m = 1 << 32

    def next(self) -> float:
        self._state = (self._a * self._state + self._c) % self._m
        return self._state / self._m

    def normal(self) -> float:
        u1 = max(self.next(), NUMERICS.safe_log_floor)
        u2 = self.next()
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2.0 * math.pi * u2)


__all__ = ["PriorGeometry", "magic4_matrix", "build_magic_covariance",
           "simplex_lattice_points", "calibration_designs",
           "canalization_mask", "whale_boundary_word",
           "orbit_representatives"]
