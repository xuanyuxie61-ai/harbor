"""
lattice.py — 晶格几何与 k 点网格生成
======================================

本模块实现 1D 周期晶格的几何描述和 Brillouin 区采样。
融合了以下种子项目的算法:

  - 054_asa299 (simplex lattice enumeration):
    k 点星的整数格点枚举, 用于生成对称性不等价 k 点轨道
  - 742_mcnuggets (Diophantine enumeration / DP):
    能带占据数的组合计数, 费米面交叉的整数约束
  - 681_line_integrals (1D 线段几何与单式积分):
    线段元胞长度、单式积参考值
  - 559_hypercube_integrals (超立方体积分):
    高维 BZ 的推广积分 (虽然本模型为 1D, 但保留高维接口)

核心物理:
  对于 1D 晶格常数 a, 倒格子矢量 G = 2πn/a (n ∈ ℤ)
  第一 Brillouin 区: k ∈ [-π/a, π/a]
  高对称点: Γ = 0, X = π/a

  Monkhorst-Pack k 点网格:
    k_i = (2i - N - 1) / (2N) · (2π/a)  for i = 1,...,N
  或等价的均匀偏移网格。
"""

import numpy as np
from typing import Tuple, List, Optional
from physical_constants import PI, TWO_PI


# ============================================================
# 晶格基本类
# ============================================================

class Lattice1D:
    """
    1D 布拉维格子。

    物理描述:
      实空间: 原胞 [0, a], 周期性边界条件
      倒空间: 第一 BZ [-π/a, π/a], 倒格矢 G_n = 2πn/a

    物理量 (原子单位):
      a: 晶格常数 [Bohr]
      G: 倒格矢 [1/Bohr]
      Ω: 原胞 "体积" (1D 中即长度) [Bohr]
    """

    def __init__(self, a: float):
        """
        Parameters
        ----------
        a : float
            晶格常数, 必须 > 0
        """
        if a <= 0:
            raise ValueError(f"晶格常数必须 > 0, 得到 a={a}")
        self.a = a
        self.volume = a  # 1D: 原胞体积 = 长度

    @property
    def reciprocal_lattice_vector(self) -> float:
        """第一倒格矢 b₁ = 2π/a [1/Bohr]"""
        return TWO_PI / self.a

    @property
    def bz_boundary(self) -> float:
        """第一 Brillouin 区边界 k_max = π/a"""
        return PI / self.a

    def reciprocal_vector(self, n: int) -> float:
        """第 n 个倒格矢 G_n = 2πn/a"""
        return n * TWO_PI / self.a

    def real_space_grid(self, n_points: int) -> Tuple[np.ndarray, float]:
        """
        生成实空间均匀网格。

        Parameters
        ----------
        n_points : int
            网格点数 (包含左边界, 不含右边界以维持周期性)

        Returns
        -------
        x : np.ndarray, shape (n_points,)
            网格坐标 x_i = i·dx, i=0,...,N-1
        dx : float
            网格间距 dx = a/N
        """
        if n_points < 4:
            raise ValueError(f"网格点数至少 4, 得到 {n_points}")
        dx = self.a / n_points
        x = np.arange(n_points) * dx
        return x, dx


# ============================================================
# k 点网格生成
# ============================================================

def monkhorst_pack_grid(n_kpoints: int, a: float,
                        shift: float = 0.0) -> np.ndarray:
    """
    生成 Monkhorst-Pack k 点网格 (1D)。

    Monkhorst-Pack 方案 [Monkhorst & Pack, PRB 13, 5188 (1976)]:
      k_i = (2i - N - 1 + 2·shift) / (2N) · (2π/a)
      for i = 1, 2, ..., N

    当 shift=0 时为标准的 MP 网格 (不含 Γ 点, N 偶数)。
    当 shift=0.5 时为包含 Γ 点的网格。

    k 点权重: 均匀采样时每个 k 点权重 w_i = 1/N。

    Parameters
    ----------
    n_kpoints : int
        k 点数量 N ≥ 1
    a : float
        晶格常数
    shift : float, optional
        偏移量 (0 或 0.5), 默认 0

    Returns
    -------
    kpoints : np.ndarray, shape (N,)
        k 点坐标 [1/Bohr], 已排序
    """
    if n_kpoints < 1:
        raise ValueError(f"k 点数至少为 1, 得到 {n_kpoints}")

    N = n_kpoints
    indices = np.arange(1, N + 1, dtype=float)
    kpoints = (2.0 * indices - N - 1.0 + 2.0 * shift) / (2.0 * N) * (TWO_PI / a)
    return np.sort(kpoints)


def kpoint_weights_uniform(n_kpoints: int) -> np.ndarray:
    """
    均匀 k 点网格的权重。

    对于 Monkhorst-Pack 网格, 所有 k 点等权:
      w_i = 1/N

    对于包含高对称点的非均匀网格, 权重由
    Voronoi 胞体积决定 (1D 中即相邻 k 点间距的一半之和)。

    Parameters
    ----------
    n_kpoints : int
        k 点数量

    Returns
    -------
    weights : np.ndarray, shape (N,)
        归一化权重, Σ w_i = 1
    """
    return np.ones(n_kpoints) / n_kpoints


def kpoint_weights_tetrahedron(kpoints: np.ndarray, a: float) -> np.ndarray:
    """
    四面体方法的 k 点权重 (1D 简化版本 = 梯形积分权重)。

    在 1D 中, 四面体方法简化为梯形法则:
      w_i = (k_{i+1} - k_{i-1}) / (2 · |BZ|)
    对端点:
      w_0 = (k_1 - k_0) / (2 · |BZ|)
      w_{N-1} = (k_{N-1} - k_{N-2}) / (2 · |BZ|)

    其中 |BZ| = 2π/a 为 Brillouin 区长度。

    Parameters
    ----------
    kpoints : np.ndarray, shape (N,)
        k 点坐标 (已排序)
    a : float
        晶格常数

    Returns
    -------
    weights : np.ndarray, shape (N,)
        梯形权重, Σ w_i = 1
    """
    N = len(kpoints)
    if N < 2:
        return np.array([1.0])

    bz_length = TWO_PI / a
    dk = np.diff(kpoints)

    weights = np.zeros(N)
    weights[0] = dk[0] / (2.0 * bz_length)
    weights[-1] = dk[-1] / (2.0 * bz_length)
    for i in range(1, N - 1):
        weights[i] = (dk[i - 1] + dk[i]) / (2.0 * bz_length)

    # 归一化
    weights /= np.sum(weights)
    return weights


# ============================================================
# k 点星轨道枚举 (源自 054_asa299 simplex lattice)
# ============================================================

def enumerate_kpoint_stars(n_max: int,
                           symmetry_order: int = 2) -> List[Tuple[int, int]]:
    """
    枚举 k 点星的整数轨道 (源自 054_asa299 的 simplex lattice 枚举)。

    在 1D 中, k 点星由整数 n 标记:
      k = n · π / (n_max · a)

    对称操作: k → -k (反演), 所以星 {n, -n} 合并为一个轨道。
    权重 = 该轨道中不等价 k 点的数量。

    对于 n=0 (Γ 点): 权重 = 1 (自身对称)
    对于 n=n_max (X 点, 当 n_max 偶数时): 权重 = 1
    对于其他 n: 权重 = 2 (k 和 -k 不等价但能量相同)

    推广到 d 维: 使用 simplex lattice 枚举所有满足
      Σ |n_i| ≤ n_max 的整数格点, 并按点群对称性分类。

    Parameters
    ----------
    n_max : int
        最大轨道指数
    symmetry_order : int
        点群阶数 (1D 中 = 2, 即 {E, σ})

    Returns
    -------
    stars : list of (n, multiplicity)
        每个星的指标和简并度
    """
    if n_max < 0:
        raise ValueError(f"n_max 必须 >= 0, 得到 {n_max}")

    stars = []
    seen = set()

    for n in range(n_max + 1):
        if n in seen:
            continue
        # 在 1D 中, 反演对称性 k → -k
        partner = -n
        if partner == n:
            # Γ 点或 X 点 (自对称)
            stars.append((n, 1))
        else:
            stars.append((n, symmetry_order))
            seen.add(partner)

    return stars


# ============================================================
# 能带占据组合计数 (源自 742_mcnuggets Diophantine)
# ============================================================

def band_occupation_count(n_electrons: int, n_bands: int,
                          n_kpoints: int) -> int:
    """
    计算能带占据的构型数 (源自 742_mcnuggets 的 Diophantine 枚举)。

    对于给定的电子数 N_e、能带数 N_b 和 k 点数 N_k:
    每个 (band, kpoint) 对可以容纳 0, 1, 或 2 个电子 (自旋简并)。

    总占据构型数 = 非负整数解的个数:
      Σ_{b,k} f_{b,k} = N_e
      其中 0 ≤ f_{b,k} ≤ 2

    这等价于受限 Diophantine 方程:
      x₁ + x₂ + ... + x_{N_b · N_k} = N_e
      其中 0 ≤ x_i ≤ 2

    使用生成函数 / DP 计算:
      W(N_e, N_b·N_k) = [z^{N_e}] (1 + z + z²)^{N_b · N_k}

    这等同于 742_mcnuggets 中 package_sizes=[1,2] 的
    受限分拆计数问题。

    Parameters
    ----------
    n_electrons : int
        总电子数
    n_bands : int
        能带数
    n_kpoints : int
        k 点数

    Returns
    -------
    count : int
        可能的占据构型数
    """
    if n_electrons < 0 or n_bands < 1 or n_kpoints < 1:
        return 0

    n_states = n_bands * n_kpoints  # 总状态数
    max_electrons = 2 * n_states     # 最大电子数 (每个状态 2 个自旋)

    if n_electrons > max_electrons:
        return 0

    # DP: ways[i] = 将 i 个电子放入已考虑的状态中的方式数
    # 每个状态可容纳 0, 1, 或 2 个电子
    # 等价于 742_mcnuggets 中 package_sizes=[1,2] 的计数
    ways = np.zeros(n_electrons + 1, dtype=np.int64)
    ways[0] = 1  # 0 个电子: 1 种方式

    for _ in range(n_states):
        # 更新 DP: 新 ways[i] = old ways[i] + old ways[i-1] + old ways[i-2]
        new_ways = np.zeros(n_electrons + 1, dtype=np.int64)
        for i in range(n_electrons + 1):
            new_ways[i] = ways[i]
            if i >= 1:
                new_ways[i] += ways[i - 1]
            if i >= 2:
                new_ways[i] += ways[i - 2]
        ways = new_ways

    return int(ways[n_electrons])


# ============================================================
# Brillouin 区积分参考值 (源自 681_line_integrals)
# ============================================================

def line_bz_monomial_integral(e: int, a: float) -> float:
    """
    计算 Brillouin 区上 k^e 的精确积分 (源自 681_line_integrals)。

    对于第一 BZ [-π/a, π/a]:
      ∫_{-π/a}^{π/a} k^e dk = [k^{e+1}/(e+1)]_{-π/a}^{π/a}

    当 e 为奇数: 积分为 0 (被积函数为奇函数)
    当 e 为偶数: 积分 = 2 · (π/a)^{e+1} / (e+1)

    Parameters
    ----------
    e : int
        幂次 (>= 0)
    a : float
        晶格常数

    Returns
    -------
    integral : float
        精确积分值
    """
    if e < 0:
        raise ValueError(f"幂次必须 >= 0, 得到 e={e}")

    if e % 2 == 1:
        return 0.0  # 奇函数在对称区间上积分为零

    kmax = PI / a
    return 2.0 * kmax ** (e + 1) / (e + 1)


# ============================================================
# 高维 BZ 积分 (源自 559_hypercube_integrals)
# ============================================================

def hypercube_bz_monomial_integral(exponents: np.ndarray,
                                   lattice_constants: np.ndarray) -> float:
    """
    计算高维 Brillouin 区上单式函数的精确积分。

    对于 d 维正交晶格, BZ = Π_i [-π/a_i, π/a_i]:
      ∫_{BZ} Π_i k_i^{e_i} d^d k
        = Π_i ∫_{-π/a_i}^{π/a_i} k_i^{e_i} dk_i

    每个因子由 line_bz_monomial_integral 给出。

    Parameters
    ----------
    exponents : np.ndarray, shape (d,)
        每个维度的幂次
    lattice_constants : np.ndarray, shape (d,)
        每个维度的晶格常数

    Returns
    -------
    integral : float
        精确积分值
    """
    d = len(exponents)
    if d != len(lattice_constants):
        raise ValueError("exponents 和 lattice_constants 维度不匹配")

    result = 1.0
    for i in range(d):
        result *= line_bz_monomial_integral(int(exponents[i]),
                                             lattice_constants[i])
    return result


# ============================================================
# 简约 k 点生成 (利用对称性)
# ============================================================

def generate_irreducible_kpoints(n_kpoints: int, a: float
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成不可约 k 点及其权重。

    利用 1D 反演对称性 k → -k, 只保留 BZ 的右半部分:
      k ∈ [0, π/a]

    每个不可约 k 点的权重 = 其在完整 BZ 中的简并度。

    Parameters
    ----------
    n_kpoints : int
        完整 BZ 中的 k 点数 (应取偶数)
    a : float
        晶格常数

    Returns
    -------
    k_irr : np.ndarray
        不可约 k 点
    w_irr : np.ndarray
        对应权重 (归一化)
    """
    k_full = monkhorst_pack_grid(n_kpoints, a)

    # 利用对称性: 只保留 k >= 0
    irf_mask = k_full >= -1e-14
    k_irr = k_full[irf_mask]

    # 计算权重
    w_full = kpoint_weights_uniform(n_kpoints)

    # 对于 k ≠ 0 且 k ≠ π/a 的点, 权重加倍
    w_irr = np.zeros(len(k_irr))
    for i, k in enumerate(k_irr):
        # 找到在完整网格中的对应点
        mask_pos = np.abs(k_full - k) < 1e-12
        mask_neg = np.abs(k_full + k) < 1e-12
        # 权重 = 所有对称等价 k 点的权重之和
        w_irr[i] = np.sum(w_full[mask_pos]) + np.sum(w_full[mask_neg])

    # 避免重复计算 (当 k=0 时, +k 和 -k 是同一点)
    for i, k in enumerate(k_irr):
        if abs(k) < 1e-12:
            # Γ 点, 不需要加两次
            mask_neg = np.abs(k_full + k) < 1e-12
            w_irr[i] = np.sum(w_full[mask_pos])
        elif abs(abs(k) - PI / a) < 1e-12:
            # X 点 (BZ 边界), 不需要加两次
            pass

    return k_irr, w_irr
