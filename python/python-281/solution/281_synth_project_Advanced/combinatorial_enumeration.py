"""
combinatorial_enumeration.py
============================
组合枚举与格点计数模块。
融合项目: 892_polyiamonds (多格骨牌边界枚举),
         568_i4lib (整数工具库),
         198_collatz_polynomial (迭代序列)

核心内容:
  1. 格点配置枚举: Li 在 NMC 晶格中的占据构型计数
  2. 组合数计算 (项目568: i4_choose)
  3. 格点邻域枚举 (项目892: ijk_neighbors)
  4. Collatz 型迭代用于构型空间探索 (项目198)
  5. 排列与组合的整数运算 (项目568)

应用:
  电池电极中 Li 离子的格点模型:
  - NMC 晶格有 N 个可用 Li 位点
  - 嵌入 x*N 个 Li 离子 (x = SOC)
  - 构型数 Ω = C(N, x*N) (组合数)
  - 构型熵 S = k_B * ln(Ω)
"""

import math
import numpy as np
from electrode_constants import C_MAX


# ==================== 整数工具函数 (项目568: i4lib) ====================

def i4_choose(n, k):
    """
    二项式系数 C(n, k) 的整数计算。
    项目568: i4_choose.m

    C(n,k) = n! / (k! * (n-k)!)

    使用递推避免溢出:
    C(n,k) = C(n-1,k-1) * n / k

    Parameters
    ----------
    n : int
        总数 (n ≥ 0)
    k : int
        选取数 (0 ≤ k ≤ n)

    Returns
    -------
    int
        二项式系数
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # 利用对称性: C(n,k) = C(n, n-k)
    k = min(k, n - k)

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)

    return result


def i4_choose_log(n, k):
    """
    对数二项式系数 ln(C(n,k))。
    项目568: i4_choose_log.m

    用于大数计算, 避免整数溢出。
    ln(C(n,k)) = ln(n!) - ln(k!) - ln((n-k)!)

    Parameters
    ----------
    n : int
        总数
    k : int
        选取数

    Returns
    -------
    float
        ln(C(n,k))
    """
    if k < 0 or k > n:
        return float('-inf')
    if k == 0 or k == n:
        return 0.0

    # 使用 lgamma: ln(n!) = lgamma(n+1)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def i4_factorial(n):
    """
    阶乘 n! 的整数计算。
    项目568: i4_factorial.m

    Parameters
    ----------
    n : int
        非负整数

    Returns
    -------
    int
        n!
    """
    if n < 0:
        raise ValueError("阶乘参数不能为负")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def i4_is_prime(n):
    """
    素性检验。
    项目568: i4_is_prime.m

    Parameters
    ----------
    n : int
        正整数

    Returns
    -------
    bool
        是否为素数
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def i4_partition_count(n, k):
    """
    整数 n 恰好分为 k 个正整数之和的分拆数 p(n,k)。

    递推: p(n,k) = p(n-1,k-1) + p(n-k,k)
    边界: p(n,1) = 1, p(n,n) = 1

    应用: 能量量子在格点模式间的分配。

    Parameters
    ----------
    n : int
        被分拆的整数
    k : int
        分拆的份数

    Returns
    -------
    int
        分拆数 p(n,k)
    """
    if n <= 0 or k <= 0:
        return 0
    if k > n:
        return 0
    if k == 1 or k == n:
        return 1

    # 动态规划
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][1] = 1
        for j in range(2, min(i, k) + 1):
            dp[i][j] = dp[i-1][j-1] + (dp[i-j][j] if i >= j else 0)

    return dp[n][k]


# ==================== 格点构型枚举 ====================

def lattice_site_count(soc, n_sites_per_unit=6):
    """
    给定 SOC, 计算 Li 占据的格点数。

    NMC 单位晶格有 n_sites_per_unit 个可用 Li 位点 (3a 位)。

    Parameters
    ----------
    soc : float
        荷电状态 (0 ≤ soc ≤ 1)
    n_sites_per_unit : int
        每个单位晶格的 Li 位点数

    Returns
    -------
    n_occupied : int
        被占据的格点数
    n_total : int
        总格点数 (单位晶格)
    """
    soc = max(0.0, min(1.0, soc))
    n_occupied = max(0, min(n_sites_per_unit, round(soc * n_sites_per_unit)))
    return n_occupied, n_sites_per_unit


def configuration_entropy(soc, n_sites=6):
    """
    格点模型的构型熵。

    S_config = k_B * ln(C(N, n_Li))
             = k_B * [ln(N!) - ln(n_Li!) - ln((N-n_Li)!)]

    对于大 N, 使用 Stirling 近似:
    S_config ≈ -k_B * N * [x*ln(x) + (1-x)*ln(1-x)]
    其中 x = n_Li/N

    Parameters
    ----------
    soc : float
        荷电状态
    n_sites : int
        可用格点数

    Returns
    -------
    S_config : float
        构型熵 [J/K] (乘以 k_B 前为无量纲)
    S_dimensionless : float
        无量纲构型熵 S/k_B
    """
    n_occupied, n_total = lattice_site_count(soc, n_sites)

    if n_occupied <= 0 or n_occupied >= n_total:
        return 0.0, 0.0

    # 精确组合数
    log_comb = i4_choose_log(n_total, n_occupied)
    S_dimensionless = log_comb

    # Stirling 近似 (连续版本)
    x = n_occupied / n_total
    if 0 < x < 1:
        S_stirling = -n_total * (x * math.log(x) + (1-x) * math.log(1-x))
    else:
        S_stirling = 0.0

    return S_dimensionless, S_stirling


def enumerate_lattice_configurations(n_sites, n_occupied):
    """
    枚举所有格点占据构型。
    融合项目 892: 边界枚举思想。

    对于小系统 (n_sites ≤ 20), 可以完全枚举。
    构型数 = C(n_sites, n_occupied)

    Parameters
    ----------
    n_sites : int
        格点总数
    n_occupied : int
        被占据的格点数

    Returns
    -------
    configs : list of tuple
        所有可能的构型 (每个构型为 0/1 元组)
    """
    if n_occupied > n_sites or n_occupied < 0:
        return []
    if n_occupied == 0:
        return [tuple([0] * n_sites)]
    if n_occupied == n_sites:
        return [tuple([1] * n_sites)]

    configs = []
    _enumerate_recursive(n_sites, n_occupied, 0, [], configs)
    return configs


def _enumerate_recursive(n_sites, n_remaining, idx, current, configs):
    """递归枚举辅助函数。"""
    if n_remaining == 0:
        configs.append(tuple(current + [0] * (n_sites - idx)))
        return
    if idx >= n_sites:
        return
    if n_sites - idx < n_remaining:
        return  # 剩余位置不够

    # 当前位置放 Li (1)
    current.append(1)
    _enumerate_recursive(n_sites, n_remaining - 1, idx + 1, current, configs)
    current.pop()

    # 当前位置不放 Li (0)
    current.append(0)
    _enumerate_recursive(n_sites, n_remaining, idx + 1, current, configs)
    current.pop()


# ==================== 格点邻域操作 (项目892) ====================

def ijk_neighbors(i, j, k, nx, ny, nz):
    """
    三维网格中 (i,j,k) 的 6-邻域。
    融合项目 892: ijk_neighbors.m

    Parameters
    ----------
    i, j, k : int
        当前格点坐标
    nx, ny, nz : int
        网格尺寸

    Returns
    -------
    neighbors : list of tuple
        相邻格点列表
    """
    neighbors = []
    for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
            neighbors.append((ni, nj, nk))
    return neighbors


def ijk_to_linear(i, j, k, ny, nz):
    """
    三维索引 → 一维线性索引。
    融合项目 892: ijk_to_ij.m

    Parameters
    ----------
    i, j, k : int
        三维索引
    ny, nz : int
        网格在 y, z 方向的尺寸

    Returns
    -------
    int
        线性索引
    """
    return i * ny * nz + j * nz + k


def linear_to_ijk(idx, ny, nz):
    """
    一维线性索引 → 三维索引。
    融合项目 892: ij_to_xy.m 的3D推广。

    Parameters
    ----------
    idx : int
        线性索引
    ny, nz : int
        y, z 方向尺寸

    Returns
    -------
    i, j, k : int
        三维索引
    """
    k = idx % nz
    j = (idx // nz) % ny
    i = idx // (ny * nz)
    return i, j, k


# ==================== Collatz 型构型空间探索 (项目198) ====================

def collatz_lattice_step(config, rule='odd_expand'):
    """
    Collatz 型格点构型迭代。
    融合项目 198: collatz_polynomial 的迭代思想。

    将格点构型视为多项式:
    p(x) = Σ c_i * x^i,  c_i ∈ {0, 1}

    迭代规则:
    - 如果 popcount(p) 为偶: p → p/2 (右移)
    - 如果 popcount(p) 为奇: p → 3p + 1 (类比 Collatz)

    用于探索构型空间的连通性。

    Parameters
    ----------
    config : tuple of int
        当前构型 (0/1 序列)
    rule : str
        迭代规则

    Returns
    -------
    next_config : tuple
        下一步构型
    """
    # 将构型视为二进制数
    value = 0
    for bit in config:
        value = value * 2 + bit

    if value == 0:
        return config

    popcount = bin(value).count('1')

    if rule == 'odd_expand':
        if popcount % 2 == 0:
            # 偶数 popcount: 右移
            value = value >> 1
        else:
            # 奇数 popcount: 3x+1
            value = 3 * value + 1
    elif rule == 'even_contract':
        if popcount % 2 == 0:
            value = 3 * value + 1
        else:
            value = value >> 1

    # 转回构型
    n_bits = len(config)
    result = []
    for _ in range(n_bits):
        result.append(value & 1)
        value >>= 1
    return tuple(reversed(result))


def collatz_sequence_length(config, max_steps=1000):
    """
    计算 Collatz 型迭代到达不动点的步数。

    Parameters
    ----------
    config : tuple
        初始构型
    max_steps : int
        最大步数

    Returns
    -------
    int
        到达不动点的步数 (-1 表示未收敛)
    """
    visited = set()
    current = config

    for step in range(max_steps):
        if current in visited:
            return step  # 检测到循环
        visited.add(current)

        next_config = collatz_lattice_step(current)
        if next_config == current:
            return step
        current = next_config

    return -1


def site_energy_landscape(config, site_energies):
    """
    计算格点构型的能量。

    E(config) = Σ_i c_i * ε_i + Σ_{<ij>} c_i * c_j * V_{ij}

    第一项: 格点能量 (外场)
    第二项: 近邻相互作用

    Parameters
    ----------
    config : tuple
        占据构型
    site_energies : ndarray
        各格点的能量

    Returns
    -------
    float
        构型总能量
    """
    n = len(config)
    energy = 0.0

    # 单格点能量
    for i in range(n):
        energy += config[i] * site_energies[i]

    # 近邻相互作用 (一维链近似)
    V_nn = 0.01  # 近邻交互能 [eV]
    for i in range(n - 1):
        energy += config[i] * config[i+1] * V_nn

    return energy
