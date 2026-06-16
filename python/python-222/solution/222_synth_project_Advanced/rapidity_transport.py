# -*- coding: utf-8 -*-
"""
rapidity_transport.py
=====================

快度空间 Walsh-Hadamard 变换与 parton 密度输运。

融合种子项目:
    - 1400_walsh_transform: Fast Walsh Transform (FWT)

物理背景:
    在高能碰撞中, parton 的快度 y = 0.5 ln((E+p_z)/(E-p_z)) 是自然的演化变量。
    Parton 密度 rho(y) 在快度空间的演化满足扩散型方程:
        d rho / dY = D d^2 rho / dy^2
    其中 Y = ln(Q^2/Q0^2) 为演化"时间", D 为扩散系数。

    Walsh 变换提供正交二值基分解:
        rho(y) = (1/N) sum_k W_k * wal_k(y)
    其中 wal_k 为 Walsh 函数, W_k 为 Walsh 谱系数。

    该分解的优势:
    1) 快速 O(N log N) 变换
    2) 二值性质天然适合 parton 级联的二叉树分裂结构
    3) 谱截断可提供有效的红外正则化
"""

from __future__ import annotations
import math
from typing import List, Tuple
import constants as C


# ======================================================================
# Fast Walsh-Hadamard Transform (来自 1400_walsh_transform/fwt.m)
# ======================================================================
def fast_walsh_transform(x: List[float]) -> List[float]:
    """
    快速 Walsh-Hadamard 变换 (sequency 序)。
    N 必须为 2 的幂。
    算法源自 Ken Beauchamp, Walsh functions and their applications (1975)。

    Parameters
    ----------
    x : list of float
        长度为 2^k 的输入序列

    Returns
    -------
    list of float
        Walsh 谱系数
    """
    n = len(x)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"输入长度必须是 2 的幂, 收到 {n}")

    result = list(x)
    # Haar 蝶形运算
    length = 1
    while length < n:
        half = length
        length *= 2
        for start in range(0, n, length):
            for j in range(start, start + half):
                a = result[j]
                b = result[j + half]
                result[j] = a + b
                result[j + half] = a - b
    return result


def inverse_walsh_transform(w: List[float]) -> List[float]:
    """
    逆 Walsh 变换: W^{-1} = (1/N) W^T = (1/N) W (自逆性)。
    """
    n = len(w)
    fwd = fast_walsh_transform(w)
    return [x / n for x in fwd]


# ======================================================================
# Walsh 函数求值
# ======================================================================
def walsh_function(k: int, y: float, y_max: float = 5.0) -> float:
    """
    Walsh 函数 wal_k(t) 在 t = (y + y_max) / (2 y_max) 处求值。
    wal_k(t) = prod_{j} (-1)^{b_j(t) b_j(k)}
    其中 b_j 为二进制位。

    返回 +1 或 -1。
    """
    n_bits = max(1, k.bit_length())
    t_norm = (y + y_max) / (2.0 * y_max)
    t_norm = max(0.0, min(1.0 - 1e-12, t_norm))

    # 将 t 离散化到 2^n_bits 个 bin
    N = 1 << n_bits
    t_disc = int(t_norm * N)

    result = 1
    kk = k
    tt = t_disc
    while kk > 0:
        if (kk & 1) and (tt & 1):
            result *= -1
        kk >>= 1
        tt >>= 1
    return float(result)


# ======================================================================
# 快度空间 parton 密度
# ======================================================================
def rapidity_density_initial(y_grid: List[float],
                              beam_rapidity: float = 5.0,
                              temperature: float = 1.0) -> List[float]:
    """
    初始快度分布 (热化模型):
        rho(y) = A * exp(-y^2 / (2 sigma^2))
    其中 sigma ~ T / m_T, 反映热涨落。

    在 RHIC/LHC 中, 中心快度平台由 gluon saturation 产生:
        rho_flat(y) ~ const for |y| < Y_beam
    """
    rho = []
    sigma = temperature
    for y in y_grid:
        # 组合: Gauss 中心 + 平台
        gauss = math.exp(-y * y / (2.0 * sigma * sigma))
        plateau = 1.0 if abs(y) < beam_rapidity * 0.8 else 0.0
        rho.append(0.6 * gauss + 0.4 * plateau)
    return rho


def walsh_spectrum(rho: List[float]) -> List[float]:
    """
    将快度密度 rho(y) 投影到 Walsh 基。
    首先 pad 到 2 的幂长度。
    """
    n = len(rho)
    # pad 到 2^k
    k = 0
    while (1 << k) < n:
        k += 1
    N = 1 << k
    rho_padded = list(rho) + [0.0] * (N - n)
    return fast_walsh_transform(rho_padded)


def walsh_inverse_spectrum(spectrum: List[float], original_n: int) -> List[float]:
    """从 Walsh 谱重构密度"""
    rho_padded = inverse_walsh_transform(spectrum)
    return rho_padded[:original_n]


# ======================================================================
# 快度扩散演化 (在 Walsh 谱空间)
# ======================================================================
def walsh_sequency(k: int, N: int) -> int:
    """
    Walsh 函数 k 的 sequency (零交叉数)。
    对应在快度空间的"频率"。
    """
    # Gray code 转换
    gray = k ^ (k >> 1)
    seq = 0
    g = gray
    while g > 0:
        seq += g & 1
        g >>= 1
    return seq


def evolve_rapidity_diffusion(rho0: List[float], diffusivity: float,
                               dY: float, n_steps: int) -> List[List[float]]:
    """
    快度扩散演化:
        d rho / dY = D * d^2 rho / dy^2

    在 Walsh 谱空间, 扩散方程为对角形式:
        W_k(Y) = W_k(0) * exp(-D * s_k^2 * Y)
    其中 s_k 为 sequency。

    这展示了 Walsh 变换对 diffusion 方程的天然对角化优势。
    """
    n = len(rho0)
    k = 0
    while (1 << k) < n:
        k += 1
    N = 1 << k
    rho_padded = list(rho0) + [0.0] * (N - n)

    # Walsh 谱
    W = fast_walsh_transform(rho_padded)

    trajectory = [list(rho0)]

    for step in range(n_steps):
        Y = (step + 1) * dY
        W_evolved = []
        for k_idx in range(N):
            s = walsh_sequency(k_idx, N)
            # 扩散衰减因子
            decay = math.exp(-diffusivity * s * s * Y)
            W_evolved.append(W[k_idx] * decay)

        # 逆变换回快度空间
        rho_evolved = inverse_walsh_transform(W_evolved)
        trajectory.append(rho_evolved[:n])

    return trajectory


# ======================================================================
# 谱截断正则化
# ======================================================================
def walsh_spectral_cutoff(spectrum: List[float], max_sequency: int) -> List[float]:
    """
    Walsh 谱截断: 将 sequency > max_sequency 的模式置零。
    这是一种有效的红外正则化, 抑制高频涨落。
    """
    N = len(spectrum)
    result = []
    for k_idx in range(N):
        s = walsh_sequency(k_idx, N)
        if s > max_sequency:
            result.append(0.0)
        else:
            result.append(spectrum[k_idx])
    return result


def walsh_energy_spectrum(spectrum: List[float]) -> List[Tuple[int, float]]:
    """
    按 sequency 分组的能量谱:
        E(s) = sum_{k: seq(k) = s} |W_k|^2
    """
    N = len(spectrum)
    max_seq = 0
    seq_list = []
    for k_idx in range(N):
        s = walsh_sequency(k_idx, N)
        seq_list.append(s)
        max_seq = max(max_seq, s)

    energy = [0.0] * (max_seq + 1)
    for k_idx in range(N):
        s = seq_list[k_idx]
        energy[s] += spectrum[k_idx] ** 2
    return list(enumerate(energy))


# ======================================================================
# Walsh 卷积 (用于 parton 分裂核的快速卷积)
# ======================================================================
def walsh_convolution(a: List[float], b: List[float]) -> List[float]:
    """
    Walsh 卷积定理:
        (a *_W b)(t) = IDWT( DWT(a) .* DWT(b) / N )
    其中 .* 为逐元素乘法。

    用于在快度空间计算分裂核的卷积:
        P (x) rho = int P(y) rho(x-y) dy
    """
    n = len(a)
    assert len(b) == n, "输入长度必须相等"
    Wa = fast_walsh_transform(a)
    Wb = fast_walsh_transform(b)
    Wc = [Wa[i] * Wb[i] / n for i in range(n)]
    return inverse_walsh_transform(Wc)


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Walsh Transform Self-test ===")
    # 正交性: W W^T = N I
    n = 8
    test = [float(i + 1) for i in range(n)]
    W = fast_walsh_transform(test)
    back = inverse_walsh_transform(W)
    err = max(abs(test[i] - back[i]) for i in range(n))
    print(f"  Roundtrip error: {err:.2e}")

    # 快度扩散
    ny = 16
    y_grid = [-4.0 + 8.0 * i / (ny - 1) for i in range(ny)]
    rho0 = rapidity_density_initial(y_grid)
    print(f"  Initial rho integral: {sum(rho0) * 8.0 / ny:.4f}")

    traj = evolve_rapidity_diffusion(rho0, 0.5, 0.1, 5)
    for k, rho in enumerate(traj):
        integral = sum(rho) * 8.0 / ny
        print(f"  Step {k}: integral = {integral:.6f}")

    # Walsh 能量谱
    spec = walsh_spectrum(rho0)
    es = walsh_energy_spectrum(spec)
    total_e = sum(e for _, e in es)
    print(f"  Walsh total energy: {total_e:.4f}")
