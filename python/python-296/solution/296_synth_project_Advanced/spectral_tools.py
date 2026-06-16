# -*- coding: utf-8 -*-
"""
spectral_tools.py
=================
Hilbert 曲线索引与素数谱滤波.

Hilbert 曲线 (来自 537_hilbert_curve_display):
----------------------------------------------
Hilbert 曲线是一种空间填充曲线, 将 1D 索引映射到 2D 坐标,
保持局部性: 相邻 1D 索引对应相邻 2D 坐标.

在 fast ignition 中:
- 等离子体网格 (2D) → Hilbert 索引 (1D)
- 提高 cache 命中率, 改善 FD 模板访问模式
- 用于谱分析中的模式排序

算法 (来自 537_hilbert_curve_display/d2xy, rot):
    d2xy(m, d): 1D Hilbert 坐标 d → 2D (x, y)
    rot(n, x, y, rx, ry): 象限旋转/翻转

素数谱滤波 (来自 910_prime):
----------------------------
素数索引模式选择用于谱滤波:
- 在 FD 求解后, 对解进行离散 Fourier 变换
- 保留素数波数模式 (去除合数模式的噪声)
- 独特方法论: 素数模式对应 "不可分解" 的相干结构

物理动机:
- 合数模式可分解为低阶模式的乘积, 更可能包含非线性混频噪声
- 素数模式保持独立性, 对应物理上最基本的相干结构
"""

import math


# ============================================================
# Hilbert 曲线: d2xy (来自 537_hilbert_curve_display/d2xy)
# ============================================================
def d2xy(m, d):
    """
    1D Hilbert 坐标 → 2D Cartesian (来自 537_hilbert_curve_display/d2xy).

    参数:
        m: Hilbert 曲线阶数, 网格大小 N = 2^m
        d: 1D Hilbert 坐标, 0 ≤ d < N^2
    返回:
        (x, y): 2D 坐标, 0 ≤ x, y < N
    """
    if m <= 0:
        raise ValueError("m 必须 > 0")
    n = 1 << m
    if d < 0 or d >= n * n:
        raise ValueError("d 越界: 0 <= d < {}".format(n * n))

    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = (t // 2) % 2
        if rx == 0:
            ry = t % 2
        else:
            ry = (t ^ rx) % 2
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def xy2d(m, x, y):
    """
    2D Cartesian → 1D Hilbert 坐标 (逆映射).

    参数:
        m: Hilbert 阶数
        x, y: 2D 坐标
    返回:
        d: 1D Hilbert 坐标
    """
    n = 1 << m
    if x < 0 or x >= n or y < 0 or y >= n:
        raise ValueError("x, y 越界")
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s >>= 1
    return d


def rot(n, x, y, rx, ry):
    """
    象限旋转/翻转 (来自 537_hilbert_curve_display/rot).

    参数:
        n     : 当前尺度
        x, y  : 坐标
        rx, ry: 旋转控制位
    返回:
        (x_new, y_new)
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


# ============================================================
# Hilbert 索引在等离子体网格上的应用
# ============================================================
def hilbert_order_2d_grid(nx, ny):
    """
    为 nx × ny 二维网格生成 Hilbert 排序索引.

    步骤:
    1. 找到最小 m 使得 2^m ≥ max(nx, ny)
    2. 对每个网格点 (i,j), 计算 Hilbert 坐标 d
    3. 返回排序后的网格点索引

    参数:
        nx, ny: 网格尺寸
    返回:
        list[tuple]: (i, j, d) 按 d 排序
    """
    # 找 m
    N = max(nx, ny)
    m = 0
    while (1 << m) < N:
        m += 1
    actual_N = 1 << m

    # 为每个网格点计算 Hilbert 坐标
    points = []
    for j in range(ny):
        for i in range(nx):
            # 映射到 [0, actual_N) 范围
            xi = int(i * actual_N / nx)
            yi = int(j * actual_N / ny)
            xi = min(xi, actual_N - 1)
            yi = min(yi, actual_N - 1)
            d = xy2d(m, xi, yi)
            points.append((i, j, d))

    # 按 Hilbert 坐标排序
    points.sort(key=lambda p: p[2])
    return points


def hilbert_reorder_field(field_2d, nx, ny):
    """
    将二维场按 Hilbert 顺序重排为一维数组.

    参数:
        field_2d: ny × nx 的场
        nx, ny  : 网格尺寸
    返回:
        list: 按 Hilbert 顺序重排的值
    """
    order = hilbert_order_2d_grid(nx, ny)
    return [field_2d[j][i] for i, j, d in order]


def hilbert_restore_field(hilbert_array, nx, ny):
    """
    从 Hilbert 顺序恢复到二维场.

    参数:
        hilbert_array: Hilbert 排序的一维数组
        nx, ny       : 网格尺寸
    返回:
        list[list]: ny × nx 二维场
    """
    order = hilbert_order_2d_grid(nx, ny)
    field = [[0.0] * nx for _ in range(ny)]
    for k, (i, j, d) in enumerate(order):
        if k < len(hilbert_array):
            field[j][i] = hilbert_array[k]
    return field


# ============================================================
# 素数谱滤波 (来自 910_prime)
# ============================================================
def prime_sieve(n):
    """
    Eratosthenes 筛法 (来自 910_prime/prime_sieve).
    返回 ≤ n 的所有素数.
    """
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    i = 2
    while i * i <= n:
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
        i += 1
    return [k for k in range(2, n + 1) if is_prime[k]]


def discrete_fourier_transform_1d(signal):
    """
    一维 DFT (直接使用定义, 不依赖 numpy.fft):
        X_k = Σ_{n=0}^{N-1} x_n · exp(-2πi kn/N)

    参数:
        signal: 实数信号列表
    返回:
        list[complex]: DFT 系数
    """
    N = len(signal)
    X = []
    for k in range(N):
        re = 0.0
        im = 0.0
        for n in range(N):
            angle = -2.0 * math.pi * k * n / N
            re += signal[n] * math.cos(angle)
            im += signal[n] * math.sin(angle)
        X.append(complex(re, im))
    return X


def inverse_dft_1d(X):
    """一维逆 DFT."""
    N = len(X)
    x = []
    for n in range(N):
        re = 0.0
        for k in range(N):
            angle = 2.0 * math.pi * k * n / N
            re += X[k].real * math.cos(angle) - X[k].imag * math.sin(angle)
        x.append(re / N)
    return x


def prime_mode_filter(signal, keep_prime_modes=True):
    """
    素数模式滤波器.

    步骤:
    1. 对信号进行 DFT
    2. 保留 (或去除) 素数波数模式
    3. 逆 DFT 得到滤波后信号

    参数:
        signal          : 输入信号
        keep_prime_modes: True 保留素数模式, False 去除
    返回:
        filtered: 滤波后信号
        spectrum: 原始谱
        filtered_spectrum: 滤波后谱
    """
    N = len(signal)
    # DFT
    X = discrete_fourier_transform_1d(signal)
    # 素数列表
    primes = set(prime_sieve(N - 1))
    primes.add(0)  # 保留直流分量

    # 滤波
    X_filtered = list(X)
    for k in range(N):
        if keep_prime_modes:
            if k not in primes:
                X_filtered[k] = complex(0.0, 0.0)
        else:
            if k in primes and k > 0:
                X_filtered[k] = complex(0.0, 0.0)

    # 逆 DFT
    filtered = inverse_dft_1d(X_filtered)
    return filtered, X, X_filtered


def spectral_energy(spectrum):
    """
    谱能量: E_k = |X_k|^2 / N
    """
    N = len(spectrum)
    return [abs(Xk) ** 2 / max(N, 1) for Xk in spectrum]


def print_spectral_summary(signal, filtered_signal, spectrum, filtered_spectrum):
    """打印谱分析摘要."""
    print("\n" + "=" * 72)
    print("Hilbert 索引 + 素数谱滤波")
    print("=" * 72)
    N = len(signal)
    primes = prime_sieve(N - 1)

    # 信号统计
    orig_rms = math.sqrt(sum(s ** 2 for s in signal) / N)
    filt_rms = math.sqrt(sum(s ** 2 for s in filtered_signal) / N)
    print("  信号长度   : {}".format(N))
    print("  素数数量   : {} (≤ {})".format(len(primes), N - 1))
    print("  素数模式   : {}".format(primes[:min(20, len(primes))]))
    print("  原始 RMS   : {:.6e}".format(orig_rms))
    print("  滤波后 RMS : {:.6e}".format(filt_rms))
    print("  能量保持率 : {:.4f} %".format(
        100.0 * filt_rms / max(orig_rms, 1.0e-300)))

    # 谱能量分布
    orig_energy = spectral_energy(spectrum)
    filt_energy = spectral_energy(filtered_spectrum)
    total_orig = sum(orig_energy)
    total_filt = sum(filt_energy)
    prime_energy = sum(orig_energy[k] for k in primes if k < N)
    print("  总谱能量   : {:.6e}".format(total_orig))
    print("  素数模式能量占比: {:.4f} %".format(
        100.0 * prime_energy / max(total_orig, 1.0e-300)))
    print("  滤波后总能量  : {:.6e}".format(total_filt))
    print("=" * 72)


# ============================================================
# Hilbert 排序的 cache 友好性演示
# ============================================================
def hilbert_locality_score(nx, ny):
    """
    比较 Hilbert 排序 vs 行主序的局部性得分.

    局部性 = Σ_{相邻点对} |index_diff|
    越小表示局部性越好.
    """
    # 行主序: 相邻点索引差 = 1 (水平) 或 nx (垂直)
    row_major_score = 0
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                row_major_score += 1  # 水平
            if j + 1 < ny:
                row_major_score += nx  # 垂直

    # Hilbert: 相邻点索引差 = 1 (在 Hilbert 顺序中)
    hilbert_score = 0
    order = hilbert_order_2d_grid(nx, ny)
    for k in range(len(order) - 1):
        i1, j1, d1 = order[k]
        i2, j2, d2 = order[k + 1]
        # 2D 距离
        dist = abs(i2 - i1) + abs(j2 - j1)
        hilbert_score += dist

    return {
        "row_major_score": row_major_score,
        "hilbert_score": hilbert_score,
        "improvement": row_major_score / max(hilbert_score, 1),
    }
