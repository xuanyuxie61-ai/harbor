# -*- coding: utf-8 -*-
"""
utils.py
数值计算工具与辅助函数

核心算法来源：
- prime_factors: 整数质因数分解（用于谱方法最优节点数选取）
- imshow_numeric: 数值归一化与边界处理思想
"""

import numpy as np
from math import gcd
from functools import reduce


def prime_factors(n):
    """
    返回整数 n 的质因数分解列表（含重复因子）。

    在谱方法中，Chebyshev 多项式展开阶数 N 的选取常受 FFT 长度约束。
    通过质因数分解可构造最优截断阶数，使得 N+1 仅含小质因数（2, 3, 5），
    从而保障快速 Chebyshev 变换（FCT）的数值效率。

    参数:
        n (int): 待分解的整数，n >= 1

    返回:
        list[int]: 质因数列表（升序）

    公式:
        n = \prod_{i=1}^{k} p_i^{\alpha_i}
    """
    if not isinstance(n, int):
        raise TypeError("prime_factors: 输入必须为整数")
    if n < 1:
        raise ValueError("prime_factors: 输入整数必须 >= 1")

    factors = []
    i = 2
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def optimal_chebyshev_order(target_n, max_prime=5):
    """
    基于质因数分解选取最优 Chebyshev 展开阶数。

    寻找不小于 target_n 的最小整数 N，使得 N+1 的所有质因数均不超过 max_prime。
    这保证在快速 Chebyshev 变换中可使用混合基 FFT，降低计算复杂度从 O(N^2) 到 O(N \log N)。

    参数:
        target_n (int): 目标阶数下限
        max_prime (int): 允许的最大质因数，默认 5

    返回:
        int: 最优阶数 N
    """
    N = target_n
    while True:
        factors = prime_factors(N + 1)
        if all(p <= max_prime for p in set(factors)):
            return N
        N += 1


def normalize_array(a, method="minmax"):
    """
    数组归一化，借鉴 imshow_numeric 的数值边界处理思想。

    对 CFD 数据场进行归一化，避免除零，并处理恒值场。

    参数:
        a (np.ndarray): 输入数组
        method (str): "minmax" 或 "zscore"

    返回:
        np.ndarray: 归一化后的数组
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    a_min = np.min(a)
    a_max = np.max(a)
    if method == "minmax":
        if abs(a_max - a_min) < 1e-15:
            return np.zeros_like(a)
        return (a - a_min) / (a_max - a_min)
    elif method == "zscore":
        mu = np.mean(a)
        sigma = np.std(a)
        if sigma < 1e-15:
            return np.zeros_like(a)
        return (a - mu) / sigma
    else:
        raise ValueError(f"未知归一化方法: {method}")


def safe_divide(a, b, fill_value=0.0):
    """
    安全除法，避免被零除。

    参数:
        a (np.ndarray or float): 分子
        b (np.ndarray or float): 分母
        fill_value (float): 分母为零时的填充值

    返回:
        np.ndarray: a / b（含边界保护）
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # 广播到相同形状
    a, b = np.broadcast_arrays(a, b)
    result = np.empty_like(a, dtype=np.float64)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def blasius_function(eta, n_terms=50):
    """
    计算不可压缩 Blasius 边界层相似解 f(η)。

    Blasius 方程:
        f''' + (1/2) f f'' = 0
    边界条件:
        f(0) = f'(0) = 0,  f'(∞) = 1

    采用级数展开结合数值修正:
        f(η) ≈ η - 1.7208 + ...  (大 η 渐近)

    参数:
        eta (float or np.ndarray): 相似变量 η
        n_terms (int): 级数截断项数

    返回:
        tuple: (f, fp, fpp) 分别为 f, f', f''
    """
    eta = np.asarray(eta, dtype=np.float64)
    f = np.zeros_like(eta)
    fp = np.zeros_like(eta)
    fpp = np.zeros_like(eta)

    # Blasius 壁面曲率 f''(0) ≈ 0.332057
    alpha = 0.332057336215196

    # 小 η 级数展开: f(η) = Σ_{n=0}^∞ (-1)^n a_n η^{3n+2}
    # 这里用经验拟合+边界层修正
    mask_small = eta < 5.0
    e = eta[mask_small]
    f[mask_small] = (alpha / 2.0) * e**2 - (alpha**2 / 240.0) * e**5 \
                    + (11.0 * alpha**3 / 161280.0) * e**8
    fp[mask_small] = alpha * e - (alpha**2 / 48.0) * e**4 \
                     + (11.0 * alpha**3 / 20160.0) * e**7
    fpp[mask_small] = alpha - (alpha**2 / 12.0) * e**3 \
                      + (11.0 * alpha**3 / 2880.0) * e**6

    # 大 η 渐近: f(η) ≈ η - β + exp
    mask_large = ~mask_small
    e = eta[mask_large]
    beta = 1.7207876575205
    f[mask_large] = e - beta
    fp[mask_large] = 1.0 - np.exp(-beta * (e - beta))
    fpp[mask_large] = beta * np.exp(-beta * (e - beta))

    return f, fp, fpp


def compressible_blasius_velocity(eta, Ma, gamma=1.4, Pr=0.72, Tw_Te=1.0):
    """
    可压缩 Blasius 边界层速度剖面（绝热壁修正）。

    采用 Crocco-Busemann 能量积分:
        T + (γ-1)/2 * Ma^2 * u^2 = T_w + (T_e - T_w) * u + (γ-1)/2 * Ma^2 * u * (1 - u)

    参数:
        eta (np.ndarray): 相似变量
        Ma (float): 马赫数
        gamma (float): 比热比
        Pr (float): 普朗特数
        Tw_Te (float): 壁温与边界层外缘温度比

    返回:
        tuple: (u, T, mu) 速度、温度、粘性系数剖面
    """
    _, fp, _ = blasius_function(eta)
    u = np.clip(fp, 0.0, 1.0)

    # Crocco-Busemann 关系
    r = Pr ** (1.0 / 3.0)  # 恢复因子
    Tr_Te = 1.0 + r * (gamma - 1.0) / 2.0 * Ma**2
    T = Tw_Te + (Tr_Te - Tw_Te) * u - (gamma - 1.0) / 2.0 * Ma**2 * u**2
    T = np.clip(T, 0.1, None)

    # Sutherland 粘性定律
    mu = sutherland_viscosity(T)
    return u, T, mu


def sutherland_viscosity(T, T_ref=300.0, mu_ref=1.7894e-5, S=110.4):
    """
    Sutherland 粘性系数公式。

        μ / μ_ref = (T / T_ref)^{3/2} * (T_ref + S) / (T + S)

    参数:
        T (np.ndarray): 温度 [K]
        T_ref (float): 参考温度
        mu_ref (float): 参考粘性系数
        S (float): Sutherland 常数

    返回:
        np.ndarray: 动力粘性系数
    """
    T = np.asarray(T, dtype=np.float64)
    ratio = T / T_ref
    mu = mu_ref * ratio**1.5 * (T_ref + S) / np.maximum(T + S, 1e-10)
    return mu


def chebyshev_nodes(n, a=-1.0, b=1.0):
    """
    生成 n 阶 Chebyshev-Gauss-Lobatto 节点。

        x_j = cos(j * π / n),  j = 0, ..., n

    参数:
        n (int): 阶数
        a, b (float): 映射到区间 [a, b]

    返回:
        np.ndarray: 节点坐标
    """
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def chebyshev_diff_matrix(n, a=-1.0, b=1.0):
    """
    Chebyshev 谱微分矩阵（Gauss-Lobatto 节点）。

    对于节点 x_j = cos(jπ/n)，微分矩阵元素:
        D_{ij} = (c_i / c_j) * (-1)^{i+j} / (x_i - x_j),  i ≠ j
        D_{ii} = -x_i / (2(1-x_i^2)),  i = 1,...,n-1
        D_{00} = (2n^2+1)/6
        D_{nn} = -(2n^2+1)/6

    其中 c_0 = c_n = 2, c_j = 1 (1 ≤ j ≤ n-1)。

    参数:
        n (int): 阶数
        a, b (float): 区间端点

    返回:
        np.ndarray: (n+1) × (n+1) 微分矩阵
    """
    # TODO: 实现 Chebyshev 谱微分矩阵的核心计算
    # 提示: 需先获取 Gauss-Lobatto 节点，再按经典公式构造微分矩阵 D，
    #       最后根据区间 [a, b] 进行缩放。
    #       该函数被 stability_analysis.CompressibleLST 直接调用。
    raise NotImplementedError("chebyshev_diff_matrix: 请根据 Chebyshev 谱微分矩阵的经典公式完成实现")
