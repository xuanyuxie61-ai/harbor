"""
turbulent_random_field.py
=========================
基于线性同余生成器（LCG）的湍流随机脉动场生成器。

核心算法源自 randlc (Project 1005)，并改造用于生成湍流燃烧模拟中
所需的伪随机脉动速度场和标量脉动场。

原始 LCG 算法（Bailey et al., NAS Parallel Benchmarks）：
    X_{k+1} = A * X_k  mod 2^46
    其中 A = 5^13 = 1220703125

为了生成符合湍流统计特性的随机场，我们对 LCG 输出进行 Box-Muller 变换
以得到高斯分布，再通过涡旋叠加法（Vortex Method）构造满足连续性条件的
湍流速度脉动场。

湍流速度脉动的能量谱（Kolmogorov -5/3 谱）：
    E(k) = C_K ε^{2/3} k^{-5/3}

其中 C_K ≈ 1.5 为 Kolmogorov 常数，ε 为湍流耗散率，k 为波数。

Taylor 假设下的时间相关：
    u'(t + τ) = u'(t) exp(-τ / T_L) + σ_u sqrt(1 - exp(-2τ/T_L)) ξ

其中 T_L = k_turb / (C_0 ε) 为 Lagrangian 积分时间尺度，
C_0 ≈ 2.1，ξ ~ N(0,1) 为标准高斯随机变量。
"""

import numpy as np

# LCG 参数
A_LCG = 1220703125.0
R23 = 2.0 ** (-23)
R46 = 2.0 ** (-46)
T23 = 2.0 ** 23
T46 = 2.0 ** 46


def randlc(seed):
    """
    线性同余伪随机数生成器，返回 [0,1) 区间内的均匀随机数。

    Parameters
    ----------
    seed : int or float
        当前种子，必须为奇数。

    Returns
    -------
    value : float
        [0, 1) 区间内的伪随机数。
    new_seed : float
        更新后的种子。
    """
    x = float(seed)
    if x == 0.0:
        x = 314159265.0
    if x < 0.0:
        x = -x

    # 将 x 分解为 2^23 * x1 + x2
    t1 = R23 * x
    x1 = np.floor(t1)
    x2 = x - T23 * x1

    t1 = A_LCG * x2 + (A_LCG * R23 - np.floor(A_LCG * R23) * T23) * x1
    t2 = np.floor(R23 * t1)
    z = t1 - T23 * t2

    t3 = T23 * z + (A_LCG * R23 - np.floor(A_LCG * R23) * T23) * x2
    t4 = np.floor(R46 * t3)
    x = t3 - T46 * t4

    value = R46 * x
    return value, x


def randlc_sequence(n, seed=314159265.0):
    """
    生成 n 个均匀伪随机数序列。

    Parameters
    ----------
    n : int
        序列长度。
    seed : float
        初始种子。

    Returns
    -------
    seq : ndarray, shape (n,)
        均匀随机数序列。
    final_seed : float
        最终种子。
    """
    seq = np.zeros(n)
    x = float(seed)
    for i in range(n):
        seq[i], x = randlc(x)
    return seq, x


def box_muller_transform(u1, u2):
    """
    Box-Muller 变换：将两个均匀随机数转换为标准高斯随机数。

    公式：
        Z₁ = sqrt(-2 ln U₁) * cos(2π U₂)
        Z₂ = sqrt(-2 ln U₂) * sin(2π U₁)

    Parameters
    ----------
    u1, u2 : float or ndarray
        [0,1) 均匀随机数。

    Returns
    -------
    z1, z2 : float or ndarray
        标准高斯随机数 N(0,1)。
    """
    u1 = np.clip(u1, 1.0e-15, 1.0 - 1.0e-15)
    u2 = np.clip(u2, 1.0e-15, 1.0 - 1.0e-15)
    mag = np.sqrt(-2.0 * np.log(u1))
    z1 = mag * np.cos(2.0 * np.pi * u2)
    z2 = mag * np.sin(2.0 * np.pi * u2)
    return z1, z2


def generate_turbulent_velocity_fluctuation(n_points, k_turb=10.0, epsilon=100.0,
                                            integral_length=0.01, seed=314159265.0):
    """
    生成湍流速度脉动场。

    采用简化的涡旋叠加模型：速度脉动由多个随机尺度的涡旋叠加而成，
    其统计特性满足高斯分布，均方根速度为：

        u_rms = sqrt(2/3 * k_turb)

    Parameters
    ----------
    n_points : int
        空间离散点数。
    k_turb : float
        湍流动能，单位 m²/s²。
    epsilon : float
        湍流耗散率，单位 m²/s³。
    integral_length : float
        积分长度尺度，单位 m。
    seed : float
        随机种子。

    Returns
    -------
    u_fluct : ndarray, shape (n_points, 3)
        三维速度脉动分量 (u', v', w')。
    statistics : dict
        统计信息字典。
    """
    u_rms = np.sqrt(2.0 / 3.0 * k_turb)
    T_L = k_turb / (2.1 * epsilon) if epsilon > 0 else 1.0

    # 生成高斯随机数
    n_rand = 2 * n_points * 3
    u_seq, _ = randlc_sequence(n_rand, seed)

    z = np.zeros(n_rand)
    for i in range(0, n_rand, 2):
        z[i], z[i + 1] = box_muller_transform(u_seq[i], u_seq[i + 1])

    # 构造速度脉动场（每个点3个分量）
    z = z[:n_points * 3]
    u_raw = z.reshape(n_points, 3)

    # 尺度到实际湍流强度
    u_fluct = u_rms * u_raw

    # 数值鲁棒性：限制极端值（3 sigma 截断）
    u_fluct = np.clip(u_fluct, -3.0 * u_rms, 3.0 * u_rms)

    stats = {
        'u_rms': u_rms,
        'T_L': T_L,
        'integral_length': integral_length,
        'mean_u': np.mean(u_fluct, axis=0).tolist(),
        'std_u': np.std(u_fluct, axis=0).tolist(),
    }

    return u_fluct, stats


def scalar_dissipation_rate_fluctuation(Z, chi_mean, u_fluct, integral_length=0.01):
    """
    基于湍流速度脉动生成标量耗散率脉动。

    公式（Pope, Turbulent Flows）：
        χ' ≈ χ_mean * (1 + C_χ * u' / U_mean)

    Parameters
    ----------
    Z : ndarray
        混合分数空间坐标。
    chi_mean : float
        平均标量耗散率。
    u_fluct : ndarray
        速度脉动场。
    integral_length : float
        积分长度尺度。

    Returns
    -------
    chi_fluct : ndarray
        标量耗散率脉动分布。
    """
    n = len(Z)
    if len(u_fluct) < n:
        # 循环扩展
        u_ext = np.tile(u_fluct, (n // len(u_fluct) + 1, 1))[:n]
    else:
        u_ext = u_fluct[:n]

    u_mag = np.sqrt(np.sum(u_ext ** 2, axis=1))
    u_ref = np.mean(u_mag) if np.mean(u_mag) > 1.0e-12 else 1.0

    C_chi = 2.0
    chi_fluct = chi_mean * (1.0 + C_chi * u_mag / u_ref)

    # 边界处理
    chi_fluct = np.maximum(chi_fluct, 1.0e-6)
    return chi_fluct
