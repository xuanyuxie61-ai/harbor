"""
sampling_pattern.py
===================
基于素数理论和圆几何的非相干采样模式设计模块

科学背景：
---------
在压缩感知（Compressed Sensing, CS）框架下，测量矩阵 \Phi 必须满足约束等距性
（Restricted Isometry Property, RIP）。Candès 和 Tao 证明了：若采样矩阵的行数 m
满足 m \geq C \cdot s \cdot \log(N/s)（其中 s 为稀疏度，N 为信号维度），则高概率
恢复原始信号。

对于图像重建问题，k-空间采样轨迹的设计至关重要。本模块结合：
1. 素数长度采样模式（来自 599_is_prime）：利用素数 p 生成长度为 p 的采样序列，
   避免周期性混叠，保证采样模式的互不相干性（mutual incoherence）。
2. 圆几何采样（来自 185_circles）：在 k-空间中构造同心圆环采样轨迹，
   模拟 MRI 中的径向采样（radial sampling），其数学描述为：
   
       k_r(\theta) = r \cdot (\cos\theta, \sin\theta), \quad \theta \in [0, 2\pi)

核心公式：
---------
- 不相干系数：\mu(\Phi, \Psi) = \sqrt{N} \max_{i,j} |\langle \phi_i, \psi_j \rangle|
- 素数采样序列：s_n = (n \cdot p_k) \mod N,  n = 0, 1, ..., m-1
  其中 p_k 为第 k 个素数，保证序列在 Z_N 上均匀分布。
- 径向采样密度：\rho(r) \propto r^{d-1}  （d 为空间维度）
"""

import numpy as np
from typing import Tuple, List


def is_prime(n: int) -> bool:
    """
    判断整数 n 是否为素数（来自项目 599_is_prime）。

    数学定义：
        素数是大于 1 的自然数，除了 1 和它本身外没有其他正因数。
        即：\forall d \in \mathbb{Z}^+, 1 < d < n \Rightarrow n \mod d \neq 0

    参数:
        n: 待检测的整数
    返回:
        若 n 为素数则返回 True，否则返回 False
    """
    if not isinstance(n, int):
        raise TypeError("is_prime(): 输入必须是整数")
    if n < 0:
        return False
    if n == 0 or n == 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # 只需检查到 sqrt(n)
    bound = int(np.sqrt(n)) + 1
    for i in range(3, bound, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(count: int) -> List[int]:
    """
    生成前 count 个素数序列。

    参数:
        count: 需要生成的素数个数
    返回:
        前 count 个素数的列表
    """
    if count <= 0:
        return []
    primes = []
    candidate = 2
    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def prime_sampling_indices(signal_length: int, num_samples: int, prime_index: int = 5) -> np.ndarray:
    """
    利用素数生成不相干采样索引。

    算法原理：
        选取第 prime_index 个素数 p，构造采样序列
            s_k = (k \cdot p) \mod N,  k = 0, 1, ..., m-1
        由于 p 与 N 互素（当 p 为素数且 p \nmid N 时），序列 s_k 在 Z_N 中均匀分布。

    参数:
        signal_length: 信号总长度 N
        num_samples: 采样点数 m
        prime_index: 使用的素数序号（默认为第 5 个素数 11）
    返回:
        采样索引数组，形状为 (num_samples,)
    """
    if signal_length <= 0:
        raise ValueError("signal_length 必须为正整数")
    if num_samples <= 0 or num_samples > signal_length:
        raise ValueError("num_samples 必须在 [1, signal_length] 范围内")

    primes = generate_primes(prime_index + 1)
    p = primes[prime_index]

    # 确保素数与信号长度互素；若不互素，尝试下一个素数
    while np.gcd(p, signal_length) != 1 and prime_index < len(primes) - 1:
        prime_index += 1
        p = primes[prime_index]

    indices = np.mod(np.arange(num_samples) * p, signal_length)
    # 去重并保持顺序
    _, unique_idx = np.unique(indices, return_index=True)
    indices = np.sort(indices[np.sort(unique_idx)])

    # 若去重后不足 num_samples，补充随机采样
    if len(indices) < num_samples:
        remaining = np.setdiff1d(np.arange(signal_length), indices)
        extra = np.random.choice(remaining, num_samples - len(indices), replace=False)
        indices = np.sort(np.concatenate([indices, extra]))

    return indices.astype(int)


def circle_kspace_sampling(image_shape: Tuple[int, int], num_radial_lines: int,
                           samples_per_line: int, max_radius: float = None) -> np.ndarray:
    """
    基于圆几何的 k-空间径向采样轨迹设计（来自项目 185_circles）。

    数学模型：
        在二维 k-空间中，采样点位于同心圆上：
            k_x = r \cos\theta_j, \quad k_y = r \sin\theta_j
        其中角度 \theta_j = 2\pi j / N_{\theta}, j = 0, ..., N_{\theta}-1
        半径 r 按黄金角增量分布，以最小化相干伪影：
            \Delta\theta = \pi \cdot (3 - \sqrt{5}) \approx 2.39996 \text{ rad}

    参数:
        image_shape: 图像尺寸 (H, W)
        num_radial_lines: 径向线数量 N_\theta
        samples_per_line: 每条径向线上的采样点数
        max_radius: 最大采样半径（默认取 min(H,W)/2）
    返回:
        采样坐标数组，形状为 (num_radial_lines * samples_per_line, 2)，
        每行为 (kx, ky) 的浮点坐标
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape 必须是二元组 (H, W)")
    H, W = image_shape
    if H <= 0 or W <= 0:
        raise ValueError("图像尺寸必须为正")
    if num_radial_lines <= 0 or samples_per_line <= 0:
        raise ValueError("径向线数量和每线采样点数必须为正")

    if max_radius is None:
        max_radius = min(H, W) / 2.0

    # 黄金角分布
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    coords = []
    for j in range(num_radial_lines):
        theta = j * golden_angle
        # 半径从中心向外均匀分布，考虑径向采样密度 \rho(r) \propto r
        radii = np.linspace(0.0, max_radius, samples_per_line)
        # 避免全零半径导致重复
        if samples_per_line > 1:
            radii[0] = 1e-6
        for r in radii:
            kx = r * np.cos(theta)
            ky = r * np.sin(theta)
            coords.append([kx, ky])

    return np.array(coords, dtype=float)


def build_incoherent_mask(image_shape: Tuple[int, int], sampling_ratio: float) -> np.ndarray:
    """
    构造二维非相干采样掩码，结合素数采样与径向采样的混合策略。

    采样策略：
        1. 低频区域（中心区域）全采样，因为低频包含主要能量；
        2. 高频区域采用径向稀疏采样，利用圆几何设计；
        3. 在角向利用素数序列避免周期性混叠。

    参数:
        image_shape: 图像尺寸 (H, W)
        sampling_ratio: 采样比例，范围 (0, 1]
    返回:
        二维布尔掩码，True 表示该频域位置被采样
    """
    if not (0.0 < sampling_ratio <= 1.0):
        raise ValueError("sampling_ratio 必须在 (0, 1] 范围内")

    H, W = image_shape
    mask = np.zeros((H, W), dtype=bool)

    # 中心低频全采样（约占总采样能量的 80%）
    low_freq_ratio = 0.15
    cy, cx = H // 2, W // 2
    low_r = int(min(H, W) * low_freq_ratio / 2)
    y_grid, x_grid = np.ogrid[:H, :W]
    center_mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= low_r ** 2
    mask[center_mask] = True

    # 高频区域采用径向稀疏采样
    num_radial = max(8, int(np.sqrt(H * W * sampling_ratio) / 4))
    samples_per_radial = max(4, int(min(H, W) * sampling_ratio * 2))
    coords = circle_kspace_sampling(image_shape, num_radial, samples_per_radial)

    # 将浮点坐标映射到离散网格
    for kx, ky in coords:
        ix = int(round(kx)) + cx
        iy = int(round(ky)) + cy
        if 0 <= ix < W and 0 <= iy < H:
            mask[iy, ix] = True

    # 若采样比例仍不足，在剩余位置按素数步长补充
    total_needed = int(H * W * sampling_ratio)
    current = np.count_nonzero(mask)
    if current < total_needed:
        remaining = np.column_stack(np.where(~mask))
        if len(remaining) > 0:
            step = max(1, len(remaining) // (total_needed - current))
            # 利用素数步长选取补充采样点
            primes = generate_primes(20)
            p = primes[7 % len(primes)]
            idx = np.mod(np.arange(0, len(remaining), step) * p, len(remaining))
            idx = np.unique(idx)[:total_needed - current]
            mask[remaining[idx, 0], remaining[idx, 1]] = True

    return mask
