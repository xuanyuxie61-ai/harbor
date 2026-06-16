"""
bz_sampling.py — Brillouin 区采样与 CRT 索引映射
=================================================

融合种子项目:
  - 170_chinese_remainder_theorem: CRT 重构
    f = sum s_i * t_i * r_i, s_i = M/m_i, Bezout: s_i*t_i + m_i*u_i = 1
  - 067_ball_grid: 球内网格生成 -> Brillouin 球内 k 点采样
  - 341_eternity_tile: 邻接矩阵 -> 不可约 Brillouin 区边界

物理背景:
  Brillouin 区 (BZ) 采样是声子计算的核心:
    omega_n(k) = eigenvalues of D(k)
    热力学量 = (1/V_BZ) * integral_BZ f(omega_n(k)) dk

  Monkhorst-Pack 方案: 均匀 k 网格, 利用对称性减少独立 k 点数。
  CRT 索引: 三维网格索引 (i,j,k) 可用中国剩余定理
    映射为一维索引 n, 提高内存局部性。

  不可约 Brillouin 区 (IBZ):
    利用晶格的点群对称性, 仅需在 1/|G| 的 BZ 内采样。
    对 FCC: |G| = 48 (Oh 群)
    对 BCC: |G| = 48
    对 Diamond: |G| = 48 (Oh)
"""

import numpy as np
from typing import Tuple, List, Dict
from math import gcd


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    扩展欧几里得算法 (融合 chinese_remainder_theorem)。
    返回 (g, x, y) 满足 a*x + b*y = g = gcd(a,b)
    """
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def crt_reconstruct(moduli: List[int], remainders: List[int]) -> int:
    """
    中国剩余定理重构 (融合 chinese_remainder_theorem 的 crt.m)。

    给定互素模数 m_1, ..., m_k 和余数 r_1, ..., r_k,
    求唯一 f 满足 f mod m_i = r_i, 0 <= f < M = prod(m_i)。

    算法:
      1. s_i = M / m_i
      2. t_i: s_i * t_i ≡ 1 (mod m_i) [Bezout 系数]
      3. f = sum(s_i * t_i * r_i) mod M

    参数:
        moduli: 互素模数列表
        remainders: 余数列表

    返回:
        f: CRT 重构结果
    """
    # 验证互素性
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError(
                    f"模数 {moduli[i]} 和 {moduli[j]} 不互素 (gcd={gcd(moduli[i], moduli[j])})"
                )

    M = 1
    for m in moduli:
        M *= m

    f = 0
    for m_i, r_i in zip(moduli, remainders):
        s_i = M // m_i
        g, t_i, _ = extended_gcd(s_i % m_i, m_i)
        if g != 1:
            raise ValueError(f"Bezout 系数不存在: gcd({s_i % m_i}, {m_i}) = {g}")
        f += s_i * t_i * r_i

    return f % M


def crt_index_to_grid(
    linear_index: int,
    grid_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    使用 CRT 将一维索引映射到三维网格 (当 N1, N2, N3 互素时)。

    若 N1, N2, N3 互素, 则 Z/(N1*N2*N3)Z ≅ Z/N1 × Z/N2 × Z/N3

    参数:
        linear_index: 一维索引 [0, N1*N2*N3 - 1]
        grid_shape: (N1, N2, N3) 互素网格尺寸

    返回:
        (i, j, k): 三维网格索引
    """
    n1, n2, n3 = grid_shape
    # CRT: linear_index mod n1, mod n2, mod n3
    i = linear_index % n1
    j = linear_index % n2
    k = linear_index % n3
    return i, j, k


def grid_to_crt_index(
    i: int, j: int, k: int,
    grid_shape: Tuple[int, int, int],
) -> int:
    """
    CRT 反向映射: 三维网格索引 -> 一维索引。
    """
    n1, n2, n3 = grid_shape
    return crt_reconstruct([n1, n2, n3], [i, j, k])


def generate_irreducible_bz_points(
    grid_shape: Tuple[int, int, int],
    point_group: str = 'Oh',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成不可约 Brillouin 区 k 点 (利用点群对称性)。

    点群操作将 BZ 分为 |G| 个等价区域,
    仅保留不等价 k 点。

    参数:
        grid_shape: MP 网格尺寸
        point_group: 点群类型 ('Oh', 'D6h', 'C4v', ...)

    返回:
        k_irred: (N_irred, 3) 不可约 k 点
        weights: (N_irred,) 每个 k 点的权重
    """
    # 生成全 MP 网格
    n1, n2, n3 = grid_shape
    k_all = []
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                k = np.array([
                    (2 * i1 - n1 + 1) / (2.0 * n1),
                    (2 * i2 - n2 + 1) / (2.0 * n2),
                    (2 * i3 - n3 + 1) / (2.0 * n3),
                ])
                k_all.append(k)
    k_all = np.array(k_all)

    # 获取点群对称操作
    ops = get_point_group_operations(point_group)
    n_ops = len(ops)

    # 标记等价类
    assigned = np.zeros(len(k_all), dtype=bool)
    k_irred = []
    weights = []

    for idx in range(len(k_all)):
        if assigned[idx]:
            continue
        k_ref = k_all[idx]
        star_count = 0

        for op_idx in range(len(k_all)):
            if assigned[op_idx]:
                continue
            k_test = k_all[op_idx]
            # 检查是否与 k_ref 等价
            is_equiv = False
            for R in ops:
                k_rot = R @ k_ref
                diff = k_rot - k_test
                diff -= np.round(diff)  # 模倒格矢
                if np.linalg.norm(diff) < 1e-10:
                    is_equiv = True
                    break
            if is_equiv:
                assigned[op_idx] = True
                star_count += 1

        k_irred.append(k_ref)
        weights.append(star_count)

    k_irred = np.array(k_irred)
    weights = np.array(weights, dtype=float)
    weights /= np.sum(weights)

    return k_irred, weights


def get_point_group_operations(point_group: str) -> List[np.ndarray]:
    """
    获取点群的对称操作矩阵 (3x3 正交矩阵)。

    Oh 群 (48 个操作): FCC/Diamond 的点群
    包含: E, 8C3, 6C4, 3C2, 6C2', i, 8S6, 6S4, 3sigma_h, 6sigma_d
    """
    ops = []
    if point_group == 'Oh':
        # 生成 Oh 群的所有 48 个操作
        # 先生成 O 群 (24 个纯旋转)
        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                          (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
                R = np.zeros((3, 3))
                for i in range(3):
                    R[i, perm[i]] = signs[i]
                if abs(np.linalg.det(R) - 1.0) < 0.1:  # 纯旋转
                    ops.append(R)
                if abs(np.linalg.det(R) + 1.0) < 0.1:  # 反演 * 旋转
                    ops.append(-R)
    elif point_group == 'D6h':
        # 简化: 使用 12 个主要操作
        for angle in np.arange(0, 2 * np.pi, np.pi / 3):
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            ops.append(R)
            ops.append(R @ np.diag([1, -1, 1]))
            ops.append(-R)
            ops.append(-R @ np.diag([1, -1, 1]))
    else:
        # 默认: 仅恒等
        ops.append(np.eye(3))

    # 去重
    unique_ops = [ops[0]]
    for R in ops[1:]:
        is_new = True
        for Ru in unique_ops:
            if np.linalg.norm(R - Ru) < 1e-10:
                is_new = False
                break
        if is_new:
            unique_ops.append(R)

    return unique_ops


def high_symmetry_path(
    lattice_type: str,
    n_points_per_segment: int = 50,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    高对称性 k 路径 (用于声子色散绘图)。

    常见路径:
      FCC:  Gamma - X - W - K - Gamma - L - U - W - L - K
      BCC:  Gamma - H - N - Gamma - P - H
      SC:   Gamma - X - M - Gamma - R - X
      Diamond: 同 FCC

    返回:
        k_path: (N_total, 3) k 点路径
        labels: 高对称点名称
        boundary_indices: 各段分界索引
    """
    paths = {
        'fcc': {
            'points': {
                'Gamma': [0.0, 0.0, 0.0],
                'X': [0.5, 0.0, 0.5],
                'W': [0.5, 0.25, 0.75],
                'K': [0.375, 0.375, 0.75],
                'L': [0.5, 0.5, 0.5],
                'U': [0.625, 0.25, 0.625],
            },
            'path': ['Gamma', 'X', 'W', 'K', 'Gamma', 'L', 'U', 'W', 'L', 'K'],
        },
        'bcc': {
            'points': {
                'Gamma': [0.0, 0.0, 0.0],
                'H': [0.5, -0.5, 0.5],
                'N': [0.0, 0.0, 0.5],
                'P': [0.25, 0.25, 0.25],
            },
            'path': ['Gamma', 'H', 'N', 'Gamma', 'P', 'H'],
        },
        'sc': {
            'points': {
                'Gamma': [0.0, 0.0, 0.0],
                'X': [0.5, 0.0, 0.0],
                'M': [0.5, 0.5, 0.0],
                'R': [0.5, 0.5, 0.5],
            },
            'path': ['Gamma', 'X', 'M', 'Gamma', 'R', 'X'],
        },
        'diamond': {
            'points': {
                'Gamma': [0.0, 0.0, 0.0],
                'X': [0.5, 0.0, 0.5],
                'W': [0.5, 0.25, 0.75],
                'K': [0.375, 0.375, 0.75],
                'L': [0.5, 0.5, 0.5],
                'U': [0.625, 0.25, 0.625],
            },
            'path': ['Gamma', 'X', 'W', 'K', 'Gamma', 'L', 'U', 'W', 'L', 'K'],
        },
    }

    if lattice_type not in paths:
        lattice_type = 'fcc'

    info = paths[lattice_type]
    k_points_list = []
    labels = []
    boundaries = [0]

    for seg_idx in range(len(info['path']) - 1):
        p1_name = info['path'][seg_idx]
        p2_name = info['path'][seg_idx + 1]
        p1 = np.array(info['points'][p1_name])
        p2 = np.array(info['points'][p2_name])

        if seg_idx == 0:
            labels.append(p1_name)

        for i in range(n_points_per_segment):
            t = i / n_points_per_segment
            k_points_list.append(p1 + t * (p2 - p1))

        boundaries.append(len(k_points_list))
        labels.append(p2_name)

    k_path = np.array(k_points_list)
    return k_path, labels, boundaries


def ball_grid_bz_sampling(
    radius: float,
    n_divisions: int = 10,
) -> np.ndarray:
    """
    球坐标 BZ 采样 (融合 ball_grid 的球内网格策略)。

    用于计算 DOS 时对 BZ 球内的体积积分。
    使用正八分面枚举 + 反射对称。
    """
    points = []
    for i in range(n_divisions + 1):
        for j in range(n_divisions + 1):
            for k in range(n_divisions + 1):
                x = radius * 2 * i / (2 * n_divisions + 1)
                y = radius * 2 * j / (2 * n_divisions + 1)
                z = radius * 2 * k / (2 * n_divisions + 1)
                r2 = x ** 2 + y ** 2 + z ** 2
                if r2 <= radius ** 2:
                    # 正卦限点
                    points.append([x, y, z])
                    if i > 0:
                        points.append([-x, y, z])
                    if j > 0:
                        points.append([x, -y, z])
                    if k > 0:
                        points.append([x, y, -z])
                    if i > 0 and j > 0:
                        points.append([-x, -y, z])
                    if i > 0 and k > 0:
                        points.append([-x, y, -z])
                    if j > 0 and k > 0:
                        points.append([x, -y, -z])
                    if i > 0 and j > 0 and k > 0:
                        points.append([-x, -y, -z])

    # 去重
    points = np.array(points)
    unique_points = np.unique(np.round(points, 10), axis=0)
    return unique_points
