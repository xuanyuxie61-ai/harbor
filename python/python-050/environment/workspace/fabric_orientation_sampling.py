"""
fabric_orientation_sampling.py
冰晶优选取向随机采样 — 正半球蒙特卡洛统计

基于种子项目 1125_sphere_positive_distance 的球面蒙特卡洛采样方法，
应用于冰晶 c 轴取向分布的统计分析与各向异性张量构建。

核心数学:
  1. 冰晶 c 轴取向分布函数 (Orientation Distribution Function, ODF):
       对于单轴拉伸变形，ODF 常用 Watson 分布描述:

       g(\mathbf{c}) = \frac{1}{4\pi} \frac{\kappa^{1/2}}{{}_1F_1(1/2, 3/2, \kappa)}
                       \exp\left( \kappa (\mathbf{c} \cdot \mathbf{m})^2 \right)

       其中 \mathbf{m} 为优选取向方向，\kappa 为集中参数，
       {}_1F_1 为合流超几何函数 (Kummer 函数)。

  2. 二阶取向张量:
       a_{ij}^{(2)} = \oint_{S^2} c_i c_j \, g(\mathbf{c}) \, d\Omega

       特征值分解: \lambda_1 \ge \lambda_2 \ge \lambda_3,
       满足 \lambda_1 + \lambda_2 + \lambda_3 = 1。

  3. 正半球 Monte Carlo 估计:
       由于冰晶 c 轴无方向性 (c \equiv -c)，只需在上半球采样。
       均匀采样方法: 生成 x \sim N(0, I_3)，取 p = |x| / \|x\| (限制到正象限)。

  4. 距离统计 (类比种子项目):
       对随机取向对 (c_1, c_2)，计算夹角 \theta:
       d_{angle} = \arccos(|\mathbf{c}_1 \cdot \mathbf{c}_2|)

       统计量: 平均夹角、方差、分布矩。

应用场景:
  - 由冰芯样本估计晶格各向异性
  - 为 Glen 流动律提供增强因子输入
  - 验证 ODF 理论模型与实测 EBSD 数据
"""

import numpy as np
from typing import Tuple


def uniform_on_positive_hemisphere(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    在三维单位球面的正半球 (z >= 0) 上均匀随机采样。

    方法:
        x \sim N(0, I_3),  p = x / \|x\|,  然后取 z = |p_z| 投影到正半球。
        对于全正卦限 (x>0, y>0, z>0)，取各分量绝对值。

    参数:
        n_samples: 采样点数
        seed: 随机种子

    返回:
        points: (n_samples, 3) 单位向量，z >= 0
    """
    rng = np.random.default_rng(seed)
    xyz = rng.standard_normal((n_samples, 3), dtype=np.float64)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = xyz / norms

    # 取绝对值映射到正卦限 (first octant)
    points = np.abs(points)

    # 再归一化确保单位长度
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return points


def watson_odf_sample(n_samples: int, concentration: float = 5.0,
                      preferred_direction: np.ndarray = None,
                      seed: int = 42) -> np.ndarray:
    """
    从 Watson 分布采样晶格取向。

    Watson 分布 (girdle 型, \kappa < 0) 或 (cluster 型, \kappa > 0)。
    这里采用 rejection sampling 的近似方法。

    参数:
        n_samples: 采样数
        concentration: \kappa 参数 (>0 为 cluster)
        preferred_direction: 优选方向 (3,), 默认 z 轴
        seed: 随机种子

    返回:
        samples: (n_samples, 3)
    """
    rng = np.random.default_rng(seed)
    if preferred_direction is None:
        m = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        m = np.asarray(preferred_direction, dtype=np.float64)
        m = m / np.linalg.norm(m)

    samples = []
    max_trials = n_samples * 100
    trial = 0

    # 简单 rejection sampling: 在半球均匀采样，按概率接受
    while len(samples) < n_samples and trial < max_trials:
        trial += 1
        # 在半球上采样 (z >= 0)
        xyz = rng.standard_normal(3)
        xyz[2] = abs(xyz[2])
        c = xyz / np.linalg.norm(xyz)

        # Watson 概率密度 (正比于)
        prob = np.exp(concentration * (np.dot(c, m) ** 2))
        if rng.random() < prob / np.exp(concentration):
            samples.append(c)

    if len(samples) < n_samples:
        # 若 rejection sampling 效率低，补充均匀样本
        while len(samples) < n_samples:
            xyz = rng.standard_normal(3)
            xyz[2] = abs(xyz[2])
            samples.append(xyz / np.linalg.norm(xyz))

    return np.array(samples, dtype=np.float64)


def compute_second_order_tensor(orientations: np.ndarray) -> np.ndarray:
    """
    计算二阶取向张量 a^{(2)}_{ij} = <c_i c_j>。

    参数:
        orientations: (N, 3) 单位取向向量

    返回:
        a2: (3, 3) 对称张量
    """
    c = np.asarray(orientations, dtype=np.float64)
    N = len(c)
    if N == 0:
        return np.eye(3, dtype=np.float64) / 3.0

    # 外积平均
    a2 = np.zeros((3, 3), dtype=np.float64)
    for ci in c:
        a2 += np.outer(ci, ci)
    a2 /= N

    # 对称化与迹归一化
    a2 = 0.5 * (a2 + a2.T)
    trace = np.trace(a2)
    if trace > 0:
        a2 = a2 / trace
    return a2


def angular_distance_stats(orientations: np.ndarray) -> dict:
    """
    计算取向对之间的角距离统计量。

    角距离定义:
        d_{ij} = \arccos(|\mathbf{c}_i \cdot \mathbf{c}_j|)

    参数:
        orientations: (N, 3) 取向向量

    返回:
        stats: {'mean_angle_rad', 'std_angle_rad', 'mean_cos2', 'j_index'}
    """
    c = np.asarray(orientations, dtype=np.float64)
    N = len(c)
    if N < 2:
        return {
            'mean_angle_rad': np.pi / 2.0,
            'std_angle_rad': 0.0,
            'mean_cos2': 1.0 / 3.0,
            'j_index': 1.0,
        }

    # 计算所有对的内积 (为避免 O(N^2) 过大，采样估计)
    max_pairs = 10000
    if N * (N - 1) // 2 > max_pairs:
        rng = np.random.default_rng(42)
        idx1 = rng.integers(0, N, max_pairs)
        idx2 = rng.integers(0, N, max_pairs)
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]
    else:
        idx1, idx2 = np.triu_indices(N, k=1)

    dots = np.abs(np.sum(c[idx1] * c[idx2], axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    angles = np.arccos(dots)

    mean_angle = float(np.mean(angles))
    std_angle = float(np.std(angles))
    mean_cos2 = float(np.mean(dots ** 2))

    # J 指数 (无序度指标, J=1 为完全随机, J->\infty 为完全有序)
    j_index = float(np.mean(1.0 / (dots + 0.1)))

    return {
        'mean_angle_rad': mean_angle,
        'std_angle_rad': std_angle,
        'mean_cos2': mean_cos2,
        'j_index': j_index,
    }


def fabric_anisotropy_indices(a2_tensor: np.ndarray) -> dict:
    """
    由二阶取向张量计算各向异性指标。

    特征值 \lambda_1 \ge \lambda_2 \ge \lambda_3 (满足和为 1):

    1. 单晶度 (Single Maximum):
       S = 2\lambda_1 - 1

    2. 环带度 (Girdle):
       G = 2(\lambda_1 + \lambda_2) - 1 = 1 - 2\lambda_3

    3. 强度因子 ( eigenvalue 方差):
       I_s = \sqrt{ \frac{3}{2} \sum (\lambda_i - 1/3)^2 }

    参数:
        a2_tensor: (3, 3) 二阶取向张量

    返回:
        indices: 各向异性指标字典
    """
    a2 = np.asarray(a2_tensor, dtype=np.float64)
    a2 = 0.5 * (a2 + a2.T)

    evals = np.linalg.eigvalsh(a2)
    evals = np.sort(evals)[::-1]  # 降序

    # 归一化确保和为 1
    evals = evals / np.maximum(np.sum(evals), 1e-15)

    S = 2.0 * evals[0] - 1.0
    G = 1.0 - 2.0 * evals[2]
    I_s = np.sqrt(1.5 * np.sum((evals - 1.0 / 3.0) ** 2))

    return {
        'eigenvalues': evals.tolist(),
        'single_maximum': float(S),
        'girdle': float(G),
        'strength_index': float(I_s),
    }


def monte_carlo_fabric_simulation(n_samples: int = 10000,
                                   concentration: float = 5.0,
                                   seed: int = 42) -> dict:
    """
    完整的蒙特卡洛冰晶取向模拟流程。

    返回:
        results: 包含取向样本、二阶张量、统计量、各向异性指标
    """
    orientations = watson_odf_sample(n_samples, concentration, seed=seed)
    a2 = compute_second_order_tensor(orientations)
    stats = angular_distance_stats(orientations)
    indices = fabric_anisotropy_indices(a2)

    return {
        'orientations': orientations,
        'second_order_tensor': a2,
        'angular_stats': stats,
        'anisotropy_indices': indices,
    }
