"""
sampling_quality.py
蒙特卡洛相空间采样质量评估体系

基于 958_quality 项目重构:
  将几何点集质量度量转化为高维相空间采样质量评估
  
评估指标:
  - Gamma 度量: 最近邻距离的最大最小比 (均匀性)
  - Beta 度量: 最近邻距离的变异系数 (离散度)
  - R0 能量: Riesz s=0 对数能量 (点间排斥势)
  - Chi 度量: Voronoi  cell 面积方差
  - D/H 度量: 覆盖半径与填充半径比
"""
import numpy as np
from constants import TINY

# ============================================================
# 1. 最近邻距离计算
# ============================================================
def nearest_neighbor_distances(points):
    """
    计算每个点的最近邻距离
    
    对于 N 个点在 D 维空间，使用 O(N^2) 精确计算
    d_{ij} = ||x_i - x_j||_2
    """
    n = points.shape[0]
    nn_dist = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(points[i] - points[j])
                if dist < nn_dist[i]:
                    nn_dist[i] = dist
    return nn_dist


# ============================================================
# 2. Gamma 度量 (均匀性指标)
# ============================================================
def gamma_measure(points):
    """
    Gamma 度量 = d_max / d_min, 其中 d 为最近邻距离
    
    理论下界: Gamma >= 1
    理想均匀分布: Gamma -> 1
    
    边界处理:
      - 若所有点重合 (d_min = 0), 返回 inf 并警告
    """
    nn = nearest_neighbor_distances(points)
    d_min = np.min(nn)
    d_max = np.max(nn)
    if d_min < TINY:
        return np.inf
    return d_max / d_min


# ============================================================
# 3. Beta 度量 (变异系数)
# ============================================================
def beta_measure(points):
    """
    Beta 度量 = sigma_d / mu_d, 最近邻距离的标准差/均值
    
    理想均匀采样: Beta -> 0
    """
    nn = nearest_neighbor_distances(points)
    mu = np.mean(nn)
    if mu < TINY:
        return np.inf
    return np.std(nn) / mu


# ============================================================
# 4. R0 能量 (对数 Riesz 能量)
# ============================================================
def r0_measure(points):
    """
    Riesz s=0 能量 (对数能量):
      E_0 = (2 / (N*(N-1))) * sum_{i<j} log(1 / d_{ij})
          = -(2 / (N*(N-1))) * sum_{i<j} log(d_{ij})
    
    物理意义: 点集在库仑势下的平均势能，越小表示分布越分散均匀
    
    边界处理: d_{ij} < TINY 时截断为 TINY 避免 log(0)
    """
    n = points.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            dist = max(dist, TINY)
            total += np.log(dist)
            count += 1
    return -total / count if count > 0 else 0.0


# ============================================================
# 5. Chi 度量 (Voronoi 单元近似)
# ============================================================
def chi_measure(points, n_samples=5000):
    """
    使用蒙特卡洛采样近似 Voronoi 单元体积方差
    
    方法:
      1. 在包围盒内均匀采样 n_samples 个点
      2. 将每个采样点分配给最近的生成点 (find_closest)
      3. 统计每个生成点分配的样本数 N_i
      4. Chi = std(N_i) / mean(N_i)
    
    理想均匀: Chi -> 0
    """
    n = points.shape[0]
    dim = points.shape[1]
    
    # 构建包围盒
    lows = np.min(points, axis=0)
    highs = np.max(points, axis=0)
    # 扩展边界
    margin = 0.05 * (highs - lows)
    margin[margin < TINY] = TINY
    lows -= margin
    highs += margin
    
    # 蒙特卡洛采样
    samples = np.random.uniform(0.0, 1.0, (n_samples, dim))
    for d in range(dim):
        span = highs[d] - lows[d]
        if span > TINY:
            samples[:, d] = lows[d] + samples[:, d] * span
    
    # 分配到最近的生成点
    counts = np.zeros(n)
    for s in samples:
        dists = np.linalg.norm(points - s, axis=1)
        idx = np.argmin(dists)
        counts[idx] += 1.0
    
    mean_count = np.mean(counts)
    if mean_count < TINY:
        return np.inf
    return np.std(counts) / mean_count


# ============================================================
# 6. Q 度量 (三角剖面质量, 2D/3D 专用)
# ============================================================
def q_measure_2d(points):
    """
    2D 点集的 Q 度量: 基于 Delaunay 三角剖面的内切圆/外接圆半径比
    
    由于实现完整 Delaunay 较复杂，此处用近似:
      对每个点，取最近邻三点构成近似三角形，计算其质量
    
    单个三角形质量:
      q = 4 * sqrt(3) * A / (a^2 + b^2 + c^2)
      其中 A 为面积, a,b,c 为边长
      等边三角形: q = 1; 退化三角形: q -> 0
    """
    if points.shape[1] != 2:
        return None
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    q_values = []
    for i in range(n):
        # 找最近邻的两个不同点
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        idx = np.argsort(dists)[:2]
        if dists[idx[1]] == np.inf:
            continue
        p0, p1, p2 = points[i], points[idx[0]], points[idx[1]]
        
        # 边长
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p0 - p2)
        c = np.linalg.norm(p0 - p1)
        if a < TINY or b < TINY or c < TINY:
            continue
        
        # 海伦公式面积
        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(max(area_sq, 0.0))
        
        denom = a * a + b * b + c * c
        if denom > TINY:
            q = 4.0 * np.sqrt(3.0) * area / denom
            q_values.append(q)
    
    return np.min(q_values) if q_values else 0.0


# ============================================================
# 7. 综合采样质量报告
# ============================================================
def sampling_quality_report(points):
    """
    对相空间采样点集进行全面质量评估
    
    参数:
        points: numpy array, shape (N, D)
    返回:
        dict: 各项质量指标
    """
    report = {
        "n_points": points.shape[0],
        "dimension": points.shape[1],
        "gamma": gamma_measure(points),
        "beta": beta_measure(points),
        "r0_energy": r0_measure(points),
        "chi": chi_measure(points),
    }
    if points.shape[1] == 2:
        report["q_2d"] = q_measure_2d(points)
    return report


def evaluate_phase_space_sampling(events):
    """
    对生成的事件相空间进行采样质量评估
    
    提取特征: (m_z1, m_z2, cos_theta1, cos_theta2)
    其中 theta 为 Z 衰变轴相对于母粒子运动方向的夹角
    """
    features = []
    for e in events:
        # 简化特征: Z1质量、Z2质量、以及两个Z的动量大小
        pz1_mag = np.linalg.norm(e["pz1"][1:])
        pz2_mag = np.linalg.norm(e["pz2"][1:])
        features.append([e["m_z1"], e["m_z2"], pz1_mag, pz2_mag])
    
    points = np.array(features)
    # 标准化到单位超立方体以便度量计算
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = maxs - mins
    spans[spans < TINY] = 1.0
    normalized = (points - mins) / spans
    
    return sampling_quality_report(normalized)
