"""
dos_calculator.py
=================
光子态密度 (DOS) 与局域态密度 (LDOS) 计算器

融合原项目:
  - 1081_simplex_monte_carlo : 单纯形上的蒙特卡洛积分
  - 538_histogram_data_2d_sample : 二维离散 PDF/CDF 采样

本模块实现:
  1. 基于四面体/单纯形方法的布里渊区态密度积分
  2. 基于直方图采样的随机 k 点蒙特卡洛 DOS
  3. Van Hove 奇异点分析
"""

import numpy as np
from physics_core import C_0, local_density_of_states_3d


# =============================================================================
# 基于 1081_simplex_monte_carlo 的单纯形积分
# =============================================================================

def simplex_unit_sample(m, n_samples):
    """
    在单位单纯形上均匀采样 —— 基于 simplex_unit_sample.m
    
    m 维单位单纯形:
        S = {x ∈ R^m | x_i ≥ 0, Σ x_i ≤ 1}
    
    采样方法 (Rubinstein):
        生成 m+1 个指数分布随机数 e_i ~ Exp(1)
        x_j = e_j / Σ e_i
    
    Parameters
    ----------
    m : int
        空间维数
    n_samples : int
        采样点数
    
    Returns
    -------
    x : ndarray, shape (m, n_samples)
        单纯形内采样点
    """
    if m < 1 or n_samples < 1:
        raise ValueError("维数和采样数必须 >= 1")
    
    x = np.zeros((m, n_samples))
    for j in range(n_samples):
        e = -np.log(np.random.rand(m + 1))
        total = np.sum(e)
        if total < 1e-18:
            total = 1.0
        x[:, j] = e[:m] / total
    
    return x


def simplex_unit_to_general(m, n_samples, t_vertices, ref_points):
    """
    将单位单纯形映射到一般单纯形 —— 基于 simplex_unit_to_general.m
    
    线性映射:
        phy = t₀ + Σ_{v=1}^{m} (t_v - t₀) · ref_v
    
    Parameters
    ----------
    m : int
        维数
    n_samples : int
        采样点数
    t_vertices : ndarray, shape (m, m+1)
        一般单纯形顶点
    ref_points : ndarray, shape (m, n_samples)
        单位单纯形内点
    
    Returns
    -------
    phy : ndarray, shape (m, n_samples)
        一般单纯形内点
    """
    phy = np.zeros((m, n_samples))
    for dim in range(m):
        phy[dim, :] = t_vertices[dim, 0]
        for vertex in range(1, m + 1):
            phy[dim, :] += (t_vertices[dim, vertex] - t_vertices[dim, 0]) * ref_points[vertex - 1, :]
    
    return phy


def simplex_general_sample(m, n_samples, t_vertices):
    """
    在一般单纯形上均匀采样 —— 基于 simplex_general_sample.m
    
    Parameters
    ----------
    m : int
        维数
    n_samples : int
        采样点数
    t_vertices : ndarray, shape (m, m+1)
        单纯形顶点
    
    Returns
    -------
    x : ndarray, shape (m, n_samples)
        采样点
    """
    ref = simplex_unit_sample(m, n_samples)
    return simplex_unit_to_general(m, n_samples, t_vertices, ref)


def simplex_volume(m, t_vertices):
    """
    计算 m 维单纯形的体积
    
    公式:
        V = |det(T)| / m!
    
    其中 T 为 (t₁-t₀, t₂-t₀, ..., t_m-t₀) 构成的矩阵。
    
    Parameters
    ----------
    m : int
        维数
    t_vertices : ndarray, shape (m, m+1)
        顶点坐标
    
    Returns
    -------
    float
        体积
    """
    if m < 1:
        raise ValueError("维数必须 >= 1")
    
    T = np.zeros((m, m))
    for i in range(m):
        T[:, i] = t_vertices[:, i + 1] - t_vertices[:, 0]
    
    det = np.linalg.det(T)
    factorial = 1
    for i in range(2, m + 1):
        factorial *= i
    
    return abs(det) / factorial


def monte_carlo_dos_brillouin(omega_bands, k_points, n_samples=10000):
    """
    用蒙特卡洛方法计算布里渊区态密度
    
    态密度定义:
        ρ(ω) = Σ_n ∫_{BZ} δ(ω - ω_n(k)) d²k / V_BZ
    
    采用直方图法近似 δ 函数:
        δ(ω - ω₀) ≈ 1/(Δω·√π) exp(-(ω-ω₀)²/Δω²)
    
    Parameters
    ----------
    omega_bands : ndarray, shape (N_k, n_bands)
        能带频率
    k_points : ndarray, shape (N_k, 2)
        k 点坐标
    n_samples : int
        布里渊区随机采样点数
    
    Returns
    -------
    omega_bins : ndarray
        频率分箱中心
    dos : ndarray
        态密度
    """
    N_k, n_bands = omega_bands.shape
    if N_k < 2:
        raise ValueError("k 点数量必须 >= 2")
    
    # 确定频率范围
    omega_min = np.min(omega_bands)
    omega_max = np.max(omega_bands)
    if omega_max <= omega_min:
        return np.array([omega_min]), np.array([0.0])
    
    n_bins = max(50, N_k // 2)
    omega_bins = np.linspace(omega_min, omega_max, n_bins)
    dos = np.zeros(n_bins)
    
    # 高斯展宽参数
    domega = (omega_max - omega_min) / n_bins
    sigma = max(domega * 2.0, 1e-12)
    
    # 计算 k 点凸包作为布里渊区近似
    k_min = np.min(k_points, axis=0)
    k_max = np.max(k_points, axis=0)
    area_bz = (k_max[0] - k_min[0]) * (k_max[1] - k_min[1])
    
    if area_bz < 1e-18:
        return omega_bins, dos
    
    # 在 k 点凸包内随机采样
    k_samples = np.random.uniform(k_min, k_max, size=(n_samples, 2))
    
    # 对每个采样 k 点，插值得到频率
    for ks in k_samples:
        # 找到最近的 k 点并插值
        distances = np.sum((k_points - ks) ** 2, axis=1)
        idx_sorted = np.argsort(distances)[:4]  # 最近 4 个点
        
        if np.sum(distances[idx_sorted]) < 1e-18:
            weights = np.ones(len(idx_sorted)) / len(idx_sorted)
        else:
            inv_dist = 1.0 / (distances[idx_sorted] + 1e-12)
            weights = inv_dist / np.sum(inv_dist)
        
        for band in range(n_bands):
            omega_interp = np.sum(omega_bands[idx_sorted, band] * weights)
            # 加入高斯展宽
            dos += np.exp(-((omega_bins - omega_interp) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    dos /= (n_samples * n_bands)
    
    return omega_bins, dos


# =============================================================================
# 基于 538_histogram_data_2d_sample 的离散 PDF/CDF 采样
# =============================================================================

def set_discrete_cdf(n1, n2, pdf):
    """
    由离散 PDF 构建 CDF —— 基于 set_discrete_cdf.m
    
    CDF 按行优先顺序累积:
        C(i,j) = Σ_{i'≤i, j'≤j} PDF(i',j')
    
    Parameters
    ----------
    n1, n2 : int
        PDF 网格维度
    pdf : ndarray, shape (n1, n2)
        概率密度函数 (已归一化或任意正数)
    
    Returns
    -------
    cdf : ndarray, shape (n1, n2)
        累积分布函数
    """
    if n1 < 1 or n2 < 1:
        raise ValueError("维度必须 >= 1")
    pdf = np.asarray(pdf)
    if pdf.shape != (n1, n2):
        raise ValueError("pdf 形状不匹配")
    if np.any(pdf < 0):
        raise ValueError("PDF 值必须非负")
    
    cdf = np.zeros((n1, n2))
    total = 0.0
    for j in range(n2):
        for i in range(n1):
            total += pdf[i, j]
            cdf[i, j] = total
    
    # 归一化
    if total > 1e-18:
        cdf /= total
    
    return cdf


def discrete_cdf_to_xy(n1, n2, cdf, n_samples, u=None):
    """
    由离散 CDF 反演采样 —— 基于 discrete_cdf_to_xy.m
    
    给定 CDF 和均匀随机数 u∈[0,1]，找到对应的 (i,j) 单元格，
    并在该单元格内均匀随机取点。
    
    Parameters
    ----------
    n1, n2 : int
        网格维度
    cdf : ndarray, shape (n1, n2)
        累积分布函数
    n_samples : int
        采样点数
    u : ndarray, optional
        均匀随机数 [0,1]，若 None 则自动生成
    
    Returns
    -------
    xy : ndarray, shape (2, n_samples)
        采样点坐标 (归一化到 [0,1]²)
    """
    if u is None:
        u = np.random.rand(n_samples)
    else:
        u = np.asarray(u)
        if len(u) != n_samples:
            raise ValueError("u 长度必须等于 n_samples")
    
    u = np.clip(u, 0.0, 1.0)
    xy = np.zeros((2, n_samples))
    
    cdf_flat = cdf.flatten()
    
    for k in range(n_samples):
        # 二分查找 CDF 反演
        idx = np.searchsorted(cdf_flat, u[k])
        idx = min(idx, n1 * n2 - 1)
        i = idx % n1
        j = idx // n1
        
        # 在单元格内均匀采样
        r = np.random.rand(2)
        xy[0, k] = (i + r[0]) / n1
        xy[1, k] = (j + r[1]) / n2
    
    return xy


def importance_sampled_dos(omega_bands, k_points, n1=20, n2=20, n_samples=5000):
    """
    重要性采样计算态密度
    
    将能带频率分布作为 PDF，在频率高的区域加密采样。
    
    Parameters
    ----------
    omega_bands : ndarray
        能带频率
    k_points : ndarray
        k 点坐标
    n1, n2 : int
        二维 PDF 网格分辨率
    n_samples : int
        采样数
    
    Returns
    -------
    omega_bins : ndarray
        频率分箱
    dos : ndarray
        态密度
    """
    N_k, n_bands = omega_bands.shape
    
    # 构建二维频率分布直方图作为 PDF
    kx_min, kx_max = np.min(k_points[:, 0]), np.max(k_points[:, 0])
    ky_min, ky_max = np.min(k_points[:, 1]), np.max(k_points[:, 1])
    
    pdf = np.zeros((n1, n2))
    for band in range(n_bands):
        for ik in range(N_k):
            i = int(np.clip((k_points[ik, 0] - kx_min) / (kx_max - kx_min) * (n1 - 1), 0, n1 - 1))
            j = int(np.clip((k_points[ik, 1] - ky_min) / (ky_max - ky_min) * (n2 - 1), 0, n2 - 1))
            pdf[i, j] += np.sum(omega_bands[ik, band])
    
    pdf = np.maximum(pdf, 1e-12)
    cdf = set_discrete_cdf(n1, n2, pdf)
    
    # 重要性采样
    u = np.random.rand(n_samples)
    xy_samples = discrete_cdf_to_xy(n1, n2, cdf, n_samples, u)
    
    # 将采样点映射回 k 空间
    k_samples = np.zeros((n_samples, 2))
    k_samples[:, 0] = kx_min + xy_samples[0, :] * (kx_max - kx_min)
    k_samples[:, 1] = ky_min + xy_samples[1, :] * (ky_max - ky_min)
    
    # 计算 DOS
    omega_min = np.min(omega_bands)
    omega_max = np.max(omega_bands)
    n_bins = 50
    omega_bins = np.linspace(omega_min, omega_max, n_bins)
    dos = np.zeros(n_bins)
    
    sigma = (omega_max - omega_min) / n_bins * 2.0
    
    for ks in k_samples:
        distances = np.sum((k_points - ks) ** 2, axis=1)
        idx_sorted = np.argsort(distances)[:4]
        
        inv_dist = 1.0 / (distances[idx_sorted] + 1e-12)
        weights = inv_dist / np.sum(inv_dist)
        
        for band in range(n_bands):
            omega_interp = np.sum(omega_bands[idx_sorted, band] * weights)
            dos += np.exp(-((omega_bins - omega_interp) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    dos /= (n_samples * n_bands)
    
    return omega_bins, dos


# =============================================================================
# Van Hove 奇异点分析
# =============================================================================

def van_hove_singularity_type(omega_bands, k_path_distance):
    """
    识别 Van Hove 奇异点类型
    
    在二维系统中，Van Hove 奇异点对应态密度的发散/不连续:
        - M₀: 极小值点 → DOS 从 0 开始连续增长
        - M₁: 鞍点     → DOS 对数发散
        - M₂: 极大值点 → DOS 突变后下降
    
    判定依据:
        d²ω/dk² 的符号变化
    
    Parameters
    ----------
    omega_bands : ndarray, shape (N_k, n_bands)
        能带频率
    k_path_distance : ndarray, shape (N_k,)
        沿 k 路径的累积距离
    
    Returns
    -------
    singularities : list of dict
        奇异点信息列表
    """
    N_k, n_bands = omega_bands.shape
    if N_k < 5:
        return []
    
    singularities = []
    
    for band in range(n_bands):
        omega = omega_bands[:, band]
        k = k_path_distance
        
        # 二阶数值微分
        d2omega = np.zeros(N_k)
        dk = k[1] - k[0]
        if abs(dk) < 1e-18:
            continue
        
        d2omega[0] = (omega[2] - 2 * omega[1] + omega[0]) / dk ** 2
        d2omega[-1] = (omega[-1] - 2 * omega[-2] + omega[-3]) / dk ** 2
        for i in range(1, N_k - 1):
            d2omega[i] = (omega[i + 1] - 2 * omega[i] + omega[i - 1]) / dk ** 2
        
        # 寻找 d²ω/dk² 的零点 (拐点)
        for i in range(N_k - 1):
            if d2omega[i] * d2omega[i + 1] < 0:
                # 线性插值找到更精确的零点位置
                t = abs(d2omega[i]) / (abs(d2omega[i]) + abs(d2omega[i + 1]))
                k_sing = k[i] + t * (k[i + 1] - k[i])
                omega_sing = omega[i] + t * (omega[i + 1] - omega[i])
                
                # 判断类型
                if d2omega[i] > 0:
                    vh_type = 'M1_saddle'
                else:
                    vh_type = 'M0_or_M2_extremum'
                
                singularities.append({
                    'band': band,
                    'k_position': k_sing,
                    'omega': omega_sing,
                    'type': vh_type,
                    'curvature_change': 'positive_to_negative' if d2omega[i] > 0 else 'negative_to_positive'
                })
    
    return singularities
